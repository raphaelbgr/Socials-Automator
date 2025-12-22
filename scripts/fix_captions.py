#!/usr/bin/env python
"""Fix empty Instagram reel captions using Chrome browser automation.

This script connects to Chrome with remote debugging and automates the process
of editing reel captions on Instagram's web interface.

Each profile is automatically assigned a unique Chrome port and user-data-dir
to prevent conflicts when running multiple instances simultaneously.

Usage:
    py scripts/fix_captions.py                    # Interactive profile selection
    py scripts/fix_captions.py ai.for.mortals     # Fix specific profile
    py scripts/fix_captions.py --help             # Show help

Prerequisites:
    1. Run `sync-captions` first to identify empty captions
    2. Chrome browser installed
    3. pip install selenium typer rich
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

# Check dependencies
try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table
    from rich.prompt import Prompt, Confirm
except ImportError:
    print("[X] Missing dependencies. Run: pip install typer rich")
    sys.exit(1)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
except ImportError:
    print("[X] Selenium not installed. Run: pip install selenium")
    sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================

app = typer.Typer(
    name="fix-captions",
    help="Fix empty Instagram reel captions using Chrome automation",
    add_completion=False,
    invoke_without_command=True,
)
console = Console()

# Base port for Chrome debugging (profiles get port = BASE + hash % RANGE)
CHROME_PORT_BASE = 9200
CHROME_PORT_RANGE = 100

# Log file for debugging
LOG_FILE = Path("logs/caption_fix.log")


def log_debug(message: str):
    """Write debug message to log file."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {message}\n")
    except Exception:
        pass

# Chrome user data directory base
CHROME_DATA_BASE = Path.home() / "ChromeDebug"

# Tracking file for fixed captions
TRACKING_FILE = Path("docs/empty_captions/fixed_captions.json")


# =============================================================================
# DATA CLASSES
# =============================================================================


class FixStatus(str, Enum):
    """Status of a caption fix attempt."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProfileInfo:
    """Information about a profile for caption fixing."""
    name: str
    path: Path
    port: int
    user_data_dir: Path
    empty_count: int = 0
    chrome_running: bool = False


@dataclass
class ReelToFix:
    """A reel that needs caption fixing."""
    reel_path: Path
    reel_name: str
    permalink: str
    local_caption: str
    needs_regeneration: bool = False


@dataclass
class FixResult:
    """Result of fixing a single reel."""
    reel_name: str
    permalink: str
    status: FixStatus
    error: Optional[str] = None


@dataclass
class FixSummary:
    """Summary of all fix operations."""
    profile: str
    total: int
    success: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[FixResult] = field(default_factory=list)


# =============================================================================
# PORT AND CHROME PATH UTILITIES
# =============================================================================


def get_profile_port(profile_name: str) -> int:
    """Get deterministic Chrome debug port for a profile.

    Uses hash of profile name to ensure same profile always gets same port.
    """
    hash_bytes = hashlib.md5(profile_name.encode()).digest()
    hash_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    return CHROME_PORT_BASE + (hash_int % CHROME_PORT_RANGE)


def get_chrome_user_data_dir(profile_name: str) -> Path:
    """Get Chrome user data directory for a profile."""
    return CHROME_DATA_BASE / profile_name


def get_chrome_path() -> Optional[Path]:
    """Find Chrome executable path based on OS."""
    system = platform.system()

    if system == "Windows":
        paths = [
            Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "Application" / "chrome.exe",
        ]
    elif system == "Darwin":  # macOS
        paths = [
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        ]
    else:  # Linux
        paths = [
            Path("/usr/bin/google-chrome"),
            Path("/usr/bin/google-chrome-stable"),
            Path("/usr/bin/chromium"),
            Path("/usr/bin/chromium-browser"),
        ]
        # Also check PATH
        chrome_in_path = shutil.which("google-chrome") or shutil.which("chromium")
        if chrome_in_path:
            paths.insert(0, Path(chrome_in_path))

    for path in paths:
        if path.exists():
            return path

    return None


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def is_chrome_running_on_port(port: int) -> bool:
    """Check if Chrome is accessible on the debug port."""
    try:
        options = Options()
        options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")
        driver = webdriver.Chrome(options=options)
        driver.quit()
        return True
    except Exception:
        return False


# =============================================================================
# CHROME MANAGEMENT
# =============================================================================


def launch_chrome(port: int, user_data_dir: Path) -> tuple[bool, str]:
    """Launch Chrome with remote debugging enabled.

    Returns:
        tuple: (success, error_message)
    """
    chrome_path = get_chrome_path()

    if not chrome_path:
        return False, "Chrome not found. Please install Google Chrome."

    # Create user data dir if needed
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = [
        str(chrome_path),
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
    ]

    try:
        # Launch Chrome in background
        if platform.system() == "Windows":
            subprocess.Popen(
                cmd,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait for Chrome to start
        for _ in range(10):
            time.sleep(1)
            if is_port_in_use(port):
                return True, ""

        return False, "Chrome started but debug port not available"

    except Exception as e:
        return False, f"Failed to launch Chrome: {e}"


def connect_to_chrome(port: int) -> tuple[Optional[webdriver.Chrome], str]:
    """Connect to Chrome on the specified debug port.

    Returns:
        tuple: (driver or None, error_message)
    """
    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")

    try:
        driver = webdriver.Chrome(options=options)
        return driver, ""
    except WebDriverException as e:
        error = str(e)
        if "cannot connect" in error.lower():
            return None, f"Cannot connect to Chrome on port {port}. Is Chrome running with --remote-debugging-port={port}?"
        return None, f"WebDriver error: {error}"
    except Exception as e:
        return None, f"Failed to connect: {e}"


# =============================================================================
# PROFILE AND REEL DISCOVERY
# =============================================================================


def find_profiles() -> list[ProfileInfo]:
    """Find all profiles with their port assignments."""
    profiles_dir = Path("profiles")
    if not profiles_dir.exists():
        return []

    profiles = []
    for profile_dir in sorted(profiles_dir.iterdir()):
        if not profile_dir.is_dir():
            continue
        if not (profile_dir / "metadata.json").exists():
            continue

        port = get_profile_port(profile_dir.name)
        user_data_dir = get_chrome_user_data_dir(profile_dir.name)

        info = ProfileInfo(
            name=profile_dir.name,
            path=profile_dir,
            port=port,
            user_data_dir=user_data_dir,
        )

        # Count empty captions
        info.empty_count = len(find_empty_caption_reels(profile_dir))

        # Check if Chrome is running
        info.chrome_running = is_port_in_use(port)

        profiles.append(info)

    return profiles


def find_empty_caption_reels(profile_path: Path) -> list[ReelToFix]:
    """Find all reels with empty Instagram captions."""
    reels_dir = profile_path / "reels"
    if not reels_dir.exists():
        return []

    empty_reels = []

    for year_dir in reels_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            posted_dir = month_dir / "posted"
            if not posted_dir.exists():
                continue

            for reel_dir in posted_dir.glob("*"):
                if not reel_dir.is_dir():
                    continue

                metadata_path = reel_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    continue

                # Check if we have synced caption data
                instagram_data = metadata.get("instagram", {})
                actual_caption = instagram_data.get("actual_caption")

                # Only include if caption was synced AND is empty
                if actual_caption is not None and actual_caption == "":
                    platform_status = metadata.get("platform_status", {})
                    ig_status = platform_status.get("instagram", {})
                    permalink = ig_status.get("permalink", "")

                    # Load local caption
                    caption_path = reel_dir / "caption+hashtags.txt"
                    local_caption = ""
                    if caption_path.exists():
                        try:
                            with open(caption_path, encoding="utf-8") as f:
                                local_caption = f.read().strip()
                        except Exception:
                            pass

                    if permalink:
                        empty_reels.append(ReelToFix(
                            reel_path=reel_dir,
                            reel_name=reel_dir.name,
                            permalink=permalink,
                            local_caption=local_caption,
                            needs_regeneration=not local_caption,
                        ))

    return sorted(empty_reels, key=lambda x: x.reel_name)


# =============================================================================
# TRACKING
# =============================================================================


def load_tracking() -> dict:
    """Load tracking data for fixed captions."""
    if TRACKING_FILE.exists():
        try:
            with open(TRACKING_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"fixed": []}


def save_tracking(data: dict) -> None:
    """Save tracking data."""
    TRACKING_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKING_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def mark_as_fixed(tracking: dict, profile: str, reel_name: str, permalink: str) -> None:
    """Mark a reel as fixed."""
    tracking["fixed"].append({
        "profile": profile,
        "reel_name": reel_name,
        "permalink": permalink,
        "fixed_at": datetime.now().isoformat(),
    })
    save_tracking(tracking)


# =============================================================================
# CAPTION SANITIZATION
# =============================================================================


def sanitize_caption(caption: str) -> str:
    """Sanitize caption for Instagram and Selenium compatibility."""
    if not caption:
        return ""

    # Normalize Unicode to NFC
    caption = unicodedata.normalize('NFC', caption)

    # Replace problematic characters
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2014': '-',   # Em dash
        '\u2013': '-',   # En dash
        '\u2026': '...', # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\u200c': '',    # Zero-width non-joiner
        '\u200d': '',    # Zero-width joiner
        '\ufeff': '',    # BOM
        '\u00ad': '',    # Soft hyphen
        '\u2028': '\n',  # Line separator
        '\u2029': '\n',  # Paragraph separator
    }

    for old, new in replacements.items():
        caption = caption.replace(old, new)

    # Remove control characters (keep newlines and tabs)
    cleaned = []
    for char in caption:
        if char in '\n\r\t':
            cleaned.append(char)
        elif unicodedata.category(char) in ('Cc', 'Cf'):
            continue
        else:
            cleaned.append(char)

    caption = ''.join(cleaned)

    # Remove emojis (Selenium can't handle non-BMP characters)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    caption = emoji_pattern.sub('', caption)

    # Normalize whitespace
    while '\n\n\n' in caption:
        caption = caption.replace('\n\n\n', '\n\n')

    caption = re.sub(r'  +', ' ', caption)

    return caption.strip()


def remove_hashtags(caption: str) -> str:
    """Remove all hashtags from caption."""
    caption = re.sub(r'#\w+\s*', '', caption)
    caption = re.sub(r'\n\s*\n', '\n\n', caption)
    return caption.strip()


# =============================================================================
# INSTAGRAM AUTOMATION
# =============================================================================


def check_for_instagram_error(driver) -> Optional[str]:
    """Check if Instagram is showing an error message."""
    error_selectors = [
        '//*[contains(text(), "Something went wrong")]',
        '//*[contains(text(), "Try again")]',
        '//*[contains(text(), "couldn\'t be saved")]',
        '//*[@role="alert"]',
    ]

    for selector in error_selectors:
        try:
            elements = driver.find_elements(By.XPATH, selector)
            for elem in elements:
                if elem.is_displayed():
                    text = elem.text.strip()
                    if text and len(text) < 300:
                        return text
        except Exception:
            pass
    return None


def wait_and_click(driver, by, value, timeout=10) -> bool:
    """Wait for element and click it."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )
        time.sleep(0.3)
        element.click()
        return True
    except Exception:
        return False


def fix_single_reel(
    driver: webdriver.Chrome,
    reel: ReelToFix,
    use_hashtags: bool = True,
    progress_callback=None,
) -> tuple[bool, str]:
    """Fix caption for a single reel.

    Supports English and Portuguese (Brazilian) Instagram interfaces.

    Returns:
        tuple: (success, error_message)
    """
    log_debug(f"=== Starting fix for: {reel.reel_name} ===")
    log_debug(f"Permalink: {reel.permalink}")
    log_debug(f"Use hashtags: {use_hashtags}")

    caption = sanitize_caption(reel.local_caption)

    if not use_hashtags:
        caption = remove_hashtags(caption)

    log_debug(f"Caption length: {len(caption)} chars")

    def update_progress(step: str):
        if progress_callback:
            progress_callback(step)

    # Step 1: Load the reel page
    update_progress("Loading reel page...")
    try:
        driver.get(reel.permalink)
    except Exception as e:
        return False, f"Failed to load page: {e}"

    time.sleep(3)

    # Check if page loaded correctly
    try:
        current_url = driver.current_url
        if "login" in current_url.lower():
            return False, "Session expired - please log in to Instagram"
        page_source = driver.page_source.lower()
        if "sorry" in page_source or "page isn't available" in page_source:
            return False, "Reel not found or deleted"
    except Exception as e:
        return False, f"Error checking page: {e}"

    # Step 2: Click "..." (more options) button
    # English: "More options", "More"
    # Portuguese: "Mais opcoes", "Mais"
    update_progress("Opening menu...")
    more_selectors = [
        # Portuguese (check first)
        '//div[@role="button"][.//svg[@aria-label="Mais opções" or @aria-label="Mais"]]',
        '//*[@aria-label="Mais opções"]',
        '//*[@aria-label="Mais"]',
        '//svg[@aria-label="Mais opções"]/parent::div[@role="button"]',
        '//svg[@aria-label="Mais"]/parent::*',
        # English
        '//div[@role="button"][.//svg[@aria-label="More options" or @aria-label="More"]]',
        '//*[@aria-label="More options"]',
        '//*[@aria-label="More"]',
        # Generic fallbacks - look for SVG buttons near the reel
        '//div[@role="button"][.//svg[contains(@aria-label, "opções")]]',
        '//div[@role="button"][.//svg[contains(@aria-label, "ptions")]]',
        # Last resort - any button with an SVG that could be a menu (3 dots usually have a specific path)
        '//article//div[@role="button"][.//svg]',
    ]

    clicked = False
    for selector in more_selectors:
        try:
            elements = driver.find_elements(By.XPATH, selector)
            for elem in elements:
                if elem.is_displayed():
                    elem.click()
                    clicked = True
                    break
            if clicked:
                break
        except Exception:
            continue

    if not clicked:
        return False, "Could not find 'More options' button (Mais opções)"

    time.sleep(1)

    # Step 3: Click "Manage" / "Gerenciar post"
    update_progress("Clicking Manage/Gerenciar...")
    manage_selectors = [
        # Portuguese (check first since user reported this)
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Gerenciar post")]]',
        '//*[contains(text(), "Gerenciar post")]',
        '//span[contains(text(), "Gerenciar post")]',
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Gerenciar")]]',
        '//*[contains(text(), "Gerenciar")]',
        # English
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Manage")]]',
        '//*[contains(text(), "Manage")]',
    ]

    clicked = False
    for selector in manage_selectors:
        if wait_and_click(driver, By.XPATH, selector, timeout=3):
            clicked = True
            break

    if not clicked:
        return False, "Could not find 'Manage' button (Gerenciar)"

    time.sleep(1)

    # Step 4: Click "Edit" / "Editar"
    update_progress("Clicking Edit/Editar...")
    edit_selectors = [
        # English
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Edit")]]',
        '//*[contains(text(), "Edit")]',
        # Portuguese
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Editar")]]',
        '//*[contains(text(), "Editar")]',
    ]

    clicked = False
    for selector in edit_selectors:
        if wait_and_click(driver, By.XPATH, selector, timeout=3):
            clicked = True
            break

    if not clicked:
        log_debug(f"[{reel.reel_name}] FAILED: Could not find Edit button")
        return False, "Could not find 'Edit' button (Editar)"

    log_debug(f"[{reel.reel_name}] Clicked Edit, waiting for dialog...")
    time.sleep(3)  # Increased wait for dialog to fully load

    # Step 5: Set caption text
    # English: "Write a caption..."
    # Portuguese: "Escreva uma legenda..."
    update_progress("Setting caption...")
    log_debug(f"[{reel.reel_name}] Step 5: Looking for caption field...")

    caption_selectors = [
        '//div[@aria-label="Write a caption..."][@contenteditable="true"]',
        '//div[@aria-label="Escreva uma legenda..."][@contenteditable="true"]',
        '//div[contains(@aria-label, "caption") or contains(@aria-label, "legenda")][@contenteditable="true"]',
        '//div[@contenteditable="true"][@role="textbox"]',
        # More generic - any editable textbox
        '//div[@role="textbox"][@contenteditable="true"]',
        '//div[@data-lexical-editor="true"][@contenteditable="true"]',
    ]

    caption_div = None
    # Retry finding caption field up to 3 times with increasing wait
    for attempt in range(3):
        for selector in caption_selectors:
            try:
                log_debug(f"[{reel.reel_name}] Trying selector: {selector[:60]}...")
                caption_div = WebDriverWait(driver, 3).until(
                    EC.presence_of_element_located((By.XPATH, selector))
                )
                if caption_div and caption_div.is_displayed():
                    log_debug(f"[{reel.reel_name}] Found caption field with selector: {selector[:60]}")
                    break
                caption_div = None
            except TimeoutException:
                continue
            except Exception as e:
                log_debug(f"[{reel.reel_name}] Error with selector: {e}")
                continue

        if caption_div:
            break

        # Wait and retry
        log_debug(f"[{reel.reel_name}] Caption field not found, attempt {attempt + 1}/3, waiting...")
        time.sleep(2)

    if not caption_div:
        # Try to get page source for debugging
        try:
            page_source = driver.page_source[:2000]
            log_debug(f"[{reel.reel_name}] Page source (first 2000 chars): {page_source}")
        except Exception:
            pass
        return False, "Caption field not found (Escreva uma legenda...)"

    try:
        if not caption_div.is_displayed():
            log_debug(f"[{reel.reel_name}] Caption field found but not visible")
            return False, "Caption field not visible"

        # Click to focus
        caption_div.click()
        time.sleep(0.5)

        # Clear existing content
        actions = ActionChains(driver)
        actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
        time.sleep(0.2)
        actions.send_keys(Keys.DELETE).perform()
        time.sleep(0.3)

        # Type caption line by line
        lines = caption.split('\n')
        for i, line in enumerate(lines):
            if line:
                caption_div.send_keys(line)
                time.sleep(0.05)

            if i < len(lines) - 1:
                actions = ActionChains(driver)
                actions.key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
                time.sleep(0.05)

        # Wait for autocomplete to settle
        time.sleep(4)

    except TimeoutException:
        return False, "Caption field not found"
    except Exception as e:
        return False, f"Error setting caption: {e}"

    # Step 6: Click "Done" / "Concluir"
    update_progress("Saving/Salvando...")
    done_selectors = [
        # Portuguese (check first - "Concluir" is the actual button text)
        '//div[@role="button"][text()="Concluir"]',
        '//div[@role="button"][contains(text(), "Concluir")]',
        '//*[@role="button"][normalize-space()="Concluir"]',
        '//div[@role="button"][text()="Concluído"]',
        '//div[@role="button"][contains(text(), "Concluído")]',
        '//div[@role="button"][text()="Pronto"]',
        '//div[@role="button"][contains(text(), "Pronto")]',
        # English
        '//div[@role="button"][text()="Done"]',
        '//div[@role="button"][contains(text(), "Done")]',
        '//*[@role="button"][normalize-space()="Done"]',
    ]

    clicked = False
    for selector in done_selectors:
        if wait_and_click(driver, By.XPATH, selector, timeout=3):
            clicked = True
            break

    if not clicked:
        return False, "Could not find 'Done' button (Concluir)"

    # Wait and check for errors
    time.sleep(5)

    error = check_for_instagram_error(driver)
    if error:
        log_debug(f"[{reel.reel_name}] FAILED: Instagram error - {error}")
        return False, f"Instagram error: {error}"

    log_debug(f"[{reel.reel_name}] SUCCESS: Caption updated!")
    return True, ""


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================


def show_header():
    """Show script header."""
    console.print()
    console.print(Panel(
        "[bold]Instagram Reel Caption Fixer[/bold]\n\n"
        "Fixes empty captions on posted Instagram reels using browser automation.\n"
        "Each profile uses a dedicated Chrome instance to prevent conflicts.\n\n"
        "[dim]Supported languages: English, Portuguese (Brazilian)[/dim]",
        title=">>> FIX CAPTIONS",
        border_style="cyan",
    ))


def show_profiles_table(profiles: list[ProfileInfo]) -> None:
    """Show table of available profiles."""
    table = Table(title="Available Profiles", border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Profile", style="cyan")
    table.add_column("Empty Captions", justify="right")
    table.add_column("Port", justify="right", style="dim")
    table.add_column("Chrome", justify="center")

    for idx, profile in enumerate(profiles, 1):
        chrome_status = "[green]Running[/green]" if profile.chrome_running else "[dim]Stopped[/dim]"
        empty_style = "red" if profile.empty_count > 0 else "green"

        table.add_row(
            str(idx),
            profile.name,
            f"[{empty_style}]{profile.empty_count}[/{empty_style}]",
            str(profile.port),
            chrome_status,
        )

    console.print()
    console.print(table)


def show_chrome_instructions(profile: ProfileInfo):
    """Show Chrome launch instructions for a profile."""
    console.print()
    console.print(Panel(
        f"[bold]Chrome Setup for {profile.name}[/bold]\n\n"
        f"Port: [cyan]{profile.port}[/cyan]\n"
        f"User Data: [dim]{profile.user_data_dir}[/dim]\n\n"
        "[yellow]Chrome will be launched automatically.[/yellow]",
        title=">>> CHROME",
        border_style="yellow",
    ))


def show_fix_progress(current: int, total: int, reel_name: str, status: str, step: str = ""):
    """Show progress for current reel being fixed."""
    status_icon = {
        "working": "[yellow]...[/yellow]",
        "success": "[green][OK][/green]",
        "failed": "[red][X][/red]",
        "retry": "[yellow][RETRY][/yellow]",
    }.get(status, "[?]")

    step_text = f" - {step}" if step else ""
    console.print(f"  [{current:3d}/{total}] {status_icon} {reel_name[:45]}{step_text}")


def show_summary(summary: FixSummary):
    """Show final summary of fix operations."""
    console.print()

    # Stats panel
    stats = (
        f"[bold]Profile:[/bold] {summary.profile}\n"
        f"[bold]Total:[/bold] {summary.total}\n"
        f"[green]Success:[/green] {summary.success}\n"
        f"[red]Failed:[/red] {summary.failed}\n"
        f"[dim]Skipped:[/dim] {summary.skipped}"
    )

    border_color = "green" if summary.failed == 0 else "yellow" if summary.success > 0 else "red"

    console.print(Panel(
        stats,
        title=">>> SUMMARY",
        border_style=border_color,
    ))

    # Show failed reels
    if summary.failed > 0:
        console.print()
        console.print("[red]Failed reels:[/red]")
        for result in summary.results:
            if result.status == FixStatus.FAILED:
                console.print(f"  [red][X][/red] {result.reel_name}")
                console.print(f"      [dim]{result.permalink}[/dim]")
                console.print(f"      [dim]Error: {result.error}[/dim]")


# =============================================================================
# MAIN COMMAND
# =============================================================================


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Fix empty Instagram reel captions using Chrome automation."""
    # If no command specified, run fix with defaults (interactive mode)
    if ctx.invoked_subcommand is None:
        ctx.invoke(fix, profile=None, dry_run=False, no_hashtags=False, limit=None)


@app.command()
def fix(
    profile: Optional[str] = typer.Argument(
        default=None,
        help="Profile name to fix (interactive selection if not provided)",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-d",
        help="Show what would be fixed without making changes",
    ),
    no_hashtags: bool = typer.Option(
        False, "--no-hashtags",
        help="Remove hashtags from captions (use if hashtags cause errors)",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n",
        help="Limit number of reels to fix",
    ),
):
    """Fix empty Instagram reel captions using Chrome automation.

    Prerequisites:

    1. Run `sync-captions` first to identify reels with empty captions

    2. Chrome will be launched automatically with the correct settings

    3. Log in to Instagram in the Chrome window when prompted

    Examples:

        py scripts/fix_captions.py                     # Interactive selection

        py scripts/fix_captions.py ai.for.mortals      # Fix specific profile

        py scripts/fix_captions.py ai.for.mortals -n 5 # Fix only 5 reels

        py scripts/fix_captions.py --dry-run           # Preview without fixing
    """
    show_header()

    # Find profiles
    profiles = find_profiles()

    if not profiles:
        console.print("[red]No profiles found in profiles/ directory[/red]")
        raise typer.Exit(1)

    # Select profile
    selected_profile: Optional[ProfileInfo] = None

    if profile:
        # Find by name
        for p in profiles:
            if p.name == profile:
                selected_profile = p
                break

        if not selected_profile:
            console.print(f"[red]Profile not found: {profile}[/red]")
            console.print("\nAvailable profiles:")
            for p in profiles:
                console.print(f"  - {p.name}")
            raise typer.Exit(1)
    else:
        # Interactive selection
        show_profiles_table(profiles)

        # Filter to profiles with empty captions
        profiles_with_empty = [p for p in profiles if p.empty_count > 0]

        if not profiles_with_empty:
            console.print("\n[green]No profiles have empty captions![/green]")
            console.print("[dim]Run sync-captions first if you haven't already.[/dim]")
            raise typer.Exit(0)

        console.print()
        choice = Prompt.ask(
            "Select profile number",
            choices=[str(i) for i in range(1, len(profiles) + 1)],
        )
        selected_profile = profiles[int(choice) - 1]

    # Check for empty captions
    reels = find_empty_caption_reels(selected_profile.path)

    if not reels:
        console.print(f"\n[green]No empty captions found for {selected_profile.name}[/green]")
        console.print("[dim]Run sync-captions first to identify empty captions.[/dim]")
        raise typer.Exit(0)

    # Filter out reels needing regeneration
    fixable_reels = [r for r in reels if not r.needs_regeneration]
    needs_regen = [r for r in reels if r.needs_regeneration]

    if needs_regen:
        console.print(f"\n[yellow]Warning: {len(needs_regen)} reels have no local caption (skipped)[/yellow]")
        for r in needs_regen[:5]:
            console.print(f"  [dim]- {r.reel_name}[/dim]")
        if len(needs_regen) > 5:
            console.print(f"  [dim]... and {len(needs_regen) - 5} more[/dim]")

    if not fixable_reels:
        console.print("\n[yellow]No reels can be fixed (all need caption regeneration)[/yellow]")
        raise typer.Exit(1)

    # Apply limit
    if limit:
        fixable_reels = fixable_reels[:limit]

    console.print(f"\n[cyan]Found {len(fixable_reels)} reels to fix for {selected_profile.name}[/cyan]")

    if dry_run:
        console.print("\n[yellow]Dry run - showing reels that would be fixed:[/yellow]")
        for idx, reel in enumerate(fixable_reels, 1):
            console.print(f"  [{idx:3d}] {reel.reel_name}")
            console.print(f"        [dim]{reel.permalink}[/dim]")
        raise typer.Exit(0)

    # Confirm
    if not Confirm.ask(f"\nFix {len(fixable_reels)} reels?"):
        raise typer.Exit(0)

    # Check/launch Chrome
    show_chrome_instructions(selected_profile)

    if not is_port_in_use(selected_profile.port):
        console.print(f"\n[yellow]Launching Chrome on port {selected_profile.port}...[/yellow]")
        success, error = launch_chrome(selected_profile.port, selected_profile.user_data_dir)

        if not success:
            console.print(f"[red]Failed to launch Chrome: {error}[/red]")
            console.print("\n[dim]Try launching Chrome manually:[/dim]")
            chrome_path = get_chrome_path() or "chrome"
            console.print(f'  "{chrome_path}" --remote-debugging-port={selected_profile.port} --user-data-dir="{selected_profile.user_data_dir}"')
            raise typer.Exit(1)

        console.print("[green]Chrome launched successfully[/green]")
    else:
        console.print(f"\n[green]Chrome already running on port {selected_profile.port}[/green]")

    # Wait for user to log in
    console.print()
    console.print(Panel(
        f"[bold yellow]ACTION REQUIRED[/bold yellow]\n\n"
        f"1. A Chrome window should now be open\n"
        f"2. Log in to Instagram with the account for [bold cyan]{selected_profile.name}[/bold cyan]\n"
        f"3. Make sure you're on instagram.com and logged in\n"
        f"4. Come back here and confirm when ready\n\n"
        f"[dim]Supported languages: English or Portuguese (Brazilian)[/dim]",
        title=">>> LOGIN",
        border_style="yellow",
    ))
    console.print()
    if not Confirm.ask(f"Are you logged in to Instagram as [cyan]{selected_profile.name}[/cyan]?"):
        raise typer.Exit(0)

    # Connect to Chrome
    driver, error = connect_to_chrome(selected_profile.port)

    if not driver:
        console.print(f"[red]Failed to connect to Chrome: {error}[/red]")
        raise typer.Exit(1)

    console.print("[green]Connected to Chrome[/green]")

    # Load tracking
    tracking = load_tracking()

    # Fix reels
    console.print(f"\n[bold]Fixing {len(fixable_reels)} reels...[/bold]\n")

    summary = FixSummary(
        profile=selected_profile.name,
        total=len(fixable_reels),
    )

    for idx, reel in enumerate(fixable_reels, 1):
        current_step = ""

        def progress_callback(step: str):
            nonlocal current_step
            current_step = step

        # Show working status
        console.print(f"  [{idx:3d}/{len(fixable_reels)}] [yellow]...[/yellow] {reel.reel_name[:45]}", end="\r")

        # Try with hashtags first
        success, error = fix_single_reel(driver, reel, use_hashtags=not no_hashtags, progress_callback=progress_callback)

        # Retry without hashtags if failed
        if not success and not no_hashtags:
            console.print(f"  [{idx:3d}/{len(fixable_reels)}] [yellow][RETRY][/yellow] {reel.reel_name[:45]} - without hashtags")
            time.sleep(2)
            success, error = fix_single_reel(driver, reel, use_hashtags=False, progress_callback=progress_callback)

        if success:
            console.print(f"  [{idx:3d}/{len(fixable_reels)}] [green][OK][/green] {reel.reel_name[:45]}")
            summary.success += 1
            mark_as_fixed(tracking, selected_profile.name, reel.reel_name, reel.permalink)
            summary.results.append(FixResult(
                reel_name=reel.reel_name,
                permalink=reel.permalink,
                status=FixStatus.SUCCESS,
            ))
        else:
            console.print(f"  [{idx:3d}/{len(fixable_reels)}] [red][X][/red] {reel.reel_name[:45]}")
            console.print(f"        [dim]Error: {error}[/dim]")
            summary.failed += 1
            summary.results.append(FixResult(
                reel_name=reel.reel_name,
                permalink=reel.permalink,
                status=FixStatus.FAILED,
                error=error,
            ))

        # Delay between reels
        time.sleep(2)

    # Show summary
    show_summary(summary)

    if summary.failed > 0:
        raise typer.Exit(1)


@app.command()
def status():
    """Show status of all profiles and their Chrome port assignments."""
    show_header()

    profiles = find_profiles()

    if not profiles:
        console.print("[yellow]No profiles found[/yellow]")
        raise typer.Exit(0)

    show_profiles_table(profiles)

    # Show Chrome commands
    console.print()
    console.print("[dim]To manually launch Chrome for a profile:[/dim]")

    chrome_path = get_chrome_path()
    if chrome_path:
        for profile in profiles[:3]:
            console.print(f'  [dim]"{chrome_path}" --remote-debugging-port={profile.port} --user-data-dir="{profile.user_data_dir}"[/dim]')
        if len(profiles) > 3:
            console.print(f"  [dim]... and {len(profiles) - 3} more profiles[/dim]")


@app.command()
def launch_chrome_cmd(
    profile: str = typer.Argument(..., help="Profile name"),
):
    """Launch Chrome with correct settings for a profile."""
    profiles = find_profiles()

    selected = None
    for p in profiles:
        if p.name == profile:
            selected = p
            break

    if not selected:
        console.print(f"[red]Profile not found: {profile}[/red]")
        raise typer.Exit(1)

    if is_port_in_use(selected.port):
        console.print(f"[yellow]Chrome already running on port {selected.port}[/yellow]")
        raise typer.Exit(0)

    console.print(f"Launching Chrome for [cyan]{profile}[/cyan] on port [cyan]{selected.port}[/cyan]...")

    success, error = launch_chrome(selected.port, selected.user_data_dir)

    if success:
        console.print("[green]Chrome launched successfully[/green]")
    else:
        console.print(f"[red]Failed: {error}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
