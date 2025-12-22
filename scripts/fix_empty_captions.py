"""Fix empty Instagram captions using existing Chrome session.

This script connects to your already-logged-in Chrome browser and automates
the process of editing reel captions.

SETUP:
1. Close ALL Chrome windows completely
2. Start Chrome with remote debugging (use ABSOLUTE path for user-data-dir):

   Windows:
   "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\\ChromeDebug"

   macOS:
   "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port=9222 --user-data-dir="$HOME/ChromeDebug"

3. Log in to Instagram in that Chrome window
4. Run this script:
   py scripts/fix_empty_captions.py ai.for.mortals
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
except ImportError:
    print("Selenium not installed. Run: pip install selenium")
    sys.exit(1)

try:
    import pyperclip
except ImportError:
    pyperclip = None

import unicodedata


def sanitize_caption_for_instagram(caption: str) -> str:
    """Sanitize caption to remove problematic characters for Instagram.

    Handles:
    - Unicode normalization (NFC)
    - Zero-width characters
    - Control characters (except newlines)
    - Problematic Unicode that Selenium can't type
    - Smart quotes -> regular quotes
    - Non-breaking spaces -> regular spaces

    Args:
        caption: Raw caption text

    Returns:
        Sanitized caption safe for Instagram upload
    """
    if not caption:
        return ""

    # Normalize Unicode to NFC (composed form)
    caption = unicodedata.normalize('NFC', caption)

    # Replace smart quotes with regular quotes
    replacements = {
        '\u2018': "'",  # Left single quote
        '\u2019': "'",  # Right single quote
        '\u201c': '"',  # Left double quote
        '\u201d': '"',  # Right double quote
        '\u2014': '-',  # Em dash
        '\u2013': '-',  # En dash
        '\u2026': '...',  # Ellipsis
        '\u00a0': ' ',  # Non-breaking space
        '\u200b': '',   # Zero-width space
        '\u200c': '',   # Zero-width non-joiner
        '\u200d': '',   # Zero-width joiner (keep for emoji sequences? removing for safety)
        '\ufeff': '',   # BOM
        '\u00ad': '',   # Soft hyphen
        '\u2028': '\n', # Line separator
        '\u2029': '\n', # Paragraph separator
    }

    for old, new in replacements.items():
        caption = caption.replace(old, new)

    # Remove control characters (keep newlines and tabs)
    cleaned = []
    for char in caption:
        if char in '\n\r\t':
            cleaned.append(char)
        elif unicodedata.category(char) == 'Cc':  # Control characters
            continue
        elif unicodedata.category(char) == 'Cf':  # Format characters
            continue
        else:
            cleaned.append(char)

    caption = ''.join(cleaned)

    # Normalize multiple newlines to max 2
    while '\n\n\n' in caption:
        caption = caption.replace('\n\n\n', '\n\n')

    # Strip whitespace from each line
    lines = caption.split('\n')
    lines = [line.strip() for line in lines]
    caption = '\n'.join(lines)

    # Final strip
    caption = caption.strip()

    return caption


# Tracking file for fixed captions
TRACKING_FILE = Path("docs/empty_captions/fixed_captions.json")


def load_tracking() -> dict:
    """Load tracking data for fixed captions."""
    if TRACKING_FILE.exists():
        with open(TRACKING_FILE, encoding="utf-8") as f:
            return json.load(f)
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


def remove_from_fixed(tracking: dict, permalink: str) -> None:
    """Remove a reel from the fixed list (revert on failure)."""
    tracking["fixed"] = [item for item in tracking["fixed"] if item["permalink"] != permalink]
    save_tracking(tracking)


def remove_hashtags(caption: str) -> str:
    """Remove all hashtags from caption."""
    # Remove hashtags (# followed by word characters)
    caption_no_hashtags = re.sub(r'#\w+\s*', '', caption)
    # Clean up extra whitespace
    caption_no_hashtags = re.sub(r'\n\s*\n', '\n\n', caption_no_hashtags)
    return caption_no_hashtags.strip()


def remove_emojis(text: str) -> str:
    """Remove emojis from text (ChromeDriver can't handle non-BMP characters)."""
    # Remove emojis and other non-BMP Unicode characters
    # BMP is U+0000 to U+FFFF, emojis are typically above U+1F000
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    # Clean up any double spaces left behind
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def find_empty_caption_reels(profile_path: Path) -> list[dict]:
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
                        empty_reels.append({
                            "reel_path": reel_dir,
                            "reel_name": reel_dir.name,
                            "permalink": permalink,
                            "local_caption": local_caption,
                            "needs_regeneration": not local_caption,
                        })

    return sorted(empty_reels, key=lambda x: x["reel_name"])


def connect_to_chrome(port: int = 9222) -> webdriver.Chrome:
    """Connect to existing Chrome instance with remote debugging."""
    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")

    try:
        driver = webdriver.Chrome(options=options)
        print(f"[OK] Connected to Chrome on port {port}")
        return driver
    except Exception as e:
        print(f"[X] Failed to connect to Chrome on port {port}: {e}")
        print("\nMake sure Chrome is running with remote debugging:")
        print(f'  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port={port} --user-data-dir="%USERPROFILE%\\ChromeDebug_{port}"')
        sys.exit(1)


def wait_and_click(driver, by, value, timeout=10, description="element"):
    """Wait for element and click it."""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )
        time.sleep(0.3)
        element.click()
        return True
    except TimeoutException:
        return False
    except Exception:
        return False


def check_for_instagram_error(driver) -> str | None:
    """Check if Instagram is showing an error message. Returns error text or None."""
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
        except:
            pass
    return None


def fix_reel_caption(driver, reel: dict, use_hashtags: bool = True) -> tuple[bool, str]:
    """Fix caption for a single reel.

    Returns:
        tuple: (success: bool, error_message: str)
    """
    permalink = reel["permalink"]
    caption = reel["local_caption"]

    # Sanitize caption for Instagram (fixes encoding issues, control chars, etc.)
    caption = sanitize_caption_for_instagram(caption)

    # Always remove emojis (ChromeDriver can't handle non-BMP characters)
    caption = remove_emojis(caption)

    if not use_hashtags:
        caption = remove_hashtags(caption)
        print("    [i] Using caption WITHOUT hashtags")

    # =========================================================================
    # STEP 1: Load the reel page
    # =========================================================================
    print("    [Step 1/6] Loading reel page...")
    try:
        driver.get(permalink)
    except Exception as e:
        return False, f"Failed to load page: {e}"

    time.sleep(3)

    # Check if page loaded correctly
    try:
        current_url = driver.current_url
        if "login" in current_url.lower():
            return False, "Redirected to login - session expired"
        page_source = driver.page_source.lower()
        if "sorry" in page_source or "page isn't available" in page_source:
            return False, "Reel not found or deleted"
    except Exception as e:
        return False, f"Error checking page: {e}"

    # =========================================================================
    # STEP 2: Click "..." (more options) button
    # =========================================================================
    print("    [Step 2/6] Clicking more options (...)...")
    more_button_selectors = [
        '//div[@role="button"][.//svg[@aria-label="More options" or @aria-label="More"]]',
        '//*[@aria-label="More options"]',
        '//*[@aria-label="More"]',
    ]

    clicked = False
    for selector in more_button_selectors:
        try:
            elements = driver.find_elements(By.XPATH, selector)
            for elem in elements:
                if elem.is_displayed():
                    elem.click()
                    clicked = True
                    break
            if clicked:
                break
        except:
            continue

    if not clicked:
        print("    [!] Could not find more options button")
        print("    [!] Please click '...' manually, then press Enter")
        input()

    time.sleep(1)

    # =========================================================================
    # STEP 3: Click "Manage"
    # =========================================================================
    print("    [Step 3/6] Clicking Manage...")
    manage_clicked = wait_and_click(
        driver, By.XPATH,
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Manage")]]',
        timeout=5, description="Manage"
    )
    if not manage_clicked:
        manage_clicked = wait_and_click(
            driver, By.XPATH, '//*[contains(text(), "Manage")]',
            timeout=3, description="Manage"
        )
    if not manage_clicked:
        print("    [!] Please click 'Manage' manually, then press Enter")
        input()

    time.sleep(1)

    # =========================================================================
    # STEP 4: Click "Edit"
    # =========================================================================
    print("    [Step 4/6] Clicking Edit...")
    edit_clicked = wait_and_click(
        driver, By.XPATH,
        '//div[@role="button" or @role="menuitem"][.//span[contains(text(), "Edit")]]',
        timeout=5, description="Edit"
    )
    if not edit_clicked:
        edit_clicked = wait_and_click(
            driver, By.XPATH, '//*[contains(text(), "Edit")]',
            timeout=3, description="Edit"
        )
    if not edit_clicked:
        print("    [!] Please click 'Edit' manually, then press Enter")
        input()

    time.sleep(2)

    # =========================================================================
    # STEP 5: Set caption text
    # =========================================================================
    print("    [Step 5/6] Setting caption text...")
    textarea_found = False

    try:
        caption_div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((
                By.XPATH,
                '//div[@aria-label="Write a caption..."][@contenteditable="true"]'
            ))
        )

        if caption_div.is_displayed():
            # Click to focus
            caption_div.click()
            time.sleep(0.5)

            # Clear existing content with Ctrl+A, Delete
            actions = ActionChains(driver)
            actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
            time.sleep(0.2)
            actions.send_keys(Keys.DELETE).perform()
            time.sleep(0.3)

            # Type caption line by line with Shift+Enter for newlines
            lines = caption.split('\n')
            for i, line in enumerate(lines):
                if line:
                    caption_div.send_keys(line)
                    time.sleep(0.05)

                if i < len(lines) - 1:
                    actions = ActionChains(driver)
                    actions.key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
                    time.sleep(0.05)

            textarea_found = True

            # Wait for hashtag autocomplete to settle
            print("    [..] Waiting for autocomplete to settle...")
            time.sleep(4)

    except TimeoutException:
        pass
    except Exception as e:
        print(f"    [X] Error setting caption: {e}")

    if not textarea_found:
        print("    [!] Could not find caption field")
        print("    [!] Please paste caption manually, then press Enter")
        if pyperclip:
            pyperclip.copy(caption)
            print("    [i] Caption copied to clipboard (Ctrl+V)")
        input()

    # =========================================================================
    # STEP 6: Click "Done"
    # =========================================================================
    print("    [Step 6/6] Clicking Done...")
    done_clicked = wait_and_click(
        driver, By.XPATH, '//div[@role="button"][text()="Done"]',
        timeout=5, description="Done"
    )
    if not done_clicked:
        done_clicked = wait_and_click(
            driver, By.XPATH, '//div[@role="button"][contains(text(), "Done")]',
            timeout=3, description="Done"
        )
    if not done_clicked:
        done_clicked = wait_and_click(
            driver, By.XPATH, '//*[@role="button"][normalize-space()="Done"]',
            timeout=3, description="Done"
        )
    if not done_clicked:
        print("    [!] Please click 'Done' manually, then press Enter")
        input()

    # Wait for save and check for errors
    time.sleep(5)

    # Check for Instagram error
    error = check_for_instagram_error(driver)
    if error:
        return False, f"Instagram error: {error}"

    return True, ""


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix empty Instagram captions using Chrome automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py scripts/fix_empty_captions.py ai.for.mortals
  py scripts/fix_empty_captions.py ai.for.mortals --port 9222
  py scripts/fix_empty_captions.py news.but.quick --port 9223 --only-empty

Multi-account setup (run in separate terminals):
  Terminal 1: Chrome on port 9222 for Account A
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%USERPROFILE%\\ChromeDebug_9222"
    py scripts/fix_empty_captions.py ai.for.mortals --port 9222

  Terminal 2: Chrome on port 9223 for Account B
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9223 --user-data-dir="%USERPROFILE%\\ChromeDebug_9223"
    py scripts/fix_empty_captions.py news.but.quick --port 9223
        """
    )
    parser.add_argument("profile", help="Profile name (e.g., ai.for.mortals)")
    parser.add_argument("--port", "-p", type=int, default=9222,
                        help="Chrome remote debugging port (default: 9222)")
    parser.add_argument("--only-empty", "-e", action="store_true",
                        help="Only process reels with empty captions (skip already fixed)")

    args = parser.parse_args()

    profile_name = args.profile
    chrome_port = args.port
    profile_path = Path("profiles") / profile_name

    if not profile_path.exists():
        print(f"[X] Profile not found: {profile_name}")
        sys.exit(1)

    # Load tracking
    tracking = load_tracking()

    # Find empty caption reels
    print(f"\nScanning {profile_name} for empty captions...")
    empty_reels = find_empty_caption_reels(profile_path)

    # After sync, metadata.actual_caption is the source of truth
    pending_reels = empty_reels

    # Check for reels needing caption regeneration
    needs_regen = [r for r in pending_reels if r.get("needs_regeneration")]
    can_fix = [r for r in pending_reels if not r.get("needs_regeneration")]

    print(f"Found {len(empty_reels)} empty captions, {len(can_fix)} can be fixed")

    if needs_regen:
        print(f"\n[!] WARNING: {len(needs_regen)} reels have BOTH IG and local caption empty!")
        for r in needs_regen:
            print(f"      - {r['reel_name']}")
        print("    Run caption regeneration first.")
        pending_reels = can_fix

    if not pending_reels:
        print("No reels to fix!")
        return

    print("\n" + "=" * 60)
    print(f"CHROME AUTOMATION (Port {chrome_port})")
    print("=" * 60)
    print("\nMake sure Chrome is running with remote debugging.")
    print("Close ALL Chrome windows first, then run one of these commands:")
    print("-" * 60)
    print("\n[Windows] (use absolute path for user-data-dir):")
    print(f'  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port={chrome_port} --user-data-dir="%USERPROFILE%\\ChromeDebug_{chrome_port}"')
    print("\n[macOS]")
    print(f'  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port={chrome_port} --user-data-dir="$HOME/ChromeDebug_{chrome_port}"')
    print("\n[Linux]")
    print(f'  google-chrome --remote-debugging-port={chrome_port} --user-data-dir="$HOME/ChromeDebug_{chrome_port}"')
    print("-" * 60)
    print(f"\nLog in to Instagram in that Chrome window (port {chrome_port}), then press Enter...")
    input()

    # Connect to Chrome
    driver = connect_to_chrome(port=chrome_port)

    print(f"\nFixing {len(pending_reels)} reels...")
    print("=" * 60)

    fixed_count = 0
    failed_reels = []

    for idx, reel in enumerate(pending_reels, 1):
        print(f"\n[{idx}/{len(pending_reels)}] {reel['reel_name']}")
        print(f"    URL: {reel['permalink']}")

        # First attempt: with hashtags
        success, error = fix_reel_caption(driver, reel, use_hashtags=True)

        if not success:
            print(f"    [X] Failed: {error}")

            # Second attempt: without hashtags (retry from step 1)
            print("\n    [..] Retrying WITHOUT hashtags...")
            time.sleep(2)
            success, error = fix_reel_caption(driver, reel, use_hashtags=False)

            if not success:
                print(f"    [X] Failed again: {error}")
                failed_reels.append({
                    "reel_name": reel["reel_name"],
                    "permalink": reel["permalink"],
                    "error": error,
                })
                # Make sure it's NOT in the fixed list
                remove_from_fixed(tracking, reel["permalink"])
                continue

        # Success!
        mark_as_fixed(tracking, profile_name, reel["reel_name"], reel["permalink"])
        fixed_count += 1
        print("    [OK] Caption updated and marked as fixed!")

        # Delay between reels
        time.sleep(2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"DONE! Fixed {fixed_count}/{len(pending_reels)} reels")
    print(f"Tracking saved to: {TRACKING_FILE}")

    if failed_reels:
        print(f"\n[X] FAILED REELS ({len(failed_reels)}):")
        for f in failed_reels:
            print(f"    - {f['reel_name']}")
            print(f"      URL: {f['permalink']}")
            print(f"      Error: {f['error']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
