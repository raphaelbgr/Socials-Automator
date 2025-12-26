"""TikTok browser-based uploader using Chrome with remote debugging.

This module provides browser automation for uploading videos to TikTok
using Selenium with Chrome in remote debugging mode (persistent sessions).

Authentication is handled via a persistent Chrome profile per account.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger("tiktok_browser")

# TikTok Studio upload URL
TIKTOK_STUDIO_URL = "https://www.tiktok.com/tiktokstudio/upload?from=creator_center"

# TikTok max hashtags
TIKTOK_MAX_HASHTAGS = 2


def _sanitize_for_chromedriver(text: str) -> str:
    """Remove characters that ChromeDriver can't handle (non-BMP Unicode like emojis).

    ChromeDriver only supports characters in the Basic Multilingual Plane (BMP).
    This removes emojis and other characters above U+FFFF.

    Args:
        text: Original text.

    Returns:
        Text with non-BMP characters removed.
    """
    import re
    import unicodedata

    if not text:
        return text

    # Normalize Unicode to NFC (composed form)
    text = unicodedata.normalize('NFC', text)

    # Replace smart quotes and special characters with ASCII equivalents
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
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove emojis and other non-BMP Unicode characters (above U+FFFF)
    # BMP is U+0000 to U+FFFF
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

    # Remove any remaining non-BMP characters (anything above U+FFFF)
    text = ''.join(c for c in text if ord(c) <= 0xFFFF)

    # Clean up double spaces
    text = re.sub(r'  +', ' ', text)

    return text


def _limit_hashtags(text: str, max_hashtags: int = TIKTOK_MAX_HASHTAGS) -> str:
    """Limit the number of hashtags in text while preserving all newlines.

    Args:
        text: Original text with hashtags.
        max_hashtags: Maximum number of hashtags to keep.

    Returns:
        Text with hashtags limited to max_hashtags.
    """
    import re

    if not text:
        return text

    # Find all hashtags
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, text)

    if len(hashtags) <= max_hashtags:
        return text

    # Keep only the first max_hashtags
    hashtags_to_keep = set(hashtags[:max_hashtags])
    hashtags_to_remove = set(hashtags[max_hashtags:])

    # Remove excess hashtags (but keep all newlines intact)
    result = text
    for tag in hashtags_to_remove:
        # Remove the hashtag and any trailing space, but not newlines
        result = re.sub(re.escape(tag) + r' ?', '', result)

    # Clean up any double spaces that may have been created
    result = re.sub(r'  +', ' ', result)

    return result


@dataclass
class TikTokUploadResult:
    """Result of a TikTok upload attempt."""

    success: bool
    video_id: Optional[str] = None
    error: Optional[str] = None
    video_url: Optional[str] = None


def get_chrome_profile_dir(profile_name: str, project_root: Optional[Path] = None) -> Path:
    """Get the Chrome profile directory path for a specific account.

    Creates a persistent Chrome profile per social media account inside
    the project's profile folder: profiles/<profile>/tiktok/browser/

    Args:
        profile_name: The profile name (e.g., 'ai.for.mortals')
        project_root: Optional project root path. If None, uses current working directory.

    Returns:
        Path to the Chrome user data directory for this profile.
    """
    if project_root is None:
        # Try to find project root by looking for 'profiles' directory
        cwd = Path.cwd()
        if (cwd / "profiles").exists():
            project_root = cwd
        else:
            # Fall back to the module's location
            project_root = Path(__file__).parent.parent.parent.parent

    chrome_profile_path = project_root / "profiles" / profile_name / "tiktok" / "browser"

    # Create the directory if it doesn't exist
    chrome_profile_path.mkdir(parents=True, exist_ok=True)

    return chrome_profile_path


def get_chrome_launch_command(profile_name: str, port: int = 9333) -> dict[str, str]:
    """Get Chrome launch commands for each platform.

    Args:
        profile_name: The profile name (e.g., 'ai.for.mortals')
        port: Remote debugging port (default: 9333 for TikTok)

    Returns:
        Dict with 'windows', 'macos', 'linux' keys containing launch commands.
    """
    profile_dir = get_chrome_profile_dir(profile_name)
    # Use absolute path for Chrome command
    profile_dir_abs = str(profile_dir.absolute())

    return {
        "windows": f'"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port={port} --user-data-dir="{profile_dir_abs}"',
        "macos": f'"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --remote-debugging-port={port} --user-data-dir="{profile_dir_abs}"',
        "linux": f'google-chrome --remote-debugging-port={port} --user-data-dir="{profile_dir_abs}"',
    }


def print_chrome_instructions(profile_name: str, port: int = 9333) -> None:
    """Print instructions for starting Chrome with remote debugging.

    Args:
        profile_name: The profile name (e.g., 'ai.for.mortals')
        port: Remote debugging port
    """
    commands = get_chrome_launch_command(profile_name, port)
    profile_dir = get_chrome_profile_dir(profile_name)

    print()
    print("=" * 70)
    print(f"CHROME SETUP FOR TIKTOK ({profile_name})")
    print("=" * 70)
    print()
    print("1. Close ALL Chrome windows completely")
    print()
    print("2. Open a terminal/command prompt and run ONE of these commands:")
    print()
    print("-" * 70)
    print()
    print("[Windows PowerShell / CMD]:")
    print(f"  {commands['windows']}")
    print()
    print("[macOS Terminal]:")
    print(f"  {commands['macos']}")
    print()
    print("[Linux Terminal]:")
    print(f"  {commands['linux']}")
    print()
    print("-" * 70)
    print()
    print(f"3. Chrome profile stored at:")
    print(f"   profiles/{profile_name}/tiktok/browser/")
    print()
    print("4. Log in to TikTok in that Chrome window")
    print("   (Your login will persist for future uploads)")
    print()
    print("5. Press Enter here when ready...")
    print()
    print("=" * 70)


class TikTokBrowserUploader:
    """Upload videos to TikTok using Chrome with remote debugging.

    Uses Selenium to connect to an existing Chrome instance running
    with remote debugging enabled. This allows using a persistent
    logged-in session.
    """

    def __init__(
        self,
        profile_name: str,
        port: int = 9333,
        headless: bool = False,  # Not used, kept for API compatibility
    ):
        """Initialize the uploader.

        Args:
            profile_name: Profile name for the Chrome session (e.g., 'ai.for.mortals')
            port: Chrome remote debugging port (default: 9333)
            headless: Not used (Chrome must be visible for login)
        """
        self.profile_name = profile_name
        self.port = port
        self.driver = None

    def connect(self) -> bool:
        """Connect to Chrome with remote debugging.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
        except ImportError:
            logger.error("Selenium not installed. Run: pip install selenium")
            return False

        options = Options()
        options.add_experimental_option("debuggerAddress", f"127.0.0.1:{self.port}")

        try:
            self.driver = webdriver.Chrome(options=options)
            logger.info(f"Connected to Chrome on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Chrome on port {self.port}: {e}")
            return False

    def ensure_logged_in(self) -> bool:
        """Check if user is logged in to TikTok.

        Returns:
            True if logged in, False if login required.
        """
        if not self.driver:
            return False

        try:
            # Navigate to TikTok Studio
            self.driver.get(TIKTOK_STUDIO_URL)
            time.sleep(3)

            # Check if redirected to login
            current_url = self.driver.current_url.lower()
            if "login" in current_url or "passport" in current_url:
                return False

            # Check for upload button as sign of logged-in state
            page_source = self.driver.page_source
            if "Upload" in page_source or "upload" in page_source:
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking login status: {e}")
            return False

    def upload(
        self,
        video_path: Path,
        description: str,
        thumbnail_path: Optional[Path] = None,
        video_name: Optional[str] = None,
        schedule: Optional[str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> TikTokUploadResult:
        """Upload a video to TikTok.

        Args:
            video_path: Path to the video file (MP4).
            description: Caption/description for the video.
            thumbnail_path: Optional path to thumbnail image (JPG/PNG).
            video_name: Optional name for logging (e.g., reel folder name).
            schedule: Optional schedule time (not implemented yet).
            progress_callback: Optional callback for progress updates.

        Returns:
            TikTokUploadResult with success status and details.
        """
        # Set up logging prefix for this video
        log_prefix = f"[{video_name}] " if video_name else ""
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from selenium.common.exceptions import TimeoutException
        except ImportError:
            return TikTokUploadResult(
                success=False,
                error="Selenium not installed. Run: pip install selenium",
            )

        video_path = Path(video_path)
        if not video_path.exists():
            return TikTokUploadResult(
                success=False,
                error=f"Video file not found: {video_path}",
            )

        if not self.driver:
            return TikTokUploadResult(
                success=False,
                error="Not connected to Chrome. Call connect() first.",
            )

        if progress_callback:
            progress_callback("Navigating to TikTok Studio...", 5)

        try:
            # Step 1: Navigate to upload page
            self.driver.get(TIKTOK_STUDIO_URL)
            time.sleep(3)

            # Check if logged in
            current_url = self.driver.current_url.lower()
            if "login" in current_url or "passport" in current_url:
                return TikTokUploadResult(
                    success=False,
                    error="Not logged in. Please log in to TikTok in the Chrome window.",
                )

            if progress_callback:
                progress_callback("Looking for file input...", 10)

            # Step 2: Find file input and send file directly (no button click needed)
            # TikTok has a hidden file input that accepts the file path
            time.sleep(2)  # Wait for page to fully load

            file_input = None
            file_input_selectors = [
                'input[type="file"]',
                'input[accept*="video"]',
                'input[accept="video/*"]',
            ]

            for selector in file_input_selectors:
                try:
                    inputs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    for inp in inputs:
                        # File inputs are usually hidden but still functional
                        file_input = inp
                        break
                    if file_input:
                        break
                except Exception:
                    continue

            if not file_input:
                return TikTokUploadResult(
                    success=False,
                    error="Could not find file upload input. TikTok UI may have changed.",
                )

            if progress_callback:
                progress_callback("Uploading video file...", 20)

            # Send file path directly to the input (no file picker dialog)
            try:
                file_input.send_keys(str(video_path.absolute()))
            except Exception as e:
                return TikTokUploadResult(
                    success=False,
                    error=f"Could not upload file: {e}",
                )

            if progress_callback:
                progress_callback("Waiting for video to upload...", 30)

            # Step 4: Wait for video upload to complete
            # Poll for "Enviado" (Uploaded) status in the page
            upload_complete = False
            max_wait_seconds = 300  # 5 minutes max
            wait_start = time.time()

            while not upload_complete and (time.time() - wait_start) < max_wait_seconds:
                try:
                    page_source = self.driver.page_source
                    # Check for upload complete indicators (multiple languages)
                    # "Enviado" = Portuguese, "Uploaded" = English, "Subido" = Spanish
                    if any(indicator in page_source for indicator in ["Enviado", "Uploaded", "Subido"]):
                        # Also verify it's not still in progress
                        if "Enviando" not in page_source and "Uploading" not in page_source:
                            upload_complete = True
                            logger.info(f"{log_prefix}Video upload complete (found 'Enviado' status)")
                            break
                except Exception as e:
                    logger.warning(f"{log_prefix}Error checking upload status: {e}")

                # Update progress
                elapsed = int(time.time() - wait_start)
                if progress_callback:
                    progress_callback(f"Uploading video... ({elapsed}s)", 30 + min(elapsed // 3, 20))

                time.sleep(1)  # Poll every 1 second

            if not upload_complete:
                logger.error(f"{log_prefix}Video upload timed out after 5 minutes")
                return TikTokUploadResult(
                    success=False,
                    error=f"{log_prefix}Video upload timed out after 5 minutes",
                )

            if progress_callback:
                progress_callback("Video uploaded successfully", 55)

            time.sleep(2)  # Brief pause after upload completes

            if progress_callback:
                progress_callback("Adding caption...", 60)

            # Step 5: Add caption/description
            # Wait a bit more for the form to be ready after video upload
            time.sleep(3)

            # Sanitize caption: remove emojis (ChromeDriver can't handle non-BMP chars)
            # and limit hashtags to 2 for TikTok
            caption_text = _sanitize_for_chromedriver(description)
            caption_text = _limit_hashtags(caption_text, max_hashtags=2)

            caption_added = False
            try:
                from selenium.webdriver.common.action_chains import ActionChains

                # TikTok Studio description field - try multiple approaches
                # The description is usually a contenteditable div or a text editor

                # First, try to find by placeholder text
                caption_selectors = [
                    # Look for elements with description-related placeholders
                    '//*[@data-placeholder and contains(@data-placeholder, "escri")]',  # Describe/description
                    '//*[contains(@placeholder, "escri")]',
                    # Contenteditable divs (TikTok uses these for rich text)
                    '//div[@contenteditable="true" and @role="textbox"]',
                    '//div[@contenteditable="true" and contains(@class, "editor")]',
                    '//div[@contenteditable="true" and contains(@class, "caption")]',
                    '//div[@contenteditable="true" and contains(@class, "description")]',
                    # Draft.js editor (common in React apps)
                    '//div[contains(@class, "DraftEditor-root")]//div[@contenteditable="true"]',
                    '//div[contains(@class, "public-DraftEditor-content")]',
                    # Generic contenteditable (last resort - pick the largest one)
                    '//div[@contenteditable="true"]',
                ]

                caption_field = None
                for selector in caption_selectors:
                    try:
                        elements = self.driver.find_elements(By.XPATH, selector)
                        for elem in elements:
                            if elem.is_displayed():
                                # Found a visible contenteditable element
                                caption_field = elem
                                logger.info(f"{log_prefix}Found caption field with selector: {selector}")
                                break
                        if caption_field:
                            break
                    except Exception:
                        continue

                if caption_field:
                    # Click to focus
                    caption_field.click()
                    time.sleep(0.5)

                    # Clear existing content
                    actions = ActionChains(self.driver)
                    actions.key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()
                    time.sleep(0.2)
                    actions.send_keys(Keys.DELETE).perform()
                    time.sleep(0.3)

                    # Type caption preserving ALL newlines with Shift+Enter
                    lines = caption_text.split('\n')
                    for i, line in enumerate(lines):
                        # Always type the line (even if empty, to preserve blank lines)
                        if line:
                            caption_field.send_keys(line)
                        time.sleep(0.05)
                        # Add newline after each line except the last
                        if i < len(lines) - 1:
                            actions = ActionChains(self.driver)
                            actions.key_down(Keys.SHIFT).send_keys(Keys.ENTER).key_up(Keys.SHIFT).perform()
                            time.sleep(0.05)

                    caption_added = True
                    time.sleep(2)  # Wait for any autocomplete to settle
                    if progress_callback:
                        progress_callback("Caption added", 70)
                else:
                    logger.warning(f"{log_prefix}Could not find caption field - no matching element found")
                    if progress_callback:
                        progress_callback("Caption field not found - please add manually", 70)

            except Exception as e:
                logger.warning(f"{log_prefix}Could not set caption: {e}")
                if progress_callback:
                    progress_callback(f"Caption error: {e}", 70)
                # Continue - video can still be posted without caption

            # Step 6: Upload thumbnail (if provided)
            if thumbnail_path and Path(thumbnail_path).exists():
                if progress_callback:
                    progress_callback("Uploading thumbnail...", 72)

                thumbnail_uploaded = False
                try:
                    from selenium.webdriver.common.action_chains import ActionChains

                    # Step 6a: Click "Editar capa" / "Edit cover"
                    edit_cover_selectors = [
                        '//div[contains(@class, "edit-container") and contains(text(), "Editar capa")]',
                        '//div[contains(text(), "Editar capa")]',
                        '//div[contains(text(), "Edit cover")]',
                        '//div[contains(text(), "Edit thumbnail")]',
                        '//*[contains(@class, "edit-container")]',
                    ]

                    edit_cover_clicked = False
                    for selector in edit_cover_selectors:
                        try:
                            elements = self.driver.find_elements(By.XPATH, selector)
                            for elem in elements:
                                if elem.is_displayed():
                                    elem.click()
                                    edit_cover_clicked = True
                                    logger.info(f"{log_prefix}Clicked 'Edit cover' with selector: {selector}")
                                    time.sleep(1)
                                    break
                            if edit_cover_clicked:
                                break
                        except Exception:
                            continue

                    if edit_cover_clicked:
                        # Step 6b: Click "Carregar capa" / "Upload cover" tab
                        time.sleep(1)
                        upload_cover_selectors = [
                            '//div[contains(@class, "cover-edit-tab") and contains(text(), "Carregar capa")]',
                            '//div[contains(text(), "Carregar capa")]',
                            '//div[contains(text(), "Upload cover")]',
                            '//div[contains(text(), "Upload thumbnail")]',
                            '//div[contains(@class, "cover-edit-tab")]',
                        ]

                        upload_tab_clicked = False
                        for selector in upload_cover_selectors:
                            try:
                                elements = self.driver.find_elements(By.XPATH, selector)
                                for elem in elements:
                                    if elem.is_displayed():
                                        elem.click()
                                        upload_tab_clicked = True
                                        logger.info(f"{log_prefix}Clicked 'Upload cover' tab with selector: {selector}")
                                        time.sleep(1)
                                        break
                                if upload_tab_clicked:
                                    break
                            except Exception:
                                continue

                        if upload_tab_clicked:
                            # Step 6c: Find file input and send thumbnail
                            time.sleep(1)
                            thumb_input = None
                            thumb_input_selectors = [
                                'input[type="file"][accept*="image"]',
                                'input[type="file"][accept="image/*"]',
                                'input[type="file"][accept*="jpg"]',
                                'input[type="file"][accept*="png"]',
                                'input[type="file"]',
                            ]

                            for selector in thumb_input_selectors:
                                try:
                                    inputs = self.driver.find_elements(By.CSS_SELECTOR, selector)
                                    for inp in inputs:
                                        # Look for inputs in the cover upload area
                                        thumb_input = inp
                                        break
                                    if thumb_input:
                                        break
                                except Exception:
                                    continue

                            if thumb_input:
                                try:
                                    thumb_input.send_keys(str(Path(thumbnail_path).absolute()))
                                    logger.info(f"{log_prefix}Sent thumbnail file to input")
                                    time.sleep(2)

                                    # Step 6d: Click "Confirmar" / "Confirm"
                                    # Wait a moment for the button to become clickable
                                    time.sleep(1)

                                    confirm_selectors = [
                                        # Most specific - TUXButton with primary class containing Confirmar
                                        '//button[contains(@class, "TUXButton--primary")][.//div[contains(@class, "TUXButton-label")][contains(text(), "Confirmar")]]',
                                        '//button[contains(@class, "TUXButton--primary")][.//div[contains(text(), "Confirmar")]]',
                                        '//button[contains(@class, "TUXButton")][.//div[contains(text(), "Confirmar")]]',
                                        # English variants
                                        '//button[contains(@class, "TUXButton--primary")][.//div[contains(text(), "Confirm")]]',
                                        '//button[contains(@class, "TUXButton")][.//div[contains(text(), "Confirm")]]',
                                        # Fallback - any button with Confirmar/Confirm text
                                        '//button[.//div[contains(text(), "Confirmar")]]',
                                        '//button[.//div[contains(text(), "Confirm")]]',
                                        '//button[contains(text(), "Confirmar")]',
                                        '//button[contains(text(), "Confirm")]',
                                    ]

                                    confirm_clicked = False
                                    for selector in confirm_selectors:
                                        try:
                                            elements = self.driver.find_elements(By.XPATH, selector)
                                            logger.debug(f"{log_prefix}Confirm selector '{selector}' found {len(elements)} elements")
                                            for elem in elements:
                                                if elem.is_displayed() and elem.is_enabled():
                                                    try:
                                                        elem.click()
                                                        confirm_clicked = True
                                                        thumbnail_uploaded = True
                                                        logger.info(f"{log_prefix}Clicked 'Confirm' button with selector: {selector}")
                                                        time.sleep(2)
                                                        break
                                                    except Exception as click_err:
                                                        logger.warning(f"{log_prefix}Click failed on confirm button: {click_err}")
                                                        # Try JavaScript click as fallback
                                                        try:
                                                            self.driver.execute_script("arguments[0].click();", elem)
                                                            confirm_clicked = True
                                                            thumbnail_uploaded = True
                                                            logger.info(f"{log_prefix}Clicked 'Confirm' via JS with selector: {selector}")
                                                            time.sleep(2)
                                                            break
                                                        except Exception as js_err:
                                                            logger.warning(f"{log_prefix}JS click also failed: {js_err}")
                                            if confirm_clicked:
                                                break
                                        except Exception as e:
                                            logger.debug(f"{log_prefix}Selector failed: {selector} - {e}")
                                            continue

                                    if not confirm_clicked:
                                        logger.error(f"{log_prefix}Failed to click 'Confirm' button - no selector worked")

                                except Exception as e:
                                    logger.warning(f"{log_prefix}Error sending thumbnail file: {e}")
                            else:
                                logger.warning(f"{log_prefix}Could not find thumbnail file input")
                        else:
                            logger.warning(f"{log_prefix}Could not find 'Upload cover' tab")
                    else:
                        logger.warning(f"{log_prefix}Could not find 'Edit cover' button")

                except Exception as e:
                    logger.warning(f"{log_prefix}Error uploading thumbnail: {e}")

                if thumbnail_uploaded:
                    if progress_callback:
                        progress_callback("Thumbnail uploaded", 75)
                else:
                    if progress_callback:
                        progress_callback("Thumbnail upload failed - continuing without", 75)

            # Step 7: Re-check video upload status before publishing
            if progress_callback:
                progress_callback("Verifying upload status...", 78)

            time.sleep(1)
            try:
                page_source = self.driver.page_source
                if not any(indicator in page_source for indicator in ["Enviado", "Uploaded", "Subido"]):
                    logger.warning(f"{log_prefix}Video upload status not confirmed before publish")
                else:
                    logger.info(f"{log_prefix}Video upload status confirmed before publish")
            except Exception:
                pass

            if progress_callback:
                progress_callback("Posting video...", 80)

            # Step 8: Click Post button
            time.sleep(2)
            try:
                post_button_selectors = [
                    # TikTok Studio specific - data-e2e attribute (most reliable)
                    '//button[@data-e2e="post_video_button"]',
                    # Multilingual support (Publicar = Spanish, Post = English, etc.)
                    '//button[contains(text(), "Publicar")]',
                    '//button[contains(text(), "Post")]',
                    '//button[contains(text(), "Publish")]',
                    '//button[.//div[contains(text(), "Publicar")]]',
                    '//button[.//div[contains(text(), "Post")]]',
                    '//button[.//div[contains(text(), "Publish")]]',
                    # Button with specific classes
                    '//button[contains(@class, "Button__root--type-primary")]',
                    '//button[contains(@class, "post")]',
                    '//button[contains(@class, "publish")]',
                    '//button[contains(@class, "submit")]',
                    # Div buttons
                    '//div[@role="button"][contains(text(), "Post")]',
                    '//div[@role="button"][contains(text(), "Publish")]',
                    # TikTok specific - look for primary buttons
                    '//button[contains(@class, "TUXButton") and contains(@class, "primary")]',
                    # Last resort - any button that looks like a submit
                    '//button[@type="submit"]',
                ]

                posted = False
                for selector in post_button_selectors:
                    try:
                        buttons = self.driver.find_elements(By.XPATH, selector)
                        for btn in buttons:
                            if btn.is_displayed() and btn.is_enabled():
                                btn_text = btn.text.lower()
                                # Make sure it's actually a post button, not cancel/discard
                                if 'cancel' in btn_text or 'discard' in btn_text or 'back' in btn_text:
                                    continue
                                btn.click()
                                posted = True
                                logger.info(f"{log_prefix}Clicked post button with selector: {selector}")
                                break
                        if posted:
                            break
                    except Exception:
                        continue

                if not posted:
                    # Try one more approach - find by button text directly
                    try:
                        all_buttons = self.driver.find_elements(By.TAG_NAME, "button")
                        for btn in all_buttons:
                            if btn.is_displayed() and btn.is_enabled():
                                btn_text = btn.text.strip().lower()
                                if btn_text in ['post', 'publish', 'upload']:
                                    btn.click()
                                    posted = True
                                    logger.info(f"{log_prefix}Clicked button with text: {btn_text}")
                                    break
                    except Exception:
                        pass

                if not posted:
                    if progress_callback:
                        progress_callback("Post button not found - please click manually", 85)
                    return TikTokUploadResult(
                        success=False,
                        error="Could not find Post button. Please post manually.",
                    )

                # Step 6b: Handle confirmation modal (if it appears)
                # TikTok may show a "Publish now" / "Publicar agora" confirmation
                time.sleep(2)

                confirm_selectors = [
                    # TikTok modal confirm buttons
                    '//button[contains(@class, "TUXButton--primary")]//div[contains(text(), "Publicar agora")]/..',
                    '//button[contains(@class, "TUXButton--primary")]//div[contains(text(), "Publish now")]/..',
                    '//button[contains(@class, "TUXButton--primary")][.//div[contains(text(), "Publicar")]]',
                    '//button[contains(@class, "TUXButton--primary")][.//div[contains(text(), "Publish")]]',
                    '//button[.//div[contains(text(), "Publicar agora")]]',
                    '//button[.//div[contains(text(), "Publish now")]]',
                    '//button[contains(text(), "Publicar agora")]',
                    '//button[contains(text(), "Publish now")]',
                    # Generic modal confirm
                    '//div[contains(@class, "modal")]//button[contains(@class, "primary")]',
                ]

                for selector in confirm_selectors:
                    try:
                        confirm_buttons = self.driver.find_elements(By.XPATH, selector)
                        for btn in confirm_buttons:
                            if btn.is_displayed() and btn.is_enabled():
                                btn.click()
                                logger.info(f"{log_prefix}Clicked confirmation button: {selector}")
                                if progress_callback:
                                    progress_callback("Confirmed publish", 85)
                                time.sleep(1)
                                break
                    except Exception:
                        continue

            except Exception as e:
                return TikTokUploadResult(
                    success=False,
                    error=f"Error clicking Post button: {e}",
                )

            if progress_callback:
                progress_callback("Waiting for confirmation...", 90)

            # Step 9: Check for error toasts (rate limiting, etc.)
            time.sleep(3)

            # Check for error toast messages
            error_toast_selectors = [
                '//div[contains(@class, "TUXTopToast")]//div[contains(@class, "TUXTopToast-content")]',
                '//div[contains(@class, "TUXToast")]//div[contains(text(), "tentativas")]',
                '//div[contains(@class, "TUXToast")]//div[contains(text(), "attempts")]',
                '//div[contains(@class, "TUXToast")]//div[contains(text(), "error")]',
                '//div[contains(@class, "TUXToast")]//div[contains(text(), "erro")]',
            ]

            # Rate limit error messages in different languages
            rate_limit_messages = [
                "muitas tentativas",  # Portuguese
                "too many",           # English
                "demasiadas",         # Spanish
                "try again later",
                "tente novamente",
                "rate limit",
            ]

            for selector in error_toast_selectors:
                try:
                    toasts = self.driver.find_elements(By.XPATH, selector)
                    for toast in toasts:
                        if toast.is_displayed():
                            toast_text = toast.text.lower()
                            logger.warning(f"{log_prefix}Toast message detected: {toast.text}")

                            # Check if it's a rate limit error
                            if any(msg in toast_text for msg in rate_limit_messages):
                                error_msg = f"Rate limit error: {toast.text}"
                                logger.error(f"{log_prefix}{error_msg}")
                                return TikTokUploadResult(
                                    success=False,
                                    error=error_msg,
                                )

                            # Check for other errors
                            if "error" in toast_text or "erro" in toast_text or "failed" in toast_text:
                                error_msg = f"Upload error: {toast.text}"
                                logger.error(f"{log_prefix}{error_msg}")
                                return TikTokUploadResult(
                                    success=False,
                                    error=error_msg,
                                )
                except Exception:
                    continue

            # Step 10: Final success check
            time.sleep(2)

            # The upload is considered successful if we:
            # 1. Successfully uploaded the file
            # 2. Clicked the post button
            # 3. (Optionally) clicked the confirm button
            # 4. No error toasts were detected
            # We already passed all these steps, so it's a success

            logger.info(f"{log_prefix}Upload completed successfully")

            # Step 11: Try to capture video URL
            # After successful upload, TikTok may show a link to the video
            # or we can navigate to the content page to find the latest video
            video_url = None
            video_id = None

            try:
                if progress_callback:
                    progress_callback("Capturing video URL...", 95)

                # Wait a moment for any success modal/redirect
                time.sleep(3)

                # Method 1: Check current URL (TikTok may redirect to the video page)
                current_url = self.driver.current_url
                if "/video/" in current_url:
                    video_url = current_url
                    match = re.search(r'/video/(\d+)', current_url)
                    if match:
                        video_id = match.group(1)
                    logger.info(f"{log_prefix}Captured video URL from redirect: {video_url}")

                # Method 2: Look for video link in success modal or page
                if not video_url:
                    try:
                        # TikTok may show a "View" or "See post" link after upload
                        video_link_selectors = [
                            '//a[contains(@href, "/video/")]',
                            '//a[contains(text(), "View")]',
                            '//a[contains(text(), "Ver")]',  # Portuguese
                            '//button[contains(text(), "View")]/..//a',
                        ]
                        for selector in video_link_selectors:
                            try:
                                links = self.driver.find_elements(By.XPATH, selector)
                                for link in links:
                                    href = link.get_attribute('href')
                                    if href and '/video/' in href:
                                        video_url = href
                                        match = re.search(r'/video/(\d+)', href)
                                        if match:
                                            video_id = match.group(1)
                                        logger.info(f"{log_prefix}Captured video URL from page: {video_url}")
                                        break
                            except Exception:
                                continue
                            if video_url:
                                break
                    except Exception as e:
                        logger.debug(f"{log_prefix}Could not find video link in page: {e}")

                # Method 3: Navigate to content page and get latest video
                if not video_url:
                    try:
                        logger.debug(f"{log_prefix}Trying to get video URL from content page...")
                        self.driver.get("https://www.tiktok.com/tiktokstudio/content")
                        time.sleep(3)

                        # Find first video link (most recent)
                        video_links = self.driver.find_elements(By.XPATH, '//a[contains(@href, "/video/")]')
                        if video_links:
                            href = video_links[0].get_attribute('href')
                            if href and '/video/' in href:
                                video_url = href
                                match = re.search(r'/video/(\d+)', href)
                                if match:
                                    video_id = match.group(1)
                                logger.info(f"{log_prefix}Captured video URL from content page: {video_url}")
                    except Exception as e:
                        logger.debug(f"{log_prefix}Could not get video URL from content page: {e}")

            except Exception as e:
                logger.warning(f"{log_prefix}Could not capture video URL: {e}")

            if progress_callback:
                progress_callback("Upload complete!", 100)

            return TikTokUploadResult(
                success=True,
                video_id=video_id,
                video_url=video_url,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"{log_prefix}TikTok upload failed: {error_msg}")
            return TikTokUploadResult(
                success=False,
                error=error_msg,
            )

    def close(self) -> None:
        """Close the browser connection (but not the browser itself)."""
        if self.driver:
            try:
                # Don't quit - just disconnect, leaving browser open
                self.driver = None
            except Exception:
                pass

    @staticmethod
    def get_setup_instructions(profile_name: str, port: int = 9333) -> str:
        """Get setup instructions for TikTok browser upload.

        Args:
            profile_name: Profile name for the Chrome session
            port: Chrome remote debugging port

        Returns:
            Formatted instructions string.
        """
        commands = get_chrome_launch_command(profile_name, port)
        profile_dir = get_chrome_profile_dir(profile_name)

        return f"""
TikTok Browser Upload Setup for '{profile_name}'
{'=' * 60}

This command uses Chrome with remote debugging for persistent login.
Your TikTok session will be saved and reused for future uploads.

FIRST TIME SETUP:
-----------------
1. Close ALL Chrome windows completely

2. Open a terminal and run ONE of these commands:

   [Windows PowerShell / CMD]:
   {commands['windows']}

   [macOS Terminal]:
   {commands['macos']}

   [Linux Terminal]:
   {commands['linux']}

3. Chrome will open with a dedicated profile at:
   {profile_dir}

4. Navigate to: {TIKTOK_STUDIO_URL}

5. Log in to TikTok (your session will persist!)

6. Run the upload command again

SUBSEQUENT UPLOADS:
-------------------
Just make sure Chrome is running with the same command before uploading.
Your login session will be remembered.
"""


def get_cookies_path(profile_path: Path) -> Optional[Path]:
    """Find TikTok cookies file in profile directory.

    Note: This is kept for backwards compatibility but the new
    approach uses Chrome persistent profiles instead of cookies.

    Args:
        profile_path: Path to the profile directory.

    Returns:
        Path to cookies file if found, None otherwise.
    """
    # Check multiple possible locations and names
    possible_paths = [
        profile_path / "tiktok_cookies.txt",
        profile_path / "tiktok_cookies.json",
        profile_path / "cookies.txt",
        profile_path / "data" / "tiktok_cookies.txt",
        profile_path / "data" / "tiktok_cookies.json",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None
