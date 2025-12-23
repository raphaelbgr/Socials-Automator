"""Headless browser screenshot fallback for image downloads.

When direct image downloads fail (403, hotlink protection), this module
uses Playwright to take a screenshot of the image from its source page.

Uses a sandboxed Chromium browser stored in the project's .playwright folder.
Runs Playwright in a subprocess to avoid asyncio compatibility issues on Windows.

Usage:
    from .headless_screenshot import capture_image_screenshot

    path = await capture_image_screenshot(
        image_url="https://example.com/image.jpg",
        source_page_url="https://example.com/article",
        output_path=Path("image.jpg"),
    )

Dependencies:
    pip install playwright
    # Browser binaries auto-install to .playwright folder on first use
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("video.pipeline")

# Sandboxed browser location (project-local)
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
PLAYWRIGHT_BROWSERS_PATH = _PROJECT_ROOT / ".playwright"

# Thread pool for subprocess operations
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="playwright")

# Flag to track if Playwright is known to be broken on this platform
_playwright_broken = False
_playwright_broken_reason = ""
_playwright_lock = threading.Lock()


def _get_browsers_path() -> Path:
    """Get the sandboxed browser path."""
    return PLAYWRIGHT_BROWSERS_PATH


def _ensure_browsers_installed() -> bool:
    """Ensure browser binaries are installed.

    Downloads Chromium to sandboxed location if not present.

    Returns:
        True if browsers are available.
    """
    browsers_path = _get_browsers_path()

    # Check if chromium folder exists
    chromium_folders = list(browsers_path.glob("chromium-*"))
    if chromium_folders:
        return True

    # Need to install
    logger.info("HeadlessScreenshot: Installing sandboxed Chromium...")
    try:
        env = os.environ.copy()
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(browsers_path)

        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )

        if result.returncode == 0:
            logger.info("HeadlessScreenshot: Chromium installed successfully")
            return True
        else:
            logger.error(f"HeadlessScreenshot: Failed to install Chromium: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"HeadlessScreenshot: Failed to install Chromium: {e}")
        return False


# Inline Python script that runs Playwright in subprocess
# This avoids asyncio issues on Windows Python 3.14+
_PLAYWRIGHT_SCRIPT = '''
import json
import sys
import os

def capture_screenshot(image_url, source_page_url, output_path, timeout=15000):
    """Capture screenshot using Playwright."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"success": False, "error": "Playwright not installed"}

    context = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            )
            page = context.new_page()

            # Strategy 1: Try loading image URL directly
            try:
                page.goto(image_url, timeout=timeout, wait_until="load")

                img_element = page.query_selector("img")
                if img_element:
                    dimensions = img_element.evaluate(
                        "el => ({width: el.naturalWidth, height: el.naturalHeight})"
                    )

                    if dimensions["width"] > 100 and dimensions["height"] > 100:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        img_element.screenshot(path=output_path)
                        return {
                            "success": True,
                            "path": output_path,
                            "width": dimensions["width"],
                            "height": dimensions["height"],
                            "method": "direct",
                        }

                # Check if whole page is the image
                body = page.query_selector("body")
                if body:
                    body_html = body.inner_html()
                    if "<img" in body_html.lower() and body_html.count("<") <= 3:
                        img = page.query_selector("img")
                        if img:
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            img.screenshot(path=output_path)
                            return {
                                "success": True,
                                "path": output_path,
                                "method": "raw_page",
                            }

            except Exception as e:
                pass  # Fall through to strategy 2

            # Strategy 2: Load source page and find the image
            if source_page_url and source_page_url != image_url:
                try:
                    import re
                    page.goto(source_page_url, timeout=timeout, wait_until="domcontentloaded")
                    page.wait_for_timeout(1000)

                    # Try to find by exact src
                    img_element = page.query_selector(f'img[src="{image_url}"]')

                    if not img_element:
                        # Try partial match
                        match = re.search(r"/([^/]+\\.(jpg|jpeg|png|webp|gif))", image_url, re.I)
                        if match:
                            filename = match.group(1)
                            img_element = page.query_selector(f'img[src*="{filename}"]')

                    if not img_element:
                        # Find any large image
                        all_images = page.query_selector_all("img")
                        for img in all_images:
                            try:
                                dims = img.evaluate(
                                    "el => ({width: el.naturalWidth, height: el.naturalHeight, src: el.src})"
                                )
                                if dims["width"] > 200 and dims["height"] > 200:
                                    if any(part in dims["src"] for part in image_url.split("/")[-2:]):
                                        img_element = img
                                        break
                            except:
                                continue

                    if img_element:
                        img_element.scroll_into_view_if_needed()
                        page.wait_for_timeout(500)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        img_element.screenshot(path=output_path)

                        dimensions = img_element.evaluate(
                            "el => ({width: el.naturalWidth, height: el.naturalHeight})"
                        )
                        return {
                            "success": True,
                            "path": output_path,
                            "width": dimensions["width"],
                            "height": dimensions["height"],
                            "method": "source_page",
                        }

                except Exception as e:
                    pass

            return {"success": False, "error": "Could not capture image"}

    except Exception as e:
        return {"success": False, "error": str(e)}

    finally:
        if context:
            try:
                context.close()
            except:
                pass


if __name__ == "__main__":
    # Read args from stdin as JSON
    args = json.loads(sys.stdin.read())
    result = capture_screenshot(
        image_url=args["image_url"],
        source_page_url=args.get("source_page_url"),
        output_path=args["output_path"],
        timeout=args.get("timeout", 15000),
    )
    print(json.dumps(result))
'''


def _run_playwright_subprocess(
    image_url: str,
    source_page_url: Optional[str],
    output_path: Path,
    timeout: int = 15000,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Run Playwright screenshot in a subprocess.

    This avoids asyncio compatibility issues on Windows Python 3.14+.

    Args:
        image_url: URL of the image to capture.
        source_page_url: URL of the page containing the image (fallback).
        output_path: Path to save the screenshot.
        timeout: Timeout in milliseconds.
        log_callback: Optional callback for progress logging.

    Returns:
        Path to the saved screenshot, or None if failed.
    """
    global _playwright_broken, _playwright_broken_reason

    def log(msg: str):
        if log_callback:
            log_callback(msg)
        logger.debug(f"HeadlessScreenshot: {msg}")

    with _playwright_lock:
        if _playwright_broken:
            log(f"Skipped: {_playwright_broken_reason}")
            return None

    # Ensure browsers are installed
    if not _ensure_browsers_installed():
        with _playwright_lock:
            _playwright_broken = True
            _playwright_broken_reason = "Chromium not installed"
        log(f"Disabled: {_playwright_broken_reason}")
        return None

    # Prepare args for subprocess
    args = {
        "image_url": image_url,
        "source_page_url": source_page_url,
        "output_path": str(output_path),
        "timeout": timeout,
    }

    env = os.environ.copy()
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(_get_browsers_path())

    try:
        log("Starting screenshot subprocess...")

        # Run the inline script as a subprocess
        result = subprocess.run(
            [sys.executable, "-c", _PLAYWRIGHT_SCRIPT],
            input=json.dumps(args),
            capture_output=True,
            text=True,
            env=env,
            timeout=60,  # 60 second timeout for entire operation
        )

        if result.returncode != 0:
            error = result.stderr.strip() if result.stderr else "Unknown error"
            # Check for known platform issues
            if "NotImplementedError" in error or "subprocess" in error.lower():
                with _playwright_lock:
                    _playwright_broken = True
                    _playwright_broken_reason = "Platform not supported"
                log(f"Disabled: {_playwright_broken_reason}")
            else:
                log(f"Subprocess failed: {error[:50]}")
            return None

        # Parse result
        try:
            output = json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            log(f"Invalid response: {result.stdout[:50]}")
            return None

        if output.get("success"):
            path = Path(output["path"])
            if path.exists():
                method = output.get("method", "unknown")
                width = output.get("width", "?")
                height = output.get("height", "?")
                log(f"Captured ({width}x{height}) via {method}")
                return path

        error = output.get("error", "Unknown error")
        log(f"Capture failed: {error[:50]}")
        return None

    except subprocess.TimeoutExpired:
        log("Subprocess timed out")
        return None
    except Exception as e:
        log(f"Subprocess error: {str(e)[:50]}")
        return None


async def capture_image_screenshot(
    image_url: str,
    source_page_url: Optional[str],
    output_path: Path,
    timeout: int = 15000,
    use_tor: bool = False,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    """Capture screenshot of an image using headless browser.

    Runs Playwright in a subprocess to avoid asyncio issues on Windows.

    Args:
        image_url: Direct URL to the image.
        source_page_url: URL of the page where image was found (for fallback).
        output_path: Path to save the screenshot.
        timeout: Page load timeout in milliseconds.
        use_tor: Unused (Playwright doesn't support embedded Tor).
        log_callback: Optional callback for logging progress.

    Returns:
        Path to saved screenshot, or None if failed.
    """
    # Run subprocess in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _run_playwright_subprocess,
        image_url,
        source_page_url,
        output_path,
        timeout,
        log_callback,
    )


async def close_browser():
    """Close the headless browser if open (no-op for subprocess mode)."""
    # Subprocess mode doesn't need cleanup - each call is independent
    pass


def is_playwright_available() -> bool:
    """Check if Playwright is installed and usable on this platform."""
    global _playwright_broken

    if _playwright_broken:
        return False

    try:
        import playwright
        return True
    except ImportError:
        return False


def get_browsers_info() -> dict:
    """Get information about sandboxed browser installation.

    Returns:
        Dict with installation status and paths.
    """
    browsers_path = _get_browsers_path()
    chromium_folders = list(browsers_path.glob("chromium-*"))

    return {
        "browsers_path": str(browsers_path),
        "playwright_installed": is_playwright_available(),
        "chromium_installed": len(chromium_folders) > 0,
        "chromium_versions": [f.name for f in chromium_folders],
        "subprocess_mode": True,
    }
