"""Typer app configuration and logging setup."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import typer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress httpx/asyncio cleanup warnings (cosmetic issue, doesn't affect functionality)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Fix for Windows asyncio event loop issues with httpx
import sys
import asyncio

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass  # Policy not available in this Python version

# Suppress "Task exception was never retrieved" for httpx cleanup
def _silence_event_loop_closed(func):
    """Wrapper to silence 'Event loop is closed' errors on Windows."""
    from functools import wraps
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
    return wrapper

# Patch all transports to silence the error
try:
    from asyncio.proactor_events import _ProactorBasePipeTransport
    _ProactorBasePipeTransport.__del__ = _silence_event_loop_closed(_ProactorBasePipeTransport.__del__)
except (ImportError, AttributeError):
    pass

try:
    from asyncio.selector_events import _SelectorTransport
    original_close = _SelectorTransport.close
    def patched_close(self):
        try:
            original_close(self)
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise
    _SelectorTransport.close = patched_close
except (ImportError, AttributeError):
    pass

# Custom exception handler to silence "Task exception was never retrieved" for httpx cleanup
def _custom_exception_handler(loop, context):
    """Silence httpx/anyio cleanup errors when event loop is closing."""
    exception = context.get("exception")
    message = context.get("message", "")

    # Silence "Event loop is closed" errors from httpx cleanup
    if exception and isinstance(exception, RuntimeError):
        if "Event loop is closed" in str(exception):
            return  # Silently ignore

    # Silence task cleanup errors
    if "Task exception was never retrieved" in message:
        if exception and "Event loop is closed" in str(exception):
            return  # Silently ignore

    # For all other exceptions, use default handler
    loop.default_exception_handler(context)

# Apply custom exception handler to new event loops
_original_new_event_loop = asyncio.new_event_loop
def _patched_new_event_loop():
    loop = _original_new_event_loop()
    loop.set_exception_handler(_custom_exception_handler)
    return loop
asyncio.new_event_loop = _patched_new_event_loop

# Also patch get_event_loop for cases where it creates a new loop
_original_get_event_loop = asyncio.get_event_loop
def _patched_get_event_loop():
    try:
        loop = _original_get_event_loop()
        if loop.get_exception_handler() is None:
            loop.set_exception_handler(_custom_exception_handler)
        return loop
    except RuntimeError:
        # No running event loop, create a new one
        return _patched_new_event_loop()
asyncio.get_event_loop = _patched_get_event_loop

# Create Typer app
app = typer.Typer(
    name="socials",
    help="AI-powered Instagram content generator",
    add_completion=False,
)


def register_commands() -> None:
    """Register all commands from feature modules."""
    # Import and register reel commands
    from .reel.commands import generate_reel, upload_reel

    app.command(name="generate-reel")(generate_reel)
    app.command(name="upload-reel")(upload_reel)

    # Import and register post commands
    from .post.commands import generate_post, upload_post

    app.command(name="generate-post")(generate_post)
    app.command(name="upload-post")(upload_post)

    # Import and register profile commands
    from .profile.commands import fix_thumbnails, list_profiles

    app.command(name="list-profiles")(list_profiles)
    app.command(name="fix-thumbnails")(fix_thumbnails)

    # Import and register queue commands
    from .queue.commands import queue, schedule

    app.command(name="queue")(queue)
    app.command(name="schedule")(schedule)

    # Import and register maintenance commands
    from .maintenance.commands import (
        cleanup_posted_reels,
        init,
        list_niches,
        migrate_platform_status,
        new_profile,
        status,
        token,
        update_artifacts,
    )

    app.command(name="init")(init)
    app.command(name="token")(token)
    app.command(name="status")(status)
    app.command(name="new-profile")(new_profile)
    app.command(name="list-niches")(list_niches)
    app.command(name="update-artifacts")(update_artifacts)
    app.command(name="migrate-platform-status")(migrate_platform_status)
    app.command(name="cleanup-reels")(cleanup_posted_reels)


def setup_logging() -> None:
    """Configure logging for CLI.

    - Suppresses console output from libraries
    - Sets up file logging for AI calls and Instagram API
    """
    # Get log directory
    log_dir = Path(__file__).parent.parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove any default console handlers from root logger
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.CRITICAL)  # Suppress root logger output

    # Suppress loggers that might print to console
    for logger_name in ["httpx", "httpcore", "urllib3", "asyncio"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    # Setup ai_calls logger with FileHandler for full AI request/response logging
    ai_calls_logger = logging.getLogger("ai_calls")
    ai_calls_logger.setLevel(logging.DEBUG)
    ai_calls_logger.propagate = False
    ai_calls_logger.handlers = []  # Clear any existing handlers
    ai_file_handler = logging.FileHandler(log_dir / "ai_calls.log", encoding="utf-8")
    ai_file_handler.setLevel(logging.DEBUG)
    ai_file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    ai_calls_logger.addHandler(ai_file_handler)

    # Ensure instagram_api logger doesn't propagate to console
    instagram_logger = logging.getLogger("instagram_api")
    instagram_logger.propagate = False


# Initialize logging on module import
setup_logging()

# Register all commands
register_commands()


def main() -> None:
    """CLI entry point."""
    app()
