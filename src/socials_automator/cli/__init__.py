"""Refactored CLI package - modular, feature-based, stateless architecture.

This package provides a clean separation of concerns:
- core/: Shared utilities (types, validators, parsers, paths)
- reel/: Video reel generation and upload
- post/: Carousel post generation and upload
- profile/: Profile management
- queue/: Post queue management
- maintenance/: Utility and maintenance commands

Usage:
    python -m socials_automator.cli --help
    python -m socials_automator.cli generate-reel <profile>
    python -m socials_automator.cli generate-post <profile>
"""

from .app import app, main

__all__ = ["app", "main"]
