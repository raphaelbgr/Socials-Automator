"""Services module for cross-cutting concerns.

Provides services that handle responsibilities across the application:
- ProgressManager: Tracks and reports generation progress
- OutputService: Handles saving posts to disk
- StructuredExtractor: Extracts structured JSON from AI using Instructor

These services follow the Single Responsibility Principle - each handles
one specific concern.
"""

from .progress import ProgressManager
from .output import OutputService
from .extractor import StructuredExtractor, get_extractor

__all__ = [
    "ProgressManager",
    "OutputService",
    "StructuredExtractor",
    "get_extractor",
]
