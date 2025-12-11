"""Design module for image composition and carousel creation."""

from .composer import SlideComposer
from .templates import SlideTemplate, HookSlideTemplate, ContentSlideTemplate, CTASlideTemplate

__all__ = [
    "SlideComposer",
    "SlideTemplate",
    "HookSlideTemplate",
    "ContentSlideTemplate",
    "CTASlideTemplate",
]
