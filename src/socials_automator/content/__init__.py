"""Content generation module for creating carousel posts.

This module provides a SOLID-compliant architecture for generating
Instagram carousel posts.

Architecture:
- ContentOrchestrator: Main coordinator that wires components together
- ContentPlanner: Plans post structure and content
- SlideJob classes: Handle individual slide generation
- Services: Cross-cutting concerns (progress, output)

SOLID Principles:
- Single Responsibility: Each class has one job
- Open/Closed: Add new slide types without modifying existing code
- Liskov Substitution: All SlideJobs are interchangeable
- Interface Segregation: Components depend only on what they need
- Dependency Inversion: High-level modules don't depend on low-level details
"""

from .planner import ContentPlanner
from .orchestrator import ContentOrchestrator
from .models import (
    CarouselPost,
    SlideContent,
    PostPlan,
    SlideType,
    HookType,
    GenerationProgress,
)
from .slides import (
    SlideJob,
    SlideJobContext,
    SlideJobResult,
    SlideJobFactory,
    HookSlideJob,
    ContentSlideJob,
    CTASlideJob,
)

__all__ = [
    # Main classes
    "ContentOrchestrator",
    "ContentPlanner",
    # Models
    "CarouselPost",
    "SlideContent",
    "PostPlan",
    "SlideType",
    "HookType",
    "GenerationProgress",
    # Slide jobs
    "SlideJob",
    "SlideJobContext",
    "SlideJobResult",
    "SlideJobFactory",
    "HookSlideJob",
    "ContentSlideJob",
    "CTASlideJob",
]
