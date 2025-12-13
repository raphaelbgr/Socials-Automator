"""Slide generation jobs module.

Provides SlideJob classes for each slide type following SOLID principles:
- Single Responsibility: Each job handles one slide type
- Open/Closed: New slide types can be added by creating new jobs
- Liskov Substitution: All jobs inherit from SlideJob base
- Interface Segregation: Jobs only depend on what they need
- Dependency Inversion: Jobs depend on abstractions (providers, composers)

Usage:
    from socials_automator.content.slides import SlideJobFactory, SlideJobContext

    # Create a job for a specific slide type
    job = SlideJobFactory.create(
        slide_type=SlideType.HOOK,
        image_provider=image_provider,
        composer=composer,
    )

    # Execute the job
    context = SlideJobContext(...)
    result = await job.execute(context)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import SlideJob, SlideJobContext, SlideJobResult, ProgressCallback
from .hook_job import HookSlideJob
from .content_job import ContentSlideJob
from .cta_job import CTASlideJob

if TYPE_CHECKING:
    from ..models import SlideType
    from ...providers import ImageProvider
    from ...design import SlideComposer


class SlideJobFactory:
    """Factory for creating slide jobs.

    Uses the Factory pattern to create appropriate job classes based on
    slide type. Supports runtime registration of new slide types.

    Usage:
        # Get a job for a specific slide type
        job = SlideJobFactory.create(
            slide_type=SlideType.HOOK,
            image_provider=provider,
            composer=composer,
        )

        # Register a custom slide type
        SlideJobFactory.register(SlideType.CUSTOM, CustomSlideJob)
    """

    # Registry of slide types to job classes
    _job_classes: dict["SlideType", type[SlideJob]] = {}

    @classmethod
    def _ensure_registered(cls) -> None:
        """Ensure default job classes are registered."""
        if not cls._job_classes:
            from ..models import SlideType
            cls._job_classes = {
                SlideType.HOOK: HookSlideJob,
                SlideType.CONTENT: ContentSlideJob,
                SlideType.CTA: CTASlideJob,
            }

    @classmethod
    def register(cls, slide_type: "SlideType", job_class: type[SlideJob]) -> None:
        """Register a new slide job type.

        Args:
            slide_type: SlideType enum value.
            job_class: SlideJob class to use for this type.
        """
        cls._ensure_registered()
        cls._job_classes[slide_type] = job_class

    @classmethod
    def create(
        cls,
        slide_type: "SlideType",
        image_provider: "ImageProvider",
        composer: "SlideComposer",
        progress_callback: ProgressCallback = None,
    ) -> SlideJob:
        """Create a slide job for the given type.

        Args:
            slide_type: Type of slide to create.
            image_provider: Provider for image generation.
            composer: Composer for slide rendering.
            progress_callback: Optional progress callback.

        Returns:
            Appropriate SlideJob instance.

        Raises:
            ValueError: If no job is registered for the slide type.
        """
        cls._ensure_registered()

        job_class = cls._job_classes.get(slide_type)
        if not job_class:
            raise ValueError(f"No job registered for slide type: {slide_type}")

        return job_class(
            image_provider=image_provider,
            composer=composer,
            progress_callback=progress_callback,
        )

    @classmethod
    def get_registered_types(cls) -> list["SlideType"]:
        """Get list of registered slide types.

        Returns:
            List of registered SlideType values.
        """
        cls._ensure_registered()
        return list(cls._job_classes.keys())


__all__ = [
    # Base classes
    "SlideJob",
    "SlideJobContext",
    "SlideJobResult",
    "ProgressCallback",
    # Job classes
    "HookSlideJob",
    "ContentSlideJob",
    "CTASlideJob",
    # Factory
    "SlideJobFactory",
]
