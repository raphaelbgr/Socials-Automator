"""Base classes for slide generation jobs.

Implements the Single Responsibility Principle - each SlideJob handles
only one type of slide generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import SlideContent, SlideType
    from ...providers import ImageProvider
    from ...design import SlideComposer


# Type for progress callback
ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


@dataclass
class SlideJobContext:
    """Context for slide generation.

    Contains all information needed to generate a single slide.
    """

    post_id: str
    slide_number: int
    topic: str
    outline: dict[str, Any]
    profile_config: dict[str, Any]
    design_config: dict[str, Any]
    logo_path: str | None = None

    def get_handle(self) -> str:
        """Get Instagram handle from profile config."""
        return self.profile_config.get("profile", {}).get("instagram_handle", "")

    def get_image_style(self) -> str:
        """Get image style suffix from design config."""
        return self.design_config.get("image_generation", {}).get(
            "style_prompt_suffix",
            "minimal, clean, tech aesthetic, dark mode"
        )


@dataclass
class SlideJobResult:
    """Result of slide generation.

    Contains the generated slide content and metadata.
    """

    slide_content: "SlideContent"
    image_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if the slide was generated successfully."""
        return self.image_bytes is not None and len(self.image_bytes) > 0


class SlideJob(ABC):
    """Base class for slide generation jobs.

    Each slide type (Hook, Content, CTA) has its own job class that
    inherits from this base. This follows the Open/Closed Principle -
    new slide types can be added without modifying existing code.

    Usage:
        job = HookSlideJob(image_provider, composer)
        result = await job.execute(context)
    """

    def __init__(
        self,
        image_provider: "ImageProvider",
        composer: "SlideComposer",
        progress_callback: ProgressCallback = None,
    ):
        """Initialize the slide job.

        Args:
            image_provider: Provider for generating background images.
            composer: Composer for rendering the final slide.
            progress_callback: Optional callback for progress updates.
        """
        self.image_provider = image_provider
        self.composer = composer
        self.progress_callback = progress_callback

    @abstractmethod
    async def execute(self, context: SlideJobContext) -> SlideJobResult:
        """Execute the slide generation job.

        Args:
            context: Context with all slide generation parameters.

        Returns:
            SlideJobResult with generated content and image.
        """
        pass

    @abstractmethod
    def get_slide_type(self) -> "SlideType":
        """Return the slide type this job handles.

        Returns:
            SlideType enum value.
        """
        pass

    async def _emit_progress(self, update: dict[str, Any]) -> None:
        """Emit a progress update if callback is set.

        Args:
            update: Progress update dictionary.
        """
        if self.progress_callback:
            await self.progress_callback(update)

    async def _generate_image(
        self,
        prompt: str,
        context: SlideJobContext,
        task: str = "content_images",
    ) -> bytes | None:
        """Generate a background image.

        Args:
            prompt: Image generation prompt.
            context: Slide context for configuration.
            task: Task type for provider selection.

        Returns:
            Image bytes or None if generation fails.
        """
        try:
            # Add style suffix to prompt
            full_prompt = f"{prompt}, {context.get_image_style()}"

            await self._emit_progress({
                "action": "generating_image",
                "slide_number": context.slide_number,
                "prompt_preview": full_prompt[:100],
            })

            image_bytes = await self.image_provider.generate(
                prompt=full_prompt,
                size="square",  # Instagram square format
                task=task,
            )

            return image_bytes

        except Exception as e:
            await self._emit_progress({
                "action": "image_error",
                "slide_number": context.slide_number,
                "error": str(e),
            })
            return None

    def _get_heading(
        self,
        outline: dict[str, Any],
        default: str,
    ) -> str:
        """Extract heading from outline with fallback.

        Args:
            outline: Slide outline dictionary.
            default: Default heading if not found.

        Returns:
            Heading string.
        """
        heading = outline.get("heading") or outline.get("title") or ""
        return heading if heading else default

    def _get_body(self, outline: dict[str, Any]) -> str | None:
        """Extract body text from outline.

        Args:
            outline: Slide outline dictionary.

        Returns:
            Body string or None.
        """
        return outline.get("body") or outline.get("content") or outline.get("text")
