"""Content slide job for generating middle slides of a carousel.

Content slides deliver the main value - tips, steps, information.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import SlideJob, SlideJobContext, SlideJobResult

if TYPE_CHECKING:
    from ..models import SlideType
    from ...providers import ImageProvider
    from ...design import SlideComposer


class ContentSlideJob(SlideJob):
    """Generates content (middle) slides.

    Content slides are the core of the carousel, delivering:
    - Tips, steps, or key points
    - Educational content
    - Value-driven information
    - Optional numbered formatting

    Usage:
        job = ContentSlideJob(image_provider, composer)
        context = SlideJobContext(
            post_id="test-001",
            slide_number=2,
            topic="5 AI Tools",
            outline={"heading": "Tool #1: ChatGPT", "body": "Best for..."},
            ...
        )
        result = await job.execute(context)
    """

    def get_slide_type(self) -> "SlideType":
        """Return the content slide type."""
        from ..models import SlideType
        return SlideType.CONTENT

    async def execute(self, context: SlideJobContext) -> SlideJobResult:
        """Generate a content slide.

        Args:
            context: Slide generation context.

        Returns:
            SlideJobResult with content slide.
        """
        from ..models import SlideContent, SlideType
        from ...design import ContentSlideTemplate

        outline = context.outline

        # Build slide content model
        heading = self._get_heading(
            outline,
            f"Point {context.slide_number - 1}"  # Exclude hook in numbering
        )
        body = self._get_body(outline)

        slide = SlideContent(
            number=context.slide_number,
            slide_type=SlideType.CONTENT,
            heading=heading,
            body=body,
            has_background_image=outline.get("needs_image", False),
        )

        # Generate background image if needed
        image_bytes: bytes | None = None
        image_prompt: str | None = None

        if slide.has_background_image and outline.get("image_description"):
            image_prompt = outline["image_description"]
            image_bytes = await self._generate_image(
                prompt=image_prompt,
                context=context,
                task="content_images",
            )

            if image_bytes:
                slide.image_prompt = f"{image_prompt}, {context.get_image_style()}"
            else:
                slide.has_background_image = False

        # Emit progress for composition
        await self._emit_progress({
            "action": "composing_slide",
            "slide_number": context.slide_number,
            "slide_type": "content",
        })

        # Get logo path
        logo_path = Path(context.logo_path) if context.logo_path else None

        # Compose the slide image
        template = ContentSlideTemplate()
        composed_bytes = await self.composer.create_content_slide(
            heading=slide.heading,
            body=slide.body,
            number=context.slide_number - 1 if context.slide_number > 1 else None,
            template=template,
            background_image=image_bytes,
            logo_path=logo_path,
        )

        slide.image_bytes = composed_bytes

        return SlideJobResult(
            slide_content=slide,
            image_bytes=composed_bytes,
            metadata={
                "image_prompt": image_prompt,
                "display_number": context.slide_number - 1,
            },
        )
