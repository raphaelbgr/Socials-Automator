"""Hook slide job for generating the first slide of a carousel.

The hook slide captures attention and draws viewers into the content.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import SlideJob, SlideJobContext, SlideJobResult

if TYPE_CHECKING:
    from ..models import SlideType
    from ...providers import ImageProvider
    from ...design import SlideComposer


class HookSlideJob(SlideJob):
    """Generates hook (first) slide.

    The hook slide is designed to:
    - Capture attention immediately
    - Create curiosity or interest
    - Include a compelling headline
    - Optionally include a background image

    Usage:
        job = HookSlideJob(image_provider, composer)
        context = SlideJobContext(
            post_id="test-001",
            slide_number=1,
            topic="5 AI Tools",
            outline={"heading": "Amazing Tools", "needs_image": True},
            ...
        )
        result = await job.execute(context)
    """

    def get_slide_type(self) -> "SlideType":
        """Return the hook slide type."""
        from ..models import SlideType
        return SlideType.HOOK

    async def execute(self, context: SlideJobContext) -> SlideJobResult:
        """Generate a hook slide.

        Args:
            context: Slide generation context.

        Returns:
            SlideJobResult with hook slide content.
        """
        from ..models import SlideContent, SlideType
        from ...design import HookSlideTemplate

        outline = context.outline

        # Build slide content model
        heading = self._get_heading(outline, context.topic)
        body = self._get_body(outline)

        slide = SlideContent(
            number=1,
            slide_type=SlideType.HOOK,
            heading=heading,
            body=body,
            has_background_image=outline.get("needs_image", True),
        )

        # Generate background image if needed
        image_bytes: bytes | None = None
        image_prompt: str | None = None

        if slide.has_background_image:
            image_prompt = outline.get(
                "image_description",
                f"Person working at cozy cafe with natural light, warm lifestyle aesthetic"
            )
            image_bytes = await self._generate_image(
                prompt=image_prompt,
                context=context,
                task="hook_images",
            )

            # Update slide with image info
            if image_bytes:
                slide.image_prompt = f"{image_prompt}, {context.get_image_style()}"
            else:
                slide.has_background_image = False

        # Emit progress for composition
        await self._emit_progress({
            "action": "composing_slide",
            "slide_number": context.slide_number,
            "slide_type": "hook",
        })

        # Get logo path
        logo_path = Path(context.logo_path) if context.logo_path else None

        # Compose the slide image
        template = HookSlideTemplate()
        composed_bytes = await self.composer.create_hook_slide(
            text=slide.heading,
            subtext=slide.body,
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
                "has_background": image_bytes is not None,
            },
        )
