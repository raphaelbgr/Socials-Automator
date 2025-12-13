"""CTA slide job for generating the last slide of a carousel.

The CTA (Call-to-Action) slide drives engagement and follows.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .base import SlideJob, SlideJobContext, SlideJobResult

if TYPE_CHECKING:
    from ..models import SlideType
    from ...providers import ImageProvider
    from ...design import SlideComposer


class CTASlideJob(SlideJob):
    """Generates CTA (last) slide.

    The CTA slide encourages viewers to:
    - Follow the account
    - Save the post
    - Share with others
    - Take action

    Usage:
        job = CTASlideJob(image_provider, composer)
        context = SlideJobContext(
            post_id="test-001",
            slide_number=7,
            topic="5 AI Tools",
            outline={"heading": "Follow for more!", "needs_image": True},
            ...
        )
        result = await job.execute(context)
    """

    def get_slide_type(self) -> "SlideType":
        """Return the CTA slide type."""
        from ..models import SlideType
        return SlideType.CTA

    async def execute(self, context: SlideJobContext) -> SlideJobResult:
        """Generate a CTA slide.

        Args:
            context: Slide generation context.

        Returns:
            SlideJobResult with CTA slide content.
        """
        from ..models import SlideContent, SlideType
        from ...design import CTASlideTemplate

        outline = context.outline
        handle = context.get_handle()

        # Build slide content model
        heading = self._get_heading(outline, "Follow for more!")
        body = self._get_body(outline)

        slide = SlideContent(
            number=context.slide_number,
            slide_type=SlideType.CTA,
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
                f"Abstract CTA background for {context.topic}, inspiring, motivational"
            )
            image_bytes = await self._generate_image(
                prompt=image_prompt,
                context=context,
                task="cta_images",
            )

            if image_bytes:
                slide.image_prompt = f"{image_prompt}, {context.get_image_style()}"
            else:
                slide.has_background_image = False

        # Emit progress for composition
        await self._emit_progress({
            "action": "composing_slide",
            "slide_number": context.slide_number,
            "slide_type": "cta",
        })

        # Get logo path
        logo_path = Path(context.logo_path) if context.logo_path else None

        # Compose the slide image
        template = CTASlideTemplate()
        composed_bytes = await self.composer.create_cta_slide(
            text=slide.heading,
            handle=handle,
            secondary_text=slide.body,
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
                "handle": handle,
            },
        )
