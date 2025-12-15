"""Slide composer for creating Instagram carousel images with Pillow."""

from __future__ import annotations

import math
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .templates import (
    SlideTemplate,
    HookSlideTemplate,
    ContentSlideTemplate,
    CTASlideTemplate,
    TextAlignment,
    VerticalAlignment,
    ColorScheme,
)


class SlideComposer:
    """Compose carousel slides using Pillow.

    Creates professional Instagram carousel slides with:
    - Text overlays with custom fonts
    - Gradient backgrounds
    - Image backgrounds with overlays
    - Numbered content slides
    - Logo/watermark placement

    Usage:
        composer = SlideComposer()

        # Create hook slide
        hook = await composer.create_hook_slide(
            text="5 ChatGPT tricks that save me 2 HOURS daily",
            subtext="Copy these prompts today",
            background_image=image_bytes,  # optional
        )

        # Create content slide
        content = await composer.create_content_slide(
            number=1,
            heading="Use the 'Act As' prompt",
            body="Tell ChatGPT to act as an expert...",
        )

        # Create CTA slide
        cta = await composer.create_cta_slide(
            text="Follow for more AI tips",
            handle="@ai.for.mortals",
        )
    """

    # Default font paths - will try these in order
    FONT_PATHS = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]

    def __init__(
        self,
        fonts_dir: Path | None = None,
        default_template: SlideTemplate | None = None,
    ):
        """Initialize the composer.

        Args:
            fonts_dir: Directory containing custom fonts.
            default_template: Default template to use.
        """
        self.fonts_dir = fonts_dir
        self.default_template = default_template or SlideTemplate(name="default")
        self._font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def _hex_to_rgb(self, hex_color: str) -> tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    def _get_font(self, font_name: str, size: int) -> ImageFont.FreeTypeFont:
        """Get a font, with caching."""
        cache_key = (font_name, size)
        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = None

        # Try custom fonts directory first
        if self.fonts_dir:
            font_path = self.fonts_dir / f"{font_name}.ttf"
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size)

        # Try font name directly (might be a path)
        if font is None:
            try:
                font = ImageFont.truetype(font_name, size)
            except (OSError, IOError):
                pass

        # Try system fonts
        if font is None:
            for path in self.FONT_PATHS:
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except (OSError, IOError):
                    continue

        # Fall back to default
        if font is None:
            font = ImageFont.load_default()

        self._font_cache[cache_key] = font
        return font

    def _create_gradient_background(
        self,
        width: int,
        height: int,
        color_start: str,
        color_end: str,
        direction: str = "diagonal",
    ) -> Image.Image:
        """Create a gradient background image."""
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        r1, g1, b1 = self._hex_to_rgb(color_start)
        r2, g2, b2 = self._hex_to_rgb(color_end)

        if direction == "diagonal":
            # Diagonal gradient (top-left to bottom-right)
            for y in range(height):
                for x in range(width):
                    # Calculate position along diagonal
                    t = (x + y) / (width + height)
                    r = int(r1 + (r2 - r1) * t)
                    g = int(g1 + (g2 - g1) * t)
                    b = int(b1 + (b2 - b1) * t)
                    draw.point((x, y), fill=(r, g, b))
        elif direction == "vertical":
            for y in range(height):
                t = y / height
                r = int(r1 + (r2 - r1) * t)
                g = int(g1 + (g2 - g1) * t)
                b = int(b1 + (b2 - b1) * t)
                draw.line([(0, y), (width, y)], fill=(r, g, b))
        else:  # horizontal
            for x in range(width):
                t = x / width
                r = int(r1 + (r2 - r1) * t)
                g = int(g1 + (g2 - g1) * t)
                b = int(b1 + (b2 - b1) * t)
                draw.line([(x, 0), (x, height)], fill=(r, g, b))

        return img

    def _create_solid_background(
        self,
        width: int,
        height: int,
        color: str,
    ) -> Image.Image:
        """Create a solid color background."""
        return Image.new("RGB", (width, height), self._hex_to_rgb(color))

    def _add_image_background(
        self,
        base: Image.Image,
        image_bytes: bytes,
        overlay_opacity: float = 0.7,
    ) -> Image.Image:
        """Add an image as background with dark overlay."""
        # Load and resize image to fit
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")

        # Resize to cover
        img_ratio = img.width / img.height
        base_ratio = base.width / base.height

        if img_ratio > base_ratio:
            # Image is wider - fit height
            new_height = base.height
            new_width = int(new_height * img_ratio)
        else:
            # Image is taller - fit width
            new_width = base.width
            new_height = int(new_width / img_ratio)

        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_width - base.width) // 2
        top = (new_height - base.height) // 2
        img = img.crop((left, top, left + base.width, top + base.height))

        # Add dark overlay
        overlay = Image.new("RGB", (base.width, base.height), (0, 0, 0))
        img = Image.blend(img, overlay, overlay_opacity)

        return img

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
    ) -> list[str]:
        """Wrap text to fit within max_width."""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = font.getbbox(test_line)
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def _draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        color: str,
        x: int,
        y: int,
        max_width: int,
        align: TextAlignment = TextAlignment.CENTER,
        line_height: float = 1.3,
    ) -> int:
        """Draw wrapped text and return the total height used."""
        lines = self._wrap_text(text, font, max_width)
        rgb_color = self._hex_to_rgb(color)

        total_height = 0
        for line in lines:
            bbox = font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            line_height_px = bbox[3] - bbox[1]

            # Calculate x position based on alignment
            if align == TextAlignment.CENTER:
                line_x = x + (max_width - line_width) // 2
            elif align == TextAlignment.RIGHT:
                line_x = x + max_width - line_width
            else:
                line_x = x

            draw.text((line_x, y + total_height), line, font=font, fill=rgb_color)
            total_height += int(line_height_px * line_height)

        return total_height

    def _add_logo(
        self,
        img: Image.Image,
        logo_path: Path | None = None,
        position: str = "bottom-right",
        size: int = 60,
        padding: int = 40,
    ) -> Image.Image:
        """Add logo/watermark to image."""
        if logo_path is None or not logo_path.exists():
            return img

        logo = Image.open(logo_path)
        logo = logo.convert("RGBA")

        # Resize logo maintaining aspect ratio
        ratio = size / max(logo.width, logo.height)
        new_size = (int(logo.width * ratio), int(logo.height * ratio))
        logo = logo.resize(new_size, Image.Resampling.LANCZOS)

        # Calculate position
        if position == "bottom-right":
            x = img.width - logo.width - padding
            y = img.height - logo.height - padding
        elif position == "bottom-left":
            x = padding
            y = img.height - logo.height - padding
        elif position == "top-right":
            x = img.width - logo.width - padding
            y = padding
        else:  # top-left
            x = padding
            y = padding

        # Paste with alpha
        img.paste(logo, (x, y), logo)
        return img

    async def create_hook_slide(
        self,
        text: str,
        subtext: str | None = None,
        template: HookSlideTemplate | None = None,
        background_image: bytes | None = None,
        logo_path: Path | None = None,
    ) -> bytes:
        """Create a hook/first slide.

        Args:
            text: Main hook text.
            subtext: Optional subtitle.
            template: Slide template to use.
            background_image: Optional background image bytes.
            logo_path: Path to logo file.

        Returns:
            Slide image as JPEG bytes.
        """
        template = template or HookSlideTemplate()

        # Create background
        if background_image:
            base = self._create_solid_background(
                template.width, template.height, template.colors.background
            )
            img = self._add_image_background(
                base, background_image, template.background_image_overlay
            )
        elif template.background_type == "gradient":
            img = self._create_gradient_background(
                template.width,
                template.height,
                template.colors.gradient_start,
                template.colors.gradient_end,
            )
        else:
            img = self._create_solid_background(
                template.width, template.height, template.colors.background
            )

        draw = ImageDraw.Draw(img)

        # Get fonts
        hook_font = self._get_font(template.typography.heading_font, template.hook_font_size)
        subtext_font = self._get_font(template.typography.body_font, template.subtext_font_size)

        # Calculate text area
        text_area_width = template.width - (template.padding_x * 2)
        text_area_height = template.height - (template.padding_y * 2)

        # Process text (optionally uppercase)
        display_text = text.upper() if template.use_all_caps else text

        # Wrap and measure text
        lines = self._wrap_text(display_text, hook_font, text_area_width)
        line_height = hook_font.getbbox("Ay")[3] - hook_font.getbbox("Ay")[1]
        total_text_height = len(lines) * int(line_height * template.typography.line_height)

        if subtext:
            # Wrap subtext too for proper height calculation
            subtext_lines = self._wrap_text(subtext, subtext_font, text_area_width)
            subtext_line_height = subtext_font.getbbox("Ay")[3] - subtext_font.getbbox("Ay")[1]
            subtext_height = len(subtext_lines) * int(subtext_line_height * template.typography.line_height)
            total_text_height += subtext_height + 40  # 40px gap

        # Calculate starting Y position for vertical centering
        start_y = (template.height - total_text_height) // 2

        # Draw main text
        current_y = start_y
        for line in lines:
            bbox = hook_font.getbbox(line)
            line_width = bbox[2] - bbox[0]
            x = (template.width - line_width) // 2
            draw.text((x, current_y), line, font=hook_font, fill=self._hex_to_rgb(template.colors.text_primary))
            current_y += int(line_height * template.typography.line_height)

        # Draw subtext - with same horizontal padding as main text
        # This is critical for hook slide since Instagram displays it in 4:3 container
        if subtext:
            current_y += 40  # Gap
            # Wrap subtext within the same text_area_width as main text
            subtext_lines = self._wrap_text(subtext, subtext_font, text_area_width)
            subtext_line_height = subtext_font.getbbox("Ay")[3] - subtext_font.getbbox("Ay")[1]

            for line in subtext_lines:
                bbox = subtext_font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                # Center within the padded text area
                x = template.padding_x + (text_area_width - line_width) // 2
                draw.text((x, current_y), line, font=subtext_font, fill=self._hex_to_rgb(template.subtext_color))
                current_y += int(subtext_line_height * template.typography.line_height)

        # Add logo
        if logo_path and template.show_logo:
            img = self._add_logo(img, logo_path, template.logo_position, template.logo_size, template.logo_padding)

        # Convert to bytes (high quality JPEG)
        output = BytesIO()
        img.save(output, format="JPEG", quality=98, subsampling=0, optimize=True)
        return output.getvalue()

    async def create_content_slide(
        self,
        heading: str,
        body: str | None = None,
        number: int | None = None,
        template: ContentSlideTemplate | None = None,
        background_image: bytes | None = None,
        logo_path: Path | None = None,
    ) -> bytes:
        """Create a content/information slide.

        Args:
            heading: Main heading text.
            body: Optional body text.
            number: Optional number for numbered lists.
            template: Slide template to use.
            background_image: Optional background image.
            logo_path: Path to logo file.

        Returns:
            Slide image as JPEG bytes.
        """
        template = template or ContentSlideTemplate()

        # Create background
        if background_image:
            base = self._create_solid_background(
                template.width, template.height, template.colors.background
            )
            img = self._add_image_background(
                base, background_image, template.background_image_overlay
            )
        else:
            img = self._create_solid_background(
                template.width, template.height, template.colors.background
            )

        draw = ImageDraw.Draw(img)

        # Draw large background number if enabled
        if number is not None and template.show_number and template.number_position == "background":
            number_font = self._get_font(template.typography.heading_font, template.number_font_size)
            number_text = str(number)
            bbox = number_font.getbbox(number_text)
            number_height = bbox[3] - bbox[1]

            # Position in top-left with proper offset to avoid cutoff
            number_x = template.padding_x
            number_y = template.padding_y + 20  # Extra offset to prevent cutoff

            # Draw with low opacity (create semi-transparent layer)
            number_color = self._hex_to_rgb(template.number_color)
            alpha_color = tuple(list(number_color) + [int(255 * template.number_opacity)])

            # Create overlay for number
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.text((number_x, number_y), number_text, font=number_font, fill=alpha_color)
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
            draw = ImageDraw.Draw(img)

        # Calculate text area
        text_area_width = template.width - (template.padding_x * 2)
        max_content_height = template.height - (template.padding_y * 2) - 200  # Reserve space for number/logo

        # Dynamic font sizing based on content length for mobile readability
        heading_len = len(heading)
        body_len = len(body) if body else 0
        total_len = heading_len + body_len

        if total_len < 50:
            # Very short - use extra large fonts
            heading_size = int(template.heading_font_size * 1.5)
            body_size = int(template.body_font_size * 1.4)
        elif total_len < 100:
            # Short - use large fonts
            heading_size = int(template.heading_font_size * 1.3)
            body_size = int(template.body_font_size * 1.25)
        elif total_len < 200:
            # Medium - use medium-large fonts
            heading_size = int(template.heading_font_size * 1.15)
            body_size = int(template.body_font_size * 1.1)
        elif total_len < 300:
            # Long content - use default fonts
            heading_size = template.heading_font_size
            body_size = template.body_font_size
        else:
            # Very long content - use smaller fonts
            heading_size = int(template.heading_font_size * 0.85)
            body_size = int(template.body_font_size * 0.85)

        # Get fonts with dynamic sizes
        heading_font = self._get_font(template.typography.heading_font, heading_size)
        body_font = self._get_font(template.typography.body_font, body_size)

        # Pre-calculate total content height for proper vertical centering
        heading_lines = self._wrap_text(heading, heading_font, text_area_width)
        heading_line_height = heading_font.getbbox("Ay")[3] - heading_font.getbbox("Ay")[1]
        total_heading_height = len(heading_lines) * int(heading_line_height * template.typography.line_height)

        body_height = 0
        gap = 50 if total_len < 100 else 40
        if body:
            body_lines = self._wrap_text(body, body_font, text_area_width)
            body_line_height = body_font.getbbox("Ay")[3] - body_font.getbbox("Ay")[1]
            body_height = len(body_lines) * int(body_line_height * template.typography.line_height)

        total_content_height = total_heading_height + (gap + body_height if body else 0)

        # Calculate vertical position - true center for square format
        # Account for the background number taking up some space at top
        available_height = template.height - (template.padding_y * 2)
        if number is not None and template.show_number:
            # Shift content down slightly to account for number
            content_start_y = (template.height - total_content_height) // 2 + 30
        else:
            content_start_y = (template.height - total_content_height) // 2

        # Ensure minimum padding from top
        min_start_y = template.padding_y + 150 if number else template.padding_y
        content_start_y = max(content_start_y, min_start_y)

        # Draw heading
        current_y = content_start_y
        heading_height = self._draw_text(
            draw, heading, heading_font,
            template.colors.text_primary,
            template.padding_x, current_y,
            text_area_width,
            template.text_align,
            template.typography.line_height,
        )

        # Draw body
        if body:
            current_y += heading_height + gap
            self._draw_text(
                draw, body, body_font,
                template.body_color,
                template.padding_x, current_y,
                text_area_width,
                template.text_align,
                template.typography.line_height,
            )

        # Add logo
        if logo_path and template.show_logo:
            img = self._add_logo(img, logo_path, template.logo_position, template.logo_size, template.logo_padding)

        # Convert to bytes (high quality JPEG)
        output = BytesIO()
        img.save(output, format="JPEG", quality=98, subsampling=0, optimize=True)
        return output.getvalue()

    async def create_cta_slide(
        self,
        text: str,
        handle: str | None = None,
        secondary_text: str | None = None,
        template: CTASlideTemplate | None = None,
        background_image: bytes | None = None,
        logo_path: Path | None = None,
    ) -> bytes:
        """Create a call-to-action slide.

        Args:
            text: Main CTA text.
            handle: Instagram handle to display.
            secondary_text: Optional secondary text.
            template: Slide template to use.
            background_image: Optional background image bytes.
            logo_path: Path to logo file.

        Returns:
            Slide image as JPEG bytes.
        """
        template = template or CTASlideTemplate()

        # Create background - use provided image or template setting
        if background_image:
            # Use provided background image with overlay
            base = self._create_solid_background(
                template.width, template.height, template.colors.background
            )
            img = self._add_image_background(
                base, background_image, overlay_opacity=0.7  # Heavier overlay for CTA readability
            )
        elif template.background_type == "solid":
            # Use solid black background
            img = self._create_solid_background(
                template.width,
                template.height,
                template.colors.background,  # Default is #0a0a0f (near black)
            )
        else:
            # Use gradient background
            img = self._create_gradient_background(
                template.width,
                template.height,
                template.colors.gradient_start,
                template.colors.gradient_end,
            )

        draw = ImageDraw.Draw(img)

        # Get fonts
        cta_font = self._get_font(template.typography.heading_font, template.cta_font_size)
        secondary_font = self._get_font(template.typography.body_font, template.secondary_font_size)
        handle_font = self._get_font(template.typography.body_font, template.handle_font_size)

        # Calculate text area width with padding
        text_area_width = template.width - (template.padding_x * 2)

        # Calculate total height for centering
        total_height = 0
        elements = []

        # Main CTA text - wrapped
        cta_lines = self._wrap_text(text, cta_font, text_area_width)
        cta_line_height = cta_font.getbbox("Ay")[3] - cta_font.getbbox("Ay")[1]
        cta_height = len(cta_lines) * int(cta_line_height * template.typography.line_height)
        elements.append(("cta", cta_lines, cta_font, template.colors.text_primary, cta_line_height))
        total_height += cta_height

        # Secondary text - NOW WRAPPED properly
        if secondary_text:
            total_height += 40  # Gap
            secondary_lines = self._wrap_text(secondary_text, secondary_font, text_area_width)
            secondary_line_height = secondary_font.getbbox("Ay")[3] - secondary_font.getbbox("Ay")[1]
            secondary_height = len(secondary_lines) * int(secondary_line_height * template.typography.line_height)
            elements.append(("secondary", secondary_lines, secondary_font, template.secondary_color, secondary_line_height))
            total_height += secondary_height

        # Handle
        if handle and template.show_handle:
            total_height += 60  # Larger gap before handle
            handle_line_height = handle_font.getbbox(handle)[3] - handle_font.getbbox(handle)[1]
            elements.append(("handle", [handle], handle_font, template.handle_color, handle_line_height))
            total_height += handle_line_height

        # Draw elements centered
        current_y = (template.height - total_height) // 2

        for elem_type, lines, font, color, line_height in elements:
            if elem_type == "secondary":
                current_y += 40
            elif elem_type == "handle":
                current_y += 60

            for line in lines:
                bbox = font.getbbox(line)
                line_width = bbox[2] - bbox[0]
                x = (template.width - line_width) // 2
                draw.text((x, current_y), line, font=font, fill=self._hex_to_rgb(color))
                current_y += int(line_height * template.typography.line_height)

        # Add logo
        if logo_path and template.show_logo:
            img = self._add_logo(img, logo_path, template.logo_position, template.logo_size, template.logo_padding)

        # Convert to bytes (high quality JPEG)
        output = BytesIO()
        img.save(output, format="JPEG", quality=98, subsampling=0, optimize=True)
        return output.getvalue()

    async def save_slide(self, slide_bytes: bytes, output_path: Path) -> Path:
        """Save slide to file.

        Args:
            slide_bytes: Slide image bytes.
            output_path: Path to save the image.

        Returns:
            Path to saved file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(slide_bytes)
        return output_path
