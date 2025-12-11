"""Slide templates for Instagram carousels."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class TextAlignment(str, Enum):
    """Text alignment options."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class VerticalAlignment(str, Enum):
    """Vertical alignment options."""

    TOP = "top"
    CENTER = "center"
    BOTTOM = "bottom"


@dataclass
class ColorScheme:
    """Color scheme for slides."""

    background: str = "#0a0a0f"
    text_primary: str = "#ffffff"
    text_secondary: str = "#a1a1aa"
    accent: str = "#6366f1"
    accent_secondary: str = "#8b5cf6"

    # Gradient colors
    gradient_start: str = "#6366f1"
    gradient_end: str = "#8b5cf6"

    def get_gradient_colors(self) -> tuple[str, str]:
        """Get gradient color tuple."""
        return (self.gradient_start, self.gradient_end)


@dataclass
class Typography:
    """Typography settings."""

    heading_font: str = "Inter-Bold"
    body_font: str = "Inter-Regular"
    accent_font: str = "JetBrainsMono-Regular"

    # Font sizes
    hook_size: int = 72
    heading_size: int = 56
    body_size: int = 36
    caption_size: int = 28
    number_size: int = 120

    # Line heights (as multiplier)
    line_height: float = 1.3


@dataclass
class SlideTemplate:
    """Base template for carousel slides."""

    name: str
    width: int = 1080
    height: int = 1080  # Square format for Instagram

    # Colors
    colors: ColorScheme = field(default_factory=ColorScheme)

    # Typography
    typography: Typography = field(default_factory=Typography)

    # Padding
    padding_x: int = 80
    padding_y: int = 80

    # Text alignment
    text_align: TextAlignment = TextAlignment.CENTER
    vertical_align: VerticalAlignment = VerticalAlignment.CENTER

    # Background
    background_type: Literal["solid", "gradient", "image"] = "solid"
    background_image_overlay: float = 0.7  # Opacity of dark overlay on images

    # Logo/Branding
    show_logo: bool = True
    logo_position: Literal["top-left", "top-right", "bottom-left", "bottom-right"] = "bottom-right"
    logo_size: int = 60
    logo_padding: int = 40

    # Handle
    show_handle: bool = False
    handle_position: Literal["top", "bottom"] = "bottom"


@dataclass
class HookSlideTemplate(SlideTemplate):
    """Template for hook/first slide."""

    name: str = "hook"
    background_type: Literal["solid", "gradient", "image"] = "gradient"
    text_align: TextAlignment = TextAlignment.CENTER
    vertical_align: VerticalAlignment = VerticalAlignment.CENTER

    # Increased padding for Instagram grid view (text gets cut on edges)
    padding_x: int = 140
    padding_y: int = 100

    # Hook-specific settings - larger fonts for mobile readability
    hook_font_size: int = 78
    subtext_font_size: int = 36
    subtext_color: str = "#a1a1aa"

    # Emphasis styling
    emphasis_color: str = "#f59e0b"  # Amber for highlighted words
    use_all_caps: bool = False


@dataclass
class ContentSlideTemplate(SlideTemplate):
    """Template for content/information slides."""

    name: str = "content"
    background_type: Literal["solid", "gradient", "image"] = "solid"
    text_align: TextAlignment = TextAlignment.LEFT
    vertical_align: VerticalAlignment = VerticalAlignment.CENTER

    # Number styling (for numbered lists)
    show_number: bool = True
    number_font_size: int = 140
    number_color: str = "#6366f1"  # Gradient or solid
    number_opacity: float = 0.3
    number_position: Literal["left", "background"] = "background"

    # Content text - larger fonts for mobile readability
    heading_font_size: int = 58
    body_font_size: int = 40
    body_color: str = "#d4d4d8"

    # Icon (optional)
    show_icon: bool = False
    icon_size: int = 48
    icon_color: str = "#6366f1"


@dataclass
class CTASlideTemplate(SlideTemplate):
    """Template for call-to-action/final slide."""

    name: str = "cta"
    background_type: Literal["solid", "gradient", "image"] = "solid"  # Black background
    text_align: TextAlignment = TextAlignment.CENTER
    vertical_align: VerticalAlignment = VerticalAlignment.CENTER

    # CTA-specific - larger fonts for mobile readability
    cta_font_size: int = 64
    secondary_font_size: int = 42
    secondary_color: str = "#a1a1aa"

    # Handle display
    show_handle: bool = True
    handle_font_size: int = 48
    handle_color: str = "#ffffff"

    # Action icons
    show_icons: bool = True  # Save, Share, Follow icons


# Pre-built template presets
TEMPLATE_PRESETS = {
    "minimal_dark": {
        "colors": ColorScheme(
            background="#0a0a0f",
            text_primary="#ffffff",
            accent="#6366f1",
        ),
    },
    "minimal_light": {
        "colors": ColorScheme(
            background="#ffffff",
            text_primary="#18181b",
            text_secondary="#71717a",
            accent="#6366f1",
        ),
    },
    "gradient_purple": {
        "colors": ColorScheme(
            background="#0a0a0f",
            gradient_start="#6366f1",
            gradient_end="#8b5cf6",
        ),
    },
    "gradient_blue": {
        "colors": ColorScheme(
            background="#0a0a0f",
            gradient_start="#3b82f6",
            gradient_end="#06b6d4",
        ),
    },
}
