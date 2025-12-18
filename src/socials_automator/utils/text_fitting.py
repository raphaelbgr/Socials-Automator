"""Text fitting utilities for thumbnails and images.

Provides functions to auto-fit text within image containers:
- Calculate max characters per line based on font/width
- Truncate text intelligently (keeping important words)
- Scale font size to fit content
- Handle multi-line text layouts

Usage:
    from socials_automator.utils.text_fitting import TextFitter

    fitter = TextFitter(
        max_width=1080,
        margin_percent=0.06,
        font_path="fonts/Montserrat-Bold.ttf",
    )

    result = fitter.fit_text(
        lines=["JUST IN", "-> Netflix acquires Warner Bros"],
        target_font_size=90,
        min_font_size=60,
    )

    # result.lines = fitted/truncated lines
    # result.font_size = optimal font size
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass
class FitResult:
    """Result of text fitting operation."""

    lines: list[str]
    font_size: int
    total_height: int
    fits: bool
    truncated: bool  # True if any line was truncated


class TextFitter:
    """Fits text within a container by scaling and truncating.

    Uses a hybrid approach:
    1. Try to fit at target font size with truncation
    2. If lines are too many, scale down font size
    3. Respect minimum font size for readability
    """

    # Default truncation settings
    ELLIPSIS = "..."
    MIN_WORD_LENGTH = 2  # Don't truncate words shorter than this

    def __init__(
        self,
        max_width: int,
        max_height: Optional[int] = None,
        margin_percent: float = 0.06,
        font_path: Optional[Path] = None,
        line_spacing_ratio: float = 0.3,
    ):
        """Initialize text fitter.

        Args:
            max_width: Maximum width in pixels.
            max_height: Maximum height in pixels (for multi-line).
            margin_percent: Horizontal margin as percentage (0.06 = 6%).
            font_path: Path to font file.
            line_spacing_ratio: Line spacing as ratio of font size.
        """
        self.max_width = max_width
        self.max_height = max_height
        self.margin_percent = margin_percent
        self.font_path = font_path
        self.line_spacing_ratio = line_spacing_ratio

        # Calculate usable width
        self.margin_px = int(max_width * margin_percent)
        self.usable_width = max_width - (2 * self.margin_px)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load font at specified size."""
        if self.font_path and Path(self.font_path).exists():
            return ImageFont.truetype(str(self.font_path), size)
        # Fallback to default
        try:
            return ImageFont.truetype("arial.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def _measure_text(self, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
        """Measure text width and height.

        Args:
            text: Text to measure.
            font: Font to use.

        Returns:
            Tuple of (width, height).
        """
        # Create dummy image for measurement
        dummy = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    def _estimate_chars_per_line(self, font: ImageFont.FreeTypeFont) -> int:
        """Estimate maximum characters that fit per line.

        Uses average character width for estimation.

        Args:
            font: Font to use.

        Returns:
            Estimated max characters per line.
        """
        # Sample text for measurement (mix of wide and narrow chars)
        sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        width, _ = self._measure_text(sample, font)
        avg_char_width = width / len(sample)

        # Calculate max chars with some safety margin
        max_chars = int(self.usable_width / avg_char_width * 0.95)
        return max(10, max_chars)  # Minimum 10 chars

    def _truncate_line(
        self,
        text: str,
        max_chars: int,
        preserve_prefix: bool = True,
    ) -> tuple[str, bool]:
        """Truncate a line to fit max characters.

        Tries to truncate at word boundaries intelligently.

        Args:
            text: Text to truncate.
            max_chars: Maximum characters allowed.
            preserve_prefix: Keep line prefix like "-> " intact.

        Returns:
            Tuple of (truncated_text, was_truncated).
        """
        if len(text) <= max_chars:
            return text, False

        # Handle prefix preservation (e.g., "-> ")
        prefix = ""
        content = text
        if preserve_prefix and text.startswith("-> "):
            prefix = "-> "
            content = text[3:]
            max_chars -= 3  # Account for prefix

        # Reserve space for ellipsis
        target_length = max_chars - len(self.ELLIPSIS)

        if target_length <= 0:
            return prefix + self.ELLIPSIS, True

        # Try to truncate at word boundary
        words = content.split()
        truncated = ""

        for word in words:
            test = truncated + (" " if truncated else "") + word
            if len(test) <= target_length:
                truncated = test
            else:
                break

        # If we couldn't fit any words, hard truncate
        if not truncated:
            truncated = content[:target_length]

        return prefix + truncated.strip() + self.ELLIPSIS, True

    def _truncate_to_fit(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
    ) -> tuple[str, bool]:
        """Truncate text to fit within usable width.

        Uses actual pixel measurement for accuracy.

        Args:
            text: Text to truncate.
            font: Font to use.

        Returns:
            Tuple of (truncated_text, was_truncated).
        """
        width, _ = self._measure_text(text, font)
        if width <= self.usable_width:
            return text, False

        # Handle prefix preservation
        prefix = ""
        content = text
        if text.startswith("-> "):
            prefix = "-> "
            content = text[3:]

        # Binary search for optimal length
        words = content.split()
        low, high = 0, len(words)
        best_fit = ""

        while low <= high:
            mid = (low + high) // 2
            test_text = " ".join(words[:mid])
            test_full = prefix + test_text + (self.ELLIPSIS if mid < len(words) else "")
            width, _ = self._measure_text(test_full, font)

            if width <= self.usable_width:
                best_fit = test_full
                low = mid + 1
            else:
                high = mid - 1

        if not best_fit or best_fit == prefix + self.ELLIPSIS:
            # Hard truncate character by character
            for i in range(len(content), 0, -1):
                test = prefix + content[:i] + self.ELLIPSIS
                width, _ = self._measure_text(test, font)
                if width <= self.usable_width:
                    return test, True
            return prefix + self.ELLIPSIS, True

        return best_fit, best_fit.endswith(self.ELLIPSIS)

    def fit_text(
        self,
        lines: list[str],
        target_font_size: int = 90,
        min_font_size: int = 60,
        max_lines: Optional[int] = None,
    ) -> FitResult:
        """Fit text lines within container.

        Hybrid approach:
        1. Try target font size with truncation
        2. Scale down if needed for height
        3. Respect minimum font size

        Args:
            lines: Text lines to fit.
            target_font_size: Starting font size.
            min_font_size: Minimum readable font size.
            max_lines: Maximum lines allowed (truncate if exceeded).

        Returns:
            FitResult with fitted lines and optimal font size.
        """
        if not lines:
            return FitResult(
                lines=[],
                font_size=target_font_size,
                total_height=0,
                fits=True,
                truncated=False,
            )

        # Limit lines if max specified
        if max_lines and len(lines) > max_lines:
            lines = lines[:max_lines]

        current_font_size = target_font_size
        any_truncated = False

        while current_font_size >= min_font_size:
            font = self._load_font(current_font_size)
            fitted_lines = []
            truncated_this_pass = False

            # Fit each line
            for line in lines:
                fitted_line, was_truncated = self._truncate_to_fit(line, font)
                fitted_lines.append(fitted_line)
                if was_truncated:
                    truncated_this_pass = True

            # Calculate total height
            line_height = current_font_size
            spacing = int(current_font_size * self.line_spacing_ratio)
            total_height = (line_height * len(fitted_lines)) + (spacing * (len(fitted_lines) - 1))

            # Check if fits in max_height
            if self.max_height is None or total_height <= self.max_height:
                return FitResult(
                    lines=fitted_lines,
                    font_size=current_font_size,
                    total_height=total_height,
                    fits=True,
                    truncated=truncated_this_pass,
                )

            # Reduce font size and try again
            current_font_size -= 5
            any_truncated = truncated_this_pass

        # Couldn't fit even at min size - return best effort
        font = self._load_font(min_font_size)
        fitted_lines = []
        for line in lines:
            fitted_line, was_truncated = self._truncate_to_fit(line, font)
            fitted_lines.append(fitted_line)
            if was_truncated:
                any_truncated = True

        line_height = min_font_size
        spacing = int(min_font_size * self.line_spacing_ratio)
        total_height = (line_height * len(fitted_lines)) + (spacing * (len(fitted_lines) - 1))

        return FitResult(
            lines=fitted_lines,
            font_size=min_font_size,
            total_height=total_height,
            fits=self.max_height is None or total_height <= self.max_height,
            truncated=any_truncated,
        )

    def fit_teaser_text(
        self,
        header: str,
        headlines: list[str],
        header_font_size: int = 90,
        headline_font_size: int = 70,
        min_font_size: int = 50,
        max_headlines: int = 3,
    ) -> FitResult:
        """Fit teaser-style text (header + bullet headlines).

        Specialized for news thumbnail format:
        - Large header (e.g., "JUST IN")
        - Smaller headlines with "-> " prefix

        Args:
            header: Header text (e.g., "JUST IN").
            headlines: List of headlines.
            header_font_size: Font size for header.
            headline_font_size: Font size for headlines.
            min_font_size: Minimum headline font size.
            max_headlines: Maximum headlines to show.

        Returns:
            FitResult with all lines and headline font size.
        """
        # Limit headlines
        headlines = headlines[:max_headlines]

        # Fit header (usually short, no truncation needed)
        header_font = self._load_font(header_font_size)
        header_fitted, header_truncated = self._truncate_to_fit(header, header_font)
        header_height = header_font_size

        # Calculate remaining height for headlines
        remaining_height = None
        if self.max_height:
            spacing = int(header_font_size * self.line_spacing_ratio)
            remaining_height = self.max_height - header_height - spacing

        # Fit headlines with their own fitter
        headline_fitter = TextFitter(
            max_width=self.max_width,
            max_height=remaining_height,
            margin_percent=self.margin_percent,
            font_path=self.font_path,
            line_spacing_ratio=self.line_spacing_ratio,
        )

        # Add "-> " prefix if not present
        prefixed_headlines = []
        for h in headlines:
            if not h.startswith("-> "):
                h = f"-> {h}"
            prefixed_headlines.append(h)

        headline_result = headline_fitter.fit_text(
            lines=prefixed_headlines,
            target_font_size=headline_font_size,
            min_font_size=min_font_size,
        )

        # Combine results
        all_lines = [header_fitted] + headline_result.lines
        total_height = header_height + int(header_font_size * self.line_spacing_ratio) + headline_result.total_height

        return FitResult(
            lines=all_lines,
            font_size=headline_result.font_size,  # Return headline font size
            total_height=total_height,
            fits=headline_result.fits,
            truncated=header_truncated or headline_result.truncated,
        )


def fit_thumbnail_text(
    lines: list[str],
    image_width: int = 1080,
    margin_percent: float = 0.06,
    font_path: Optional[Path] = None,
    target_font_size: int = 90,
    min_font_size: int = 60,
    max_lines: int = 4,
) -> FitResult:
    """Convenience function to fit text for thumbnails.

    Args:
        lines: Text lines to fit.
        image_width: Image width in pixels.
        margin_percent: Horizontal margin percentage.
        font_path: Path to font file.
        target_font_size: Starting font size.
        min_font_size: Minimum font size.
        max_lines: Maximum lines.

    Returns:
        FitResult with fitted text.
    """
    fitter = TextFitter(
        max_width=image_width,
        margin_percent=margin_percent,
        font_path=font_path,
    )
    return fitter.fit_text(
        lines=lines,
        target_font_size=target_font_size,
        min_font_size=min_font_size,
        max_lines=max_lines,
    )


def fit_news_teaser(
    header: str,
    headlines: list[str],
    image_width: int = 1080,
    margin_percent: float = 0.06,
    font_path: Optional[Path] = None,
    header_font_size: int = 90,
    headline_font_size: int = 70,
    min_font_size: int = 50,
    max_headlines: int = 3,
) -> FitResult:
    """Convenience function to fit news teaser text.

    Args:
        header: Header text (e.g., "JUST IN").
        headlines: List of headlines.
        image_width: Image width in pixels.
        margin_percent: Horizontal margin percentage.
        font_path: Path to font file.
        header_font_size: Font size for header.
        headline_font_size: Font size for headlines.
        min_font_size: Minimum headline font size.
        max_headlines: Maximum headlines to show.

    Returns:
        FitResult with fitted text.
    """
    fitter = TextFitter(
        max_width=image_width,
        margin_percent=margin_percent,
        font_path=font_path,
    )
    return fitter.fit_teaser_text(
        header=header,
        headlines=headlines,
        header_font_size=header_font_size,
        headline_font_size=headline_font_size,
        min_font_size=min_font_size,
        max_headlines=max_headlines,
    )
