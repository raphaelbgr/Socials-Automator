"""Image overlay planning using AI.

Analyzes video script segments to determine:
- Which segments need image overlays
- What type of match is required (exact vs illustrative)
- Search queries for Pexels fallback
- Timing information for each overlay

The AI prioritizes EXACT matches for specific content (TV shows, products,
brands, people) and ILLUSTRATIVE matches for generic concepts.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .base import (
    IImageOverlayPlanner,
    ImageOverlay,
    ImageOverlayScript,
    ImageOverlayError,
    PipelineContext,
    VideoScript,
)
from ...providers.text import TextProvider

logger = logging.getLogger("ai_calls")


# System prompt for image overlay planning
IMAGE_OVERLAY_SYSTEM_PROMPT = """You are an AI that analyzes video scripts to determine where images should appear to illustrate the narration.

Your task is to identify segments that would benefit from visual support and determine the best image to show.

## IMAGE TYPES

1. **EXACT MATCH** (match_type: "exact")
   - Use when discussing specific, identifiable content:
     - TV shows, movies, games (e.g., "Stranger Things", "The Witcher")
     - Products, apps, software (e.g., "ChatGPT", "iPhone 15", "Midjourney")
     - Brands, companies (e.g., "Netflix", "OpenAI", "Apple")
     - Famous people (e.g., "Elon Musk", "Sam Altman")
   - The image MUST be the actual thing being discussed
   - Set confidence HIGH (0.9+) for these

2. **ILLUSTRATIVE** (match_type: "illustrative")
   - Use for abstract concepts, actions, or generic ideas:
     - "productivity tips" -> person working at desk
     - "morning routine" -> coffee cup, sunrise
     - "staying organized" -> planner, to-do list
   - A relevant stock photo is acceptable
   - Set confidence MEDIUM (0.6-0.8)

## WHEN TO SKIP (do not add overlay)

- Opening hooks (segment 0 or first few seconds) - let the video speak
- Call-to-action segments - focus on the message
- Transitions between topics
- When the narration is self-explanatory
- When no good image would add value

## OUTPUT FORMAT

Return valid JSON with this structure:
{
  "overlays": [
    {
      "segment_index": 1,
      "start_time": 5.2,
      "end_time": 15.8,
      "topic": "Stranger Things",
      "match_type": "exact",
      "local_hint": "stranger-things",
      "pexels_query": "stranger things netflix series poster",
      "confidence": 0.95,
      "alt_text": "Stranger Things TV show poster"
    }
  ],
  "skipped": [
    {"segment_index": 0, "reason": "Opening hook - no image needed"},
    {"segment_index": 5, "reason": "CTA segment"}
  ]
}

## RULES

1. local_hint should be lowercase, hyphenated (e.g., "stranger-things", "chat-gpt")
2. pexels_query should be descriptive for stock photo search
3. Only add overlays that genuinely enhance understanding
4. For EXACT matches, be specific in the topic field
5. alt_text should describe what the image shows for accessibility
"""


class ImageOverlayPlanner(IImageOverlayPlanner):
    """Plans image overlays for video segments using AI."""

    def __init__(
        self,
        text_provider: Optional[TextProvider] = None,
        preferred_provider: Optional[str] = None,
    ):
        """Initialize image overlay planner.

        Args:
            text_provider: Optional TextProvider for AI calls.
            preferred_provider: Preferred AI provider name.
        """
        super().__init__()
        self._text_provider = text_provider
        self._preferred_provider = preferred_provider

    def _get_text_provider(self) -> TextProvider:
        """Get or create the text provider."""
        if self._text_provider:
            return self._text_provider

        self._text_provider = TextProvider(
            provider_override=self._preferred_provider
        )
        return self._text_provider

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute image overlay planning step.

        Args:
            context: Pipeline context with script.

        Returns:
            Updated context with image overlay script.
        """
        if not context.script:
            raise ImageOverlayError("No script available for image overlay planning")

        self.log_start("Planning image overlays for narration segments")

        try:
            # Plan overlays
            overlay_script = await self.plan_overlays(
                script=context.script,
                profile_path=context.profile_path,
            )

            context.image_overlays = overlay_script

            # Log results in table format
            self._log_results(overlay_script)

            self.log_success(
                f"{len(overlay_script.overlays)} overlays planned "
                f"({overlay_script.exact_count} exact, {overlay_script.illustrative_count} illustrative)"
            )

            return context

        except Exception as e:
            self.log_error(f"Image overlay planning failed: {e}")
            raise ImageOverlayError(f"Failed to plan image overlays: {e}") from e

    async def plan_overlays(
        self,
        script: VideoScript,
        profile_path: Optional[Path] = None,
    ) -> ImageOverlayScript:
        """Plan image overlays based on script content.

        Args:
            script: Video script with segments and timing.
            profile_path: Path to profile for local image library.

        Returns:
            ImageOverlayScript with planned overlays.
        """
        # Build prompt with script information
        prompt = self._build_prompt(script)

        # Get AI response
        provider = self._get_text_provider()

        self.log_progress(f"[>] Analyzing {len(script.segments)} segments...")

        response = await provider.generate(
            prompt=prompt,
            system=IMAGE_OVERLAY_SYSTEM_PROMPT,
            task_name="image_overlay_planning",
        )

        # Parse response
        overlay_script = self._parse_response(response, script)

        return overlay_script

    def _build_prompt(self, script: VideoScript) -> str:
        """Build the prompt for AI analysis.

        Args:
            script: Video script to analyze.

        Returns:
            Formatted prompt string.
        """
        # Build segment list with timing
        segments_text = []

        # Add hook as segment 0
        segments_text.append(
            f"Segment 0 (Hook): [0.0s - 3.0s]\n"
            f'"{script.hook}"'
        )

        # Add main segments
        for seg in script.segments:
            segments_text.append(
                f"Segment {seg.index}: [{seg.start_time:.1f}s - {seg.end_time:.1f}s]\n"
                f'"{seg.text}"'
            )

        # Add CTA as last segment
        if script.segments:
            last_end = script.segments[-1].end_time
        else:
            last_end = 3.0

        cta_index = len(script.segments) + 1
        segments_text.append(
            f"Segment {cta_index} (CTA): [{last_end:.1f}s - {script.total_duration:.1f}s]\n"
            f'"{script.cta}"'
        )

        prompt = f"""Analyze this video script and plan image overlays:

TITLE: {script.title}

FULL NARRATION:
{script.full_narration}

SEGMENTS WITH TIMING:
{chr(10).join(segments_text)}

TOTAL DURATION: {script.total_duration:.1f}s

Determine which segments need image overlays and return the JSON response.
Remember:
- Use "exact" match_type for specific content (shows, products, brands, people)
- Use "illustrative" for abstract concepts
- Skip hooks and CTAs
- Only add overlays that genuinely enhance the video
"""

        return prompt

    def _parse_response(
        self,
        response: str,
        script: VideoScript,
    ) -> ImageOverlayScript:
        """Parse AI response into ImageOverlayScript.

        Args:
            response: AI response text.
            script: Original script for timing reference.

        Returns:
            Parsed ImageOverlayScript.
        """
        # Extract JSON from response
        json_str = self._extract_json(response)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse AI response as JSON: {e}")
            # Return empty script on parse failure
            return ImageOverlayScript()

        overlays = []
        skipped = data.get("skipped", [])

        for overlay_data in data.get("overlays", []):
            try:
                overlay = ImageOverlay(
                    segment_index=overlay_data.get("segment_index", 0),
                    start_time=float(overlay_data.get("start_time", 0)),
                    end_time=float(overlay_data.get("end_time", 0)),
                    topic=overlay_data.get("topic", ""),
                    match_type=overlay_data.get("match_type", "illustrative"),
                    local_hint=overlay_data.get("local_hint"),
                    pexels_query=overlay_data.get("pexels_query"),
                    confidence=float(overlay_data.get("confidence", 0.5)),
                    alt_text=overlay_data.get("alt_text"),
                )

                # Validate timing
                if overlay.end_time > overlay.start_time:
                    overlays.append(overlay)
                else:
                    logger.warning(
                        f"Skipping overlay with invalid timing: "
                        f"{overlay.start_time} -> {overlay.end_time}"
                    )

            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse overlay: {e}")
                continue

        return ImageOverlayScript(overlays=overlays, skipped=skipped)

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content.

        Args:
            text: Text potentially containing JSON.

        Returns:
            Extracted JSON string.
        """
        # Try to find JSON in code blocks first
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Return original text and let JSON parser fail
        return text

    def _log_results(self, overlay_script: ImageOverlayScript) -> None:
        """Log planning results in table format.

        Args:
            overlay_script: Planned overlay script.
        """
        if not overlay_script.overlays:
            self.log_progress("  No overlays planned")
            return

        self.log_progress("")
        self.log_progress("  --- Planned Overlays ---")
        self.log_progress("  | Seg | Topic                    | Type        | Time            | Conf |")
        self.log_progress("  |-----|--------------------------|-------------|-----------------|------|")

        for overlay in overlay_script.overlays:
            topic = overlay.topic[:24].ljust(24)
            match_type = overlay.match_type[:11].ljust(11)
            time_range = f"{overlay.start_time:05.1f}s -> {overlay.end_time:05.1f}s"
            conf = f"{int(overlay.confidence * 100):3d}%"

            self.log_progress(
                f"  | {overlay.segment_index:3d} | {topic} | {match_type} | {time_range} | {conf} |"
            )

        # Log skipped segments
        if overlay_script.skipped:
            skipped_segs = ", ".join(
                f"Seg {s['segment_index']} ({s.get('reason', 'skipped')[:15]})"
                for s in overlay_script.skipped[:3]
            )
            if len(overlay_script.skipped) > 3:
                skipped_segs += f" +{len(overlay_script.skipped) - 3} more"
            self.log_progress(f"  Skipped: {skipped_segs}")

        self.log_progress("")
