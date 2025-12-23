"""Dense image overlay planning with fixed TTL.

This module provides an alternative to ImageOverlayPlanner that:
- Extracts maximum visual topics from narration (AI phase)
- Distributes overlays with fixed TTL duration (code phase)
- Uses SRT timing for natural image appearance
- Leaves gaps empty if no image found (no fallbacks)

Usage:
    planner = DenseOverlayPlanner(
        text_provider=ai_client,
        image_ttl=3.0,        # Each image shows for 3 seconds
        minimum_images=20,    # Target 20+ topics from AI
    )
    context = await planner.execute(context)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

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


# =============================================================================
# Configuration Constants
# =============================================================================

# Minimum time before any overlay can appear (skip hook/intro)
MIN_OVERLAY_START_TIME = 3.0  # seconds

# Buffer before CTA - no overlays in final N seconds
CTA_BUFFER_TIME = 4.0  # seconds

# Small gap between consecutive overlays
OVERLAY_GAP = 0.2  # seconds

# Minimum visible duration (skip if slot too short)
MIN_VISIBLE_DURATION = 1.5  # seconds


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ExtractedTopic:
    """A visual topic extracted from narration by AI."""

    topic: str                          # "Stranger Things"
    match_type: str                     # "exact" or "illustrative"
    search_query: str                   # "stranger things netflix poster"
    keywords: List[str] = field(default_factory=list)  # ["stranger", "things"]
    priority: int = 1                   # 1=must-have, 2=nice-to-have

    # Filled by time slot distribution
    srt_start: Optional[float] = None   # When first mentioned in SRT
    srt_end: Optional[float] = None     # When mention ends


@dataclass
class SrtEntry:
    """A single subtitle entry from SRT file."""

    index: int
    start_time: float
    end_time: float
    text: str


# =============================================================================
# AI System Prompt
# =============================================================================

DENSE_EXTRACTION_PROMPT = """You are an AI that extracts EVERY distinct visual subject from video narration.

Your goal is MAXIMUM TOPIC EXTRACTION - identify every subject that could have an image overlay.

NARRATION:
{narration}

TARGET: Extract at least {minimum} unique visual topics.

RULES:
1. Extract SPECIFIC subjects that can be visually represented
2. NO DUPLICATES - each topic must be unique
3. Prioritize EXACT matches (products, shows, brands, people, apps)
4. Include keywords that might appear in subtitles (for timing matching)
5. search_query should be CONTEXTUAL to what's being said about the topic

CRITICAL FOR SEARCH QUERIES:
- The search_query must reflect what the narration SAYS about the topic
- If narration says "BTS tickets sold out in minutes", query should be "BTS concert sold out tickets crowd"
- If narration says "Zootopia 2 hits $1 billion", query should be "Zootopia 2 movie billion box office success"
- If narration says "Taylor Swift touring Europe", query should be "Taylor Swift European tour concert stage"
- Include action/context words from the narration, not just the subject name
- This helps find images that match what's being discussed, not just generic images

MATCH TYPES:
- "exact": Specific identifiable content (TV shows, products, brands, people)
- "illustrative": Generic visual concepts (only if truly needed)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "topics": [
    {{
      "topic": "Stranger Things",
      "match_type": "exact",
      "search_query": "stranger things season finale netflix ending",
      "keywords": ["stranger", "things", "netflix", "season", "finale"],
      "priority": 1
    }},
    {{
      "topic": "person using laptop",
      "match_type": "illustrative",
      "search_query": "person typing laptop computer",
      "keywords": ["laptop", "computer", "typing", "work"],
      "priority": 2
    }}
  ]
}}

IMPORTANT:
- Extract MORE topics than the target (narration may have many subjects)
- Each topic MUST be distinct (don't split "Netflix" and "Netflix logo")
- Keywords should match words that appear in the narration
- search_query should capture the CONTEXT of what's being said, not just the topic name
- Prefer exact matches over illustrative
"""


# =============================================================================
# DenseOverlayPlanner
# =============================================================================

class DenseOverlayPlanner(IImageOverlayPlanner):
    """Dense image overlay planner with fixed TTL per image.

    This planner extracts maximum topics from narration and distributes
    them into fixed-duration time slots based on SRT timing.

    Attributes:
        image_ttl: Fixed display duration for each image (seconds).
        minimum_images: Target number of topics to extract from AI.
    """

    def __init__(
        self,
        text_provider: Optional[TextProvider] = None,
        image_ttl: float = 3.0,
        minimum_images: Optional[int] = None,
    ):
        """Initialize dense overlay planner.

        Args:
            text_provider: AI text provider for topic extraction.
            image_ttl: Display duration per image in seconds.
            minimum_images: Target topic count. If None, auto-calculated.
        """
        super().__init__()
        self.name = "DenseOverlayPlanner"  # Override for logging
        self._text_provider = text_provider
        self.image_ttl = image_ttl
        self.minimum_images = minimum_images

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute dense overlay planning step.

        Args:
            context: Pipeline context with script and SRT path.

        Returns:
            Updated context with image overlay script.
        """
        if not context.script:
            raise ImageOverlayError("No script available for image overlay planning")

        self.log_start(
            f"Planning dense overlays (TTL={self.image_ttl}s, "
            f"target={self.minimum_images or 'auto'})"
        )

        try:
            # Plan overlays using dense extraction
            overlay_script = await self.plan_overlays(
                script=context.script,
                profile_path=context.profile_path,
                srt_path=context.srt_path,
            )

            context.image_overlays = overlay_script

            # Log results
            self._log_results(overlay_script)

            self.log_success(
                f"{len(overlay_script.overlays)} overlays planned "
                f"(TTL={self.image_ttl}s each)"
            )

            return context

        except Exception as e:
            self.log_error(f"Dense overlay planning failed: {e}")
            raise ImageOverlayError(f"Failed to plan dense overlays: {e}") from e

    async def plan_overlays(
        self,
        script: VideoScript,
        profile_path: Optional[Path] = None,
        srt_path: Optional[Path] = None,
    ) -> ImageOverlayScript:
        """Plan image overlays using dense extraction.

        Args:
            script: Video script with narration.
            profile_path: Path to profile (unused, for interface compat).
            srt_path: Path to SRT file for timing.

        Returns:
            ImageOverlayScript with TTL-spaced overlays.
        """
        # Calculate minimum if not provided
        minimum = self._calculate_minimum(script)

        self.log_progress(f"[>] Extracting {minimum}+ topics from narration...")

        # Phase 1: AI topic extraction
        topics = await self._extract_topics(
            narration=script.full_narration,
            target_count=minimum + 5,  # Ask for extra
        )

        self.log_progress(f"[>] AI extracted {len(topics)} unique topics")

        # Phase 2: Parse SRT for timing
        srt_entries = self._parse_srt(srt_path) if srt_path else []
        if srt_entries:
            self.log_progress(f"[>] Parsed {len(srt_entries)} SRT entries for timing")

        # Phase 3: Match topics to SRT timing
        matched_topics = self._match_topics_to_srt(topics, srt_entries)
        matched_count = sum(1 for t in matched_topics if t.srt_start is not None)
        self.log_progress(f"[>] {matched_count}/{len(topics)} topics matched to SRT")

        # Phase 4: Distribute into time slots
        overlays = self._distribute_to_slots(
            topics=matched_topics,
            script=script,
        )

        return ImageOverlayScript(overlays=overlays, skipped=[])

    def _calculate_minimum(self, script: VideoScript) -> int:
        """Calculate minimum topics based on duration and TTL."""
        if self.minimum_images is not None:
            return self.minimum_images

        # Auto-calculate: available_time / ttl
        hook_end = getattr(script, 'hook_end_time', MIN_OVERLAY_START_TIME)
        cta_start = getattr(script, 'cta_start_time', script.total_duration - CTA_BUFFER_TIME)
        available = cta_start - hook_end

        return max(5, int(available / self.image_ttl))

    async def _extract_topics(
        self,
        narration: str,
        target_count: int,
    ) -> List[ExtractedTopic]:
        """Extract visual topics from narration using AI.

        Args:
            narration: Full video narration text.
            target_count: Target number of topics to extract.

        Returns:
            List of extracted topics.
        """
        if not self._text_provider:
            self.log_warning("No text provider - using fallback extraction")
            return self._fallback_extraction(narration, target_count)

        prompt = DENSE_EXTRACTION_PROMPT.format(
            narration=narration,
            minimum=target_count,
        )

        try:
            response = await self._text_provider.generate(prompt)
            return self._parse_ai_response(response)
        except Exception as e:
            self.log_warning(f"AI extraction failed: {e}, using fallback")
            return self._fallback_extraction(narration, target_count)

    def _parse_ai_response(self, response: str) -> List[ExtractedTopic]:
        """Parse AI JSON response into ExtractedTopic list."""
        # Clean response (remove markdown if present)
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        try:
            data = json.loads(clean)
            topics = []

            for item in data.get("topics", []):
                topics.append(ExtractedTopic(
                    topic=item.get("topic", ""),
                    match_type=item.get("match_type", "illustrative"),
                    search_query=item.get("search_query", item.get("topic", "")),
                    keywords=item.get("keywords", []),
                    priority=item.get("priority", 1),
                ))

            return topics

        except json.JSONDecodeError as e:
            self.log_warning(f"Failed to parse AI response: {e}")
            return []

    def _fallback_extraction(
        self,
        narration: str,
        target_count: int,
    ) -> List[ExtractedTopic]:
        """Extract topics using simple keyword extraction (no AI)."""
        # Extract capitalized multi-word phrases
        pattern = r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
        matches = re.findall(pattern, narration)

        # Filter and deduplicate
        seen = set()
        topics = []
        skip_words = {'The', 'This', 'That', 'With', 'From', 'Here', 'There',
                      'What', 'When', 'Where', 'First', 'Next', 'Follow', 'Save'}

        for match in matches:
            if match in seen or match in skip_words:
                continue
            if len(match) < 3:
                continue

            seen.add(match)
            topics.append(ExtractedTopic(
                topic=match,
                match_type="exact",
                search_query=match.lower(),
                keywords=match.lower().split(),
                priority=1,
            ))

            if len(topics) >= target_count:
                break

        return topics

    def _parse_srt(self, srt_path: Path) -> List[SrtEntry]:
        """Parse SRT file into entries."""
        if not srt_path or not srt_path.exists():
            return []

        entries = []
        try:
            content = srt_path.read_text(encoding="utf-8")
            blocks = content.strip().split("\n\n")

            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) < 3:
                    continue

                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text = " ".join(lines[2:])

                    # Parse time: "00:00:03,500 --> 00:00:07,200"
                    start_str, end_str = time_line.split(" --> ")
                    start_time = self._parse_srt_time(start_str)
                    end_time = self._parse_srt_time(end_str)

                    entries.append(SrtEntry(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                    ))
                except (ValueError, IndexError):
                    continue

        except Exception as e:
            self.log_warning(f"Failed to parse SRT: {e}")

        return entries

    def _parse_srt_time(self, time_str: str) -> float:
        """Parse SRT time string to seconds."""
        # Format: "00:00:03,500" or "00:00:03.500"
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def _match_topics_to_srt(
        self,
        topics: List[ExtractedTopic],
        srt_entries: List[SrtEntry],
    ) -> List[ExtractedTopic]:
        """Match topics to SRT entries for timing."""
        if not srt_entries:
            return topics

        for topic in topics:
            match = self._find_srt_match(topic, srt_entries)
            if match:
                topic.srt_start = match.start_time
                topic.srt_end = match.end_time

        return topics

    def _find_srt_match(
        self,
        topic: ExtractedTopic,
        srt_entries: List[SrtEntry],
    ) -> Optional[SrtEntry]:
        """Find first SRT entry containing topic keywords."""
        # Build search terms
        search_terms = [topic.topic.lower()]
        search_terms.extend([k.lower() for k in topic.keywords])

        for entry in srt_entries:
            text_lower = entry.text.lower()
            for term in search_terms:
                if term in text_lower:
                    return entry

        return None

    def _distribute_to_slots(
        self,
        topics: List[ExtractedTopic],
        script: VideoScript,
    ) -> List[ImageOverlay]:
        """Distribute topics into TTL-sized time slots.

        Topics with SRT matches are placed at their mention time.
        Topics without matches are skipped (no forced placement).
        """
        # Get timing boundaries
        hook_end = getattr(script, 'hook_end_time', MIN_OVERLAY_START_TIME)
        cta_start = getattr(script, 'cta_start_time', script.total_duration - CTA_BUFFER_TIME)

        min_start = max(MIN_OVERLAY_START_TIME, hook_end)
        max_end = min(cta_start, script.total_duration - CTA_BUFFER_TIME)

        # Filter to matched topics and sort by SRT start time
        matched = [t for t in topics if t.srt_start is not None]
        matched.sort(key=lambda t: t.srt_start)

        # Assign time slots
        overlays = []
        current_time = min_start

        for topic in matched:
            # Start at SRT mention time (or current_time if topic mentioned earlier)
            start = max(topic.srt_start, current_time)

            # Ensure within boundaries
            if start >= max_end:
                break  # No more room

            # Calculate end time based on TTL
            end = min(start + self.image_ttl, max_end)

            # Skip if slot too short
            if end - start < MIN_VISIBLE_DURATION:
                self.log_detail(f"  Skipped '{topic.topic}': slot too short ({end - start:.1f}s)")
                continue

            overlays.append(ImageOverlay(
                segment_index=0,  # Not segment-based
                start_time=start,
                end_time=end,
                topic=topic.topic,
                match_type=topic.match_type,
                pexels_query=topic.search_query,
                confidence=0.9 if topic.match_type == "exact" else 0.7,
            ))

            # Move current time past this overlay
            current_time = end + OVERLAY_GAP

        return overlays

    def _log_results(self, overlay_script: ImageOverlayScript) -> None:
        """Log overlay planning results in table format."""
        if not overlay_script.overlays:
            self.log_progress("[>] No overlays planned")
            return

        self.log_progress(f"[>] Planned {len(overlay_script.overlays)} overlays:")

        for overlay in overlay_script.overlays:
            time_range = f"{overlay.start_time:05.1f}s -> {overlay.end_time:05.1f}s"
            duration = overlay.end_time - overlay.start_time
            match_icon = "[E]" if overlay.match_type == "exact" else "[I]"

            self.log_detail(
                f"  {match_icon} {time_range} ({duration:.1f}s) | {overlay.topic[:30]}"
            )
