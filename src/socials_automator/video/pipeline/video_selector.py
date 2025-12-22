"""Video selection using AI to build optimal video sequences.

Given a pool of candidate videos and a target duration, selects the
optimal sequence of videos that:
1. Match the narration/topic thematically
2. Sum to approximately the target duration
3. Are all unique (no repetition)
4. Only the last video is trimmed to fit exactly

This is the core of the "video-first" approach where video boundaries
determine segment boundaries, rather than vice versa.
"""

import json
from typing import Optional

from .base import PipelineStep, VideoSearchError


class VideoSelector(PipelineStep):
    """Selects optimal video sequence to fill target duration."""

    def __init__(
        self,
        ai_client: Optional[object] = None,
        min_video_duration: float = 3.0,  # Minimum video length to consider
        max_video_duration: float = 20.0,  # Maximum video length to consider
    ):
        """Initialize video selector.

        Args:
            ai_client: AI client for intelligent selection.
            min_video_duration: Minimum video duration to consider.
            max_video_duration: Maximum video duration to consider.
        """
        super().__init__("VideoSelector")
        self.ai_client = ai_client
        self.min_video_duration = min_video_duration
        self.max_video_duration = max_video_duration

    async def execute(self, context):
        """Execute is not used - call select_video_sequence directly."""
        # This step is called directly, not through the pipeline
        return context

    async def select_video_sequence(
        self,
        candidates: list[dict],
        target_duration: float,
        narration: str = "",
        topic: str = "",
        used_video_ids: Optional[set[int]] = None,
    ) -> list[dict]:
        """Select optimal video sequence to fill target duration.

        Args:
            candidates: List of candidate videos with id, duration, description.
            target_duration: Target total duration in seconds.
            narration: Full narration text for thematic matching.
            topic: Topic/title for thematic matching.
            used_video_ids: Set of video IDs to exclude (already used).

        Returns:
            List of selected videos in order, with 'trim_to' field on last video.
        """
        if not candidates:
            raise VideoSearchError("No candidate videos provided")

        used_ids = used_video_ids or set()

        # Filter candidates
        filtered = []
        for c in candidates:
            vid = c.get("id") or c.get("pexels_id")
            duration = c.get("duration") or c.get("duration_seconds", 0)

            # Skip if already used
            if vid in used_ids:
                continue

            # Skip if duration out of range
            if duration < self.min_video_duration:
                continue
            if duration > self.max_video_duration:
                continue

            filtered.append({
                "id": vid,
                "duration": duration,
                "description": c.get("description") or c.get("_description", ""),
                "url": c.get("url") or c.get("pexels_url", ""),
                "video_files": c.get("video_files", []),
                "_original": c,  # Keep original for downstream
            })

        if not filtered:
            raise VideoSearchError(
                f"No suitable videos after filtering (need {self.min_video_duration}-{self.max_video_duration}s)"
            )

        self.log_detail(f"Filtered to {len(filtered)} candidates ({self.min_video_duration}-{self.max_video_duration}s)")

        # Use AI selection if available, otherwise greedy
        if self.ai_client and len(filtered) > 5:
            selected = await self._ai_select_sequence(
                filtered, target_duration, narration, topic
            )
        else:
            selected = self._greedy_select_sequence(filtered, target_duration)

        if not selected:
            raise VideoSearchError("Could not select any videos")

        # Calculate total duration and mark trim on last video
        total = sum(v["duration"] for v in selected)

        # === DURATION CONTRACT VALIDATION ===
        self.log_progress(f"[Contract] Required: {target_duration:.1f}s | Selected: {total:.1f}s")

        if total > target_duration:
            # Trim last video - good, we have enough
            excess = total - target_duration
            last_video = selected[-1]
            last_video["trim_to"] = last_video["duration"] - excess
            last_video["trimmed"] = True
            self.log_progress(f"[Contract] OK - {excess:.1f}s excess (last video will be trimmed)")
            self.log_detail(
                f"Last video trimmed: {last_video['duration']:.1f}s -> {last_video['trim_to']:.1f}s"
            )
        elif total < target_duration:
            # Not enough footage - this will require frame extension
            gap = target_duration - total
            self.log_progress(f"[Contract] WARNING - {gap:.1f}s SHORT! Last frame will be frozen to extend.")
            # Mark last video to extend
            selected[-1]["extend_by"] = gap
        else:
            self.log_progress(f"[Contract] OK - exact match")

        # Log selection summary
        durations_str = " + ".join(f"{v['duration']:.0f}s" for v in selected)
        self.log_progress(
            f"Selected {len(selected)} videos: {durations_str} = {total:.1f}s"
        )

        return selected

    def _greedy_select_sequence(
        self,
        candidates: list[dict],
        target_duration: float,
    ) -> list[dict]:
        """Greedy selection: pick videos until we reach target duration.

        Prioritizes variety by not picking similar durations consecutively.

        Args:
            candidates: Filtered candidate videos.
            target_duration: Target total duration.

        Returns:
            List of selected videos.
        """
        selected = []
        current_duration = 0.0
        used_ids = set()

        # Sort by duration (medium-length first for variety)
        sorted_candidates = sorted(
            candidates,
            key=lambda v: abs(v["duration"] - 8.0)  # Prefer ~8s videos
        )

        while current_duration < target_duration and sorted_candidates:
            # Find best next video
            remaining = target_duration - current_duration
            best = None
            best_score = -1

            for c in sorted_candidates:
                if c["id"] in used_ids:
                    continue

                duration = c["duration"]
                # Score: prefer videos that fill remaining time well
                if duration <= remaining:
                    # Video fits completely - good
                    score = duration / remaining  # Prefer longer ones that fit
                else:
                    # Video needs trimming - okay if close to remaining
                    if duration <= remaining * 1.5:
                        score = 0.5  # Acceptable
                    else:
                        score = 0.1  # Too long, will waste footage

                # Bonus for having description (better for AI matching)
                if c.get("description"):
                    score += 0.1

                if score > best_score:
                    best_score = score
                    best = c

            if not best:
                break

            selected.append(best)
            used_ids.add(best["id"])
            current_duration += best["duration"]

            # Remove from candidates
            sorted_candidates = [c for c in sorted_candidates if c["id"] != best["id"]]

        return selected

    async def _ai_select_sequence(
        self,
        candidates: list[dict],
        target_duration: float,
        narration: str,
        topic: str,
    ) -> list[dict]:
        """AI-powered selection for thematic matching.

        Args:
            candidates: Filtered candidate videos.
            target_duration: Target total duration.
            narration: Narration text for matching.
            topic: Topic for matching.

        Returns:
            List of selected videos.
        """
        if not self.ai_client:
            return self._greedy_select_sequence(candidates, target_duration)

        try:
            # Build candidate list for AI
            candidate_info = []
            for i, c in enumerate(candidates[:25]):  # Limit for prompt size
                desc = c.get("description", "no description")
                duration = c["duration"]
                candidate_info.append(f"{i+1}. [{c['id']}] \"{desc}\" ({duration:.0f}s)")

            # Request 10% buffer to ensure we don't end up short
            buffer_duration = target_duration * 1.10
            prompt = f"""Select stock videos to create a {target_duration:.0f}-second video.

TOPIC: {topic}
NARRATION: {narration[:500]}{'...' if len(narration) > 500 else ''}

AVAILABLE VIDEOS:
{chr(10).join(candidate_info)}

RULES:
1. Select videos that sum to AT LEAST {target_duration:.0f} seconds (aim for {buffer_duration:.0f}s to be safe)
2. Choose videos that MATCH the topic/narration thematically
3. Create a LOGICAL visual flow (don't jump randomly between unrelated scenes)
4. Each video can only be used ONCE
5. The LAST video will be trimmed, so going OVER is preferred to going under
6. NEVER go under {target_duration:.0f} seconds total - the audio is {target_duration:.0f}s!

Example output for 60s target:
1, 5, 3, 12, 8, 15 (videos in viewing order, totaling 65-70s)

Respond with ONLY the video numbers in order, comma-separated.
Think about: What visual story matches this narration?"""

            response = await self.ai_client.generate(prompt)
            selection_str = response.strip()

            # Parse response - extract numbers
            import re
            numbers = re.findall(r'\d+', selection_str)

            selected = []
            used_ids = set()
            current_duration = 0.0

            for num_str in numbers:
                try:
                    idx = int(num_str) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(candidates[:25]):
                        video = candidates[idx]
                        if video["id"] not in used_ids:
                            selected.append(video)
                            used_ids.add(video["id"])
                            current_duration += video["duration"]

                            # Stop if we have enough (with 10% buffer)
                            if current_duration >= target_duration * 1.10:
                                break
                except (ValueError, IndexError):
                    continue

            if selected:
                self.log_detail(f"AI selected {len(selected)} videos, {current_duration:.1f}s total")

                # If still short of target, add more videos greedily
                if current_duration < target_duration:
                    self.log_detail(f"AI selection short ({current_duration:.1f}s < {target_duration:.1f}s), adding more...")
                    remaining_needed = target_duration - current_duration
                    used_ids_set = set(v["id"] for v in selected)

                    # Try to add more candidates
                    for c in candidates:
                        if c["id"] not in used_ids_set:
                            selected.append(c)
                            used_ids_set.add(c["id"])
                            current_duration += c["duration"]
                            self.log_detail(f"  Added [{c['id']}] ({c['duration']:.0f}s) -> {current_duration:.1f}s")
                            if current_duration >= target_duration:
                                break

                return selected
            else:
                self.log_detail("AI selection failed, using greedy fallback")
                return self._greedy_select_sequence(candidates, target_duration)

        except Exception as e:
            self.log_detail(f"AI selection error: {e}, using greedy fallback")
            return self._greedy_select_sequence(candidates, target_duration)

    def create_segments_from_videos(
        self,
        selected_videos: list[dict],
        narration_sentences: list[str],
    ) -> list[dict]:
        """Create video segments based on selected video boundaries.

        Distributes narration sentences across video segments proportionally.

        Args:
            selected_videos: List of selected videos with durations.
            narration_sentences: List of narration sentences.

        Returns:
            List of segment dicts with video and narration info.
        """
        if not selected_videos:
            return []

        # Calculate total video duration (using trim_to for last if set)
        total_duration = 0.0
        for v in selected_videos:
            if v.get("trim_to"):
                total_duration += v["trim_to"]
            else:
                total_duration += v["duration"]

        # Distribute sentences across segments proportionally
        total_sentences = len(narration_sentences)
        segments = []
        current_time = 0.0
        sentence_idx = 0

        for i, video in enumerate(selected_videos):
            duration = video.get("trim_to", video["duration"])

            # Calculate how many sentences for this segment
            segment_ratio = duration / total_duration
            sentences_for_segment = max(1, round(total_sentences * segment_ratio))

            # Get sentences for this segment
            end_sentence_idx = min(sentence_idx + sentences_for_segment, total_sentences)
            segment_sentences = narration_sentences[sentence_idx:end_sentence_idx]
            sentence_idx = end_sentence_idx

            segment = {
                "index": i + 1,
                "video_id": video["id"],
                "video_url": video.get("url", ""),
                "video_description": video.get("description", ""),
                "start_time": current_time,
                "end_time": current_time + duration,
                "duration": duration,
                "narration": " ".join(segment_sentences),
                "trimmed": video.get("trimmed", False),
            }
            segments.append(segment)
            current_time += duration

        # If we have leftover sentences, add to last segment
        if sentence_idx < total_sentences:
            remaining = narration_sentences[sentence_idx:]
            if segments:
                segments[-1]["narration"] += " " + " ".join(remaining)

        return segments
