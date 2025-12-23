"""Video search using Pexels API with AI-powered selection.

Searches for vertical (9:16) stock videos matching script segments.
Uses a metadata index to accumulate video information and AI to select
the best matching video from a pool of candidates.

Flow:
1. Search Pexels API -> Add ALL results to metadata index
2. Search local metadata index for matching videos
3. Build candidate pool from both sources
4. AI selects best video from pool based on descriptions
5. Return selected video for download

Resilience:
- Retry with exponential backoff on API errors (503, 429, etc.)
- Fall back to local metadata index if API is unavailable
- Only fail if both API and local index have no results
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx

from .base import (
    IVideoSearcher,
    PipelineContext,
    VideoScript,
    VideoSearchError,
)
from .pexels_index import PexelsMetadataIndex
from socials_automator.constants import get_pexels_cache_dir


class VideoSearcher(IVideoSearcher):
    """Searches Pexels for stock videos with AI-powered selection."""

    PEXELS_API_URL = "https://api.pexels.com/videos"

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0  # seconds (will be 2s, 4s, 8s with exponential backoff)
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}  # Rate limit + server errors

    def __init__(
        self,
        api_key: Optional[str] = None,
        prefer_portrait: bool = True,
        ai_client: Optional[object] = None,
        use_ai_selection: bool = True,
    ):
        """Initialize video searcher.

        Args:
            api_key: Pexels API key. If None, reads from PEXELS_API_KEY env var.
            prefer_portrait: Whether to prefer portrait (9:16) videos.
            ai_client: Optional AI client for enhanced selection.
            use_ai_selection: Whether to use AI to select from candidate pool.
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY")
        self.prefer_portrait = prefer_portrait
        self.ai_client = ai_client
        self.use_ai_selection = use_ai_selection
        self._client: Optional[httpx.AsyncClient] = None
        # Metadata index (accumulates all seen videos)
        self._metadata_index = PexelsMetadataIndex()
        # Track used video IDs to prevent duplicates (within single run)
        self._used_video_ids: set[int] = set()
        # Persistent video history (across runs)
        self._history_file = get_pexels_cache_dir() / "video_history.json"
        self._video_history: dict[str, str] = {}  # video_id -> timestamp
        self._history_cooldown_days = 7  # Days before video can be reused
        self._load_video_history()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            if not self.api_key:
                raise VideoSearchError(
                    "Pexels API key not found. Set PEXELS_API_KEY environment variable."
                )
            self._client = httpx.AsyncClient(
                headers={"Authorization": self.api_key},
                timeout=30.0,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _load_video_history(self) -> None:
        """Load video usage history from file, removing expired entries."""
        try:
            if self._history_file.exists():
                with open(self._history_file, "r") as f:
                    all_history = json.load(f)

                # Remove entries older than cooldown period
                cutoff = datetime.now() - timedelta(days=self._history_cooldown_days)
                cutoff_str = cutoff.isoformat()

                self._video_history = {
                    vid: ts for vid, ts in all_history.items()
                    if ts > cutoff_str
                }

                # Pre-populate used_video_ids with recent history
                self._used_video_ids = {int(vid) for vid in self._video_history.keys()}

                expired = len(all_history) - len(self._video_history)
                if expired > 0:
                    self.log_detail(f"Removed {expired} expired videos from history")
                    self._save_video_history()
        except Exception as e:
            self.log_detail(f"Could not load video history: {e}")
            self._video_history = {}

    def _save_video_history(self) -> None:
        """Save video usage history to file."""
        try:
            self._history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._history_file, "w") as f:
                json.dump(self._video_history, f, indent=2)
        except Exception as e:
            self.log_detail(f"Could not save video history: {e}")

    def _mark_video_used(self, video_id: int) -> None:
        """Mark a video as used in both session and persistent history."""
        self._used_video_ids.add(video_id)
        self._video_history[str(video_id)] = datetime.now().isoformat()

    async def _api_request_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict,
    ) -> Optional[dict]:
        """Make API request with retry and exponential backoff.

        Args:
            client: HTTP client.
            url: Request URL.
            params: Query parameters.

        Returns:
            JSON response data or None if all retries failed.
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await client.get(url, params=params)

                # Check for retryable errors
                if response.status_code in self.RETRYABLE_STATUS_CODES:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    self.log_detail(
                        f"  API {response.status_code}, retry {attempt + 1}/{self.MAX_RETRIES} in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                if e.response.status_code in self.RETRYABLE_STATUS_CODES:
                    delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                    self.log_detail(
                        f"  API {e.response.status_code}, retry {attempt + 1}/{self.MAX_RETRIES} in {delay:.0f}s..."
                    )
                    last_error = e
                    await asyncio.sleep(delay)
                    continue
                else:
                    # Non-retryable error
                    last_error = e
                    break

            except httpx.RequestError as e:
                # Network errors are retryable
                delay = self.RETRY_BASE_DELAY * (2 ** attempt)
                self.log_detail(
                    f"  Network error, retry {attempt + 1}/{self.MAX_RETRIES} in {delay:.0f}s..."
                )
                last_error = e
                await asyncio.sleep(delay)
                continue

        # All retries exhausted
        if last_error:
            self.log_detail(f"  API failed after {self.MAX_RETRIES} retries: {last_error}")
        return None

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute video search step.

        Args:
            context: Pipeline context with script.

        Returns:
            Updated context with search results stored in context.
        """
        if not context.script:
            raise VideoSearchError("No script available for video search")

        self.log_start(f"Searching videos for {len(context.script.segments)} segments")

        # Log duration contract for tracking
        if context.required_video_duration:
            self.log_progress(f"[Duration Contract] {context.required_video_duration:.1f}s")
        else:
            self.log_progress("[Duration Contract] NOT SET (will be set after voice generation)")

        # Load fresh history (in case another process updated it)
        self._load_video_history()
        history_count = len(self._video_history)

        # Log metadata index stats
        index_stats = self._metadata_index.get_stats()
        self.log_detail(f"Metadata index: {index_stats['total_videos']} videos ({index_stats['9_16_videos']} are 9:16)")
        if history_count > 0:
            self.log_detail(f"History: {history_count} videos (will avoid reuse for {self._history_cooldown_days} days)")

        try:
            search_results = await self.search_videos(context.script)

            # Store results in context for download step
            context._search_results = search_results

            # Save updated history
            self._save_video_history()

            # Log stats
            new_videos = len(self._used_video_ids) - history_count
            new_index_stats = self._metadata_index.get_stats()
            new_indexed = new_index_stats['total_videos'] - index_stats['total_videos']

            self.log_progress(f"Selected {new_videos} new videos")
            if new_indexed > 0:
                self.log_detail(f"Added {new_indexed} new videos to metadata index")
            self.log_success(f"Found videos for {len(search_results)} segments")
            return context

        except Exception as e:
            self.log_error(f"Video search failed: {e}")
            raise VideoSearchError(f"Failed to search videos: {e}") from e
        finally:
            await self.close()

    async def search_videos(self, script: VideoScript) -> list[dict]:
        """Search for videos matching script segments.

        Args:
            script: Video script with segments.

        Returns:
            List of search results, one per segment.
        """
        results = []

        # Log segment durations we're searching for
        self.log_progress("--- Video Search Targets ---")
        total_needed = 0.0
        for seg in script.segments:
            self.log_progress(f"  Seg {seg.index}: {seg.duration_seconds:.1f}s needed")
            total_needed += seg.duration_seconds
        self.log_progress(f"  TOTAL: {total_needed:.1f}s of video needed")

        # Generate enhanced keywords using AI if available
        enhanced_keywords = await self._get_enhanced_keywords(script)

        for segment in script.segments:
            # Use enhanced keywords if available, otherwise use segment keywords
            keywords = enhanced_keywords.get(segment.index, segment.keywords)

            self.log_detail(
                f"Searching segment {segment.index}: {keywords[:3]}"
            )

            # Search for this segment (will skip already-used videos)
            video = await self._search_for_segment(segment, keywords)

            if video:
                # Mark video as used (persisted to avoid reuse across runs)
                video_id = video.get("id")
                if video_id:
                    # Check if already used (shouldn't happen, but log if it does)
                    if video_id in self._used_video_ids:
                        self.log_detail(f"[WARNING] Video {video_id} already used but returned again!")
                    self._mark_video_used(video_id)

                    # Log video details including dimensions
                    video_files = video.get("video_files", [])
                    if video_files:
                        vf = video_files[0]
                        width = vf.get("width", 0)
                        height = vf.get("height", 0)
                        ratio = f"{width}x{height}" if width and height else "unknown"
                        is_9_16 = "9:16" if self._is_9_16(video) else "NOT 9:16"
                        self.log_detail(f"Segment {segment.index}: video {video_id} ({ratio}, {is_9_16}, {video.get('duration', 0)}s)")
                    else:
                        self.log_detail(f"Segment {segment.index}: video {video_id} (duration: {video.get('duration', 0)}s)")

                results.append({
                    "segment_index": segment.index,
                    "video": video,
                    "keywords_used": keywords,
                    "duration_needed": segment.duration_seconds,
                })
            else:
                self.log_detail(f"No video found for segment {segment.index}, using fallback")
                fallback = await self._search_fallback()
                if fallback:
                    video_id = fallback.get("id")
                    if video_id:
                        self._mark_video_used(video_id)
                results.append({
                    "segment_index": segment.index,
                    "video": fallback,
                    "keywords_used": ["abstract", "technology"],
                    "duration_needed": segment.duration_seconds,
                })

        # Log what was found vs what was needed
        self.log_progress("--- Video Search Results ---")
        total_found = 0.0
        for r in results:
            video = r.get("video", {})
            video_duration = video.get("duration", 0)
            needed = r.get("duration_needed", 0)
            seg_idx = r.get("segment_index", 0)
            status = "[OK]" if video_duration >= needed else "[!]"
            self.log_progress(f"  Seg {seg_idx}: {video_duration:.1f}s found (needed {needed:.1f}s) {status}")
            total_found += video_duration
        self.log_progress(f"  TOTAL: {total_found:.1f}s found / {total_needed:.1f}s needed")

        if total_found < total_needed:
            shortage = total_needed - total_found
            self.log_progress(f"  [!] WARNING: Video footage SHORT BY {shortage:.1f}s")

        return results

    async def _get_enhanced_keywords(self, script: VideoScript) -> dict[int, list[str]]:
        """Use AI to generate better video search keywords.

        Args:
            script: Video script with segments.

        Returns:
            Dict mapping segment index to enhanced keywords.
        """
        if not self.ai_client:
            return {}

        try:
            import json

            # Build segment info
            segments_info = "\n".join(
                f"Segment {s.index}: \"{s.text}\" (current keywords: {s.keywords[:3]})"
                for s in script.segments
            )

            prompt = f"""Generate UNIQUE video search keywords for each segment of this video script.

Segments:
{segments_info}

CRITICAL RULES:
1. Each segment MUST have DIFFERENT keywords - no repeats across segments!
2. Keywords must be VISUAL concepts that can be filmed (not abstract)
3. Use 2-word search phrases that work on stock video sites
4. Think: "What specific scene would I see in a video?"

GOOD examples: "laptop typing", "woman smiling", "city night", "hand writing", "coffee shop"
BAD examples: "productivity", "success", "AI", "concept" (too abstract)

Format as JSON:
{{
    "1": ["keyword1", "keyword2", "keyword3"],
    "2": ["keyword1", "keyword2", "keyword3"],
    ...
}}

Each segment MUST have unique keywords. Check your output for duplicates!
Respond with ONLY valid JSON."""

            response = await self.ai_client.generate(prompt)

            # Parse JSON
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            clean = clean.strip()

            data = json.loads(clean)

            # Convert to dict with int keys
            result = {}
            for key, value in data.items():
                result[int(key)] = value

            self.log_progress(f"AI enhanced keywords for {len(result)} segments")
            return result

        except Exception as e:
            self.log_detail(f"AI keyword enhancement failed: {e}, using defaults")
            return {}

    async def _search_for_segment(
        self,
        segment,
        keywords: Optional[list[str]] = None,
    ) -> Optional[dict]:
        """Search for a video matching a segment using AI selection.

        Uses hybrid approach: tries Pexels API first, falls back to local index.

        Args:
            segment: Video segment with keywords.
            keywords: Optional enhanced keywords to use instead of segment keywords.

        Returns:
            Best matching video or None (excludes already-used videos).
        """
        client = await self._get_client()
        search_keywords = keywords if keywords else segment.keywords

        # Build candidate pool from multiple sources
        api_candidates = []
        index_candidates = []

        # Source 1: Fresh Pexels API search (with retry)
        api_failed = False
        for keyword in search_keywords[:3]:  # Limit to top 3 keywords
            fresh_videos = await self._search_with_orientation(
                client, keyword, "portrait", segment.duration_seconds, return_all=True
            )
            if fresh_videos:
                api_candidates.extend(fresh_videos)
            elif fresh_videos is None or (isinstance(fresh_videos, list) and len(fresh_videos) == 0):
                # Empty result could be API failure or no matches
                # We'll check after all keywords
                pass

        # Check if API seems to be down (no results from any keyword)
        if not api_candidates:
            api_failed = True

        # Source 2: Local metadata index (always available)
        index_results = self._metadata_index.search(
            keywords=search_keywords,
            limit=30 if api_failed else 20,  # Get more from index if API is down
            only_portrait=self.prefer_portrait,
            min_duration=segment.duration_seconds * 0.5,  # At least half the needed duration
            max_duration=segment.duration_seconds * 3,  # Not too long
        )

        # Convert index results to API-like format
        for idx_result in index_results:
            pexels_id = idx_result.get("pexels_id")
            if pexels_id and not any(c.get("id") == pexels_id for c in api_candidates):
                index_candidates.append({
                    "id": pexels_id,
                    "url": idx_result.get("pexels_url", ""),
                    "duration": idx_result.get("duration_seconds", 0),
                    "width": idx_result.get("width", 0),
                    "height": idx_result.get("height", 0),
                    "video_files": [{
                        "link": idx_result.get("video_file_url", ""),
                        "quality": idx_result.get("video_quality", "hd"),
                        "width": idx_result.get("video_file_width", 0),
                        "height": idx_result.get("video_file_height", 0),
                    }],
                    "user": {"name": idx_result.get("author", ""), "url": idx_result.get("author_url", "")},
                    "image": idx_result.get("thumbnail_url", ""),
                    "_from_index": True,
                    "_description": idx_result.get("description", ""),
                    "_match_score": idx_result.get("_match_score", 0),
                })

        # Combine candidates
        candidates = api_candidates + index_candidates

        # Log source breakdown
        if api_failed and index_candidates:
            self.log_detail(f"  [FALLBACK] API unavailable, using {len(index_candidates)} from local index")
        elif api_failed and not index_candidates:
            self.log_detail(f"  [!] API unavailable and no local index matches")

        if not candidates:
            self.log_detail(f"  No candidates found for segment {segment.index}")
            return None

        # Deduplicate by video ID
        seen_ids = set()
        unique_candidates = []
        for c in candidates:
            vid = c.get("id")
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                unique_candidates.append(c)
        candidates = unique_candidates

        # Filter out already-used videos
        available = [c for c in candidates if c.get("id") not in self._used_video_ids]
        if not available:
            self.log_detail(f"  All {len(candidates)} candidates already used, allowing reuse")
            available = candidates

        # Log pool composition
        api_count = len([c for c in available if not c.get("_from_index")])
        idx_count = len([c for c in available if c.get("_from_index")])
        self.log_detail(f"  Pool: {len(available)} candidates ({api_count} API, {idx_count} index)")

        # AI selection or fallback to duration-based selection
        if self.use_ai_selection and self.ai_client and len(available) > 1:
            selected = await self._ai_select_video(segment, available)
            if selected:
                return selected

        # Fallback: select by duration
        return self._select_best_duration(available, segment.duration_seconds)

    async def _ai_select_video(
        self,
        segment,
        candidates: list[dict],
    ) -> Optional[dict]:
        """Use AI to select the best video from candidates.

        Args:
            segment: Video segment with text and context.
            candidates: List of candidate videos.

        Returns:
            Selected video or None.
        """
        if not self.ai_client or len(candidates) < 2:
            return None

        try:
            # Build candidate descriptions for AI
            candidate_info = []
            for i, c in enumerate(candidates[:15]):  # Limit to 15 for prompt size
                pexels_id = c.get("id")
                # Get description from index or extract from URL
                if c.get("_description"):
                    description = c.get("_description")
                else:
                    meta = self._metadata_index.get(pexels_id)
                    description = meta.get("description", "") if meta else ""
                    if not description:
                        # Extract from URL
                        url = c.get("url", "")
                        description, _ = self._metadata_index._extract_description_from_url(url)

                duration = c.get("duration", 0)
                candidate_info.append(f"{i+1}. [{pexels_id}] \"{description}\" ({duration}s)")

            prompt = f"""Select the BEST stock video for this script segment.

SEGMENT TEXT: "{segment.text}"
SEGMENT KEYWORDS: {segment.keywords[:5]}
DURATION NEEDED: {segment.duration_seconds}s

AVAILABLE VIDEOS:
{chr(10).join(candidate_info)}

RULES:
1. Choose the video whose description BEST matches the segment content
2. Prefer videos with duration >= {segment.duration_seconds}s (can be trimmed)
3. Consider visual relevance - what would look good with this narration?

Respond with ONLY the video number (1-{len(candidate_info)}), nothing else."""

            response = await self.ai_client.generate(prompt)
            selection = response.strip()

            # Parse selection
            try:
                # Handle responses like "1" or "Video 1" or "1."
                num = int(''.join(filter(str.isdigit, selection.split()[0])))
                if 1 <= num <= len(candidates[:15]):
                    selected = candidates[num - 1]
                    video_id = selected.get("id")
                    meta = self._metadata_index.get(video_id)
                    desc = meta.get("description", "unknown") if meta else "unknown"
                    self.log_detail(f"  AI selected: {video_id} ({desc})")

                    # Mark as selected in index
                    if video_id:
                        self._metadata_index.mark_selected(video_id)

                    return selected
            except (ValueError, IndexError):
                self.log_detail(f"  AI response parse failed: {selection}")

        except Exception as e:
            self.log_detail(f"  AI selection error: {e}")

        return None

    async def _search_query(
        self,
        client: httpx.AsyncClient,
        query: str,
        target_duration: float,
    ) -> Optional[dict]:
        """Search Pexels with a specific query.

        Args:
            client: HTTP client.
            query: Search query.
            target_duration: Target video duration.

        Returns:
            Best matching video or None.
        """
        try:
            # Search portrait first
            if self.prefer_portrait:
                video = await self._search_with_orientation(
                    client, query, "portrait", target_duration
                )
                if video:
                    return video

            # Fallback to landscape (will be cropped)
            video = await self._search_with_orientation(
                client, query, "landscape", target_duration
            )
            return video

        except httpx.HTTPError as e:
            self.log_detail(f"Search error for '{query}': {e}")
            return None

    async def _search_with_orientation(
        self,
        client: httpx.AsyncClient,
        query: str,
        orientation: str,
        target_duration: float,
        return_all: bool = False,
    ) -> Optional[dict] | list[dict]:
        """Search with specific orientation (with retry and fallback).

        Args:
            client: HTTP client.
            query: Search query.
            orientation: Video orientation (portrait/landscape).
            target_duration: Target duration in seconds.
            return_all: If True, return all matching videos for AI selection.

        Returns:
            Best matching video, list of candidates (if return_all), or None/empty list on failure.
        """
        params = {
            "query": query,
            "orientation": orientation,
            "size": "medium",  # 1080p
            "per_page": 30,  # Fetch more to find true 9:16 videos
        }

        # Use retry helper - returns None if all retries fail
        data = await self._api_request_with_retry(
            client, f"{self.PEXELS_API_URL}/search", params
        )

        if not data:
            # API failed - return empty (will fall back to local index)
            return [] if return_all else None

        videos = data.get("videos", [])

        if not videos:
            return [] if return_all else None

        # Add ALL videos to metadata index (accumulator)
        search_keywords = query.split()
        self._metadata_index.add_batch_from_search(videos, search_keywords)

        # For portrait, filter to 9:16 videos
        if orientation == "portrait":
            perfect_matches = [v for v in videos if self._is_9_16(v)]
            if perfect_matches:
                self.log_detail(f"  Found {len(perfect_matches)}/{len(videos)} true 9:16 videos")
                if return_all:
                    return perfect_matches
                return self._select_best_duration(perfect_matches, target_duration)
            else:
                self.log_detail(f"  No true 9:16 videos in {len(videos)} portrait results")
                return [] if return_all else None

        # For landscape, return all or best match
        if return_all:
            return videos
        return self._select_best_duration(videos, target_duration)

    async def _search_fallback(self) -> dict:
        """Search for generic fallback video (prefers 9:16 vertical videos).

        Returns:
            Fallback video.
        """
        client = await self._get_client()

        # Queries more likely to return vertical/portrait videos
        fallback_queries = [
            "vertical abstract",
            "phone screen recording",
            "vertical gradient",
            "abstract technology",
            "digital background",
            "neon lights vertical",
            "particles motion",
        ]

        for query in fallback_queries:
            self.log_detail(f"Fallback search: '{query}'")
            video = await self._search_query(client, query, 10.0)
            if video:
                # Log what we're using as fallback
                video_files = video.get("video_files", [])
                if video_files:
                    vf = video_files[0]
                    width = vf.get("width", 0)
                    height = vf.get("height", 0)
                    is_9_16 = "9:16" if self._is_9_16(video) else "NOT 9:16"
                    self.log_detail(f"  Fallback: {video.get('id')} ({width}x{height}, {is_9_16})")
                return video

        raise VideoSearchError("Could not find any fallback videos")

    def _is_9_16(self, video: dict) -> bool:
        """Check if video is 9:16 aspect ratio.

        Args:
            video: Pexels video data.

        Returns:
            True if video is approximately 9:16.
        """
        video_files = video.get("video_files", [])
        if not video_files:
            return False

        vf = video_files[0]
        width = vf.get("width", 0)
        height = vf.get("height", 0)

        if width == 0 or height == 0:
            return False

        ratio = width / height
        target = 9 / 16  # 0.5625
        return abs(ratio - target) < 0.1

    def _select_best_duration(
        self,
        videos: list[dict],
        target_duration: float,
    ) -> Optional[dict]:
        """Select best video preferring LONGER clips (excluding already-used).

        Prefers videos that are longer than needed (can be trimmed) over
        shorter ones (would need slow motion).

        Args:
            videos: List of videos.
            target_duration: Target duration in seconds.

        Returns:
            Best matching video that hasn't been used yet.
        """
        if not videos:
            return None

        # Filter out already-used videos
        available_videos = [
            v for v in videos
            if v.get("id") not in self._used_video_ids
        ]

        if not available_videos:
            self.log_detail("All videos already used, allowing reuse as last resort")
            available_videos = videos

        # Prefer videos LONGER than target (can be trimmed cleanly)
        longer_videos = [v for v in available_videos if v.get("duration", 0) >= target_duration]

        if longer_videos:
            # Among longer videos, pick the one closest to target (minimize waste)
            return min(
                longer_videos,
                key=lambda v: v.get("duration", 0) - target_duration,
            )
        else:
            # No videos long enough - pick the longest available (minimize slow motion)
            return max(
                available_videos,
                key=lambda v: v.get("duration", 0),
            )

    async def get_candidate_pool(
        self,
        keywords: list[str],
        target_duration: float,
        max_candidates: int = 40,
    ) -> list[dict]:
        """Get a pool of candidate videos for video-first selection.

        Searches Pexels API and local index to build a large pool of
        candidates for the VideoSelector to choose from.

        Args:
            keywords: Search keywords for the topic.
            target_duration: Target total duration (for reference).
            max_candidates: Maximum candidates to return.

        Returns:
            List of candidate videos with id, duration, description, etc.
        """
        self.log_start(f"Building candidate pool for {target_duration:.0f}s video")

        # Load fresh history
        self._load_video_history()

        client = await self._get_client()
        candidates = []

        # Source 1: Fresh Pexels API searches
        for keyword in keywords[:5]:  # Top 5 keywords
            self.log_detail(f"Searching: '{keyword}'")
            fresh_videos = await self._search_with_orientation(
                client, keyword, "portrait", target_duration, return_all=True
            )
            if fresh_videos:
                candidates.extend(fresh_videos)
                self.log_detail(f"  Found {len(fresh_videos)} videos")

        # Source 2: Local metadata index
        index_results = self._metadata_index.search(
            keywords=keywords,
            limit=30,
            only_portrait=self.prefer_portrait,
            only_9_16=True,
            min_duration=3.0,
            max_duration=25.0,
        )
        self.log_detail(f"Local index: {len(index_results)} matches")

        # Convert index results to consistent format
        for idx_result in index_results:
            pexels_id = idx_result.get("pexels_id")
            if pexels_id and not any(c.get("id") == pexels_id for c in candidates):
                candidates.append({
                    "id": pexels_id,
                    "url": idx_result.get("pexels_url", ""),
                    "duration": idx_result.get("duration_seconds", 0),
                    "width": idx_result.get("width", 0),
                    "height": idx_result.get("height", 0),
                    "video_files": [{
                        "link": idx_result.get("video_file_url", ""),
                        "quality": idx_result.get("video_quality", "hd"),
                        "width": idx_result.get("video_file_width", 0),
                        "height": idx_result.get("video_file_height", 0),
                    }],
                    "user": {"name": idx_result.get("author", ""), "url": idx_result.get("author_url", "")},
                    "image": idx_result.get("thumbnail_url", ""),
                    "description": idx_result.get("description", ""),
                    "_from_index": True,
                })

        # Deduplicate by video ID
        seen_ids = set()
        unique_candidates = []
        for c in candidates:
            vid = c.get("id")
            if vid and vid not in seen_ids:
                seen_ids.add(vid)
                # Add description from index if not present
                if not c.get("description"):
                    meta = self._metadata_index.get(vid)
                    if meta:
                        c["description"] = meta.get("description", "")
                unique_candidates.append(c)

        # Filter out recently used videos
        available = [
            c for c in unique_candidates
            if c.get("id") not in self._used_video_ids
        ]

        if len(available) < len(unique_candidates):
            filtered_count = len(unique_candidates) - len(available)
            self.log_detail(f"Filtered {filtered_count} recently-used videos")

        # Limit to max_candidates
        result = available[:max_candidates]

        self.log_success(f"Candidate pool: {len(result)} videos")
        return result

    def mark_videos_used(self, video_ids: list[int]) -> None:
        """Mark multiple videos as used.

        Args:
            video_ids: List of Pexels video IDs to mark as used.
        """
        for vid in video_ids:
            self._mark_video_used(vid)
        self._save_video_history()
