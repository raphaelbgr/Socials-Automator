"""Pexels video metadata index.

Stores metadata for ALL videos seen from Pexels searches.
No video files - just metadata for AI selection.
Grows over time as an accumulator.

Structure:
    pexels/
        metadata_index.json   # All seen videos (metadata only)
        cache/
            index.json        # Downloaded videos only
            12345.mp4         # Actual files
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from socials_automator.constants import get_pexels_cache_dir


class PexelsMetadataIndex:
    """Metadata index for all Pexels videos seen (no downloads).

    This index accumulates metadata from every Pexels search,
    allowing AI to select from a large pool of known videos.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize metadata index.

        Args:
            base_dir: Base directory for Pexels data. Uses default if not provided.
        """
        self.base_dir = Path(base_dir) if base_dir else get_pexels_cache_dir().parent
        self.index_path = self.base_dir / "metadata_index.json"
        self._index: dict = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save index to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)

    def add_from_api_response(self, video_data: dict, search_keywords: list[str]) -> dict:
        """Add or update video metadata from Pexels API response.

        Args:
            video_data: Pexels API video response.
            search_keywords: Keywords used in the search.

        Returns:
            The metadata entry (new or updated).
        """
        pexels_id = video_data.get("id")
        if not pexels_id:
            return {}

        str_id = str(pexels_id)

        # Extract description from URL
        pexels_url = video_data.get("url", "")
        description, description_keywords = self._extract_description_from_url(pexels_url)

        # Get video file info
        video_files = video_data.get("video_files", [])
        best_file = self._get_best_video_file(video_files)

        # Check if already in index
        existing = self._index.get(str_id, {})
        existing_keywords = existing.get("search_keywords_history", [])

        # Build metadata
        metadata = {
            "pexels_id": pexels_id,
            "pexels_url": pexels_url,
            "description": description,
            "description_keywords": description_keywords,
            "duration_seconds": video_data.get("duration", 0),
            "width": video_data.get("width", 0),
            "height": video_data.get("height", 0),
            "orientation": self._get_orientation(video_data),
            "aspect_ratio": self._get_aspect_ratio(video_data),
            "is_portrait": self._is_portrait(video_data),
            "is_9_16": self._is_9_16(video_data),
            "author": video_data.get("user", {}).get("name", "Unknown"),
            "author_url": video_data.get("user", {}).get("url", ""),
            "thumbnail_url": video_data.get("image", ""),
            # Video file info (for download later)
            "video_file_url": best_file.get("link", "") if best_file else "",
            "video_quality": best_file.get("quality", "") if best_file else "",
            "video_file_width": best_file.get("width", 0) if best_file else 0,
            "video_file_height": best_file.get("height", 0) if best_file else 0,
            # Combined keywords for searching
            "all_keywords": list(set(search_keywords + description_keywords)),
            # Track search history
            "search_keywords_history": list(set(existing_keywords + search_keywords)),
            "first_seen": existing.get("first_seen", datetime.now().isoformat()),
            "last_seen": datetime.now().isoformat(),
            "times_seen": existing.get("times_seen", 0) + 1,
            # Selection tracking
            "times_selected": existing.get("times_selected", 0),
            "last_selected": existing.get("last_selected"),
        }

        self._index[str_id] = metadata
        return metadata

    def add_batch_from_search(self, videos: list[dict], search_keywords: list[str]) -> int:
        """Add multiple videos from a Pexels search response.

        Args:
            videos: List of video data from Pexels API.
            search_keywords: Keywords used in the search.

        Returns:
            Number of videos added/updated.
        """
        count = 0
        for video_data in videos:
            if self.add_from_api_response(video_data, search_keywords):
                count += 1

        if count > 0:
            self._save_index()

        return count

    def mark_selected(self, pexels_id: int) -> None:
        """Mark a video as selected by AI.

        Args:
            pexels_id: Pexels video ID.
        """
        str_id = str(pexels_id)
        if str_id in self._index:
            self._index[str_id]["times_selected"] = self._index[str_id].get("times_selected", 0) + 1
            self._index[str_id]["last_selected"] = datetime.now().isoformat()
            self._save_index()

    def get(self, pexels_id: int) -> Optional[dict]:
        """Get metadata for a video.

        Args:
            pexels_id: Pexels video ID.

        Returns:
            Metadata dict or None.
        """
        return self._index.get(str(pexels_id))

    def has(self, pexels_id: int) -> bool:
        """Check if video is in index.

        Args:
            pexels_id: Pexels video ID.

        Returns:
            True if in index.
        """
        return str(pexels_id) in self._index

    def search(
        self,
        keywords: list[str],
        limit: int = 50,
        only_portrait: bool = True,
        only_9_16: bool = False,
        min_duration: float = 0,
        max_duration: float = 999,
    ) -> list[dict]:
        """Search index by keywords.

        Args:
            keywords: Keywords to search for.
            limit: Maximum results.
            only_portrait: Filter to portrait videos only.
            only_9_16: Filter to 9:16 aspect ratio only.
            min_duration: Minimum duration in seconds.
            max_duration: Maximum duration in seconds.

        Returns:
            List of matching metadata sorted by relevance.
        """
        results = []
        keywords_lower = [k.lower() for k in keywords]

        for metadata in self._index.values():
            # Apply filters
            if only_portrait and not metadata.get("is_portrait", False):
                continue
            if only_9_16 and not metadata.get("is_9_16", False):
                continue

            duration = metadata.get("duration_seconds", 0)
            if duration < min_duration or duration > max_duration:
                continue

            # Calculate match score
            all_keywords = metadata.get("all_keywords", [])
            all_keywords_lower = [k.lower() for k in all_keywords]
            description = metadata.get("description", "").lower()

            # Keyword matches
            keyword_matches = sum(
                1 for k in keywords_lower
                if any(k in ak for ak in all_keywords_lower)
            )

            # Description matches
            description_matches = sum(
                1 for k in keywords_lower
                if k in description
            )

            # Bonus for previously selected videos (AI liked them before)
            selection_bonus = min(metadata.get("times_selected", 0) * 0.5, 2.0)

            total_score = keyword_matches + (description_matches * 0.5) + selection_bonus

            if total_score > 0:
                results.append({
                    **metadata,
                    "_match_score": total_score,
                    "_keyword_matches": keyword_matches,
                    "_description_matches": description_matches,
                })

        # Sort by score
        results.sort(key=lambda x: x.get("_match_score", 0), reverse=True)

        return results[:limit]

    def get_stats(self) -> dict:
        """Get index statistics.

        Returns:
            Stats dict.
        """
        total = len(self._index)
        portrait = sum(1 for m in self._index.values() if m.get("is_portrait"))
        nine_sixteen = sum(1 for m in self._index.values() if m.get("is_9_16"))
        selected = sum(1 for m in self._index.values() if m.get("times_selected", 0) > 0)

        return {
            "total_videos": total,
            "portrait_videos": portrait,
            "9_16_videos": nine_sixteen,
            "selected_at_least_once": selected,
            "index_path": str(self.index_path),
        }

    def _extract_description_from_url(self, url: str) -> tuple[str, list[str]]:
        """Extract description and keywords from Pexels URL slug."""
        if not url:
            return "", []

        # Extract slug from URL: /video/slug-with-words-12345/
        match = re.search(r"/video/([^/]+)-(\d+)/?$", url)
        if not match:
            match = re.search(r"/video/([^/]+)/?$", url)
            if not match:
                return "", []

        slug = match.group(1)
        description = slug.replace("-", " ")

        # Filter stop words
        stop_words = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "with",
            "and", "or", "is", "are", "was", "were", "be", "been", "being",
            "video", "footage", "clip", "stock", "free", "hd", "4k",
        }

        words = slug.split("-")
        keywords = [
            word.lower() for word in words
            if word.lower() not in stop_words and len(word) > 2
        ]

        return description, keywords

    def _get_orientation(self, video_data: dict) -> str:
        """Determine video orientation."""
        width = video_data.get("width", 0)
        height = video_data.get("height", 0)

        if width > height:
            return "landscape"
        elif height > width:
            return "portrait"
        else:
            return "square"

    def _is_portrait(self, video_data: dict) -> bool:
        """Check if video is portrait orientation."""
        width = video_data.get("width", 0)
        height = video_data.get("height", 0)
        return height > width

    def _is_9_16(self, video_data: dict) -> bool:
        """Check if video is 9:16 aspect ratio."""
        video_files = video_data.get("video_files", [])
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

    def _get_aspect_ratio(self, video_data: dict) -> str:
        """Get aspect ratio as string."""
        width = video_data.get("width", 0)
        height = video_data.get("height", 0)

        if width == 0 or height == 0:
            return "unknown"

        from math import gcd
        divisor = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def _get_best_video_file(self, video_files: list[dict]) -> Optional[dict]:
        """Get best quality video file from list."""
        if not video_files:
            return None

        # Prefer HD quality
        for vf in video_files:
            if vf.get("quality") == "hd":
                return vf

        return video_files[0]

    def migrate_from_cache(self, cache_index_path: Path) -> int:
        """Migrate existing cache index entries to metadata index.

        Args:
            cache_index_path: Path to existing cache index.json.

        Returns:
            Number of entries migrated.
        """
        if not cache_index_path.exists():
            return 0

        try:
            with open(cache_index_path, "r", encoding="utf-8") as f:
                cache_index = json.load(f)
        except (json.JSONDecodeError, IOError):
            return 0

        migrated = 0
        for str_id, cache_meta in cache_index.items():
            if str_id in self._index:
                # Already exists, just update download status
                continue

            # Convert cache metadata to index metadata
            pexels_url = cache_meta.get("pexels_url", "")
            description, description_keywords = self._extract_description_from_url(pexels_url)

            metadata = {
                "pexels_id": cache_meta.get("pexels_id"),
                "pexels_url": pexels_url,
                "description": description or cache_meta.get("description", ""),
                "description_keywords": description_keywords or cache_meta.get("description_keywords", []),
                "duration_seconds": cache_meta.get("duration_seconds", 0),
                "width": cache_meta.get("width", 0),
                "height": cache_meta.get("height", 0),
                "orientation": cache_meta.get("orientation", ""),
                "aspect_ratio": cache_meta.get("aspect_ratio", ""),
                "is_portrait": cache_meta.get("orientation") == "portrait",
                "is_9_16": cache_meta.get("aspect_ratio") == "9:16",
                "author": cache_meta.get("author", ""),
                "author_url": cache_meta.get("author_url", ""),
                "thumbnail_url": cache_meta.get("api_response", {}).get("image", ""),
                "video_file_url": cache_meta.get("video_file_url", ""),
                "video_quality": cache_meta.get("video_quality", ""),
                "all_keywords": cache_meta.get("all_keywords", cache_meta.get("keywords_matched", [])),
                "search_keywords_history": cache_meta.get("keywords_matched", []),
                "first_seen": cache_meta.get("cached_at", datetime.now().isoformat()),
                "last_seen": cache_meta.get("last_used", datetime.now().isoformat()),
                "times_seen": cache_meta.get("hit_count", 0) + 1,
                "times_selected": cache_meta.get("hit_count", 0),
                "last_selected": cache_meta.get("last_used"),
            }

            self._index[str_id] = metadata
            migrated += 1

        if migrated > 0:
            self._save_index()

        return migrated
