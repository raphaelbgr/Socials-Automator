"""Pexels video cache system.

Caches downloaded Pexels videos to reduce API calls and download time.
Uses a flat file structure with a JSON index for fast lookups.

Structure:
    pexels/cache/
        index.json          # Quick lookup by pexels_id
        4962903.mp4         # Video files named by pexels_id
        7413793.mp4
        ...
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from socials_automator.constants import get_pexels_cache_dir


class PexelsCache:
    """Manages cached Pexels videos with JSON index."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Cache directory path. Uses default if not provided.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_pexels_cache_dir()
        self.index_path = self.cache_dir / "index.json"
        self._index: dict = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load index from disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, default=str)

    def has_video(self, pexels_id: int) -> bool:
        """Check if video is in cache.

        Args:
            pexels_id: Pexels video ID.

        Returns:
            True if video is cached and file exists.
        """
        str_id = str(pexels_id)
        if str_id not in self._index:
            return False

        # Verify file actually exists
        video_path = self.cache_dir / self._index[str_id].get("filename", "")
        return video_path.exists()

    def get_video_path(self, pexels_id: int) -> Optional[Path]:
        """Get cached video path.

        Args:
            pexels_id: Pexels video ID.

        Returns:
            Path to cached video or None if not cached.
        """
        str_id = str(pexels_id)
        if not self.has_video(pexels_id):
            return None

        # Increment hit count
        self._index[str_id]["hit_count"] = self._index[str_id].get("hit_count", 0) + 1
        self._index[str_id]["last_used"] = datetime.now().isoformat()
        self._save_index()

        return self.cache_dir / self._index[str_id]["filename"]

    def get_metadata(self, pexels_id: int) -> Optional[dict]:
        """Get cached video metadata.

        Args:
            pexels_id: Pexels video ID.

        Returns:
            Metadata dict or None if not cached.
        """
        str_id = str(pexels_id)
        return self._index.get(str_id)

    def copy_to_destination(self, pexels_id: int, destination: Path) -> bool:
        """Copy cached video to destination.

        Args:
            pexels_id: Pexels video ID.
            destination: Destination path for the video.

        Returns:
            True if copied successfully, False otherwise.
        """
        cached_path = self.get_video_path(pexels_id)
        if not cached_path:
            return False

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, destination)
            return True
        except IOError:
            return False

    def add_video(
        self,
        pexels_id: int,
        source_path: Path,
        video_data: dict,
        keywords_matched: list[str],
    ) -> Path:
        """Add video to cache.

        Args:
            pexels_id: Pexels video ID.
            source_path: Path to downloaded video file.
            video_data: Pexels API video data.
            keywords_matched: Keywords used to find this video.

        Returns:
            Path to cached video.
        """
        str_id = str(pexels_id)
        filename = f"{pexels_id}.mp4"
        cache_path = self.cache_dir / filename

        # Copy to cache if not already there
        if source_path != cache_path:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, cache_path)

        # Extract video file info for metadata
        video_files = video_data.get("video_files", [])
        best_file = None
        for vf in video_files:
            if vf.get("quality") == "hd":
                best_file = vf
                break
        if not best_file and video_files:
            best_file = video_files[0]

        # Build comprehensive metadata
        metadata = {
            "pexels_id": pexels_id,
            "filename": filename,
            "title": video_data.get("user", {}).get("name", "Unknown"),
            "pexels_url": video_data.get("url", ""),
            "video_page_url": f"https://www.pexels.com/video/{pexels_id}/",
            "duration_seconds": video_data.get("duration", 0),
            "width": video_data.get("width", 0),
            "height": video_data.get("height", 0),
            "orientation": self._get_orientation(video_data),
            "author": video_data.get("user", {}).get("name", "Unknown"),
            "author_url": video_data.get("user", {}).get("url", ""),
            "keywords_matched": keywords_matched,
            "video_file_url": best_file.get("link", "") if best_file else "",
            "video_quality": best_file.get("quality", "") if best_file else "",
            "video_file_type": best_file.get("file_type", "") if best_file else "",
            "file_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
            "cached_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "hit_count": 0,
            # Store full API response for AI reference
            "api_response": {
                "id": video_data.get("id"),
                "width": video_data.get("width"),
                "height": video_data.get("height"),
                "duration": video_data.get("duration"),
                "image": video_data.get("image"),
                "video_files_count": len(video_files),
            },
        }

        # Get actual video info from file (resolution, aspect ratio, etc.)
        actual_info = self._get_actual_video_info(cache_path)
        metadata.update(actual_info)

        self._index[str_id] = metadata
        self._save_index()

        return cache_path

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

    def _get_aspect_ratio(self, width: int, height: int) -> str:
        """Calculate aspect ratio as string like '9:16'."""
        from math import gcd
        if width == 0 or height == 0:
            return "unknown"
        divisor = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    def _get_quality_label(self, height: int) -> str:
        """Get quality label based on height."""
        if height >= 1920:
            return "HD 1080p"
        elif height >= 1280:
            return "HD 720p"
        elif height >= 640:
            return "SD 360p"
        else:
            return "Low"

    def _get_actual_video_info(self, video_path: Path) -> dict:
        """Get actual video resolution and duration from file."""
        try:
            try:
                from moviepy import VideoFileClip
            except ImportError:
                from moviepy.editor import VideoFileClip

            clip = VideoFileClip(str(video_path))
            width, height = clip.size
            duration = clip.duration
            clip.close()

            return {
                "actual_width": width,
                "actual_height": height,
                "resolution": f"{width}x{height}",
                "aspect_ratio": self._get_aspect_ratio(width, height),
                "actual_duration": round(duration, 2),
                "quality_label": self._get_quality_label(height),
            }
        except Exception:
            return {}

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats.
        """
        total_videos = len(self._index)
        total_hits = sum(v.get("hit_count", 0) for v in self._index.values())
        total_size = sum(v.get("file_size_bytes", 0) for v in self._index.values())

        return {
            "total_videos": total_videos,
            "total_hits": total_hits,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def search_by_keywords(self, keywords: list[str], limit: int = 10) -> list[dict]:
        """Search cached videos by keywords.

        Args:
            keywords: Keywords to search for.
            limit: Maximum results to return.

        Returns:
            List of matching video metadata.
        """
        results = []
        keywords_lower = [k.lower() for k in keywords]

        for metadata in self._index.values():
            cached_keywords = metadata.get("keywords_matched", [])
            cached_keywords_lower = [k.lower() for k in cached_keywords]

            # Check for keyword overlap
            matches = sum(1 for k in keywords_lower if any(k in ck for ck in cached_keywords_lower))
            if matches > 0:
                results.append({
                    **metadata,
                    "_match_score": matches,
                })

        # Sort by match score
        results.sort(key=lambda x: x.get("_match_score", 0), reverse=True)

        return results[:limit]

    def clear_old_entries(self, days: int = 30) -> int:
        """Remove entries older than specified days.

        Args:
            days: Days threshold for old entries.

        Returns:
            Number of entries removed.
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        to_remove = []
        for str_id, metadata in self._index.items():
            last_used = metadata.get("last_used", metadata.get("cached_at", ""))
            if last_used:
                try:
                    last_date = datetime.fromisoformat(last_used)
                    if last_date < cutoff:
                        to_remove.append(str_id)
                except ValueError:
                    pass

        for str_id in to_remove:
            filename = self._index[str_id].get("filename", "")
            video_path = self.cache_dir / filename
            if video_path.exists():
                video_path.unlink()
            del self._index[str_id]
            removed += 1

        if removed > 0:
            self._save_index()

        return removed
