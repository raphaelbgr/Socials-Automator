"""Pexels image cache system.

Caches downloaded Pexels images to reduce API calls and download time.
Uses a flat file structure with a JSON index for fast lookups.

Structure:
    pexels/image-cache/
        index.json          # Quick lookup by pexels_id
        12345678.jpg        # Image files named by pexels_id
        87654321.jpg
        ...
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from socials_automator.constants import get_pexels_image_cache_dir


class PexelsImageCache:
    """Manages cached Pexels images with JSON index."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize cache.

        Args:
            cache_dir: Cache directory path. Uses default if not provided.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else get_pexels_image_cache_dir()
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

    def has_image(self, pexels_id: int) -> bool:
        """Check if image is in cache.

        Args:
            pexels_id: Pexels image ID.

        Returns:
            True if image is cached and file exists.
        """
        str_id = str(pexels_id)
        if str_id not in self._index:
            return False

        # Verify file actually exists
        image_path = self.cache_dir / self._index[str_id].get("filename", "")
        return image_path.exists()

    def get_image_path(self, pexels_id: int) -> Optional[Path]:
        """Get cached image path.

        Args:
            pexels_id: Pexels image ID.

        Returns:
            Path to cached image or None if not cached.
        """
        str_id = str(pexels_id)
        if not self.has_image(pexels_id):
            return None

        # Increment hit count
        self._index[str_id]["hit_count"] = self._index[str_id].get("hit_count", 0) + 1
        self._index[str_id]["last_used"] = datetime.now().isoformat()
        self._save_index()

        return self.cache_dir / self._index[str_id]["filename"]

    def get_metadata(self, pexels_id: int) -> Optional[dict]:
        """Get cached image metadata.

        Args:
            pexels_id: Pexels image ID.

        Returns:
            Metadata dict or None if not cached.
        """
        str_id = str(pexels_id)
        return self._index.get(str_id)

    def copy_to_destination(self, pexels_id: int, destination: Path) -> bool:
        """Copy cached image to destination.

        Args:
            pexels_id: Pexels image ID.
            destination: Destination path for the image.

        Returns:
            True if copied successfully, False otherwise.
        """
        cached_path = self.get_image_path(pexels_id)
        if not cached_path:
            return False

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached_path, destination)
            return True
        except IOError:
            return False

    def add_image(
        self,
        pexels_id: int,
        source_path: Path,
        image_data: dict,
        query_used: str,
    ) -> Path:
        """Add image to cache.

        Args:
            pexels_id: Pexels image ID.
            source_path: Path to downloaded image file.
            image_data: Pexels API image data.
            query_used: Search query that found this image.

        Returns:
            Path to cached image.
        """
        str_id = str(pexels_id)

        # Determine file extension from source
        ext = source_path.suffix.lower() or ".jpg"
        filename = f"{pexels_id}{ext}"
        cache_path = self.cache_dir / filename

        # Copy to cache if not already there
        if source_path != cache_path:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, cache_path)

        # Get image dimensions
        width = image_data.get("width", 0)
        height = image_data.get("height", 0)

        # Build metadata
        metadata = {
            "pexels_id": pexels_id,
            "filename": filename,
            "pexels_url": image_data.get("url", ""),
            "photographer": image_data.get("photographer", "Unknown"),
            "photographer_url": image_data.get("photographer_url", ""),
            "alt": image_data.get("alt", ""),
            "width": width,
            "height": height,
            "orientation": self._get_orientation(width, height),
            "query_used": query_used,
            "src_original": image_data.get("src", {}).get("original", ""),
            "src_large": image_data.get("src", {}).get("large", ""),
            "file_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
            "cached_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat(),
            "hit_count": 0,
        }

        self._index[str_id] = metadata
        self._save_index()

        return cache_path

    def _get_orientation(self, width: int, height: int) -> str:
        """Determine image orientation."""
        if width > height:
            return "landscape"
        elif height > width:
            return "portrait"
        else:
            return "square"

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache stats.
        """
        total_images = len(self._index)
        total_hits = sum(v.get("hit_count", 0) for v in self._index.values())
        total_size = sum(v.get("file_size_bytes", 0) for v in self._index.values())

        return {
            "total_images": total_images,
            "total_hits": total_hits,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }

    def search_by_query(self, query: str, limit: int = 10) -> list[dict]:
        """Search cached images by query text.

        Args:
            query: Query text to search for.
            limit: Maximum results to return.

        Returns:
            List of matching image metadata.
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for metadata in self._index.values():
            # Search in query_used and alt text
            cached_query = metadata.get("query_used", "").lower()
            alt_text = metadata.get("alt", "").lower()

            # Calculate match score
            score = 0

            # Exact query match
            if query_lower in cached_query:
                score += 2

            # Word matches in query
            cached_words = set(cached_query.split())
            word_matches = len(query_words & cached_words)
            score += word_matches

            # Alt text matches
            if query_lower in alt_text:
                score += 1

            if score > 0:
                results.append({
                    **metadata,
                    "_match_score": score,
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
            image_path = self.cache_dir / filename
            if image_path.exists():
                image_path.unlink()
            del self._index[str_id]
            removed += 1

        if removed > 0:
            self._save_index()

        return removed
