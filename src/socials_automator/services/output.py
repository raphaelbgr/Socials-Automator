"""Output service for saving generated posts.

Handles all file I/O for saving carousel posts, extracted from
generator.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..content.models import CarouselPost


class OutputService:
    """Handles saving generated posts to disk.

    Responsibilities:
    - Generating output paths
    - Saving slide images
    - Saving caption and hashtags
    - Saving metadata

    Usage:
        service = OutputService(profile_path, output_config)
        path = await service.save(post)
    """

    def __init__(
        self,
        profile_path: Path,
        output_config: dict[str, Any] | None = None,
    ):
        """Initialize the output service.

        Args:
            profile_path: Path to profile directory.
            output_config: Output configuration from profile.
        """
        self.profile_path = Path(profile_path)
        self.output_config = output_config or {}

    def get_output_path(
        self,
        post: "CarouselPost",
        status: str = "generated",
    ) -> Path:
        """Get output directory path for a post.

        Args:
            post: The carousel post.
            status: Post status folder - "generated", "pending-post", or "posted".

        Returns:
            Path to the post directory.
        """
        now = datetime.now()
        folder_template = self.output_config.get(
            "folder_structure",
            "posts/{year}/{month}/{status}/{day}-{post_number}-{slug}"
        )

        folder = folder_template.format(
            year=now.strftime("%Y"),
            month=now.strftime("%m"),
            day=now.strftime("%d"),
            post_number=post.id.split("-")[-1],
            slug=post.slug,
            status=status,
        )

        return self.profile_path / folder

    async def save(self, post: "CarouselPost", status: str = "generated") -> Path:
        """Save a generated post to disk.

        Args:
            post: The post to save.
            status: Post status for folder placement.

        Returns:
            Path to the saved post directory.
        """
        output_path = self.get_output_path(post, status)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save slides
        await self._save_slides(post, output_path)

        # Save caption
        await self._save_caption(post, output_path)

        # Save hashtags
        await self._save_hashtags(post, output_path)

        # Save combined caption + hashtags
        await self._save_combined(post, output_path)

        # Save alt texts
        await self._save_alt_texts(post, output_path)

        # Save metadata
        await self._save_metadata(post, output_path)

        return output_path

    async def _save_slides(self, post: "CarouselPost", output_path: Path) -> None:
        """Save slide images.

        Args:
            post: The post containing slides.
            output_path: Output directory.
        """
        file_naming = self.output_config.get("file_naming", {})
        slide_pattern = file_naming.get("slides", "slide_{number:02d}.jpg")

        for slide in post.slides:
            if slide.image_bytes:
                slide_filename = slide_pattern.format(number=slide.number)
                slide_path = output_path / slide_filename
                slide_path.write_bytes(slide.image_bytes)
                slide.image_path = str(slide_path.relative_to(output_path))

    async def _save_caption(self, post: "CarouselPost", output_path: Path) -> None:
        """Save caption file (Threads-ready, under 500 chars, no hashtags).

        Args:
            post: The post containing caption.
            output_path: Output directory.
        """
        file_naming = self.output_config.get("file_naming", {})
        caption_filename = file_naming.get("caption", "caption.txt")
        # Save ONLY the caption text (no hashtags) - Threads-ready
        caption_text = post.caption.strip()
        (output_path / caption_filename).write_text(caption_text, encoding="utf-8")

    async def _save_hashtags(self, post: "CarouselPost", output_path: Path) -> None:
        """Save hashtags file.

        Args:
            post: The post containing hashtags.
            output_path: Output directory.
        """
        file_naming = self.output_config.get("file_naming", {})
        hashtags_filename = file_naming.get("hashtags", "hashtags.txt")
        hashtags_text = " ".join(post.hashtags)
        (output_path / hashtags_filename).write_text(hashtags_text, encoding="utf-8")

    async def _save_combined(self, post: "CarouselPost", output_path: Path) -> None:
        """Save combined caption + hashtags file (full Instagram caption).

        This is the FULL caption used for Instagram posting (can be over 500 chars).
        Includes caption text plus ALL hashtags.

        Args:
            post: The post containing caption and hashtags.
            output_path: Output directory.
        """
        file_naming = self.output_config.get("file_naming", {})
        combined_filename = file_naming.get("combined", "caption+hashtags.txt")
        # Full caption with ALL hashtags for Instagram
        hashtags_text = " ".join(post.hashtags)
        combined_text = f"{post.caption.strip()}\n\n{hashtags_text}"
        (output_path / combined_filename).write_text(combined_text, encoding="utf-8")

    async def _save_alt_texts(self, post: "CarouselPost", output_path: Path) -> None:
        """Save alt texts for accessibility.

        Args:
            post: The post containing slides.
            output_path: Output directory.
        """
        file_naming = self.output_config.get("file_naming", {})
        alt_filename = file_naming.get("alt_texts", "alt_texts.json")
        alt_texts = [f"Slide {s.number}: {s.heading}" for s in post.slides]

        with open(output_path / alt_filename, "w", encoding="utf-8") as f:
            json.dump(alt_texts, f, indent=2)

    async def _save_metadata(self, post: "CarouselPost", output_path: Path) -> None:
        """Save post metadata.

        Args:
            post: The post to save metadata for.
            output_path: Output directory.
        """
        file_naming = self.output_config.get("file_naming", {})
        metadata_filename = file_naming.get("metadata", "metadata.json")

        metadata = {
            "post": {
                "id": post.id,
                "date": post.date,
                "slug": post.slug,
                "topic": post.topic,
                "content_pillar": post.content_pillar,
                "hook_type": post.hook_type.value,
                "hook_text": post.hook_text,
                "slides_count": post.slides_count,
                "status": post.status,
                "created_at": post.created_at.isoformat(),
            },
            "generation": {
                "time_seconds": post.generation_time_seconds,
                "cost_usd": post.total_cost_usd,
                "text_provider": post.text_provider,
                "image_provider": post.image_provider,
            },
            "content": {
                "slides": [
                    {
                        "number": s.number,
                        "type": s.slide_type.value,
                        "heading": s.heading,
                        "body": s.body,
                        "image_path": s.image_path,
                        "image_prompt": s.image_prompt,
                    }
                    for s in post.slides
                ],
                "caption": post.caption,
                "hashtags": post.hashtags,
            },
        }

        with open(output_path / metadata_filename, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    async def move_to_status(
        self,
        post: "CarouselPost",
        from_status: str,
        to_status: str,
    ) -> Path:
        """Move a post from one status folder to another.

        Args:
            post: The post to move.
            from_status: Current status folder.
            to_status: Target status folder.

        Returns:
            New path to the post directory.
        """
        import shutil

        old_path = self.get_output_path(post, from_status)
        new_path = self.get_output_path(post, to_status)

        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))

        return new_path
