"""Stateless service for carousel post generation and upload."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.types import Result, Success, Failure, GenerationResult


class PostGeneratorService:
    """Stateless service for carousel post generation.

    All state is passed via params - no instance state.
    """

    async def generate(
        self,
        profile_path: Path,
        topic: Optional[str],
        pillar: Optional[str],
        slides: Optional[int],
        min_slides: int,
        max_slides: int,
        text_ai: Optional[str],
        image_ai: Optional[str],
        ai_tools: bool,
        auto_retry: bool,
    ) -> Result[GenerationResult]:
        """Generate a single carousel post.

        Stateless - all configuration passed via params.

        Returns:
            Result containing GenerationResult or Failure
        """
        from socials_automator.content import ContentOrchestrator
        from socials_automator.content.output import ContentSaver

        try:
            # Load profile config
            config = self._load_profile_config(profile_path)

            # Get slide settings from profile if not overridden
            carousel_settings = config.get("content_strategy", {}).get("carousel_settings", {})
            effective_min = min_slides if min_slides != 3 else carousel_settings.get("min_slides", 3)
            effective_max = max_slides if max_slides != 10 else carousel_settings.get("max_slides", 10)

            # Select topic and pillar if not provided
            final_topic = topic or self._select_topic(config)
            final_pillar = pillar or self._select_pillar(config, final_topic)

            # Create orchestrator
            orchestrator = ContentOrchestrator(
                profile_path=profile_path,
                text_ai=text_ai,
                image_ai=image_ai,
                ai_tools=ai_tools,
            )

            # Generate post
            post = await orchestrator.generate_post(
                topic=final_topic,
                content_pillar=final_pillar,
                target_slides=slides,
                min_slides=effective_min,
                max_slides=effective_max,
            )

            # Save post
            saver = ContentSaver(profile_path)
            output_path = saver.save_carousel(post)

            return Success(GenerationResult(
                success=True,
                output_path=output_path,
                duration_seconds=0,  # Not applicable for posts
                metadata={
                    "topic": final_topic,
                    "pillar": final_pillar,
                    "slide_count": len(post.slides),
                },
            ))

        except Exception as e:
            return Failure(str(e))

    def _load_profile_config(self, profile_path: Path) -> dict:
        """Load profile configuration."""
        metadata_path = profile_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _select_topic(self, config: dict) -> str:
        """Select a topic from profile config."""
        import random

        topics = config.get("content_strategy", {}).get("topics", [])
        if topics:
            return random.choice(topics)

        niches = config.get("niches", [])
        if niches:
            return f"Tips about {random.choice(niches)}"

        return "General tips and insights"

    def _select_pillar(self, config: dict, topic: str) -> str:
        """Select a content pillar from profile config."""
        import random

        pillars = config.get("content_strategy", {}).get("pillars", [])
        if pillars:
            return random.choice(pillars)

        return "Educational"


class PostUploaderService:
    """Stateless service for carousel post upload to Instagram."""

    async def upload_all(
        self,
        profile_path: Path,
        dry_run: bool = False,
        post_one: bool = False,
    ) -> Result[list]:
        """Upload all pending posts.

        Returns:
            Result containing list of upload results or Failure
        """
        # Find pending posts
        pending = self._find_pending_posts(profile_path)
        if not pending:
            return Failure("No pending posts found")

        # Limit to one if requested
        if post_one:
            pending = pending[:1]

        # Upload each post
        results = []
        for post_path in pending:
            result = await self._upload_single(post_path, profile_path, dry_run)
            results.append(result)

        return Success(results)

    async def upload_single(
        self,
        profile_path: Path,
        post_id: str,
        dry_run: bool = False,
    ) -> Result[dict]:
        """Upload a specific post by ID.

        Returns:
            Result containing upload result or Failure
        """
        post_path = self._find_post_by_id(profile_path, post_id)
        if post_path is None:
            return Failure(f"Post not found: {post_id}")

        result = await self._upload_single(post_path, profile_path, dry_run)
        return Success(result)

    def _find_pending_posts(self, profile_path: Path) -> list[Path]:
        """Find all pending posts for profile."""
        pending_paths = []

        # Check both generated and pending-post folders
        for status in ["generated", "pending-post"]:
            base_dir = profile_path / "posts"
            if not base_dir.exists():
                continue

            for year_dir in sorted(base_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                for month_dir in sorted(year_dir.iterdir()):
                    if not month_dir.is_dir():
                        continue
                    status_dir = month_dir / status
                    if not status_dir.exists():
                        continue
                    for post_dir in sorted(status_dir.iterdir()):
                        if post_dir.is_dir() and (post_dir / "slide_01.jpg").exists():
                            pending_paths.append(post_dir)

        return pending_paths

    def _find_post_by_id(self, profile_path: Path, post_id: str) -> Optional[Path]:
        """Find a specific post by ID."""
        all_posts = self._find_pending_posts(profile_path)
        for post_path in all_posts:
            if post_id in post_path.name:
                return post_path
        return None

    async def _upload_single(
        self,
        post_path: Path,
        profile_path: Path,
        dry_run: bool,
    ) -> dict:
        """Upload a single post to Instagram."""
        result = {
            "path": post_path,
            "success": False,
            "error": None,
        }

        if dry_run:
            result["success"] = True
            result["dry_run"] = True
            return result

        try:
            from socials_automator.instagram import InstagramPoster
            from socials_automator.instagram.models import InstagramConfig

            config = InstagramConfig.from_env()
            poster = InstagramPoster(config)

            # Load post metadata
            metadata_path = post_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            # Get caption
            caption_path = post_path / "caption+hashtags.txt"
            if caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8")
            else:
                caption = metadata.get("caption", "")

            # Get image files
            images = sorted(post_path.glob("slide_*.jpg"))

            # Upload
            post_result = await poster.post_carousel(
                images=[str(img) for img in images],
                caption=caption,
            )

            result["success"] = True
            result["post_id"] = post_result.get("id")

            # Move to posted folder
            self._move_to_posted(post_path, profile_path)

        except Exception as e:
            result["error"] = str(e)

        return result

    def _move_to_posted(self, post_path: Path, profile_path: Path) -> None:
        """Move post folder to posted status."""
        import shutil

        # Determine new path
        parts = post_path.parts
        if "generated" in parts or "pending-post" in parts:
            # Replace status folder with 'posted'
            new_parts = []
            for part in parts:
                if part in ["generated", "pending-post"]:
                    new_parts.append("posted")
                else:
                    new_parts.append(part)
            new_path = Path(*new_parts)

            # Create parent directory
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Move folder
            shutil.move(str(post_path), str(new_path))
