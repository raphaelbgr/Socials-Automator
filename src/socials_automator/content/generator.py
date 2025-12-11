"""Content generator for creating complete carousel posts."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable
import re

from ..providers import TextProvider, ImageProvider

# Set up file logging for AI calls
_log_dir = Path(__file__).parent.parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_ai_logger = logging.getLogger("ai_calls")
_ai_logger.setLevel(logging.DEBUG)
_ai_file_handler = logging.FileHandler(_log_dir / "ai_calls.log", encoding="utf-8")
_ai_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
_ai_logger.addHandler(_ai_file_handler)
from ..providers.config import load_provider_config
from ..design import SlideComposer, HookSlideTemplate, ContentSlideTemplate, CTASlideTemplate
from ..knowledge import KnowledgeStore, PostRecord, PromptLog, PromptLogEntry
from .planner import ContentPlanner
from .models import CarouselPost, SlideContent, SlideType, PostPlan, HookType, GenerationProgress


ProgressCallback = Callable[[GenerationProgress], Awaitable[None]]


class ContentGenerator:
    """Generate complete carousel posts with images.

    Orchestrates:
    - Content planning
    - Text generation
    - Image generation
    - Slide composition
    - Output saving

    Usage:
        generator = ContentGenerator(profile_path)

        # Generate a single post
        post = await generator.generate_post(
            topic="5 ChatGPT tricks for email",
            content_pillar="productivity_hacks",
        )

        # Generate daily posts
        posts = await generator.generate_daily_posts(count=3)
    """

    def __init__(
        self,
        profile_path: Path,
        profile_config: dict[str, Any] | None = None,
        text_provider: TextProvider | None = None,
        image_provider: ImageProvider | None = None,
        progress_callback: ProgressCallback | None = None,
    ):
        """Initialize the content generator.

        Args:
            profile_path: Path to profile directory.
            profile_config: Profile configuration dict.
            text_provider: Text generation provider.
            image_provider: Image generation provider.
            progress_callback: Callback for progress updates.
        """
        self.profile_path = profile_path
        self.profile_config = profile_config or self._load_profile_config()
        self.progress_callback = progress_callback or self._default_progress

        # Stats tracking
        self._total_text_calls = 0
        self._total_image_calls = 0
        self._total_cost = 0.0

        # Create providers with event callbacks
        config = load_provider_config()

        async def text_event_callback(event: dict[str, Any]) -> None:
            await self._handle_ai_event(event, "text")

        async def image_event_callback(event: dict[str, Any]) -> None:
            await self._handle_ai_event(event, "image")

        self.text_provider = text_provider or TextProvider(config, event_callback=text_event_callback)
        self.image_provider = image_provider or ImageProvider(config, event_callback=image_event_callback)

        self.planner = ContentPlanner(self.profile_config, self.text_provider)
        self.composer = SlideComposer(fonts_dir=profile_path / "brand" / "fonts")
        self.knowledge = KnowledgeStore(profile_path)

        # Get design config
        self.design_config = self.profile_config.get("design", {})
        self.output_config = self.profile_config.get("output_settings", {})

        # Current progress state
        self._current_progress: GenerationProgress | None = None
        self._current_post_id: str | None = None

    def _load_profile_config(self) -> dict[str, Any]:
        """Load profile configuration from metadata.json."""
        import json

        metadata_path = self.profile_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    async def _default_progress(self, progress: GenerationProgress) -> None:
        """Default progress callback (no-op)."""
        pass

    async def _emit_progress(self, progress: GenerationProgress) -> None:
        """Emit progress update."""
        self._current_progress = progress
        if self.progress_callback:
            await self.progress_callback(progress)

    async def _handle_ai_event(self, event: dict[str, Any], source: str) -> None:
        """Handle AI events from providers and forward to progress callback."""
        if not self._current_progress:
            return

        event_type = event.get("type", "")

        # Update stats
        if event_type == "text_response":
            self._total_text_calls += 1
        elif event_type == "image_response":
            self._total_image_calls += 1

        if event.get("cost_usd"):
            self._total_cost += event["cost_usd"]

        # Log AI call to file
        post_id = self._current_post_id or "unknown"
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        duration = event.get("duration_seconds", 0)
        cost = event.get("cost_usd", 0)
        task = event.get("task", "unknown")

        if event_type in ["text_response", "image_response"]:
            prompt_preview = (event.get("prompt_preview", "") or "")[:100]
            _ai_logger.info(
                f"POST:{post_id} | {event_type.upper()} | provider:{provider} | model:{model} | "
                f"task:{task} | duration:{duration:.2f}s | cost:${cost:.4f} | prompt:{prompt_preview}..."
            )
        elif event_type in ["text_error", "image_error"]:
            error = event.get("error", "unknown")
            _ai_logger.error(f"POST:{post_id} | {event_type.upper()} | provider:{provider} | error:{error}")

        # Build update dict
        update: dict[str, Any] = {
            "event_type": event_type,
            "provider": event.get("provider"),
            "model": event.get("model"),
            "prompt_preview": event.get("prompt_preview"),
            "response_preview": event.get("response_preview"),
            "duration_seconds": event.get("duration_seconds"),
            "cost_usd": event.get("cost_usd"),
            "total_text_calls": self._total_text_calls,
            "total_image_calls": self._total_image_calls,
            "total_cost_usd": self._total_cost,
        }

        # Handle text AI events
        if event_type.startswith("text_"):
            update["text_provider"] = event.get("provider")
            update["text_model"] = event.get("model")
            update["text_prompt_preview"] = event.get("prompt_preview")
            update["text_failed_providers"] = event.get("failed_providers", [])

        # Handle image AI events
        if event_type.startswith("image_"):
            update["image_provider"] = event.get("provider")
            update["image_model"] = event.get("model")
            update["image_prompt_preview"] = event.get("prompt_preview")
            update["image_failed_providers"] = event.get("failed_providers", [])

        # Create updated progress with event details
        progress = self._current_progress.model_copy(update=update)

        await self._emit_progress(progress)

    def _generate_post_id(self) -> str:
        """Generate a unique post ID."""
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")

        # Count existing posts for today
        posts_dir = self.profile_path / "posts" / now.strftime("%Y") / now.strftime("%m")
        existing = list(posts_dir.glob(f"{now.strftime('%d')}-*")) if posts_dir.exists() else []

        post_num = len(existing) + 1
        return f"{date_str}-{post_num:03d}"

    def _create_slug(self, topic: str) -> str:
        """Create URL-friendly slug from topic."""
        slug = topic.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "-", slug)
        return slug[:50].strip("-")

    def _get_output_path(self, post: CarouselPost, status: str = "generated") -> Path:
        """Get output directory path for a post.

        Args:
            post: The carousel post
            status: Post status folder - "generated", "pending-post", or "posted"

        Returns:
            Path to the post directory
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

    async def generate_post(
        self,
        topic: str,
        content_pillar: str,
        hook_type: HookType | None = None,
        target_slides: int | None = None,
        min_slides: int | None = None,
        max_slides: int | None = None,
        research_context: str | None = None,
    ) -> CarouselPost:
        """Generate a complete carousel post.

        Args:
            topic: Topic for the post.
            content_pillar: Content pillar category.
            hook_type: Hook type to use.
            target_slides: Number of slides. If None, AI decides (between min and max).
            min_slides: Minimum slides when AI decides. Defaults to profile config or 3.
            max_slides: Maximum slides when AI decides. Defaults to profile config or 10.
            research_context: Optional research context.

        Returns:
            Complete CarouselPost.
        """
        # Get slide settings from profile config if not specified
        carousel_settings = self.profile_config.get("content_strategy", {}).get("carousel_settings", {})
        if min_slides is None:
            min_slides = carousel_settings.get("min_slides", 3)
        if max_slides is None:
            max_slides = carousel_settings.get("max_slides", 10)

        start_time = time.time()
        post_id = self._generate_post_id()
        self._current_post_id = post_id  # Track for AI call logging
        prompt_log = PromptLog(post_id=post_id)

        _ai_logger.info(f"=== NEW POST SESSION === POST:{post_id} | topic: {topic}")

        progress = GenerationProgress(
            post_id=post_id,
            status="planning",
            total_steps=4,  # Initial estimate: Planning + caption + save + buffer
        )
        await self._emit_progress(progress)

        # Step 1: Plan the post
        progress.current_step = "Planning content structure (AI deciding slide count)"
        await self._emit_progress(progress)

        plan = await self.planner.plan_post(
            topic=topic,
            content_pillar=content_pillar,
            hook_type=hook_type,
            target_slides=target_slides,
            min_slides=min_slides,
            max_slides=max_slides,
            research_context=research_context,
        )

        # Update total_steps now that we know actual slide count
        actual_slides = len(plan.slides) if plan.slides else plan.target_slides
        progress.total_steps = actual_slides + 3  # Planning + slides + caption + save

        progress.completed_steps = 1
        progress.status = "generating"
        await self._emit_progress(progress)

        # Create post object
        post = CarouselPost(
            id=post_id,
            date=datetime.now().strftime("%Y-%m-%d"),
            slug=self._create_slug(topic),
            topic=topic,
            content_pillar=content_pillar,
            hook_type=plan.hook_type,
            hook_text=plan.hook_text,
            status="generating",
        )

        # Step 2: Generate slides
        progress.total_slides = len(plan.slides)

        for i, slide_outline in enumerate(plan.slides):
            progress.current_slide = i + 1
            progress.current_step = f"Generating slide {i + 1}/{len(plan.slides)}"
            await self._emit_progress(progress)

            slide = await self._generate_slide(
                slide_outline,
                plan,
                post,
                prompt_log,
            )
            post.slides.append(slide)

            progress.completed_steps += 1
            await self._emit_progress(progress)

        # Step 3: Generate caption
        progress.current_step = "Generating caption"
        await self._emit_progress(progress)

        content_summary = " | ".join([s.heading for s in post.slides if s.slide_type == SlideType.CONTENT])

        post.caption = await self.planner.generate_caption(
            topic=topic,
            hook_text=plan.hook_text,
            content_summary=content_summary,
            hashtags=plan.keywords,  # Use keywords as base
        )

        # Extract hashtags from config
        hashtag_config = self.profile_config.get("hashtag_strategy", {})
        hashtag_sets = hashtag_config.get("hashtag_sets", {})
        all_hashtags = []
        for category in ["primary", "secondary", "niche", "branded"]:
            all_hashtags.extend(hashtag_sets.get(category, []))
        post.hashtags = all_hashtags[:10]  # Limited for Threads compatibility (500 char limit)

        progress.completed_steps += 1
        await self._emit_progress(progress)

        # Finalize
        post.generation_time_seconds = time.time() - start_time
        post.text_provider = self.text_provider.current_provider
        post.image_provider = self.image_provider.current_provider
        post.status = "generated"

        progress.status = "completed"
        progress.current_step = "Done"
        progress.completed_steps = progress.total_steps
        await self._emit_progress(progress)

        # Log to knowledge base
        self.knowledge.log_prompts(prompt_log)
        self.knowledge.add_post(PostRecord(
            id=post.id,
            date=post.date,
            topic=post.topic,
            content_pillar=post.content_pillar,
            hook_type=post.hook_type.value,
            hook_text=post.hook_text,
            slides_count=post.slides_count,
            keywords=plan.keywords,
            path=str(self._get_output_path(post).relative_to(self.profile_path)),
            text_provider=post.text_provider,
            image_provider=post.image_provider,
            generation_time_seconds=post.generation_time_seconds,
            total_cost_usd=post.total_cost_usd,
        ))

        _ai_logger.info(
            f"=== POST COMPLETE === POST:{post_id} | slides:{post.slides_count} | "
            f"time:{post.generation_time_seconds:.1f}s | cost:${post.total_cost_usd:.4f} | "
            f"text_calls:{self._total_text_calls} | image_calls:{self._total_image_calls}"
        )

        return post

    async def _generate_slide(
        self,
        outline: dict[str, Any],
        plan: PostPlan,
        post: CarouselPost,
        prompt_log: PromptLog,
    ) -> SlideContent:
        """Generate a single slide with text and image.

        Args:
            outline: Slide outline from plan.
            plan: Full post plan.
            post: The post being generated.
            prompt_log: Prompt log for tracking.

        Returns:
            SlideContent with generated content.
        """
        slide_type_str = outline.get("slide_type", "content")
        slide_type = SlideType(slide_type_str)

        # Get heading with fallback for None values
        heading = outline.get("heading") or outline.get("title") or ""
        if not heading:
            # Generate default heading based on slide type
            if slide_type == SlideType.HOOK:
                heading = post.topic
            elif slide_type == SlideType.CTA:
                heading = "Follow for more!"
            else:
                heading = f"Point {outline.get('number', 1) - 1}"

        slide = SlideContent(
            number=outline.get("number", 1),
            slide_type=slide_type,
            heading=heading,
            body=outline.get("body") or outline.get("content") or outline.get("text"),
            has_background_image=outline.get("needs_image", False),
        )

        # Generate image if needed
        image_bytes: bytes | None = None

        if slide.has_background_image and outline.get("image_description"):
            # Build image prompt
            style_suffix = self.design_config.get("image_generation", {}).get(
                "style_prompt_suffix",
                "minimal, clean, tech aesthetic, dark mode"
            )

            image_prompt = f"{outline['image_description']}, {style_suffix}"
            slide.image_prompt = image_prompt

            try:
                # Determine task type for provider selection
                task = "hook_images" if slide_type == SlideType.HOOK else "content_images"

                image_bytes = await self.image_provider.generate(
                    prompt=image_prompt,
                    size="portrait",
                    task=task,
                )
            except Exception as e:
                # Continue without image on error
                slide.has_background_image = False

        # Compose the slide image using Pillow
        logo_path = self.profile_path / "brand" / "logo.png"
        if not logo_path.exists():
            logo_path = None

        if slide_type == SlideType.HOOK:
            template = HookSlideTemplate()
            slide_bytes = await self.composer.create_hook_slide(
                text=slide.heading,
                subtext=slide.body or plan.hook_subtext,
                template=template,
                background_image=image_bytes,
                logo_path=logo_path,
            )
        elif slide_type == SlideType.CTA:
            template = CTASlideTemplate()
            handle = self.profile_config.get("profile", {}).get("instagram_handle", "")
            slide_bytes = await self.composer.create_cta_slide(
                text=slide.heading,
                handle=handle,
                secondary_text=slide.body,
                template=template,
                logo_path=logo_path,
            )
        else:
            template = ContentSlideTemplate()
            slide_bytes = await self.composer.create_content_slide(
                heading=slide.heading,
                body=slide.body,
                number=slide.number - 1 if slide.number > 1 else None,  # Adjust for hook slide
                template=template,
                background_image=image_bytes,
                logo_path=logo_path,
            )

        slide.image_bytes = slide_bytes

        return slide

    async def save_post(self, post: CarouselPost) -> Path:
        """Save a generated post to disk.

        Args:
            post: The post to save.

        Returns:
            Path to the saved post directory.
        """
        import json

        output_path = self._get_output_path(post)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save slides
        file_naming = self.output_config.get("file_naming", {})
        slide_pattern = file_naming.get("slides", "slide_{number:02d}.jpg")

        for slide in post.slides:
            if slide.image_bytes:
                slide_filename = slide_pattern.format(number=slide.number)
                slide_path = output_path / slide_filename
                slide_path.write_bytes(slide.image_bytes)
                slide.image_path = str(slide_path.relative_to(output_path))

        # Save caption
        caption_filename = file_naming.get("caption", "caption.txt")
        (output_path / caption_filename).write_text(post.caption, encoding="utf-8")

        # Save hashtags
        hashtags_filename = file_naming.get("hashtags", "hashtags.txt")
        hashtags_text = " ".join(post.hashtags)
        (output_path / hashtags_filename).write_text(
            hashtags_text,
            encoding="utf-8"
        )

        # Save combined caption + hashtags
        combined_filename = file_naming.get("combined", "caption+hashtags.txt")
        combined_text = f"{post.caption}\n\n{hashtags_text}"
        (output_path / combined_filename).write_text(combined_text, encoding="utf-8")

        # Save alt texts
        alt_texts = [f"Slide {s.number}: {s.heading}" for s in post.slides]
        alt_filename = file_naming.get("alt_texts", "alt_texts.json")
        with open(output_path / alt_filename, "w") as f:
            json.dump(alt_texts, f, indent=2)

        # Save metadata
        metadata_filename = file_naming.get("metadata", "metadata.json")

        # Create serializable metadata
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

        with open(output_path / metadata_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    async def generate_daily_posts(
        self,
        count: int = 3,
        topics: list[dict[str, Any]] | None = None,
    ) -> list[CarouselPost]:
        """Generate multiple posts for a day.

        Args:
            count: Number of posts to generate.
            topics: Optional list of topic dicts with topic and content_pillar.

        Returns:
            List of generated posts.
        """
        posts = []

        # Get content pillars from profile config
        content_pillars = [
            p.get("id")
            for p in self.profile_config.get("content_strategy", {}).get("content_pillars", [])
        ]

        # Get topics to avoid from recent post history
        avoid_topics = self.knowledge.get_topics_to_avoid(days=14)

        # Get topics if not provided
        if topics is None:
            # Try research first
            try:
                from ..research import ResearchAggregator
                aggregator = ResearchAggregator(self.profile_config)

                ideas = await aggregator.get_content_ideas(
                    content_pillars=content_pillars,
                    avoid_topics=avoid_topics,
                    count=count,
                )

                topics = [
                    {
                        "topic": idea["topic"],
                        "content_pillar": idea.get("suggested_pillar", content_pillars[0] if content_pillars else "general"),
                    }
                    for idea in ideas
                ]

                await aggregator.close()
            except Exception:
                # Research failed, topics will be empty
                topics = []

            # Fallback to AI-generated topics if research returned nothing
            if not topics:
                topics = await self._generate_topics_with_ai(
                    count=count,
                    content_pillars=content_pillars,
                    avoid_topics=avoid_topics,
                )

        # Generate posts
        for topic_info in topics[:count]:
            try:
                post = await self.generate_post(
                    topic=topic_info["topic"],
                    content_pillar=topic_info.get("content_pillar", "general"),
                )

                # Save post
                await self.save_post(post)
                posts.append(post)

            except Exception as e:
                # Continue with other posts on error
                continue

        return posts

    async def _generate_topics_with_ai(
        self,
        count: int,
        content_pillars: list[str],
        avoid_topics: list[str],
    ) -> list[dict[str, Any]]:
        """Generate topics using AI when research fails.

        Args:
            count: Number of topics to generate.
            content_pillars: Available content pillars.
            avoid_topics: Topics to avoid (recently used).

        Returns:
            List of topic dicts with topic and content_pillar.
        """
        import json as json_module
        import random

        # Get profile context
        profile = self.profile_config.get("profile", {})
        niche = profile.get("niche_id", "general")
        target_audience = profile.get("target_audience", {})

        # Get content pillar details
        pillar_details = self.profile_config.get("content_strategy", {}).get("content_pillars", [])
        pillars_info = "\n".join([
            f"- {p.get('id')}: {p.get('description', p.get('name', ''))}"
            for p in pillar_details
        ]) if pillar_details else "general content"

        # Build avoid topics string (limit to prevent prompt overflow)
        avoid_str = ""
        if avoid_topics:
            recent_topics = avoid_topics[:20]  # Limit to 20 most recent
            avoid_str = f"\n\nTOPICS TO AVOID (recently posted):\n" + "\n".join(f"- {t}" for t in recent_topics)

        # Get recent posts context for variety
        recent_context = self.knowledge.get_recent_context(days=14)

        prompt = f"""Generate {count} unique and engaging Instagram carousel post topic(s) for a {niche} account.

TARGET AUDIENCE:
{json_module.dumps(target_audience, indent=2) if target_audience else "General audience interested in the niche"}

CONTENT PILLARS (choose one for each topic):
{pillars_info}

RECENT POST HISTORY (for context and variety):
{recent_context}
{avoid_str}

REQUIREMENTS:
- Each topic should be specific and actionable (e.g., "5 ChatGPT prompts for writing emails" not "AI email tips")
- Topics should be fresh and NOT similar to recently posted topics
- Each topic should clearly fit one content pillar
- Topics should be trendy and relevant to current interests
- Mix different formats: tutorials, tips, comparisons, myths debunked, etc.

Return as JSON array:
[
    {{"topic": "Specific topic title", "content_pillar": "pillar_id"}},
    ...
]

Return ONLY the JSON array, no other text."""

        try:
            response = await self.text_provider.generate(
                prompt=prompt,
                system="You are a social media content strategist. Generate engaging, specific topic ideas that will perform well on Instagram.",
                task="topic_generation",
                temperature=0.9,  # Higher creativity for topic variety
                max_tokens=1000,
            )

            # Parse JSON response
            data = self.planner._extract_json(response)

            # Handle both array and object responses
            if isinstance(data, list):
                topics = data
            elif isinstance(data, dict) and "topics" in data:
                topics = data["topics"]
            else:
                topics = [data]

            # Validate and return topics
            valid_topics = []
            for topic in topics[:count]:
                if isinstance(topic, dict) and "topic" in topic:
                    # Ensure content_pillar is valid
                    pillar = topic.get("content_pillar", "")
                    if pillar not in content_pillars and content_pillars:
                        pillar = random.choice(content_pillars)
                    valid_topics.append({
                        "topic": topic["topic"],
                        "content_pillar": pillar or "general",
                    })

            return valid_topics

        except Exception as e:
            # Last resort: generate a simple topic
            pillar = random.choice(content_pillars) if content_pillars else "general"
            return [{
                "topic": f"Top tips for {niche.replace('-', ' ').replace('_', ' ')}",
                "content_pillar": pillar,
            }]
