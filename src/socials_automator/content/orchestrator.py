"""Content orchestrator for coordinating post generation workflow.

This is the main entry point for generating carousel posts. It coordinates
all the components following the Single Responsibility Principle - each
component handles one concern, and the orchestrator wires them together.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .slides import SlideJobFactory, SlideJobContext
from .models import CarouselPost, SlideType, HookType
from ..services import ProgressManager, OutputService
from ..tools import TOOL_SCHEMAS, ToolExecutor

if TYPE_CHECKING:
    from .planner import ContentPlanner
    from ..providers import TextProvider, ImageProvider
    from ..design import SlideComposer
    from ..knowledge import KnowledgeStore


# Logger
_logger = logging.getLogger("ai_calls")


class ContentOrchestrator:
    """Orchestrates the content generation workflow.

    This is the main coordinator that delegates to specialized components:
    - ContentPlanner: Plans post structure and content
    - SlideJobFactory: Creates slide generation jobs
    - SlideComposer: Renders final slide images
    - OutputService: Saves posts to disk
    - ProgressManager: Tracks progress

    Usage:
        orchestrator = ContentOrchestrator(
            profile_path=profile_path,
            profile_config=config,
        )

        post = await orchestrator.generate_post(
            topic="5 AI Tools",
            content_pillar="tools",
        )

        path = await orchestrator.save_post(post)
    """

    def __init__(
        self,
        profile_path: Path,
        profile_config: dict[str, Any] | None = None,
        text_provider: "TextProvider | None" = None,
        image_provider: "ImageProvider | None" = None,
        composer: "SlideComposer | None" = None,
        planner: "ContentPlanner | None" = None,
        knowledge: "KnowledgeStore | None" = None,
        progress_callback: Any = None,
        auto_retry: bool = False,
        text_provider_override: str | None = None,
        image_provider_override: str | None = None,
        ai_tools: bool = False,
    ):
        """Initialize the orchestrator.

        Args:
            profile_path: Path to profile directory.
            profile_config: Profile configuration (loaded from metadata.json if not provided).
            text_provider: Text generation provider (created if not provided).
            image_provider: Image generation provider (created if not provided).
            composer: Optional composer (created if not provided).
            planner: Optional planner (created if not provided).
            knowledge: Optional knowledge store (created if not provided).
            progress_callback: Optional progress callback.
            auto_retry: If True, retry indefinitely until valid content.
            text_provider_override: Override text provider (e.g., 'lmstudio', 'openai').
            image_provider_override: Override image provider (e.g., 'dalle', 'comfy').
            ai_tools: If True, use AI-driven tool calling for research.
        """
        self.profile_path = Path(profile_path)
        self.auto_retry = auto_retry
        self.ai_tools = ai_tools
        self.progress_callback = progress_callback

        # Load profile config if not provided
        if profile_config is None:
            profile_config = self._load_profile_config()
        self.profile_config = profile_config

        # Get configs
        self.design_config = profile_config.get("design", {})
        self.output_config = profile_config.get("output_settings", {})

        # Create providers if not provided
        from ..providers.config import load_provider_config

        config = load_provider_config()

        if text_provider is None:
            from ..providers import TextProvider
            text_provider = TextProvider(
                config,
                provider_override=text_provider_override,
            )
        self.text_provider = text_provider

        if image_provider is None:
            from ..providers import ImageProvider
            image_provider = ImageProvider(
                config,
                provider_override=image_provider_override,
            )
        self.image_provider = image_provider

        # Create composer if not provided
        if composer is None:
            from ..design import SlideComposer
            composer = SlideComposer(fonts_dir=self.profile_path / "brand" / "fonts")
        self.composer = composer

        # Create planner if not provided
        if planner is None:
            from .planner import ContentPlanner
            planner = ContentPlanner(
                profile_config=profile_config,
                text_provider=text_provider,
                progress_callback=progress_callback,
                auto_retry=auto_retry,
            )
        self.planner = planner

        # Create knowledge store if not provided
        if knowledge is None:
            from ..knowledge import KnowledgeStore
            knowledge = KnowledgeStore(self.profile_path)
        self.knowledge = knowledge

        # Create output service
        self.output_service = OutputService(self.profile_path, self.output_config)

    def _load_profile_config(self) -> dict[str, Any]:
        """Load profile configuration from metadata.json."""
        import json

        metadata_path = self.profile_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _generate_post_id(self) -> str:
        """Generate a unique post ID."""
        now = datetime.now()
        posts_dir = self.profile_path / "posts" / now.strftime("%Y") / now.strftime("%m")
        existing = list(posts_dir.glob(f"{now.strftime('%d')}-*")) if posts_dir.exists() else []
        post_num = len(existing) + 1
        return f"{now.strftime('%Y%m%d')}-{post_num:03d}"

    def _create_slug(self, topic: str) -> str:
        """Create URL-friendly slug from topic."""
        import re
        slug = topic.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s-]+", "-", slug)
        return slug[:50].strip("-")

    async def _ai_research(self, topic: str, post_id: str) -> str:
        """Perform AI-driven research using tool calling.

        The AI decides what to search for based on the topic.

        Args:
            topic: Topic to research.
            post_id: Post ID for logging.

        Returns:
            Research context string.
        """
        _logger.info(f"POST:{post_id} | AI_RESEARCH_START | topic:{topic[:80]}")

        # Create tool executor with image provider for generate_image tool
        tool_executor = ToolExecutor(
            callback=self.progress_callback,
            post_id=post_id,
            image_provider=self.image_provider,
        )

        # System prompt for research
        system_prompt = """You are a research assistant helping create engaging Instagram carousel content.
Your job is to research the given topic to find:
- Current facts and statistics
- Expert opinions and quotes
- Trending information
- Practical examples

You have access to web_search and news_search tools. Use them to gather relevant information.
After researching, provide a summary of key findings that can be used in the carousel."""

        # User prompt
        user_prompt = f"""Research the following topic for an Instagram carousel post:

TOPIC: {topic}

Use the available search tools to find:
1. Key facts and statistics about this topic
2. Recent news or developments
3. Expert tips or advice
4. Examples or case studies

After searching, summarize the most relevant findings."""

        # Initial request with tools
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        max_iterations = 5
        research_results = []

        for iteration in range(max_iterations):
            response = await self.text_provider.generate_with_tools(
                prompt=messages[-1]["content"] if messages[-1]["role"] == "user" else user_prompt,
                tools=TOOL_SCHEMAS,
                system=system_prompt if iteration == 0 else None,
                task="ai_research",
                temperature=0.7,
                max_tokens=2048,
            )

            content = response.get("content")
            tool_calls = response.get("tool_calls")
            finish_reason = response.get("finish_reason")

            # If AI returned content without tool calls, we're done
            if content and not tool_calls:
                research_results.append(content)
                break

            # If AI wants to use tools, execute them
            if tool_calls:
                _logger.info(f"POST:{post_id} | AI_TOOL_CALLS | count:{len(tool_calls)}")

                # Execute all tool calls
                tool_results = await tool_executor.execute_tool_calls(tool_calls)

                # Add tool results to context
                for tc, result in zip(tool_calls, tool_results):
                    tool_name = tc["function"]["name"]
                    if result.success:
                        research_results.append(f"[{tool_name}]\n{result.result}")
                    else:
                        _logger.warning(f"POST:{post_id} | TOOL_FAILED | tool:{tool_name} | error:{result.error}")

                # Continue conversation with tool results
                messages.append({
                    "role": "user",
                    "content": f"Here are the search results:\n\n" + "\n\n".join([
                        f"Tool: {tc['function']['name']}\nResult: {r.to_message()}"
                        for tc, r in zip(tool_calls, tool_results)
                    ]) + "\n\nNow summarize the key findings for the carousel."
                })

            # If finished for other reasons, stop
            if finish_reason == "stop" and not tool_calls:
                break

        # Combine all research into context
        research_context = "\n\n---\n\n".join(research_results) if research_results else ""

        _logger.info(
            f"POST:{post_id} | AI_RESEARCH_END | "
            f"tool_calls:{tool_executor.total_tool_calls} | "
            f"context_len:{len(research_context)}"
        )

        return research_context

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
            hook_type: Optional hook type to use.
            target_slides: Number of slides. If None, AI decides.
            min_slides: Minimum slides when AI decides.
            max_slides: Maximum slides when AI decides.
            research_context: Optional pre-fetched research context.

        Returns:
            Complete CarouselPost.
        """
        start_time = time.time()
        post_id = self._generate_post_id()

        # Initialize progress manager
        progress_manager = ProgressManager(
            post_id=post_id,
            callback=self.progress_callback,
            topic=topic,
        )

        _logger.info(f"POST:{post_id} | START | topic:{topic[:80]}")

        try:
            # Step 0: AI-driven research (if enabled and no context provided)
            if self.ai_tools and not research_context:
                await progress_manager.start_phase("Research", 0, "AI researching topic")
                research_context = await self._ai_research(topic, post_id)
                await progress_manager.complete_phase("Research")

            # Step 1: Plan the post
            await progress_manager.start_phase("Planning", 1, "Planning content structure")

            # Set planner's progress state so it can emit phase updates
            self.planner._current_progress = progress_manager.progress

            plan = await self.planner.plan_post(
                topic=topic,
                content_pillar=content_pillar,
                hook_type=hook_type,
                target_slides=target_slides,
                min_slides=min_slides,
                max_slides=max_slides,
                research_context=research_context,
            )

            await progress_manager.set_total_slides(len(plan.slides))
            await progress_manager.complete_phase("Planning")

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

            # Step 2: Enable CTA image if configured
            self._configure_cta_image(plan, topic)

            # Step 3: Generate slides using SlideJobs
            logo_path = self.profile_path / "brand" / "logo.png"

            for i, slide_outline in enumerate(plan.slides):
                await progress_manager.start_slide(i + 1)

                # Determine slide type
                slide_type_str = slide_outline.get("slide_type", "content")
                slide_type = SlideType(slide_type_str)

                # Create context for this slide
                context = SlideJobContext(
                    post_id=post_id,
                    slide_number=i + 1,
                    topic=topic,
                    outline=slide_outline,
                    profile_config=self.profile_config,
                    design_config=self.design_config,
                    logo_path=str(logo_path) if logo_path.exists() else None,
                )

                # Create and execute job
                job = SlideJobFactory.create(
                    slide_type=slide_type,
                    image_provider=self.image_provider,
                    composer=self.composer,
                )

                result = await job.execute(context)
                post.slides.append(result.slide_content)

                await progress_manager.complete_slide(i + 1)

            # Step 4: Generate caption
            await progress_manager.update(current_step="Generating caption")

            content_summary = " | ".join([
                s.heading for s in post.slides
                if s.slide_type == SlideType.CONTENT
            ])

            post.caption = await self.planner.generate_caption(
                topic=topic,
                hook_text=plan.hook_text,
                content_summary=content_summary,
                hashtags=plan.keywords,
            )

            # Extract hashtags from config
            self._configure_hashtags(post)

            # Finalize
            post.generation_time_seconds = time.time() - start_time
            post.text_provider = self.text_provider.current_provider
            post.image_provider = self.image_provider.current_provider
            post.status = "generated"

            await progress_manager.complete()

            _logger.info(
                f"POST:{post_id} | COMPLETE | slides:{post.slides_count} | "
                f"time:{post.generation_time_seconds:.1f}s"
            )

            return post

        except Exception as e:
            await progress_manager.fail(str(e))
            raise

    def _configure_cta_image(self, plan: Any, topic: str) -> None:
        """Configure CTA slide - use simple black background, no image generation.

        Args:
            plan: Post plan with slides.
            topic: Post topic (unused, kept for compatibility).
        """
        # CTA slides use simple black background - no AI image generation needed
        for slide_outline in plan.slides:
            if slide_outline.get("slide_type") == "cta":
                slide_outline["needs_image"] = False
                slide_outline["image_description"] = None

    def _configure_hashtags(self, post: CarouselPost) -> None:
        """Configure hashtags for the post.

        Args:
            post: Post to configure hashtags for.
        """
        hashtag_config = self.profile_config.get("hashtag_strategy", {})
        hashtag_sets = hashtag_config.get("hashtag_sets", {})
        all_hashtags = []

        for category in ["primary", "secondary", "niche", "branded"]:
            all_hashtags.extend(hashtag_sets.get(category, []))

        post.hashtags = all_hashtags[:10]

    async def save_post(self, post: CarouselPost) -> Path:
        """Save a generated post to disk.

        Args:
            post: The post to save.

        Returns:
            Path to saved post directory.
        """
        return await self.output_service.save(post)

    def _get_output_path(self, post: CarouselPost) -> Path:
        """Get output path for a post (for CLI compatibility).

        Args:
            post: The post to get path for.

        Returns:
            Path to post output directory.
        """
        return self.output_service.get_output_path(post)

    async def _emit_progress(self, **kwargs) -> None:
        """Emit a progress event to the callback.

        Args:
            **kwargs: Fields to include in GenerationProgress.
        """
        if self.progress_callback:
            from .models import GenerationProgress
            progress = GenerationProgress(
                post_id=kwargs.get("post_id", "prep"),
                **{k: v for k, v in kwargs.items() if k != "post_id"}
            )
            await self.progress_callback(progress)

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

        # Emit Phase 0 start
        await self._emit_progress(
            current_phase=0,
            phase_name="Preparation",
            status="preparation",
            current_step="Loading post history...",
        )

        # Get content pillars from profile config
        content_pillars = [
            p.get("id")
            for p in self.profile_config.get("content_strategy", {}).get("content_pillars", [])
        ]

        # Get topics to avoid from recent post history
        await self._emit_progress(
            current_phase=0,
            phase_name="Preparation",
            current_step="Checking recent posts to avoid duplicates...",
        )
        avoid_topics = self.knowledge.get_topics_to_avoid(days=14)
        _logger.info(f"DAILY_POSTS | avoiding {len(avoid_topics)} recent topics")

        # Get topics if not provided
        if topics is None:
            # Try research first
            await self._emit_progress(
                current_phase=0,
                phase_name="Preparation",
                current_step="Researching trending topics...",
            )

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

                if topics:
                    await self._emit_progress(
                        current_phase=0,
                        phase_name="Preparation",
                        current_step=f"Found {len(topics)} trending topic(s)",
                        current_action="Research complete",
                    )
            except Exception as e:
                # Research failed, topics will be empty
                _logger.warning(f"DAILY_POSTS | research failed: {e}")
                topics = []

            # Fallback to AI-generated topics if research returned nothing
            if not topics:
                await self._emit_progress(
                    current_phase=0,
                    phase_name="Preparation",
                    current_step="AI generating topic ideas...",
                )
                topics = await self._generate_topics_with_ai(
                    count=count,
                    content_pillars=content_pillars,
                    avoid_topics=avoid_topics,
                )
                if topics:
                    await self._emit_progress(
                        current_phase=0,
                        phase_name="Preparation",
                        current_step=f"AI generated {len(topics)} topic(s)",
                        current_action="Topic generation complete",
                    )

        # Emit topic selection
        if topics:
            topic_preview = topics[0].get("topic", "")[:50]
            await self._emit_progress(
                current_phase=0,
                phase_name="Preparation",
                current_step=f"Selected topic: {topic_preview}...",
                current_action="Starting generation",
            )

        # Generate posts
        errors = []
        for i, topic_info in enumerate(topics[:count], 1):
            topic_str = topic_info.get("topic", "Unknown")
            _logger.info(f"DAILY_POSTS | generating post {i}/{count}: {topic_str[:60]}")

            try:
                post = await self.generate_post(
                    topic=topic_info["topic"],
                    content_pillar=topic_info.get("content_pillar", "general"),
                )

                # Save post
                await self.save_post(post)
                posts.append(post)

            except RuntimeError:
                # Propagate RuntimeError (validation failures) to caller
                raise
            except Exception as e:
                # Track errors for reporting
                error_msg = f"Failed to generate post for topic '{topic_info.get('topic', 'unknown')}': {e}"
                _logger.warning(error_msg)
                errors.append((topic_info.get("topic", "unknown"), str(e)))
                # If only 1 post requested and it failed, re-raise so user sees error
                if count == 1:
                    raise RuntimeError(error_msg) from e
                continue

        # If all posts failed, raise an error
        if not posts and errors:
            error_summary = "\n".join([f"  - {t}: {e}" for t, e in errors])
            raise RuntimeError(f"All {len(errors)} post(s) failed to generate:\n{error_summary}")

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
        target_audience = self.profile_config.get("profile", {}).get("target_audience", {})

        # Get content pillar details
        pillar_details = self.profile_config.get("content_strategy", {}).get("content_pillars", [])
        pillars_info = "\n".join([
            f"- {p.get('id')}: {p.get('description', p.get('name', ''))}"
            for p in pillar_details
        ]) if pillar_details else "general content"

        # Build avoid topics string (limit to prevent prompt overflow)
        avoid_str = ""
        if avoid_topics:
            recent_topics = avoid_topics[:20]
            avoid_str = f"\n\nTOPICS TO AVOID (recently posted):\n" + "\n".join(f"- {t}" for t in recent_topics)

        # Get recent posts context for variety
        recent_context = self.knowledge.get_recent_context(days=14)

        # Get current date for context
        from datetime import datetime
        now = datetime.now()
        date_context = (
            f"IMPORTANT - CURRENT DATE: {now.strftime('%B %d, %Y')} (Year {now.year}). "
            f"All topics must be relevant to {now.year}, NOT previous years like {now.year - 1}."
        )

        prompt = f"""Generate {count} unique and engaging Instagram carousel post topic(s) for a {niche} account.

{date_context}

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
- Topics should be trendy and relevant to {now.year} (current year!)
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
                temperature=0.9,
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
            import random
            pillar = random.choice(content_pillars) if content_pillars else "general"
            return [{
                "topic": f"Top tips for {niche.replace('-', ' ').replace('_', ' ')}",
                "content_pillar": pillar,
            }]
