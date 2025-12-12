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

# Set up file logging for AI calls (console output suppressed)
_log_dir = Path(__file__).parent.parent.parent.parent / "logs"
_log_dir.mkdir(exist_ok=True)
_ai_logger = logging.getLogger("ai_calls")
_ai_logger.setLevel(logging.DEBUG)
_ai_logger.propagate = False  # Don't propagate to root logger (no console output)
_ai_file_handler = logging.FileHandler(_log_dir / "ai_calls.log", encoding="utf-8")
_ai_file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
_ai_logger.addHandler(_ai_file_handler)


from ..providers.config import load_provider_config
from ..design import SlideComposer, HookSlideTemplate, ContentSlideTemplate, CTASlideTemplate
from ..knowledge import KnowledgeStore, PostRecord, PromptLog, PromptLogEntry
from .planner import ContentPlanner
from .models import CarouselPost, SlideContent, SlideType, PostPlan, HookType, GenerationProgress


# Logging helper functions for traceable execution flow
def _log_section(post_id: str, title: str) -> None:
    """Log a section separator for readability."""
    _ai_logger.info(f"POST:{post_id} | {'='*60}")
    _ai_logger.info(f"POST:{post_id} | {title}")
    _ai_logger.info(f"POST:{post_id} | {'='*60}")


def _log_phase_start(post_id: str, phase_num: int, phase_name: str, details: str = "") -> None:
    """Log the start of a generation phase."""
    _ai_logger.info(f"POST:{post_id} | PHASE_START | phase:{phase_num} | name:{phase_name} | {details}")


def _log_phase_end(post_id: str, phase_num: int, phase_name: str, duration: float, details: str = "") -> None:
    """Log the end of a generation phase."""
    _ai_logger.info(f"POST:{post_id} | PHASE_END | phase:{phase_num} | name:{phase_name} | duration:{duration:.2f}s | {details}")


def _log_tool_call(post_id: str, tool_name: str, args_preview: str, result_preview: str = "", success: bool = True, duration_ms: int = 0) -> None:
    """Log a tool call (AI-driven research)."""
    status = "SUCCESS" if success else "FAILED"
    _ai_logger.info(f"POST:{post_id} | TOOL_CALL | tool:{tool_name} | status:{status} | duration:{duration_ms}ms | args:{args_preview[:100]} | result:{result_preview[:100]}")


def _log_ai_call(post_id: str, provider: str, model: str, task: str, prompt_preview: str, response_preview: str = "", duration: float = 0, cost: float = 0) -> None:
    """Log an AI API call with full details."""
    _ai_logger.info(
        f"POST:{post_id} | AI_CALL | provider:{provider} | model:{model} | task:{task} | "
        f"duration:{duration:.2f}s | cost:${cost:.4f} | prompt:{prompt_preview[:150]}... | response:{response_preview[:100]}..."
    )


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
        auto_retry: bool = False,
        text_provider_override: str | None = None,
        image_provider_override: str | None = None,
        ai_tools: bool = False,
    ):
        """Initialize the content generator.

        Args:
            profile_path: Path to profile directory.
            profile_config: Profile configuration dict.
            text_provider: Text generation provider.
            image_provider: Image generation provider.
            progress_callback: Callback for progress updates.
            auto_retry: If True, retry indefinitely until valid content.
            text_provider_override: Override text provider (e.g., 'lmstudio', 'openai').
            image_provider_override: Override image provider (e.g., 'dalle', 'comfy').
            ai_tools: If True, use AI-driven tool calling for research (AI decides when to search).
        """
        self.auto_retry = auto_retry
        self.ai_tools = ai_tools
        self.profile_path = profile_path
        self.profile_config = profile_config or self._load_profile_config()
        self.progress_callback = progress_callback or self._default_progress

        # Stats tracking
        self._total_text_calls = 0
        self._total_image_calls = 0
        self._total_cost = 0.0

        # Create providers with event callbacks
        config = load_provider_config()

        # Store provider overrides
        self._text_provider_override = text_provider_override
        self._image_provider_override = image_provider_override

        async def text_event_callback(event: dict[str, Any]) -> None:
            await self._handle_ai_event(event, "text")

        async def image_event_callback(event: dict[str, Any]) -> None:
            await self._handle_ai_event(event, "image")

        # Create providers with overrides passed directly
        self.text_provider = text_provider or TextProvider(
            config,
            event_callback=text_event_callback,
            provider_override=text_provider_override,
        )
        self.image_provider = image_provider or ImageProvider(
            config,
            event_callback=image_event_callback,
            provider_override=image_provider_override,
        )

        self.planner = ContentPlanner(
            self.profile_config,
            self.text_provider,
            progress_callback=self.progress_callback,
            auto_retry=self.auto_retry,
        )
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

        # Log AI call to file with improved traceability
        post_id = self._current_post_id or "unknown"
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        duration = event.get("duration_seconds", 0)
        cost = event.get("cost_usd", 0)
        task = event.get("task", "unknown")
        failed_providers = event.get("failed_providers", [])

        if event_type == "text_call":
            prompt_preview = (event.get("prompt_preview", "") or "")[:150]
            has_tools = event.get("has_tools", False)
            tool_info = f" | tools:{event.get('tool_count', 0)}" if has_tools else ""
            failed_info = f" | failed_first:{','.join(failed_providers)}" if failed_providers else ""
            _ai_logger.info(
                f"POST:{post_id} | TEXT_CALL | provider:{provider} | model:{model} | task:{task}{tool_info}{failed_info} | prompt:{prompt_preview}..."
            )
        elif event_type == "text_response":
            prompt_preview = (event.get("prompt_preview", "") or "")[:100]
            response_preview = (event.get("response_preview", "") or "")[:100]
            tool_calls_count = event.get("tool_calls_count", 0)
            tool_info = f" | tool_calls:{tool_calls_count}" if tool_calls_count else ""
            _ai_logger.info(
                f"POST:{post_id} | TEXT_RESPONSE | provider:{provider} | model:{model} | task:{task} | "
                f"duration:{duration:.2f}s | cost:${cost:.4f}{tool_info} | response:{response_preview}..."
            )
        elif event_type == "image_response":
            prompt_preview = (event.get("prompt_preview", "") or "")[:100]
            _ai_logger.info(
                f"POST:{post_id} | IMAGE_RESPONSE | provider:{provider} | model:{model} | "
                f"task:{task} | duration:{duration:.2f}s | cost:${cost:.4f} | prompt:{prompt_preview}..."
            )
        elif event_type in ["text_error", "image_error"]:
            error = event.get("error", "unknown")
            failed_info = f" | fallback_from:{','.join(failed_providers)}" if failed_providers else ""
            _ai_logger.error(f"POST:{post_id} | {event_type.upper()} | provider:{provider} | error:{error}{failed_info}")

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

    async def _research_topic(self, topic: str) -> str | None:
        """Research a topic using web search.

        Args:
            topic: The topic to research.

        Returns:
            Research context string for AI, or None if search fails.
        """
        try:
            from ..research import WebSearcher
        except ImportError:
            _ai_logger.warning("Web search unavailable: ddgs not installed")
            # Emit skipped status
            if self._current_progress:
                progress = self._current_progress.model_copy(update={
                    "web_search_status": "skipped",
                })
                await self._emit_progress(progress)
            return None

        try:
            searcher = WebSearcher(timeout=10, max_results_per_query=5)

            # Generate search queries from topic
            queries = self._generate_search_queries(topic)

            # Emit searching status
            if self._current_progress:
                progress = self._current_progress.model_copy(update={
                    "web_search_status": "searching",
                    "web_search_queries": queries,
                    "current_action": f"Searching {len(queries)} queries...",
                })
                await self._emit_progress(progress)

            # Execute parallel search
            results = await searcher.parallel_search(queries, max_results=5)

            # Emit complete status with results
            if self._current_progress:
                progress = self._current_progress.model_copy(update={
                    "web_search_status": "complete",
                    "web_search_queries": queries,
                    "web_search_results": results.total_results,
                    "web_search_sources": len(results.all_sources),
                    "web_search_domains": results.unique_domains[:5],
                    "web_search_duration_ms": results.duration_ms,
                    "current_action": f"Found {len(results.all_sources)} unique sources",
                })
                await self._emit_progress(progress)

            if results.total_results == 0:
                _ai_logger.info(f"No search results for topic: {topic}")
                return None

            # Convert to context string for AI
            return results.to_context_string(max_sources=8)

        except Exception as e:
            _ai_logger.warning(f"Web search failed: {e}")
            # Emit failed status
            if self._current_progress:
                progress = self._current_progress.model_copy(update={
                    "web_search_status": "failed",
                    "current_action": "Search failed, continuing without research",
                })
                await self._emit_progress(progress)
            return None

    async def _research_topic_with_tools(self, topic: str) -> str | None:
        """Research a topic using AI-driven tool calling.

        The AI decides when and what to search for, like InfiniteResearch.

        Args:
            topic: The topic to research.

        Returns:
            Research context string from AI tool calls, or None if no research done.
        """
        try:
            from ..tools import TOOL_SCHEMAS, ToolExecutor
        except ImportError:
            _ai_logger.warning("Tools module unavailable")
            return await self._research_topic(topic)  # Fallback to standard research

        # Create tool executor with progress callback
        async def tool_callback(event: dict) -> None:
            if not self._current_progress:
                return

            event_type = event.get("type", "")

            if event_type == "tool_call_start":
                tool_name = event.get("tool_name", "")
                args = event.get("arguments", {})

                # Update progress with tool call info
                progress = self._current_progress.model_copy(update={
                    "tool_call_status": "executing",
                    "tool_call_name": tool_name,
                    "tool_call_args": args,
                    "current_action": f"Tool: {tool_name}({', '.join(f'{k}={v}' for k, v in list(args.items())[:2])}...)",
                })
                await self._emit_progress(progress)

            elif event_type == "tool_call_complete":
                tool_name = event.get("tool_name", "")
                metadata = event.get("metadata", {})
                success = event.get("success", False)

                # Add to history
                history = list(self._current_progress.tool_calls_history)
                history.append({
                    "tool": tool_name,
                    "success": success,
                    "duration_ms": event.get("duration_ms", 0),
                    "metadata": metadata,
                })

                progress = self._current_progress.model_copy(update={
                    "tool_call_status": "complete" if success else "failed",
                    "tool_calls_history": history,
                    "total_tool_calls": len(history),
                    "current_action": f"Tool {tool_name}: {metadata.get('total_results', 0)} results" if success else f"Tool {tool_name} failed",
                })
                await self._emit_progress(progress)

        executor = ToolExecutor(callback=tool_callback, post_id=self._current_post_id)

        # Build the AI prompt with instructions to use tools
        system_prompt = """You are a research assistant helping create engaging Instagram carousel content.
You have access to tools for web search and news search.

When given a topic, decide if you need to search for:
- Current facts, statistics, or examples
- Expert opinions or quotes
- Recent news or trends
- Best practices or tips

If the topic is factual or would benefit from current data, use the web_search tool.
If the topic relates to current events or trends, use the news_search tool.
You can use both tools if helpful.

After gathering research, synthesize the key findings into a concise summary that will help create accurate, engaging content."""

        user_prompt = f"""Research this topic for an Instagram carousel post:

TOPIC: {topic}

Decide what research would help create accurate, engaging content. Use the available tools to gather information, then provide a summary of key findings.

If the topic is simple or doesn't need research, you can skip tools and provide general guidance."""

        # Emit starting status
        if self._current_progress:
            progress = self._current_progress.model_copy(update={
                "web_search_status": "searching",
                "current_action": "AI deciding research strategy...",
            })
            await self._emit_progress(progress)

        _ai_logger.info(f"POST:{self._current_post_id} | AI_RESEARCH | topic:{topic}")

        # First AI call - let it decide what tools to use
        try:
            response = await self.text_provider.generate_with_tools(
                prompt=user_prompt,
                tools=TOOL_SCHEMAS,
                system=system_prompt,
                task="research",
                temperature=0.5,  # Lower temp for more focused research decisions
                max_tokens=1500,
            )
        except Exception as e:
            _ai_logger.warning(f"AI research failed: {e}, falling back to standard search")
            return await self._research_topic(topic)

        # Build conversation messages for tool loop
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Handle tool calls in a loop (AI might call multiple tools)
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        all_research = []

        while response.get("tool_calls") and iteration < max_iterations:
            iteration += 1
            tool_calls = response["tool_calls"]

            _ai_logger.info(f"POST:{self._current_post_id} | TOOL_CALLS | iteration:{iteration} | calls:{len(tool_calls)}")

            # Add assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": response.get("content"),
                "tool_calls": tool_calls,
            })

            # Execute tools and collect results
            results = await executor.execute_tool_calls(tool_calls)

            # Add tool results to messages
            for call, result in zip(tool_calls, results):
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result.to_message(),
                })

                if result.success:
                    all_research.append(result.result)

            # Continue conversation with tool results
            try:
                response = await self.text_provider.continue_with_tool_results(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    task="research",
                    temperature=0.5,
                    max_tokens=1500,
                )
            except Exception as e:
                _ai_logger.warning(f"AI continuation failed: {e}")
                break

        # Get final content (either from last response or collected research)
        final_content = response.get("content", "")

        # Combine AI summary with raw research data
        if all_research:
            research_context = "\n\n---\n\n".join(all_research)
            if final_content:
                research_context = f"{final_content}\n\n---\nRAW RESEARCH DATA:\n{research_context}"
        else:
            research_context = final_content if final_content else None

        # Update final status
        if self._current_progress:
            progress = self._current_progress.model_copy(update={
                "web_search_status": "complete" if research_context else "skipped",
                "current_action": f"Research complete ({executor.total_tool_calls} tool calls)",
            })
            await self._emit_progress(progress)

        _ai_logger.info(
            f"POST:{self._current_post_id} | AI_RESEARCH_COMPLETE | "
            f"tool_calls:{executor.total_tool_calls} | context_len:{len(research_context or '')}"
        )

        return research_context

    def _generate_search_queries(self, topic: str) -> list[str]:
        """Generate search queries from a topic.

        Args:
            topic: The content topic.

        Returns:
            List of search queries (3-5 variations).
        """
        queries = []

        # Main topic as-is
        queries.append(topic)

        # Extract numbers and key terms for variations
        # e.g., "5 AI tools for email" -> ["AI tools for email", "best AI email tools 2024"]
        import re
        # Remove leading numbers like "5 " or "10 "
        clean_topic = re.sub(r'^\d+\s+', '', topic)
        if clean_topic != topic:
            queries.append(clean_topic)

        # Add "best" variation
        queries.append(f"best {clean_topic} 2024")

        # Add "how to" variation if applicable
        if not topic.lower().startswith("how"):
            queries.append(f"how to {clean_topic}")

        # Limit to 5 queries
        return queries[:5]

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

        _log_section(post_id, f"NEW POST SESSION | topic: {topic[:80]}...")
        _ai_logger.info(f"POST:{post_id} | CONFIG | content_pillar:{content_pillar} | slides:target={target_slides}/min={min_slides}/max={max_slides} | ai_tools:{self.ai_tools}")

        progress = GenerationProgress(
            post_id=post_id,
            status="planning",
            total_steps=5,  # Initial estimate: Research + Planning + caption + save + buffer
        )
        await self._emit_progress(progress)

        # Step 0: Web research (if no context provided)
        if research_context is None:
            _log_phase_start(post_id, 0, "Research", f"ai_tools:{self.ai_tools}")
            research_start = time.time()

            progress.current_step = "Researching topic"
            progress.current_action = "Searching the web for relevant information..."
            await self._emit_progress(progress)

            # Use AI-driven tool calling if enabled
            if self.ai_tools:
                research_context = await self._research_topic_with_tools(topic)
            else:
                research_context = await self._research_topic(topic)

            research_duration = time.time() - research_start
            if research_context:
                _log_phase_end(post_id, 0, "Research", research_duration, f"context_length:{len(research_context)}")
            else:
                _log_phase_end(post_id, 0, "Research", research_duration, "no_context")

        # Step 1: Plan the post
        progress.current_step = "Planning content structure"
        progress.current_action = "Starting content generation..."
        await self._emit_progress(progress)

        # Set planner's progress reference so it can emit updates
        self.planner._current_progress = progress

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

        # Check if using local/free image provider - if so, add image to CTA
        local_image_providers = {"comfyui", "automatic1111", "stable-diffusion-webui"}
        is_local_image = (
            self._image_provider_override
            and self._image_provider_override.lower() in local_image_providers
        )

        if is_local_image:
            # Enable image generation for CTA slide (it's free with local providers)
            for slide_outline in plan.slides:
                if slide_outline.get("slide_type") == "cta":
                    slide_outline["needs_image"] = True
                    # Generate context-aware image description based on topic
                    slide_outline["image_description"] = (
                        f"Abstract background for social media CTA about {topic}, "
                        "inspiring, motivational, modern gradient colors, subtle patterns"
                    )

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

        _log_section(post_id, "POST GENERATION COMPLETE")
        _ai_logger.info(f"POST:{post_id} | SUMMARY | slides:{post.slides_count} | hook_type:{post.hook_type.value}")
        _ai_logger.info(f"POST:{post_id} | TIMING | total:{post.generation_time_seconds:.1f}s")
        _ai_logger.info(f"POST:{post_id} | COST | total:${post.total_cost_usd:.4f}")
        _ai_logger.info(f"POST:{post_id} | CALLS | text:{self._total_text_calls} | image:{self._total_image_calls}")
        _ai_logger.info(f"POST:{post_id} | PROVIDERS | text:{post.text_provider} | image:{post.image_provider}")
        _ai_logger.info(f"POST:{post_id} | OUTPUT | path:{self._get_output_path(post)}")
        _ai_logger.info(f"POST:{post_id} | {'='*60}")

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
                background_image=image_bytes,
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

            except RuntimeError:
                # Propagate RuntimeError (validation failures) to caller
                raise
            except Exception as e:
                # Continue with other posts on non-critical errors
                _ai_logger.warning(f"Failed to generate post for topic '{topic_info.get('topic', 'unknown')}': {e}")
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
