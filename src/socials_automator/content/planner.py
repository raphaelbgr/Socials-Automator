"""Content planner for creating post outlines."""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Any, Callable, Awaitable

from pydantic import BaseModel

from ..providers import TextProvider, get_text_provider
from ..knowledge import KnowledgeStore
from .models import PostPlan, HookType, SlideType, GenerationProgress

# Get logger for file-only logging (configured in cli.py)
_logger = logging.getLogger("ai_calls")

# Progress callback type
ProgressCallback = Callable[[GenerationProgress], Awaitable[None]]


class SlideOutline(BaseModel):
    """Outline for a single slide."""

    number: int
    slide_type: str
    heading: str
    body: str | None = None
    needs_image: bool = False
    image_description: str | None = None


class CarouselPlanResponse(BaseModel):
    """Structured response for carousel planning."""

    hook_text: str
    hook_subtext: str | None = None
    slides: list[SlideOutline]
    keywords: list[str]
    caption: str
    hashtags: list[str]


class ContentPlanner:
    """Plan carousel content before generation.

    Uses AI to:
    - Generate hook options
    - Plan slide structure
    - Create content outlines
    - Suggest keywords and hashtags

    Usage:
        planner = ContentPlanner(profile_config)

        # Plan a post
        plan = await planner.plan_post(
            topic="5 ChatGPT tricks for email",
            content_pillar="productivity_hacks",
        )

        # Generate hook options
        hooks = await planner.generate_hooks(topic, hook_type="number_benefit")
    """

    def __init__(
        self,
        profile_config: dict[str, Any],
        text_provider: TextProvider | None = None,
        knowledge_store: KnowledgeStore | None = None,
        progress_callback: ProgressCallback | None = None,
        auto_retry: bool = False,
    ):
        """Initialize the content planner.

        Args:
            profile_config: Profile configuration.
            text_provider: Text generation provider.
            knowledge_store: Knowledge store for context.
            progress_callback: Callback for progress updates.
            auto_retry: If True, retry indefinitely until valid content.
        """
        self.profile = profile_config
        self.text_provider = text_provider or get_text_provider()
        self.knowledge = knowledge_store
        self.progress_callback = progress_callback
        self.auto_retry = auto_retry

        # Current progress state (set by generator)
        self._current_progress: GenerationProgress | None = None

        # Extract config
        self.content_strategy = profile_config.get("content_strategy", {})
        self.hook_strategies = profile_config.get("hook_strategies", {})
        self.ai_config = profile_config.get("ai_generation", {})

    async def _emit_progress(
        self,
        action: str,
        attempt: int = 0,
        max_attempts: int = 6,
        error: str | None = None,
        phase: int = 0,
        phase_name: str = "",
        phase_input: str = "",
        phase_output: str = "",
        content_count: int = 0,
        content_type: str = "",
        slide_titles: list[str] | None = None,
        generated_slide: dict[str, str] | None = None,
    ) -> None:
        """Emit progress update for phase-based generation."""
        if self._current_progress and self.progress_callback:
            self._current_progress.current_action = action
            self._current_progress.validation_attempt = attempt
            self._current_progress.validation_max_attempts = max_attempts
            self._current_progress.validation_error = error

            # Phase tracking
            if phase > 0:
                self._current_progress.current_phase = phase
                self._current_progress.total_phases = max_attempts
            if phase_name:
                self._current_progress.phase_name = phase_name
            if phase_input:
                self._current_progress.phase_input = phase_input[:100]
            if phase_output:
                self._current_progress.phase_output = phase_output[:100]
            if content_count > 0:
                self._current_progress.content_count = content_count
            if content_type:
                self._current_progress.content_type = content_type
            if slide_titles:
                self._current_progress.slide_titles = slide_titles
            if generated_slide:
                self._current_progress.generated_slides.append(generated_slide)

            await self.progress_callback(self._current_progress)

    def _get_system_prompt(self) -> str:
        """Get the system prompt from profile config."""
        prompts = self.ai_config.get("prompts", {})
        return prompts.get("system_context", "You are a social media content creator.")

    def _get_hook_templates(self, hook_type: HookType) -> list[str]:
        """Get hook templates for a type."""
        primary_types = self.hook_strategies.get("primary_types", [])

        for ht in primary_types:
            if ht.get("type") == hook_type.value:
                return ht.get("templates", [])

        return []

    async def generate_hooks(
        self,
        topic: str,
        hook_type: HookType | None = None,
        count: int = 5,
    ) -> list[str]:
        """Generate hook options for a topic.

        Args:
            topic: Topic to create hooks for.
            hook_type: Type of hook to generate. Random if None.
            count: Number of hook options.

        Returns:
            List of hook text options.
        """
        if hook_type is None:
            hook_type = random.choice(list(HookType))

        templates = self._get_hook_templates(hook_type)
        templates_str = "\n".join(f"- {t}" for t in templates) if templates else ""

        prompt = f"""Generate {count} hook options for an Instagram carousel about: {topic}

Hook type: {hook_type.value}

{f"Template examples to inspire (don't copy exactly):{chr(10)}{templates_str}" if templates_str else ""}

Requirements:
- Maximum 12 words per hook
- Must create curiosity or promise value
- Use specific numbers when possible
- Avoid clickbait that doesn't deliver

Return ONLY a JSON array of strings with the hooks, no other text:
["Hook 1", "Hook 2", ...]"""

        response = await self.text_provider.generate(
            prompt=prompt,
            system=self._get_system_prompt(),
            task="hook_generation",
            temperature=0.9,
        )

        # Parse JSON response
        try:
            hooks = json.loads(response.strip())
            if isinstance(hooks, list):
                return hooks[:count]
        except json.JSONDecodeError:
            # Try to extract hooks from text
            lines = response.strip().split("\n")
            hooks = [
                line.strip().strip("-").strip("0123456789.").strip()
                for line in lines
                if line.strip() and not line.startswith("[")
            ]
            return hooks[:count]

        return [topic]  # Fallback

    async def _call_with_history(
        self,
        messages: list[dict[str, str]],
        task: str = "content_planning",
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Make an AI call with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            task: Task name for provider selection
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            The assistant's response text
        """
        # Build the full prompt from history for the text provider
        # The text provider expects a single prompt, so we format history into it

        # Extract system message if present
        system_msg = None
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                conversation.append(msg)

        # Format conversation into prompt
        if len(conversation) == 1:
            # Single message - just use it as prompt
            prompt = conversation[0]["content"]
        else:
            # Multiple messages - format as conversation
            prompt_parts = []
            for msg in conversation:
                if msg["role"] == "user":
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg["role"] == "assistant":
                    prompt_parts.append(f"Assistant: {msg['content']}")
            prompt_parts.append("Assistant:")  # Prompt for next response
            prompt = "\n\n".join(prompt_parts)

        response = await self.text_provider.generate(
            prompt=prompt,
            system=system_msg,
            task=task,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.strip()

    async def _phase1_planning(
        self,
        topic: str,
        research_context: str | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Phase 1: Analyze topic and determine content structure.

        Args:
            topic: The topic for the carousel post.
            research_context: Optional web research results to inform content.

        Returns:
            Tuple of (planning_result, message_history)
        """
        system = "You are a social media content strategist. Analyze topics and plan content structure. Return ONLY valid JSON."

        # Build prompt with optional research context
        research_section = ""
        if research_context:
            research_section = f"""
RESEARCH CONTEXT (use this to inform your planning):
{research_context}

Use the research above to ensure your content is accurate and current.
"""

        prompt = f"""Analyze this Instagram carousel topic and plan the content:

TOPIC: "{topic}"
{research_section}
Determine:
1. How many content items are promised/implied (look for numbers like "5 tips", "7 tools")
2. What type of content (tips, prompts, tools, steps, etc.)
3. A refined, engaging version of the topic

Return ONLY this JSON:
{{
    "content_count": <number of items to create>,
    "content_type": "<singular type: tip, prompt, tool, step, hack, etc.>",
    "refined_topic": "<polished version of the topic>",
    "target_audience": "<who this is for>"
}}

Examples:
- "5 ChatGPT prompts" → {{"content_count": 5, "content_type": "prompt", ...}}
- "Best AI tools for writers" → {{"content_count": 5, "content_type": "tool", ...}} (default to 5 if not specified)
- "3 ways to use Claude" → {{"content_count": 3, "content_type": "way", ...}}"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        response = await self._call_with_history(messages, temperature=0.5)
        messages.append({"role": "assistant", "content": response})

        data = self._extract_json(response)

        # Ensure we have required fields with defaults
        result = {
            "content_count": data.get("content_count", 5),
            "content_type": data.get("content_type", "tip"),
            "refined_topic": data.get("refined_topic", topic),
            "target_audience": data.get("target_audience", "social media users"),
        }

        _logger.info(f"Phase 1 - Planning: {result['content_count']} {result['content_type']}s")

        return result, messages

    async def _phase2_structure(
        self,
        planning: dict[str, Any],
        hook_type: HookType,
        history: list[dict[str, str]],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Phase 2: Create hook and slide structure.

        Returns:
            Tuple of (structure_result, updated_history)
        """
        content_count = planning["content_count"]
        content_type = planning["content_type"]

        prompt = f"""Based on the planning, create the carousel structure.

Create:
1. A compelling hook (max 10 words) using {hook_type.value} style
2. {content_count} slide titles - one for each {content_type}
3. A description for the hook slide image

Return ONLY this JSON:
{{
    "hook_text": "<catchy hook headline>",
    "hook_subtext": "<optional 5-word subtext or null>",
    "hook_image_description": "<description for background image>",
    "slide_titles": [
        "<title for {content_type} 1>",
        "<title for {content_type} 2>",
        ... ({content_count} titles total)
    ]
}}

IMPORTANT: Generate EXACTLY {content_count} slide titles."""

        history.append({"role": "user", "content": prompt})
        response = await self._call_with_history(history, temperature=0.7)
        history.append({"role": "assistant", "content": response})

        data = self._extract_json(response)

        # Validate slide_titles count
        titles = data.get("slide_titles", [])
        if len(titles) != content_count:
            _logger.warning(f"Phase 2 got {len(titles)} titles instead of {content_count}, adjusting...")
            # Pad or trim to exact count
            while len(titles) < content_count:
                titles.append(f"{content_type.title()} {len(titles) + 1}")
            titles = titles[:content_count]

        result = {
            "hook_text": data.get("hook_text", planning["refined_topic"]),
            "hook_subtext": data.get("hook_subtext"),
            "hook_image_description": data.get("hook_image_description", f"Abstract background for {planning['refined_topic']}"),
            "slide_titles": titles,
        }

        _logger.info(f"Phase 2 - Structure: hook='{result['hook_text'][:30]}...', {len(titles)} titles")

        return result, history

    async def _phase3_content_slide(
        self,
        slide_number: int,
        slide_title: str,
        content_type: str,
        history: list[dict[str, str]],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Phase 3: Generate content for a single slide.

        Returns:
            Tuple of (slide_content, updated_history)
        """
        prompt = f"""Generate content for slide {slide_number}: "{slide_title}"

This is a {content_type} slide. Provide:
- A clear, specific heading (can refine the title)
- Body text with 1-2 sentences of actionable detail

Return ONLY this JSON:
{{
    "heading": "<specific heading for this {content_type}>",
    "body": "<1-2 sentences with specific, actionable details>"
}}

Make it SPECIFIC and VALUABLE - not generic filler."""

        history.append({"role": "user", "content": prompt})
        response = await self._call_with_history(history, temperature=0.7, max_tokens=300)
        history.append({"role": "assistant", "content": response})

        data = self._extract_json(response)

        result = {
            "heading": data.get("heading", slide_title),
            "body": data.get("body", ""),
        }

        _logger.info(f"Phase 3 - Slide {slide_number}: '{result['heading'][:30]}...'")

        return result, history

    async def _phase4_cta(
        self,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Phase 4: Generate CTA based on all content.

        Returns:
            CTA slide dict
        """
        prompt = """Based on all the content we've created, generate a compelling call-to-action for the final slide.

Return ONLY this JSON:
{
    "cta_text": "<short punchy CTA, 2-5 words>",
    "cta_subtext": "<optional extra line or null>"
}

Make it encourage saving, sharing, or following."""

        history.append({"role": "user", "content": prompt})
        response = await self._call_with_history(history, temperature=0.8, max_tokens=100)

        try:
            data = self._extract_json(response)
            return {
                "cta_text": data.get("cta_text", "Follow for more!"),
                "cta_subtext": data.get("cta_subtext"),
            }
        except Exception:
            return {
                "cta_text": "Follow for more!",
                "cta_subtext": None,
            }

    async def plan_post(
        self,
        topic: str,
        content_pillar: str,
        hook_type: HookType | None = None,
        target_slides: int | None = None,
        min_slides: int = 3,
        max_slides: int = 10,
        research_context: str | None = None,
    ) -> PostPlan:
        """Create a full plan for a carousel post using 4-phase AI generation.

        This approach guarantees correct slide counts by:
        1. Phase 1: Planning - extract count and type from topic
        2. Phase 2: Structure - create hook and slide titles
        3. Phase 3: Content - generate each slide individually (N calls)
        4. Phase 4: CTA - create call-to-action with full context

        Each phase builds on the history of previous phases for coherence.

        Args:
            topic: Topic for the post.
            content_pillar: Content pillar category.
            hook_type: Hook type to use.
            target_slides: Target number of slides (can override extracted count).
            min_slides: Minimum slides (default 3).
            max_slides: Maximum slides (default 10).
            research_context: Optional web research context to inform content.

        Returns:
            Complete post plan.
        """
        if hook_type is None:
            hook_type = self._select_hook_type(content_pillar)

        total_phases = 4  # Will be updated after phase 1

        # === PHASE 1: Planning ===
        await self._emit_progress(
            action="Analyzing topic...",
            phase=1,
            phase_name="Planning",
            phase_input=f'"{topic}"',
            max_attempts=total_phases,
        )

        planning, history = await self._phase1_planning(topic, research_context)
        content_count = planning["content_count"]
        content_type = planning["content_type"]

        # Override count if specified
        if target_slides is not None:
            content_count = max(1, target_slides - 2)  # Subtract hook and CTA
            planning["content_count"] = content_count

        # Clamp to min/max
        content_count = max(min_slides - 2, min(max_slides - 2, content_count))
        planning["content_count"] = content_count

        total_phases = 3 + content_count  # Phase 1, 2, N content slides, CTA

        # Emit Phase 1 result
        await self._emit_progress(
            action="Planning complete",
            phase=1,
            phase_name="Planning",
            phase_output=f'{content_count} {content_type}s identified',
            content_count=content_count,
            content_type=content_type,
            max_attempts=total_phases,
        )

        # === PHASE 2: Structure ===
        await self._emit_progress(
            action="Creating hook and structure...",
            phase=2,
            phase_name="Structure",
            phase_input=f'{content_count} {content_type}s + {hook_type.value} hook',
            max_attempts=total_phases,
        )

        structure, history = await self._phase2_structure(planning, hook_type, history)

        # Build hook slide
        hook_slide = {
            "number": 1,
            "slide_type": "hook",
            "heading": structure["hook_text"],
            "body": structure.get("hook_subtext"),
            "needs_image": True,
            "image_description": structure.get("hook_image_description", f"Abstract background for {topic}"),
        }

        # Emit Phase 2 result
        await self._emit_progress(
            action="Structure complete",
            phase=2,
            phase_name="Structure",
            phase_output=f'Hook: "{structure["hook_text"][:40]}..."',
            slide_titles=structure["slide_titles"],
            max_attempts=total_phases,
        )

        # === PHASE 3: Content Slides (one call per slide) ===
        content_slides = []
        for i, title in enumerate(structure["slide_titles"]):
            slide_num = i + 1

            await self._emit_progress(
                action=f"Generating {content_type} {slide_num}/{content_count}...",
                phase=3,
                phase_name="Content",
                phase_input=f'Slide {slide_num}: "{title[:40]}"',
                max_attempts=total_phases,
            )

            slide_content, history = await self._phase3_content_slide(
                slide_number=slide_num,
                slide_title=title,
                content_type=content_type,
                history=history,
            )

            content_slides.append({
                "number": slide_num + 1,  # +1 because hook is slide 1
                "slide_type": "content",
                "heading": slide_content["heading"],
                "body": slide_content["body"],
                "needs_image": False,
                "image_description": None,
            })

            # Emit slide completion
            await self._emit_progress(
                action=f"{content_type.title()} {slide_num} complete",
                phase=3,
                phase_name="Content",
                phase_output=f'"{slide_content["heading"][:40]}..."',
                generated_slide={"heading": slide_content["heading"], "body": slide_content["body"][:50] + "..."},
                max_attempts=total_phases,
            )

        # === PHASE 4: CTA ===
        await self._emit_progress(
            action="Creating call-to-action...",
            phase=4,
            phase_name="CTA",
            phase_input="Full conversation history",
            max_attempts=total_phases,
        )

        cta_data = await self._phase4_cta(history)
        cta_slide = {
            "number": len(content_slides) + 2,
            "slide_type": "cta",
            "heading": cta_data["cta_text"],
            "body": cta_data.get("cta_subtext"),
            "needs_image": False,
            "image_description": None,
        }

        # Combine all slides
        all_slides = [hook_slide] + content_slides + [cta_slide]

        # Generate keywords
        keywords = self._extract_keywords(topic, structure["hook_text"])

        await self._emit_progress(
            action="Generation complete!",
            phase=4,
            phase_name="Complete",
            phase_output=f'{len(all_slides)} slides ready',
            max_attempts=total_phases,
        )

        _logger.info(f"Successfully generated {len(all_slides)} slides: 1 hook + {len(content_slides)} content + 1 CTA")

        return PostPlan(
            topic=topic,
            content_pillar=content_pillar,
            hook_type=hook_type,
            hook_text=structure["hook_text"],
            hook_subtext=structure.get("hook_subtext"),
            target_slides=len(all_slides),
            slides=all_slides,
            keywords=keywords,
        )

    def _extract_keywords(self, topic: str, hook_text: str) -> list[str]:
        """Extract keywords from topic and hook text."""
        # Simple keyword extraction - split on spaces and filter
        words = (topic + " " + hook_text).lower().split()
        stop_words = {"the", "a", "an", "is", "are", "for", "to", "and", "or", "of", "in", "on", "with", "your", "you", "how", "what", "why", "when"}
        keywords = [w.strip(".,!?") for w in words if len(w) > 3 and w not in stop_words]
        return list(dict.fromkeys(keywords))[:10]  # Unique, max 10

    def _extract_json(self, response: str) -> dict[str, Any]:
        """Extract JSON from AI response, handling various formats.

        Args:
            response: Raw AI response text.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If no valid JSON found.
        """
        import re

        text = response.strip()

        # Try 1: Direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try 2: Extract from markdown code blocks
        code_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Try 3: Find JSON object by looking for { ... }
        brace_start = text.find('{')
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[brace_start:], brace_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i+1])
                        except json.JSONDecodeError:
                            break

        raise ValueError("No valid JSON found in response")

    def _create_default_slides(
        self,
        topic: str,
        hook_type: HookType,
        target_slides: int,
    ) -> list[dict[str, Any]]:
        """Create default slide structure when parsing fails.

        Args:
            topic: Post topic.
            hook_type: Type of hook.
            target_slides: Target number of slides.

        Returns:
            List of slide outline dicts.
        """
        slides = []

        # Slide 1: Hook
        slides.append({
            "number": 1,
            "slide_type": "hook",
            "heading": topic,
            "body": None,
            "needs_image": True,
            "image_description": f"Abstract tech background for topic: {topic}",
        })

        # Content slides
        content_count = max(1, target_slides - 2)
        for i in range(content_count):
            slides.append({
                "number": i + 2,
                "slide_type": "content",
                "heading": f"Point {i + 1}",
                "body": "Content to be generated",
                "needs_image": False,
                "image_description": None,
            })

        # CTA slide
        slides.append({
            "number": len(slides) + 1,
            "slide_type": "cta",
            "heading": "Follow for more tips!",
            "body": None,
            "needs_image": False,
            "image_description": None,
        })

        return slides

    def _select_hook_type(self, content_pillar: str) -> HookType:
        """Select appropriate hook type for a content pillar."""
        pillar_hook_map = {
            "tool_tutorials": HookType.CURIOSITY_GAP,
            "productivity_hacks": HookType.NUMBER_BENEFIT,
            "tool_comparisons": HookType.BOLD_STATEMENT,
            "ai_news_simplified": HookType.QUESTION,
            "prompt_templates": HookType.NUMBER_BENEFIT,
        }

        # Get from map or use weighted random
        if content_pillar in pillar_hook_map:
            return pillar_hook_map[content_pillar]

        # Weighted selection based on config
        weights = {}
        for ht in self.hook_strategies.get("primary_types", []):
            hook_type_str = ht.get("type")
            frequency = ht.get("frequency_percent", 20)
            weights[hook_type_str] = frequency

        if weights:
            types = list(weights.keys())
            probs = list(weights.values())
            selected = random.choices(types, weights=probs, k=1)[0]
            return HookType(selected)

        return random.choice(list(HookType))

    async def generate_caption(
        self,
        topic: str,
        hook_text: str,
        content_summary: str,
        hashtags: list[str] | None = None,
    ) -> str:
        """Generate Instagram caption for a post.

        Args:
            topic: Post topic.
            hook_text: The hook used.
            content_summary: Summary of post content.
            hashtags: Hashtags to include.

        Returns:
            Complete caption text.
        """
        hashtag_config = self.profile.get("hashtag_strategy", {})
        hashtag_sets = hashtag_config.get("hashtag_sets", {})

        if hashtags is None:
            # Select hashtags from config
            hashtags = []
            for category in ["primary", "secondary", "niche", "branded"]:
                tags = hashtag_sets.get(category, [])
                count = hashtag_config.get("distribution", {}).get(category.replace("_volume", ""), 5)
                hashtags.extend(random.sample(tags, min(count, len(tags))))

        hashtags_str = " ".join(hashtags[:20])  # Instagram limit

        prompt = f"""Write a SHORT Instagram caption for a carousel post.

Topic: {topic}
Hook: {hook_text}

Requirements:
- MUST be under 250 characters (very important for Threads compatibility)
- 1-2 short sentences maximum
- End with brief CTA (save/share/follow)
- 1-2 emojis max
- No hashtags (added separately)

Return ONLY the caption text, nothing else."""

        caption = await self.text_provider.generate(
            prompt=prompt,
            system=self._get_system_prompt(),
            task="content_planning",
            temperature=0.7,
        )

        # Truncate caption if too long (for Threads 500 char limit)
        caption = caption.strip()
        if len(caption) > 280:
            # Cut at last sentence boundary
            caption = caption[:280].rsplit('.', 1)[0] + '.'

        # Add hashtags (limited to ~200 chars for Threads)
        hashtags_limited = hashtags[:10]  # Limit hashtags for shorter total
        hashtags_str = " ".join(hashtags_limited)

        full_caption = f"{caption}\n\n{hashtags_str}"

        return full_caption
