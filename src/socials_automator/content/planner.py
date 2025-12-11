"""Content planner for creating post outlines."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from pydantic import BaseModel

from ..providers import TextProvider, get_text_provider
from ..knowledge import KnowledgeStore
from .models import PostPlan, HookType, SlideType


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
    ):
        """Initialize the content planner.

        Args:
            profile_config: Profile configuration.
            text_provider: Text generation provider.
            knowledge_store: Knowledge store for context.
        """
        self.profile = profile_config
        self.text_provider = text_provider or get_text_provider()
        self.knowledge = knowledge_store

        # Extract config
        self.content_strategy = profile_config.get("content_strategy", {})
        self.hook_strategies = profile_config.get("hook_strategies", {})
        self.ai_config = profile_config.get("ai_generation", {})

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
        """Create a full plan for a carousel post.

        Args:
            topic: Topic for the post.
            content_pillar: Content pillar category.
            hook_type: Hook type to use.
            target_slides: Target number of slides. If None, AI decides based on content.
            min_slides: Minimum slides when AI decides (default 3).
            max_slides: Maximum slides when AI decides (default 10).
            research_context: Optional research context.

        Returns:
            Complete post plan.
        """
        if hook_type is None:
            hook_type = self._select_hook_type(content_pillar)

        # Get context from knowledge base
        kb_context = ""
        if self.knowledge:
            kb_context = self.knowledge.get_recent_context(days=7)
            topics_to_avoid = self.knowledge.get_topics_to_avoid(days=7)
            if topics_to_avoid:
                kb_context += f"\n\nTopics to avoid (recently used): {', '.join(topics_to_avoid[:10])}"

        # Build slide count instruction
        if target_slides is not None:
            slides_instruction = f"Target slides: {target_slides} (including hook and CTA)"
            content_slides_instruction = f"2. {target_slides - 2} content slides with numbered points or steps"
        else:
            slides_instruction = f"Slide count: YOU DECIDE the optimal number of slides (minimum {min_slides}, maximum {max_slides}) based on how much valuable content the topic needs. Don't pad with filler - each slide must add real value."
            content_slides_instruction = f"2. As many content slides as needed to properly cover the topic ({min_slides - 2} to {max_slides - 2} content slides)"

        # Build planning prompt
        prompt = f"""Plan an Instagram carousel post about: {topic}

Content pillar: {content_pillar}
{slides_instruction}
Hook style: {hook_type.value}

{f"Research context:{chr(10)}{research_context}" if research_context else ""}

{f"Previous posts context:{chr(10)}{kb_context}" if kb_context else ""}

CAROUSEL STRUCTURE (MUST FOLLOW):
- Slide 1: HOOK slide with eye-catching image (this is the thumbnail people see)
- Slides 2 to N-1: CONTENT slides (1-8 content slides with the actual tips/prompts/tools)
- Slide N: CTA slide (call-to-action, always last)

Create a detailed plan with:
1. A compelling hook (first slide) - max 12 words, with background image
{content_slides_instruction}
3. A CTA slide (last slide) - "Follow for more..." or similar

CRITICAL REQUIREMENTS:
- If the hook mentions a NUMBER (e.g., "5 prompts", "7 tips"), you MUST have EXACTLY that many CONTENT slides
- Example: "5 AI Tools" = 1 hook + 5 content slides + 1 CTA = 7 total slides
- Example: "3 Quick Tips" = 1 hook + 3 content slides + 1 CTA = 5 total slides
- Each content slide MUST contain ACTUAL, SPECIFIC content that matches the topic
- If the topic is about "prompts", each content slide MUST contain an actual usable prompt
- If the topic is about "tips", each content slide MUST contain a specific, actionable tip
- If the topic is about "tools", each content slide MUST name and describe a specific tool
- NEVER use generic placeholders like "Point 1", "Main tip", "First tool", etc.
- The body text must provide real value - specific examples, explanations, or instructions

Each content slide should have:
- A clear heading (the actual tip/prompt/tool name - NOT "Point 1" or generic text)
- Supporting body text (1-2 sentences with specific details)
- Whether it needs a background image

Return as JSON:
{{
    "hook_text": "The hook text",
    "hook_subtext": "Optional subtext for hook slide",
    "slides": [
        {{"number": 1, "slide_type": "hook", "heading": "...", "body": null, "needs_image": true, "image_description": "..."}},
        {{"number": 2, "slide_type": "content", "heading": "ACTUAL TIP/PROMPT/ITEM HERE", "body": "Specific explanation with real details...", "needs_image": false, "image_description": null}},
        ...
        {{"number": N, "slide_type": "cta", "heading": "Follow for more", "body": null, "needs_image": false, "image_description": null}}
    ],
    "keywords": ["keyword1", "keyword2", ...],
    "caption": "The Instagram caption text",
    "hashtags": ["#hashtag1", "#hashtag2", ...]
}}"""

        # Try up to 4 times with different prompts
        max_attempts = 4
        last_error = None

        for attempt in range(max_attempts):
            if attempt == 0:
                # First attempt - use full prompt
                current_prompt = prompt
                system_prompt = self._get_system_prompt()
                temp = 0.7
            elif attempt == 1:
                # Second attempt - same prompt, lower temperature
                current_prompt = prompt
                system_prompt = self._get_system_prompt()
                temp = 0.5
            else:
                # Later attempts - use simpler, more explicit prompt
                current_prompt = self._get_simple_planning_prompt(
                    topic, content_pillar, hook_type, target_slides or 6
                )
                system_prompt = "You are a JSON generator. Return ONLY valid JSON, no other text."
                temp = 0.3

            response = await self.text_provider.generate(
                prompt=current_prompt,
                system=system_prompt,
                task="content_planning",
                temperature=temp,
                max_tokens=2000,
            )

            # Parse response
            try:
                data = self._extract_json(response)

                slides = data.get("slides", [])
                if slides and len(slides) >= 3:
                    # Validate that content slides have meaningful content
                    content_slides = [s for s in slides if s.get("slide_type") == "content"]
                    has_placeholder = any(
                        s.get("heading", "").startswith("Point ") or
                        s.get("body") == "Content to be generated" or
                        s.get("heading", "").lower() in ["first main point", "second main point", "third main point"]
                        for s in content_slides
                    )

                    if has_placeholder:
                        import logging
                        logging.warning(f"Attempt {attempt + 1}: Content has placeholders, retrying...")
                        last_error = "Content slides contain placeholder text instead of actual content"
                        continue

                    # Validate that if topic mentions a number, we have that many content slides
                    hook_text = data.get("hook_text", topic)
                    # Match patterns like "5 tools", "5 AI tools", "5 ChatGPT prompts", "7 quick tips", etc.
                    number_match = re.search(r'\b(\d+)\s+(?:\w+\s+)?(?:prompts?|tips?|tricks?|tools?|ways?|steps?|hacks?|ideas?|methods?|examples?|templates?|things?|apps?|features?|secrets?|reasons?)\b', hook_text, re.IGNORECASE)
                    if number_match:
                        expected_items = int(number_match.group(1))
                        actual_items = len(content_slides)
                        if actual_items != expected_items:
                            import logging
                            logging.warning(f"Attempt {attempt + 1}: Hook says '{expected_items}' items but got {actual_items} content slides, retrying...")
                            last_error = f"Hook promises {expected_items} items but only {actual_items} content slides were generated"
                            continue

                    return PostPlan(
                        topic=topic,
                        content_pillar=content_pillar,
                        hook_type=hook_type,
                        hook_text=data.get("hook_text", topic),
                        hook_subtext=data.get("hook_subtext"),
                        target_slides=len(slides) if slides else (target_slides or 6),
                        slides=slides,
                        keywords=data.get("keywords", []),
                    )
                else:
                    # Empty or too few slides - retry
                    import logging
                    logging.warning(f"Attempt {attempt + 1}: Got {len(slides)} slides, retrying...")
                    last_error = f"Only got {len(slides)} slides, need at least 3"
                    continue

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Log error for debugging
                import logging
                logging.warning(f"Attempt {attempt + 1}: Failed to parse AI response: {e}")
                if response:
                    logging.debug(f"Raw response (first 500 chars): {response[:500]}")
                last_error = str(e)
                continue

        # All retries failed - raise an error instead of using placeholders
        import logging
        logging.error(f"All {max_attempts} parsing attempts failed for topic: {topic}")
        raise RuntimeError(
            f"Failed to generate content plan after {max_attempts} attempts. "
            f"Last error: {last_error}. "
            f"Topic: {topic}"
        )

    def _get_simple_planning_prompt(
        self,
        topic: str,
        content_pillar: str,
        hook_type: HookType,
        target_slides: int,
    ) -> str:
        """Generate a simpler prompt for retry attempts."""
        # Determine content type from topic
        content_type = "tips"
        if "prompt" in topic.lower():
            content_type = "prompts"
        elif "tool" in topic.lower():
            content_type = "tools"
        elif "trick" in topic.lower():
            content_type = "tricks"
        elif "hack" in topic.lower():
            content_type = "hacks"
        elif "way" in topic.lower():
            content_type = "ways"
        elif "step" in topic.lower():
            content_type = "steps"

        return f"""Create a {target_slides}-slide Instagram carousel about "{topic}".

CRITICAL: Each content slide MUST contain REAL, SPECIFIC {content_type} - NOT placeholders!
- If topic says "5 {content_type}", include exactly 5 specific {content_type}
- Each heading must be the actual {content_type[:-1] if content_type.endswith('s') else content_type}, not "First point" or "Main tip"
- Body text must explain the specific {content_type[:-1] if content_type.endswith('s') else content_type} with real details

Return this exact JSON structure with REAL CONTENT:
```json
{{
  "hook_text": "Catchy hook headline (max 10 words)",
  "slides": [
    {{"number": 1, "slide_type": "hook", "heading": "Hook headline matching topic", "body": null, "needs_image": true, "image_description": "Description for hook image"}},
    {{"number": 2, "slide_type": "content", "heading": "ACTUAL SPECIFIC {content_type.upper()[:-1]} #1", "body": "Real explanation with specific details.", "needs_image": false, "image_description": null}},
    {{"number": 3, "slide_type": "content", "heading": "ACTUAL SPECIFIC {content_type.upper()[:-1]} #2", "body": "Real explanation with specific details.", "needs_image": false, "image_description": null}},
    {{"number": 4, "slide_type": "content", "heading": "ACTUAL SPECIFIC {content_type.upper()[:-1]} #3", "body": "Real explanation with specific details.", "needs_image": false, "image_description": null}},
    {{"number": {target_slides}, "slide_type": "cta", "heading": "Follow for more!", "body": null, "needs_image": false, "image_description": null}}
  ],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}
```

IMPORTANT: Return ONLY the JSON with REAL CONTENT, no explanation before or after."""

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
