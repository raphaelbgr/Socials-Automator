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
from ..services import StructuredExtractor
from .models import PostPlan, HookType, SlideType, GenerationProgress
from .responses import (
    PlanningResponse,
    StructureResponse,
    ContentSlideResponse,
    CTAResponse,
    HookListResponse,
    ContentValidationResponse,
)

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
        extractor: StructuredExtractor | None = None,
    ):
        """Initialize the content planner.

        Args:
            profile_config: Profile configuration.
            text_provider: Text generation provider.
            knowledge_store: Knowledge store for context.
            progress_callback: Callback for progress updates.
            auto_retry: If True, retry indefinitely until valid content.
            extractor: Structured data extractor using Instructor.
        """
        self.profile = profile_config
        self.text_provider = text_provider or get_text_provider()
        self.knowledge = knowledge_store
        self.progress_callback = progress_callback
        self.auto_retry = auto_retry

        # Create extractor with same event callback as text provider
        self.extractor = extractor or StructuredExtractor(
            event_callback=self.text_provider._event_callback,
            provider_override=self.text_provider._provider_override,
        )

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
        current_slide: int = 0,
        total_slides: int = 0,
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
            if current_slide > 0:
                self._current_progress.current_slide = current_slide
            if total_slides > 0:
                self._current_progress.total_slides = total_slides

            await self.progress_callback(self._current_progress)

    def _get_system_prompt(self) -> str:
        """Get the system prompt from profile config with current datetime context."""
        from datetime import datetime

        prompts = self.ai_config.get("prompts", {})
        base_prompt = prompts.get("system_context", "You are a social media content creator.")

        # Add current datetime context
        now = datetime.now()
        datetime_context = (
            f"\n\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} "
            f"({now.strftime('%A, %B %d, %Y')})"
        )

        return base_prompt + datetime_context

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
        system = "You are a social media content strategist. Analyze topics and plan content structure."

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

Examples:
- "5 ChatGPT prompts" -> content_count: 5, content_type: "prompt"
- "Best AI tools for writers" -> content_count: 5, content_type: "tool" (default to 5 if not specified)
- "3 ways to use Claude" -> content_count: 3, content_type: "way" """

        # Use Instructor extractor for structured extraction
        planning, history = await self.extractor.extract_with_history(
            prompt=prompt,
            response_model=PlanningResponse,
            system=system,
            task="phase1_planning",
            temperature=0.5,
        )

        result = {
            "content_count": planning.content_count,
            "content_type": planning.content_type,
            "refined_topic": planning.refined_topic,
            "target_audience": planning.target_audience,
        }

        _logger.info(f"Phase 1 - Planning: {result['content_count']} {result['content_type']}s")

        return result, history

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

IMPORTANT: Generate EXACTLY {content_count} slide titles."""

        # Use Instructor extractor for structured extraction
        structure, history = await self.extractor.extract_with_history(
            prompt=prompt,
            response_model=StructureResponse,
            history=history,
            task="phase2_structure",
            temperature=0.7,
        )

        # Validate slide_titles count
        titles = list(structure.slide_titles)
        if len(titles) != content_count:
            _logger.warning(f"Phase 2 got {len(titles)} titles instead of {content_count}, adjusting...")
            while len(titles) < content_count:
                titles.append(f"{content_type.title()} {len(titles) + 1}")
            titles = titles[:content_count]

        result = {
            "hook_text": structure.hook_text or planning["refined_topic"],
            "hook_subtext": structure.hook_subtext,
            "hook_image_description": structure.hook_image_description or f"Abstract background for {planning['refined_topic']}",
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
        topic: str = "",
        max_validation_retries: int = 2,
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        """Phase 3: Generate content for a single slide with validation.

        Generates content and validates it. If validation fails, regenerates
        with feedback about what was wrong.

        Returns:
            Tuple of (slide_content, updated_history)
        """
        # Determine if this content type requires concrete examples
        # Each entry has specific instructions AND a practical example of good output
        example_types = {
            "prompt": '''Include an ACTUAL, COPY-PASTE READY prompt.
GOOD EXAMPLE:
- Heading: "Weekly Planning Prompt"
- Body: Try: "Act as my productivity coach. Review my goals: [paste goals]. Create a prioritized weekly plan with 3 daily focuses."

BAD EXAMPLE (too vague):
- Body: "Use ChatGPT to help plan your week by asking it questions."''',

            "prompts": '''Include an ACTUAL, COPY-PASTE READY prompt.
GOOD EXAMPLE:
- Heading: "Email Response Prompt"
- Body: Use this: "Rewrite this email to be professional but friendly. Keep it under 100 words: [paste email]"

BAD EXAMPLE (too vague):
- Body: "ChatGPT can help you write better emails quickly."''',

            "tip": '''Include a SPECIFIC, ACTIONABLE tip with example.
GOOD EXAMPLE:
- Heading: "Use Voice Input"
- Body: Open ChatGPT on mobile, tap the mic icon, and speak your prompt. 3x faster than typing!

BAD EXAMPLE (too vague):
- Body: "Using voice features can help you be more productive with AI."''',

            "tips": '''Include a SPECIFIC, ACTIONABLE tip with example.
GOOD EXAMPLE:
- Heading: "Chain Your Prompts"
- Body: Start with "You are an expert in X" then follow up with "Now apply that to Y". Builds better context!

BAD EXAMPLE (too vague):
- Body: "Learning to write good prompts will improve your results."''',

            "tool": '''Include the ACTUAL TOOL NAME and what it does.
GOOD EXAMPLE:
- Heading: "Perplexity AI"
- Body: Real-time web search + AI answers. Ask "latest iPhone specs" and get sourced, up-to-date info instantly.

BAD EXAMPLE (too vague):
- Body: "There are many AI search tools that can help find information."''',

            "tools": '''Include the ACTUAL TOOL NAME and what it does.
GOOD EXAMPLE:
- Heading: "Claude for Long Docs"
- Body: Upload PDFs up to 100k tokens. Ask "summarize key points" or "find contradictions in this contract."

BAD EXAMPLE (too vague):
- Body: "AI tools can help you analyze documents faster."''',

            "step": '''Include the SPECIFIC ACTION for this step.
GOOD EXAMPLE:
- Heading: "Step 2: Set Your Role"
- Body: Start your prompt with "You are a [specific expert]." Example: "You are a senior Python developer specializing in APIs."

BAD EXAMPLE (too vague):
- Body: "Define what you want the AI to do in this step."''',

            "steps": '''Include the SPECIFIC ACTION for this step.
GOOD EXAMPLE:
- Heading: "Step 3: Add Context"
- Body: Paste relevant background. Example: "Here's my current code: [code]. I need to add error handling for API timeouts."

BAD EXAMPLE (too vague):
- Body: "Provide the AI with the information it needs."''',

            "trick": '''Include the ACTUAL TRICK with a concrete example.
GOOD EXAMPLE:
- Heading: "The Persona Trick"
- Body: Add "Explain like I'm 5" or "Explain like I'm a CEO" to any prompt. Same question, perfect detail level!

BAD EXAMPLE (too vague):
- Body: "Adjusting your prompts can give you better results."''',

            "tricks": '''Include the ACTUAL TRICK with a concrete example.
GOOD EXAMPLE:
- Heading: "The Format Trick"
- Body: End prompts with "Format as: bullet points / table / numbered list". AI follows formatting instructions perfectly.

BAD EXAMPLE (too vague):
- Body: "You can ask AI to format responses differently."''',

            "hack": '''Include the SPECIFIC HACK and how to use it.
GOOD EXAMPLE:
- Heading: "Free GPT-4 Hack"
- Body: Use Bing Chat (copilot.microsoft.com) - it's GPT-4 powered and 100% free. No subscription needed!

BAD EXAMPLE (too vague):
- Body: "There are ways to access AI tools without paying."''',

            "hacks": '''Include the SPECIFIC HACK and how to use it.
GOOD EXAMPLE:
- Heading: "Context Window Hack"
- Body: Hit the limit? Say "Continue from [last sentence]" in a new chat. Paste key context to keep going.

BAD EXAMPLE (too vague):
- Body: "Managing conversation length is important for AI tools."''',

            "way": '''Explain the SPECIFIC METHOD with example.
GOOD EXAMPLE:
- Heading: "Automate Email Replies"
- Body: Connect ChatGPT to Zapier. When emails arrive, auto-draft responses. Review and send in 1 click!

BAD EXAMPLE (too vague):
- Body: "AI can help automate your email workflow."''',

            "ways": '''Explain the SPECIFIC METHOD with example.
GOOD EXAMPLE:
- Heading: "Generate Social Posts"
- Body: Prompt: "Create 5 Twitter posts about [topic]. Include hooks, keep under 280 chars, add relevant hashtags."

BAD EXAMPLE (too vague):
- Body: "You can use AI to create content for social media."''',

            "example": '''Include a REAL, CONCRETE example.
GOOD EXAMPLE:
- Heading: "Meeting Summary"
- Body: Paste transcript + "Extract: 1) Key decisions 2) Action items with owners 3) Follow-up dates"

BAD EXAMPLE (too vague):
- Body: "AI can help summarize your meetings."''',

            "examples": '''Include a REAL, CONCRETE example.
GOOD EXAMPLE:
- Heading: "Code Review Prompt"
- Body: "Review this code for: security issues, performance problems, and style. Suggest fixes: [paste code]"

BAD EXAMPLE (too vague):
- Body: "You can use AI to help review your code."''',

            "template": '''Include an ACTUAL TEMPLATE to copy.
GOOD EXAMPLE:
- Heading: "Blog Post Template"
- Body: "Write a [word count] blog about [topic] for [audience]. Include: intro hook, 3 main points, actionable conclusion."

BAD EXAMPLE (too vague):
- Body: "Templates can help you write better content."''',

            "templates": '''Include an ACTUAL TEMPLATE to copy.
GOOD EXAMPLE:
- Heading: "Product Description"
- Body: "Write a compelling product description for [product]. Highlight: key features, benefits, target user, call-to-action."

BAD EXAMPLE (too vague):
- Body: "Using templates makes AI outputs more consistent."''',
        }

        example_instruction = example_types.get(content_type.lower(), "")

        base_prompt = f"""Generate content for slide {slide_number}: "{slide_title}"

This is a {content_type} slide. Provide:
- A SHORT heading (max 50 characters / 6-8 words) - be concise!
- Body text (max 200 characters) - 1-2 SHORT sentences

CRITICAL: The hook promised "{content_type}s" - YOU MUST DELIVER!
{example_instruction}

IMPORTANT LENGTH LIMITS:
- Heading: MAX 50 characters (will be cut off if longer!)
- Body: MAX 200 characters

The body MUST contain a real, usable {content_type} - not just a description of what the {content_type} is about.
DELIVER what the hook promised! Give readers something they can actually USE."""

        prompt = base_prompt
        working_history = history.copy()
        last_error = None

        for attempt in range(max_validation_retries + 1):
            # Add retry context if this is a retry
            if attempt > 0 and last_error:
                prompt = f"""{base_prompt}

IMPORTANT: Your previous attempt was rejected for this reason:
{last_error}

Please generate BETTER content that addresses this issue."""

            # Use Instructor extractor for structured extraction
            slide_content, working_history = await self.extractor.extract_with_history(
                prompt=prompt,
                response_model=ContentSlideResponse,
                history=working_history if attempt == 0 else history.copy(),  # Fresh history on retry
                task=f"phase3_slide_{slide_number}",
                temperature=0.7 + (attempt * 0.1),  # Slightly increase temp on retries
                max_tokens=300,
            )

            result = {
                "heading": slide_content.heading or slide_title,
                "body": slide_content.body or "",
            }

            # Validate the content
            is_valid, error = await self._validate_slide_content(
                heading=result["heading"],
                body=result["body"],
                slide_number=slide_number,
                content_type=content_type,
                topic=topic,
            )

            if is_valid:
                _logger.info(
                    f"Phase 3 - Slide {slide_number}: '{result['heading'][:30]}...' "
                    f"(validated on attempt {attempt + 1})"
                )
                return result, working_history

            # Validation failed - prepare for retry
            last_error = error
            _logger.warning(
                f"Phase 3 - Slide {slide_number} validation failed (attempt {attempt + 1}): {error}"
            )

            await self._emit_progress(
                action=f"Regenerating {content_type} {slide_number} (quality check failed)...",
                phase=3,
                phase_name="Content",
                error=error,
            )

        # All retries exhausted - use last result anyway but log warning
        _logger.warning(
            f"Phase 3 - Slide {slide_number}: accepting content after {max_validation_retries + 1} attempts"
        )
        return result, working_history

    async def _phase4_cta(
        self,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Phase 4: Generate CTA based on all content.

        Returns:
            CTA slide dict
        """
        prompt = """Generate a short call-to-action for the final slide.

Good examples:
- "Follow for more tips!"
- "Save this for later!"
- "Share with a friend!"
- "Don't miss the next one - Follow!"
- "Like if this helped you!"

Do NOT mention downloading, guides, ebooks, or any external content. Keep it simple - just encourage engagement (follow/save/share/like)."""

        try:
            # Use Instructor extractor for structured extraction
            cta, _ = await self.extractor.extract_with_history(
                prompt=prompt,
                response_model=CTAResponse,
                history=history,
                task="phase4_cta",
                temperature=0.8,
                max_tokens=100,
            )

            return {
                "cta_text": cta.cta_text or "Follow for more!",
                "cta_subtext": cta.cta_subtext,
            }
        except Exception as e:
            _logger.warning(f"Phase 4 CTA extraction failed: {e}, using default")
            return {
                "cta_text": "Follow for more!",
                "cta_subtext": None,
            }

    async def _validate_slide_content(
        self,
        heading: str,
        body: str,
        slide_number: int,
        content_type: str,
        topic: str,
    ) -> tuple[bool, str | None]:
        """Validate slide content quality using AI.

        Checks for:
        - Gibberish or nonsensical text
        - Repeated phrases
        - Content that doesn't match the topic
        - Too short or incomplete content
        - Missing actionable information
        - Missing concrete examples when content type requires them

        Args:
            heading: Slide heading text.
            body: Slide body text.
            slide_number: Slide number.
            content_type: Type of content (tip, tool, prompt, etc.).
            topic: Original topic for context.

        Returns:
            Tuple of (is_valid, error_message or None)
        """
        # Quick sanity checks first (no AI needed)
        if len(heading.strip()) < 5:
            return False, "Heading too short (less than 5 characters)"

        if len(body.strip()) < 20:
            return False, "Body text too short (less than 20 characters)"

        # Check for repeated words (sign of gibberish)
        words = body.lower().split()
        if len(words) > 5:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # More than 70% repeated words
                return False, f"Too many repeated words (unique ratio: {unique_ratio:.2f})"

        # Check for obvious gibberish patterns
        gibberish_patterns = [
            r"(.)\1{4,}",  # Same character repeated 5+ times
            r"(\w{2,})\s+\1\s+\1",  # Same word repeated 3+ times
            r"[^\x00-\x7F]{5,}",  # Long non-ASCII sequences (potential encoding issues)
        ]
        for pattern in gibberish_patterns:
            if re.search(pattern, body):
                return False, f"Detected potential gibberish pattern: {pattern}"

        # Check for content types that require concrete examples
        example_required_types = ["prompt", "prompts", "template", "templates"]
        if content_type.lower() in example_required_types:
            # Check if body contains quote marks (indicates actual prompt/template)
            if '"' not in body and "'" not in body and ":" not in body:
                return False, f"Content type '{content_type}' requires an actual example with quotes or format like 'Try: ...'"

        # Check for vague/generic content patterns
        vague_patterns = [
            r"^(This|It|The) (is|can|will|helps)",  # Starts with generic description
            r"(you can|you should|you need to) (use|try|do)",  # Generic advice without specifics
        ]
        body_lower = body.lower()
        for pattern in vague_patterns:
            if re.search(pattern, body_lower) and '"' not in body and ":" not in body:
                # Only flag if there's no actual example
                if content_type.lower() in ["prompt", "prompts", "tip", "tips", "trick", "tricks", "hack", "hacks"]:
                    return False, f"Content is too vague - needs a concrete example, not just description"

        # AI validation for semantic quality
        validation_prompt = f"""Validate this Instagram carousel slide content for quality.

TOPIC: {topic}
CONTENT TYPE: {content_type}
SLIDE NUMBER: {slide_number}

HEADING: {heading}
BODY: {body}

Check for these issues:
1. Is the text coherent and makes sense? (no gibberish)
2. Does it relate to the topic?
3. Is it specific and actionable (not generic filler)?
4. Is the grammar acceptable?
5. Does it provide real value?
6. CRITICAL: If content_type is "prompt", "template", "tip", "trick", or "hack" - does the body contain an ACTUAL EXAMPLE that readers can use? (not just a description of what to do)

IMPORTANT: For content types like "prompt" or "template", the body MUST contain an actual usable example (usually in quotes or with a format like "Try: ..."). Generic descriptions like "Use AI to help with X" are NOT acceptable - we need the ACTUAL prompt/template/tip.

If the content passes ALL checks, set is_valid=true and issues=[]
If there are problems, set is_valid=false and list the specific issues."""

        try:
            validation = await self.extractor.extract(
                prompt=validation_prompt,
                response_model=ContentValidationResponse,
                system="You are a content quality validator. Be strict about quality.",
                task="slide_validation",
                temperature=0.3,  # Low temp for consistent validation
                max_tokens=200,
            )

            if not validation.is_valid:
                issues_str = "; ".join(validation.issues) if validation.issues else "Unknown issue"
                _logger.warning(
                    f"VALIDATION_FAILED | slide:{slide_number} | severity:{validation.severity} | "
                    f"issues:{issues_str}"
                )
                return False, issues_str

            _logger.info(f"VALIDATION_PASSED | slide:{slide_number}")
            return True, None

        except Exception as e:
            # If validation fails, log but don't block (graceful degradation)
            _logger.warning(f"Validation error for slide {slide_number}: {e}, allowing content")
            return True, None

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

        # Clamp to min/max (default: min=3, max=10 total slides)
        effective_min = (min_slides if min_slides is not None else 3) - 2
        effective_max = (max_slides if max_slides is not None else 10) - 2
        content_count = max(effective_min, min(effective_max, content_count))
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
                current_slide=slide_num,
                total_slides=content_count,
            )

            slide_content, history = await self._phase3_content_slide(
                slide_number=slide_num,
                slide_title=title,
                content_type=content_type,
                history=history,
                topic=topic,  # Pass topic for validation context
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
                current_slide=slide_num,
                total_slides=content_count,
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

        # Content slides (default to 5 slides total if not specified)
        effective_slides = target_slides if target_slides is not None else 5
        content_count = max(1, effective_slides - 2)
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
        max_retries: int = 3,
    ) -> str:
        """Generate Instagram caption for a post with validation.

        Args:
            topic: Post topic.
            hook_text: The hook used.
            content_summary: Summary of post content.
            hashtags: Hashtags to include.
            max_retries: Maximum retry attempts for bad captions.

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

        # Try to generate a valid caption with retries
        for attempt in range(max_retries):
            caption = await self.text_provider.generate(
                prompt=prompt,
                system=self._get_system_prompt(),
                task="content_planning",
                temperature=0.7 + (attempt * 0.1),  # Increase temp on retries
            )

            caption = caption.strip()

            # Validate caption quality
            is_valid, error = self._validate_caption(caption, topic)
            if is_valid:
                break

            _logger.warning(f"Caption validation failed (attempt {attempt + 1}): {error}")
            _logger.debug(f"Invalid caption: {caption[:100]}...")

            if attempt == max_retries - 1:
                # Last attempt failed - use a safe default
                _logger.error(f"All caption attempts failed, using default for topic: {topic}")
                caption = f"Check out these {topic.lower()}! Save this for later."

        # Truncate caption if too long (for Threads 500 char limit)
        # Keep it under 450 chars to leave room for potential additions
        if len(caption) > 450:
            # Cut at last sentence boundary
            caption = caption[:450].rsplit('.', 1)[0] + '.'

        # Return ONLY the caption text (without hashtags)
        # Hashtags are stored separately in post.hashtags and combined in caption+hashtags.txt
        return caption

    def _validate_caption(self, caption: str, topic: str) -> tuple[bool, str | None]:
        """Validate caption quality without AI (fast checks).

        Args:
            caption: Generated caption text.
            topic: Original topic for context.

        Returns:
            Tuple of (is_valid, error_message or None)
        """
        # Check minimum length
        if len(caption) < 20:
            return False, "Caption too short (less than 20 characters)"

        # Check for incomplete sentences (signs of truncated/bad generation)
        incomplete_patterns = [
            r"^(You are|I am|The|A|An)\s+\w+\s*$",  # Incomplete start
            r"\b(the|a|an|to|for|with|and|or|but)\s*$",  # Ends with article/preposition
            r"^\s*\d+\.?\d*\s*$",  # Just a number
        ]
        for pattern in incomplete_patterns:
            if re.search(pattern, caption, re.IGNORECASE):
                return False, f"Caption appears incomplete (matched: {pattern})"

        # Check for gibberish patterns
        words = caption.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:  # Too many repeated words
                return False, f"Too many repeated words (ratio: {unique_ratio:.2f})"

        # Check for nonsensical phrases (common AI failure modes)
        nonsense_phrases = [
            "the other hand",
            "need to solve",
            "goal of the",
            "you are a new",
            "I don't know",
            "I'm not sure",
            "as an AI",
            "I cannot",
        ]
        caption_lower = caption.lower()
        for phrase in nonsense_phrases:
            if phrase in caption_lower:
                return False, f"Contains nonsense phrase: '{phrase}'"

        # Check that caption somewhat relates to topic (at least one topic word)
        topic_words = set(topic.lower().split())
        topic_words -= {"the", "a", "an", "for", "to", "and", "or", "of", "in", "on", "with"}
        caption_words = set(caption_lower.split())
        if len(topic_words) > 0 and not topic_words & caption_words:
            # No topic words in caption - might be off-topic
            # Allow if caption has common social media words
            social_words = {"save", "follow", "share", "check", "tips", "learn", "discover"}
            if not social_words & caption_words:
                return False, "Caption doesn't relate to topic"

        return True, None
