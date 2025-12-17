"""Caption and hashtag generation for video reels.

Generates Instagram-style caption and hashtags for the video
based on the script content and topic.
"""

from pathlib import Path
from typing import Any, Optional

from markdown_text_clean import clean_text as strip_markdown

from .base import (
    ArtifactStatus,
    CaptionGenerationError,
    PipelineContext,
    PipelineStep,
)
from ...services.llm_fallback import LLMFallbackManager, FallbackConfig
from ...providers.config import load_provider_config


class CaptionGenerator(PipelineStep):
    """Generates Instagram caption and hashtags for video reels.

    Note: This is different from ICaptionGenerator which is for narration.
    This generates the Instagram post caption and hashtags.

    Uses LLMFallbackManager for automatic retry and provider switching.
    """

    def __init__(
        self,
        ai_client: Optional[object] = None,
        fallback_manager: Optional[LLMFallbackManager] = None,
        preferred_provider: str | None = None,
    ):
        """Initialize caption generator.

        Args:
            ai_client: Optional AI client for enhanced caption generation (legacy).
            fallback_manager: Optional LLMFallbackManager for generation.
            preferred_provider: Preferred LLM provider (e.g., 'lmstudio').
        """
        super().__init__("CaptionGenerator")
        self.ai_client = ai_client
        self._fallback_manager = fallback_manager
        self._preferred_provider = preferred_provider

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute caption generation step.

        Args:
            context: Pipeline context with script and topic.

        Returns:
            Updated context (caption files written to output_dir).
        """
        if not context.script:
            raise CaptionGenerationError("No script available for caption generation")
        if not context.topic:
            raise CaptionGenerationError("No topic available for caption generation")

        self.log_start("Generating caption and hashtags...")

        try:
            # Generate caption and hashtags
            caption, hashtags = await self._generate_caption(context)

            # Save to files
            caption_path = context.output_dir / "caption.txt"
            caption_hashtags_path = context.output_dir / "caption+hashtags.txt"

            # Write caption only
            with open(caption_path, "w", encoding="utf-8") as f:
                f.write(caption)

            # Write caption + hashtags
            with open(caption_hashtags_path, "w", encoding="utf-8") as f:
                f.write(caption)
                f.write("\n\n")
                f.write(hashtags)

            self.log_success(f"Caption saved: {caption_path.name}")
            self.log_progress(f"Hashtags: {len(hashtags.split())} tags")

            # Update artifact tracking
            if context.metadata:
                context.metadata.artifacts.caption = ArtifactStatus(
                    status="ok",
                    file="caption.txt",
                )
                context.metadata.artifacts.hashtags = ArtifactStatus(
                    status="ok",
                    file="caption+hashtags.txt",
                )

            return context

        except Exception as e:
            self.log_error(f"Caption generation failed: {e}")
            # Update artifact tracking with error
            if context.metadata:
                context.metadata.artifacts.caption = ArtifactStatus(
                    status="failed",
                    error=str(e),
                )
                context.metadata.artifacts.hashtags = ArtifactStatus(
                    status="failed",
                    error=str(e),
                )
            raise CaptionGenerationError(f"Failed to generate caption: {e}") from e

    def _get_fallback_manager(self) -> LLMFallbackManager:
        """Get or create the LLMFallbackManager."""
        if self._fallback_manager:
            return self._fallback_manager

        # Create with config from providers.yaml
        provider_config = load_provider_config()
        config = FallbackConfig.from_provider_config(provider_config)

        self._fallback_manager = LLMFallbackManager(
            preferred_provider=self._preferred_provider,
            config=config,
            provider_config=provider_config,
        )
        return self._fallback_manager

    async def _generate_caption(
        self,
        context: PipelineContext,
    ) -> tuple[str, str]:
        """Generate caption and hashtags with validation and retry.

        Uses LLMFallbackManager for automatic retry and provider fallback.

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (caption, hashtags).
        """
        # Use LLMFallbackManager for automatic retry and fallback
        manager = self._get_fallback_manager()

        # Build the prompt
        prompt = self._build_caption_prompt(context)

        # Create validator function
        async def validate_and_parse(response: str) -> tuple[bool, str | None]:
            """Validate response and return parsed caption."""
            try:
                caption, hashtags = self._parse_caption_response(response, context)
                # Store parsed result for later retrieval
                self._last_parsed_result = (caption, hashtags)
                # Validate caption quality
                is_valid, feedback = await self._validate_caption(caption, context)
                if not is_valid:
                    return False, feedback
                return True, None
            except Exception as e:
                return False, str(e)

        # Sync wrapper for the validator (manager expects sync validator)
        import asyncio
        def sync_validator(response: str) -> tuple[bool, str | None]:
            """Sync wrapper for async validation."""
            try:
                # Parse the response first
                caption, hashtags = self._parse_caption_response(response, context)
                self._last_parsed_result = (caption, hashtags)
                # Basic validation only (AI validation is async, skip for now)
                is_valid, feedback = self._basic_validation(caption, context.script.full_narration or "")
                return is_valid, feedback
            except Exception as e:
                return False, str(e)

        self.log_start("Generating caption with LLM fallback...")

        result = await manager.generate(
            prompt=prompt,
            task="caption",
            validator=sync_validator,
        )

        if not result.success:
            raise CaptionGenerationError(
                f"Failed to generate valid caption after {result.total_attempts} attempts. "
                f"Error: {result.error}"
            )

        # Return the parsed result
        if hasattr(self, '_last_parsed_result'):
            caption, hashtags = self._last_parsed_result
            del self._last_parsed_result
            self.log_success(f"Caption generated using {result.provider_used} ({result.total_attempts} attempts)")
            return caption, hashtags

        # Fallback: parse the raw result
        caption, hashtags = self._parse_caption_response(result.result, context)
        self.log_success(f"Caption generated using {result.provider_used}")
        return caption, hashtags

    def _build_caption_prompt(self, context: PipelineContext, feedback: str | None = None) -> str:
        """Build the caption generation prompt.

        Args:
            context: Pipeline context.
            feedback: Optional feedback from previous validation failure.

        Returns:
            The prompt string.
        """
        import re

        profile_handle = context.profile.instagram_handle or ""
        profile_hashtag = f"#{context.profile.name.replace('.', '').replace('_', '').title()}" if context.profile.name else ""

        # Extract the full narration - this is the source of truth
        narration = context.script.full_narration or ""

        # Extract tool/product names mentioned in narration (e.g., "Jasper.ai", "ChatGPT", etc.)
        tool_pattern = r'\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)?(?:\'s)?)\b'
        potential_tools = re.findall(tool_pattern, narration)
        # Filter to likely tool names (capitalized, may have .ai/.io suffix)
        tools_mentioned = [t for t in potential_tools if len(t) > 2 and t not in ['The', 'This', 'With', 'When', 'First', 'Next', 'Follow', 'Save', 'Here']]
        tools_list = ", ".join(set(tools_mentioned[:10])) if tools_mentioned else "various AI tools"

        # Add feedback section if this is a retry
        feedback_section = ""
        if feedback:
            feedback_section = f"""
IMPORTANT - Your previous caption was rejected. Fix these issues:
{feedback}

Generate a NEW caption that addresses this feedback.
"""

        return f"""Write an engaging Instagram Reels caption based on this video narration.
{feedback_section}
FULL VIDEO NARRATION (read this carefully - your caption MUST reference specific content from here):
---
{narration}
---

Topic: {context.topic.topic}
Profile: {context.profile.display_name} ({profile_handle})
Tools/Products mentioned: {tools_list}

CRITICAL REQUIREMENTS:
1. Your caption MUST list 3-5 SPECIFIC tools or tips from the narration above
2. Use emoji bullet points to list what the video covers
3. Each bullet should name a specific tool AND what it does (e.g., "Jasper.ai for writing blog posts")
4. Keep caption under 200 words total
5. End with call-to-action: "Save this + follow {profile_handle} for more!"
6. Add profile hashtag: {profile_hashtag}
7. NO MARKDOWN FORMATTING - Do not use asterisks (*), underscores (_), backticks (`), or any other markdown syntax. Plain text only.

FORMAT - Return valid JSON only:
{{
    "caption": "Your caption here with emoji bullet points listing specific tools from the narration...",
    "hashtags": "#AI #ChatGPT #Productivity ... (8-12 relevant hashtags)"
}}

EXAMPLE of good caption style:
"AI productivity hacks you need to try today:

- Jasper.ai for writing blog posts in minutes
- ChatGPT for drafting professional emails
- Canva Magic Resize for quick design work
- Grammarly for error-free documents
- Otter.ai for meeting transcriptions

Save this + follow @ai.for.mortals for more! #AiForMortals"

Return ONLY the JSON, no markdown or explanation."""

    def _parse_caption_response(self, response: str, context: PipelineContext) -> tuple[str, str]:
        """Parse caption JSON response.

        Args:
            response: Raw AI response.
            context: Pipeline context.

        Returns:
            Tuple of (caption, hashtags).

        Raises:
            ValueError: If JSON parsing fails.
        """
        import json

        profile_hashtag = f"#{context.profile.name.replace('.', '').replace('_', '').title()}" if context.profile.name else ""

        clean_response = response.strip()
        if clean_response.startswith("```"):
            clean_response = clean_response.split("```")[1]
            if clean_response.startswith("json"):
                clean_response = clean_response[4:]
        clean_response = clean_response.strip()

        try:
            data = json.loads(clean_response)
            caption = data.get("caption", "")
            hashtags = data.get("hashtags", "")

            # Strip any markdown formatting from caption
            caption = strip_markdown(caption)

            # Ensure profile hashtag is in caption
            if profile_hashtag and profile_hashtag not in caption:
                caption = caption.rstrip() + f" {profile_hashtag}"

            return caption, hashtags
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")

    async def _generate_with_ai(
        self,
        context: PipelineContext,
        feedback: str | None = None,
    ) -> tuple[str, str]:
        """Generate caption using AI (legacy method, kept for compatibility).

        Args:
            context: Pipeline context.
            feedback: Optional feedback from previous validation failure.

        Returns:
            Tuple of (caption, hashtags).
        """
        if not self.ai_client:
            raise CaptionGenerationError("AI client required for _generate_with_ai")

        prompt = self._build_caption_prompt(context, feedback)
        response = await self.ai_client.generate(prompt)
        return self._parse_caption_response(response, context)

    async def _validate_caption(
        self,
        caption: str,
        context: PipelineContext,
    ) -> tuple[bool, str]:
        """Validate caption quality using AI.

        Uses LLMFallbackManager for automatic retry and provider fallback.

        Args:
            caption: The generated caption to validate.
            context: Pipeline context with narration.

        Returns:
            Tuple of (is_valid, feedback). If invalid, feedback explains why.
        """
        import json

        narration = context.script.full_narration or ""

        prompt = f"""You are a caption quality validator. Evaluate this Instagram caption.

CAPTION TO VALIDATE:
---
{caption}
---

ORIGINAL VIDEO NARRATION (the caption should summarize this):
---
{narration}
---

VALIDATION CRITERIA:
1. Does the caption mention at least 3 SPECIFIC tools/tips from the narration? (not generic phrases)
2. Does each bullet point explain what the tool DOES? (e.g., "ChatGPT for emails" not just "ChatGPT")
3. Is the caption engaging and not generic/vague?
4. Does it have a call-to-action?
5. Is it under 200 words?

RESPOND WITH JSON ONLY:
{{
    "is_valid": true or false,
    "score": 1-10,
    "feedback": "If invalid, explain what's wrong and how to fix it. If valid, say 'Good caption'"
}}

A caption is valid if score >= 7. Be strict - generic captions like "Check out these tips!" without specific content should fail."""

        try:
            # Use LLMFallbackManager for validation
            manager = self._get_fallback_manager()
            result = await manager.generate(prompt, task="caption_validation")

            if not result.success:
                self.log_detail(f"AI validation failed: {result.error}, using basic checks")
                return self._basic_validation(caption, narration)

            response = result.result

            # Parse JSON
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            data = json.loads(clean_response)
            is_valid = data.get("is_valid", False)
            score = data.get("score", 0)
            feedback = data.get("feedback", "Unknown validation error")

            # Also check score threshold
            if score < 7:
                is_valid = False

            self.log_progress(f"Validation score: {score}/10 - {'PASS' if is_valid else 'FAIL'}")

            return is_valid, feedback

        except Exception as e:
            # If validation fails, do basic programmatic checks
            self.log_detail(f"AI validation failed: {e}, using basic checks")
            return self._basic_validation(caption, narration)

    def _basic_validation(self, caption: str, narration: str) -> tuple[bool, str]:
        """Basic programmatic validation as fallback.

        Args:
            caption: Caption to validate.
            narration: Original narration.

        Returns:
            Tuple of (is_valid, feedback).
        """
        issues = []

        # Check length
        if len(caption) < 50:
            issues.append("Caption is too short (< 50 chars)")
        if len(caption) > 1000:
            issues.append("Caption is too long (> 1000 chars)")

        # Check for bullet points or list structure
        has_bullets = any(char in caption for char in ['-', '*', '1.', '2.', '3.'])
        if not has_bullets:
            issues.append("Caption should have bullet points listing specific tips/tools")

        # Check for generic phrases (bad)
        generic_phrases = [
            "check out these tips",
            "this is exactly what you need",
            "you won't believe",
            "amazing tips",
        ]
        caption_lower = caption.lower()
        for phrase in generic_phrases:
            if phrase in caption_lower:
                issues.append(f"Remove generic phrase: '{phrase}'")

        # Check that some content from narration appears
        # Extract key terms from narration
        import re
        tool_pattern = r'\b([A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)?)\b'
        narration_tools = set(re.findall(tool_pattern, narration))
        skip_words = {'The', 'This', 'With', 'When', 'First', 'Next', 'Follow', 'Save', 'Here', 'Want', 'AI'}
        narration_tools = {t for t in narration_tools if t not in skip_words and len(t) > 2}

        tools_in_caption = sum(1 for tool in narration_tools if tool in caption)
        if tools_in_caption < 2:
            issues.append(f"Caption should mention more specific tools from the video (found {tools_in_caption}, need at least 3)")

        if issues:
            return False, "; ".join(issues)
        return True, "Basic validation passed"

    def _generate_with_template(
        self,
        context: PipelineContext,
    ) -> tuple[str, str]:
        """Generate caption using template (fallback).

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (caption, hashtags).
        """
        import re

        profile_handle = context.profile.instagram_handle or ""
        profile_hashtag = f"#{context.profile.name.replace('.', '').replace('_', '').title()}" if context.profile.name else ""

        # Get narration and extract key tools/concepts
        narration = context.script.full_narration or ""

        # Extract tool names from narration (capitalized words, especially with .ai/.io)
        tool_pattern = r'\b([A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)?)\b'
        potential_tools = re.findall(tool_pattern, narration)
        # Filter common words
        skip_words = {'The', 'This', 'With', 'When', 'First', 'Next', 'Follow', 'Save', 'Here', 'Want', 'Discover', 'Use', 'AI'}
        tools = [t for t in potential_tools if t not in skip_words and len(t) > 2]
        unique_tools = list(dict.fromkeys(tools))[:5]  # Keep first 5 unique

        # Build caption from narration content
        topic_short = context.topic.topic[:60] if len(context.topic.topic) <= 60 else context.topic.topic[:57] + "..."

        if unique_tools:
            # List the tools mentioned
            tool_bullets = "\n".join(f"- {tool}" for tool in unique_tools)
            caption = (
                f"{topic_short}\n\n"
                f"Tools covered:\n{tool_bullets}\n\n"
                f"Save this + follow {profile_handle} for more! {profile_hashtag}"
            )
        else:
            # Fallback to first 2 sentences of narration as summary
            sentences = narration.split('.')[:2]
            summary = '. '.join(s.strip() for s in sentences if s.strip())
            if summary and not summary.endswith('.'):
                summary += '.'

            caption = (
                f"{summary}\n\n"
                f"Save this + follow {profile_handle} for more tips! {profile_hashtag}"
            )

        # Default hashtags based on pillar
        pillar_hashtags = {
            "tool_tutorials": "#AITools #TechTips #Tutorial #HowTo #LearnAI",
            "productivity_hacks": "#ProductivityTips #WorkSmarter #Efficiency #TimeManagement #LifeHacks",
            "ai_money_making": "#MakeMoneyOnline #SideHustle #AIBusiness #PassiveIncome #Entrepreneur",
        }

        base_hashtags = pillar_hashtags.get(
            context.topic.pillar_id,
            "#AITools #TechTips #ProductivityTips"
        )

        hashtags = f"#AI #ArtificialIntelligence #ChatGPT {base_hashtags} #FutureOfWork #Automation"

        return caption, hashtags
