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
from ...services.caption_service import sanitize_json_string
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

        Dispatches to profile-specific prompts (news vs tools).

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

        # Add feedback section if this is a retry
        feedback_section = ""
        if feedback:
            feedback_section = f"""
IMPORTANT - Your previous caption was rejected. Fix these issues:
{feedback}

Generate a NEW caption that addresses this feedback.
"""

        # Detect if this is a news profile
        is_news_profile = "news" in context.profile.name.lower() if context.profile.name else False

        if is_news_profile:
            return self._build_news_caption_prompt(
                narration, profile_handle, profile_hashtag, feedback_section, context
            )
        else:
            return self._build_tools_caption_prompt(
                narration, profile_handle, profile_hashtag, feedback_section, context
            )

    def _build_news_caption_prompt(
        self,
        narration: str,
        profile_handle: str,
        profile_hashtag: str,
        feedback_section: str,
        context: PipelineContext,
    ) -> str:
        """Build caption prompt for news profiles.

        News captions list stories with proper line breaks and source attribution.
        """
        # Extract source names from research (if available)
        source_names = []
        if context.research and context.research.sources:
            source_names = list(set(s.get("name", "") for s in context.research.sources if s.get("name")))

        sources_line = f"\nSources used: {', '.join(source_names)}" if source_names else ""

        return f"""Write an engaging Instagram Reels caption for a NEWS video.
{feedback_section}
VIDEO NARRATION (contains the news stories covered):
---
{narration}
---

Topic: {context.topic.topic}
Profile: {context.profile.display_name} ({profile_handle}){sources_line}

CRITICAL REQUIREMENTS:
1. Start with a short headline (e.g., "Night Edition: Entertainment news you can't miss!")
2. List 3-5 news stories from the narration as bullet points
3. EACH BULLET MUST BE ON ITS OWN LINE - use actual line breaks (\\n), NOT inline dashes
4. Each bullet should summarize ONE story briefly (10-15 words max)
5. After the CTA line, add "Sources: " followed by source names (e.g., TMZ, Variety)
6. End with: "Save this + follow {profile_handle} for more!"
7. Keep under 250 words total
8. NO EMOJIS in the caption text
9. NO MARKDOWN FORMATTING

IMPORTANT: In your JSON response, use \\n for line breaks between bullets. Example:
"caption": "Headline here\\n\\n- First story summary\\n- Second story summary\\n\\nSave this...\\n\\nSources: TMZ, Variety"

FORMAT - Return valid JSON only:
{{
    "caption": "Headline\\n\\n- Story 1\\n- Story 2\\n- Story 3\\n\\nSave this + follow {profile_handle} for more! {profile_hashtag}\\n\\nSources: Source1, Source2",
    "hashtags": "#Entertainment #News #Hollywood ... (5 relevant hashtags)"
}}

EXAMPLE of correct format:
{{
    "caption": "Night Edition: Entertainment news you can't miss!\\n\\n- Netflix's Terminator Zero anime brings franchise-changing storytelling\\n- Disney's Avatar 4 reveals eight-year time skip mystery\\n- Disney+ streaming offers holiday entertainment deals\\n\\nSave this + follow @news.but.quick for more! #Newsbutquick\\n\\nSources: Variety, Entertainment Weekly, Deadline",
    "hashtags": "#Entertainment #Netflix #Disney #Hollywood #Streaming"
}}

Return ONLY the JSON, no markdown or explanation."""

    def _build_tools_caption_prompt(
        self,
        narration: str,
        profile_handle: str,
        profile_hashtag: str,
        feedback_section: str,
        context: PipelineContext,
    ) -> str:
        """Build caption prompt for AI tools/tips profiles."""
        import re

        # Extract tool/product names mentioned in narration
        tool_pattern = r'\b([A-Z][a-zA-Z]*(?:\.[a-zA-Z]+)?(?:\'s)?)\b'
        potential_tools = re.findall(tool_pattern, narration)
        tools_mentioned = [t for t in potential_tools if len(t) > 2 and t not in ['The', 'This', 'With', 'When', 'First', 'Next', 'Follow', 'Save', 'Here']]
        tools_list = ", ".join(set(tools_mentioned[:10])) if tools_mentioned else "various AI tools"

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
2. EACH BULLET MUST BE ON ITS OWN LINE - use actual line breaks (\\n), NOT inline dashes
3. Each bullet should name a specific tool AND what it does (e.g., "Jasper.ai for writing blog posts")
4. Keep caption under 200 words total
5. End with call-to-action: "Save this + follow {profile_handle} for more!"
6. Add profile hashtag: {profile_hashtag}
7. NO EMOJIS in the caption text
8. NO MARKDOWN FORMATTING

IMPORTANT: In your JSON response, use \\n for line breaks between bullets. Example:
"caption": "Headline here\\n\\n- First tool\\n- Second tool\\n\\nSave this..."

FORMAT - Return valid JSON only:
{{
    "caption": "AI productivity hacks you need to try today:\\n\\n- Jasper.ai for writing blog posts\\n- ChatGPT for drafting emails\\n- Canva for quick design work\\n\\nSave this + follow {profile_handle} for more! {profile_hashtag}",
    "hashtags": "#AI #ChatGPT #Productivity ... (8-12 relevant hashtags)"
}}

Return ONLY the JSON, no markdown or explanation."""

    def _fix_inline_bullets(self, caption: str) -> str:
        """Fix inline bullets to use proper line breaks.

        AI sometimes generates captions like:
        "Headline! - First item - Second item - Third item Save this..."

        This converts them to:
        "Headline!

        - First item
        - Second item
        - Third item

        Save this..."

        Args:
            caption: Caption text that may have inline bullets.

        Returns:
            Caption with proper line breaks.
        """
        import re

        # If already has proper line breaks, don't modify
        if '\n- ' in caption or '\n* ' in caption:
            return caption

        # Pattern: sentence end (! or .) followed by space and dash
        # Example: "miss! - First" -> "miss!\n\n- First"
        if ' - ' in caption:
            # Check if this looks like inline bullets (multiple dashes)
            dash_count = caption.count(' - ')
            if dash_count >= 2:  # At least 2 inline bullets
                # Split into parts: headline and bullets
                # Find the first " - " that starts a bullet point
                # (usually after ! or . or :)
                pattern = r'([!.?:])\s*-\s+'
                match = re.search(pattern, caption)
                if match:
                    # Split at the first bullet
                    split_pos = match.end() - len(match.group(0)) + len(match.group(1))
                    headline = caption[:split_pos].strip()
                    bullets_text = caption[split_pos:].strip()

                    # Remove leading " - " from bullets_text
                    bullets_text = re.sub(r'^[\s-]+', '', bullets_text)

                    # Split remaining text by " - " pattern
                    bullets = re.split(r'\s+-\s+', bullets_text)

                    # Check if last bullet contains CTA (Save, Follow, etc.)
                    cta = ""
                    if bullets:
                        last_bullet = bullets[-1]
                        # Look for CTA pattern in last bullet
                        cta_match = re.search(
                            r'\s+(Save this|Follow|Tag|Share|Comment|Like|Check out)',
                            last_bullet,
                            re.IGNORECASE
                        )
                        if cta_match:
                            # Split last bullet from CTA
                            cta = last_bullet[cta_match.start():].strip()
                            bullets[-1] = last_bullet[:cta_match.start()].strip()

                    # Filter out empty bullets
                    bullets = [b.strip() for b in bullets if b.strip()]

                    # Reconstruct with proper line breaks
                    if bullets:
                        formatted_bullets = '\n- '.join(bullets)
                        result = f"{headline}\n\n- {formatted_bullets}"
                        if cta:
                            result += f"\n\n{cta}"
                        return result

        return caption

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
            # Sanitize JSON to handle literal newlines from local LLMs
            sanitized = sanitize_json_string(clean_response)
            data = json.loads(sanitized)
            caption = data.get("caption", "")
            hashtags = data.get("hashtags", "")

            # Strip any markdown formatting from caption
            caption = strip_markdown(caption)

            # Fix inline bullets - convert to proper line breaks
            caption = self._fix_inline_bullets(caption)

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

            # Sanitize JSON to handle literal newlines from local LLMs
            sanitized = sanitize_json_string(clean_response)
            data = json.loads(sanitized)
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

        # Check for LINE BREAKS - bullets should be on separate lines, not inline
        has_line_breaks = '\n' in caption
        if has_bullets and not has_line_breaks:
            issues.append("Bullets should be on SEPARATE LINES with actual line breaks (\\n), not inline dashes")

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

        # Check that some content from narration appears (relaxed for news)
        # Extract key terms from narration
        import re
        tool_pattern = r'\b([A-Z][a-zA-Z]+(?:\.[a-zA-Z]+)?)\b'
        narration_terms = set(re.findall(tool_pattern, narration))
        skip_words = {'The', 'This', 'With', 'When', 'First', 'Next', 'Follow', 'Save', 'Here', 'Want', 'AI'}
        narration_terms = {t for t in narration_terms if t not in skip_words and len(t) > 2}

        terms_in_caption = sum(1 for term in narration_terms if term in caption)
        # Relaxed: only require 2 terms (works for both news and tools)
        if terms_in_caption < 2:
            issues.append(f"Caption should mention more specific content from the video (found {terms_in_caption}, need at least 2)")

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
