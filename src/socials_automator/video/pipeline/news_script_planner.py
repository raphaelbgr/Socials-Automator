"""Script planning for news-based video content.

Converts a NewsBrief (curated news stories) into a VideoScript
suitable for the standard video pipeline.

Structure:
- Hook segment: "X things in entertainment you need to know"
- Story segments: One per curated story
- CTA segment: Follow for daily updates
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from .base import (
    PipelineStep,
    PipelineContext,
    ScriptPlanningError,
    VideoScript,
    VideoSegment,
    TopicInfo,
    ResearchResult,
)
from ...news.models import NewsBrief, NewsStory, NewsEdition
from ...providers.text import TextProvider

logger = logging.getLogger("ai_calls")


# =============================================================================
# Prompts
# =============================================================================

NEWS_SCRIPT_SYSTEM_PROMPT = """You are a scriptwriter for @news.but.quick, a fast-paced entertainment news account.

Your job is to write narration scripts for 60-second news briefing videos. The style is:
- Casual and conversational, like texting a friend
- Quick and punchy - no filler words
- Slightly witty but always accurate
- Transitions between stories should be natural

SPEAKING PACE: Average 150 words per minute. A 60-second video needs ~150 words total.

STRUCTURE:
1. Hook (3 seconds, ~8 words): Grab attention, state how many stories
2. Stories (10-12 seconds each, ~25-30 words each): Headline + summary + why it matters
3. CTA (3 seconds): ALWAYS end with exactly: "Follow News But Quick to keep up to date with the latest news!"

RULES:
- Use contractions (you're, it's, here's)
- Cite sources naturally ("TMZ reports...", "According to Variety...")
- Make transitions smooth ("Meanwhile...", "In other news...", "And finally...")
- End each story with the "why it matters" angle
- NO hashtags or emojis in the narration
- Keep it conversational - this will be spoken aloud"""


NEWS_SCRIPT_USER_PROMPT = """Write a narration script for this {edition} news briefing.

DATE: {current_date}
THEME: {theme}
TARGET DURATION: {target_duration} seconds (~{target_words} words)
PROFILE: {profile_name} ({profile_handle})

STORIES TO COVER:
{stories_text}

Return a JSON object with this EXACT structure:
{{
  "hook": "<opening line, ~8 words, state number of stories>",
  "segments": [
    {{
      "text": "<full narration for this story, ~25-30 words>",
      "keywords": ["<2-word visual phrase>", "<2-word visual phrase>"]
    }}
  ],
  "cta": "Follow News But Quick to keep up to date with the latest news!"
}}

KEYWORD RULES for video search:
- Must be visual, 2-word phrases
- Good: "red carpet", "concert stage", "movie premiere", "streaming service"
- Bad: "Taylor Swift" (celebrity names), "breaking news" (too generic)
- Each story should have 2-3 keywords

Return ONLY valid JSON, no markdown code blocks."""


# =============================================================================
# News Script Planner
# =============================================================================

class NewsScriptPlanner(PipelineStep):
    """Plans video scripts from news briefs.

    Converts curated NewsBrief into VideoScript format compatible
    with the standard video pipeline.

    Note: This inherits from PipelineStep (not IScriptPlanner) because
    news scripts are generated from NewsBrief, not TopicInfo/ResearchResult.
    """

    # Timing constants
    WORDS_PER_MINUTE = 150
    HOOK_DURATION = 3.0
    CTA_DURATION = 3.0
    STORY_DURATION = 11.0  # ~11 seconds per story

    def __init__(
        self,
        target_duration: float = 60.0,
        text_provider: TextProvider | None = None,
        preferred_provider: str | None = None,
    ):
        """Initialize the news script planner.

        Args:
            target_duration: Target video duration in seconds.
            text_provider: AI text provider (created if None).
            preferred_provider: Preferred LLM provider name.
        """
        super().__init__("NewsScriptPlanner")
        self.target_duration = target_duration
        self._text_provider = text_provider
        self._preferred_provider = preferred_provider

    @property
    def text_provider(self) -> TextProvider:
        """Lazy-load the text provider."""
        if self._text_provider is None:
            self._text_provider = TextProvider(
                provider_override=self._preferred_provider
            )
        return self._text_provider

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute script planning from news brief.

        Expects context to have a `news_brief` attribute (NewsBrief).
        Falls back to standard research-based planning if no news brief.
        """
        # Check for news brief in context
        news_brief: NewsBrief | None = getattr(context, "news_brief", None)

        if not news_brief:
            raise ScriptPlanningError(
                "No news brief available. Use standard ScriptPlanner for non-news content."
            )

        self.log_start(f"Planning news script: {news_brief.edition.display_name}")

        try:
            script = await self.plan_from_brief(
                brief=news_brief,
                profile_name=context.profile.display_name or context.profile.name,
                profile_handle=context.profile.instagram_handle,
            )
            context.script = script

            self.log_success(
                f"News script planned: {len(script.segments)} segments, "
                f"{script.total_duration:.1f}s"
            )
            return context

        except Exception as e:
            self.log_error(f"News script planning failed: {e}")
            raise ScriptPlanningError(f"Failed to plan news script: {e}") from e

    async def plan_from_brief(
        self,
        brief: NewsBrief,
        profile_name: str = "News But Quick",
        profile_handle: str = "@news.but.quick",
    ) -> VideoScript:
        """Plan a video script from a news brief.

        Args:
            brief: Curated news brief with stories.
            profile_name: Profile display name for CTA.
            profile_handle: Instagram handle.

        Returns:
            VideoScript ready for video pipeline.
        """
        if not brief.stories:
            raise ScriptPlanningError("News brief has no stories")

        # Calculate target words based on duration
        target_words = int(self.target_duration / 60 * self.WORDS_PER_MINUTE)

        # Format stories for prompt
        stories_text = self._format_stories_for_prompt(brief.stories)

        # Build prompt
        prompt = NEWS_SCRIPT_USER_PROMPT.format(
            edition=brief.edition.display_name,
            current_date=brief.date.strftime("%B %d, %Y"),
            theme=brief.theme,
            target_duration=int(self.target_duration),
            target_words=target_words,
            profile_name=profile_name,
            profile_handle=profile_handle,
            stories_text=stories_text,
        )

        # Get provider info for logging
        providers = self.text_provider._get_providers()
        provider_name = providers[0][0] if providers else "unknown"
        provider_config = providers[0][1] if providers else None
        model_id = provider_config.litellm_model.split("/")[-1] if provider_config else "unknown"

        print(f"  [>] {provider_name}/{model_id} (news_script)...")
        start_time = time.time()

        # Generate script with AI
        try:
            response = await self.text_provider.generate(
                prompt=prompt,
                system=NEWS_SCRIPT_SYSTEM_PROMPT,
                task="news_script",
                temperature=0.7,
                max_tokens=1500,
            )

            duration_ms = int((time.time() - start_time) * 1000)
            actual_provider = self.text_provider._current_provider or provider_name
            actual_model = self.text_provider._current_model or model_id
            print(f"  [OK] {actual_provider}/{actual_model}: OK ({duration_ms}ms)")

            script = self._parse_script_response(response, brief)

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            print(f"  [X] {provider_name}: {str(e)[:60]}...")
            logger.warning(f"AI script generation failed: {e}, using fallback")
            script = self._fallback_script(brief, profile_handle)

        return script

    def _format_stories_for_prompt(self, stories: list[NewsStory]) -> str:
        """Format stories for the AI prompt."""
        lines = []
        for i, story in enumerate(stories, 1):
            lines.append(f"STORY {i}:")
            lines.append(f"  Headline: {story.headline}")
            lines.append(f"  Summary: {story.summary}")
            lines.append(f"  Why it matters: {story.why_it_matters}")
            lines.append(f"  Source: {story.source_name}")
            lines.append(f"  Category: {story.category.value}")
            lines.append("")
        return "\n".join(lines)

    def _parse_script_response(
        self,
        response: str,
        brief: NewsBrief,
    ) -> VideoScript:
        """Parse AI response into VideoScript."""
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse script JSON: {e}")
            logger.debug(f"Response: {response[:500]}")
            return self._fallback_script(brief, "@news.but.quick")

        # Extract components
        hook = data.get("hook", brief.get_hook_text())
        cta = "Follow News But Quick to keep up to date with the latest news!"
        segments_data = data.get("segments", [])

        # Build segments
        segments = []
        current_time = self.HOOK_DURATION

        for i, seg_data in enumerate(segments_data):
            text = seg_data.get("text", "")
            keywords = seg_data.get("keywords", [])[:4]

            # Estimate duration from word count
            word_count = len(text.split())
            duration = max(8.0, min(15.0, word_count / self.WORDS_PER_MINUTE * 60))

            segment = VideoSegment(
                index=i + 1,
                text=text,
                duration_seconds=duration,
                keywords=keywords,
                start_time=current_time,
                end_time=current_time + duration,
            )
            segments.append(segment)
            current_time += duration

        # Add CTA duration
        total_duration = current_time + self.CTA_DURATION

        # Build full narration
        narration_parts = [hook]
        narration_parts.extend(s.text for s in segments)
        narration_parts.append(cta)
        full_narration = " ".join(narration_parts)

        return VideoScript(
            title=brief.theme,
            hook=hook,
            segments=segments,
            cta=cta,
            total_duration=total_duration,
            full_narration=full_narration,
        )

    def _fallback_script(
        self,
        brief: NewsBrief,
        profile_handle: str,
    ) -> VideoScript:
        """Create a basic script when AI fails."""
        hook = brief.get_hook_text()
        segments = []
        current_time = self.HOOK_DURATION

        for i, story in enumerate(brief.stories):
            # Build narration from story components
            text = f"{story.headline}. {story.summary}"

            # Use story's visual keywords or generate generic ones
            keywords = story.visual_keywords[:3] if story.visual_keywords else [
                "news broadcast", "entertainment", "media coverage"
            ]

            word_count = len(text.split())
            duration = max(8.0, min(15.0, word_count / self.WORDS_PER_MINUTE * 60))

            segment = VideoSegment(
                index=i + 1,
                text=text,
                duration_seconds=duration,
                keywords=keywords,
                start_time=current_time,
                end_time=current_time + duration,
            )
            segments.append(segment)
            current_time += duration

        cta = "Follow News But Quick to keep up to date with the latest news!"
        total_duration = current_time + self.CTA_DURATION

        narration_parts = [hook]
        narration_parts.extend(s.text for s in segments)
        narration_parts.append(cta)

        return VideoScript(
            title=brief.theme,
            hook=hook,
            segments=segments,
            cta=cta,
            total_duration=total_duration,
            full_narration=" ".join(narration_parts),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

async def plan_news_script(
    brief: NewsBrief,
    target_duration: float = 60.0,
    profile_name: str = "News But Quick",
    profile_handle: str = "@news.but.quick",
) -> VideoScript:
    """Convenience function to plan a news script.

    Args:
        brief: Curated news brief.
        target_duration: Target video duration.
        profile_name: Profile display name.
        profile_handle: Instagram handle.

    Returns:
        VideoScript ready for pipeline.
    """
    planner = NewsScriptPlanner(target_duration=target_duration)
    return await planner.plan_from_brief(
        brief=brief,
        profile_name=profile_name,
        profile_handle=profile_handle,
    )
