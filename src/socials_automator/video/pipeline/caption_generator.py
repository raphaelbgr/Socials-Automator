"""Caption and hashtag generation for video reels.

Generates Instagram-style caption and hashtags for the video
based on the script content and topic.
"""

from pathlib import Path
from typing import Optional

from .base import (
    CaptionGenerationError,
    PipelineContext,
    PipelineStep,
)


class CaptionGenerator(PipelineStep):
    """Generates Instagram caption and hashtags for video reels.

    Note: This is different from ICaptionGenerator which is for narration.
    This generates the Instagram post caption and hashtags.
    """

    def __init__(self, ai_client: Optional[object] = None):
        """Initialize caption generator.

        Args:
            ai_client: Optional AI client for enhanced caption generation.
        """
        super().__init__("CaptionGenerator")
        self.ai_client = ai_client

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

            return context

        except Exception as e:
            self.log_error(f"Caption generation failed: {e}")
            raise CaptionGenerationError(f"Failed to generate caption: {e}") from e

    async def _generate_caption(
        self,
        context: PipelineContext,
    ) -> tuple[str, str]:
        """Generate caption and hashtags.

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (caption, hashtags).
        """
        # Use AI if available
        if self.ai_client:
            try:
                return await self._generate_with_ai(context)
            except Exception as e:
                self.log_progress(f"AI caption generation failed: {e}, using template")

        # Fallback to template
        return self._generate_with_template(context)

    async def _generate_with_ai(
        self,
        context: PipelineContext,
    ) -> tuple[str, str]:
        """Generate caption using AI.

        Args:
            context: Pipeline context.

        Returns:
            Tuple of (caption, hashtags).
        """
        import json

        profile_handle = f"@{context.profile.id}" if context.profile.id else ""
        profile_hashtag = f"#{context.profile.id.replace('.', '').replace('_', '').title()}" if context.profile.id else ""

        prompt = f"""Write an Instagram Reels caption and hashtags for this video:

Topic: {context.topic.topic}
Content Pillar: {context.topic.pillar_name}
Narration Summary: {context.script.hook} ... {context.script.cta}

Profile: {context.profile.display_name} ({profile_handle})

Requirements:
1. Caption should be 1-3 sentences, engaging and emoji-rich
2. Include a call-to-action (save, share, follow)
3. End with the profile hashtag: {profile_hashtag}
4. Generate 8-12 relevant hashtags (trending + niche)

Format your response as JSON:
{{
    "caption": "Your caption here with emojis and CTA, ending with {profile_hashtag}",
    "hashtags": "#hashtag1 #hashtag2 #hashtag3 ... (8-12 hashtags)"
}}

Example caption style:
"Stop scrolling! This AI hack will save you hours every week. Try it and thank me later! Save this for later and follow for more tips! {profile_hashtag}"

Respond with ONLY valid JSON, no other text."""

        response = await self.ai_client.generate(prompt)

        # Parse JSON
        try:
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            data = json.loads(clean_response)
            caption = data.get("caption", "")
            hashtags = data.get("hashtags", "")

            # Ensure profile hashtag is in caption
            if profile_hashtag and profile_hashtag not in caption:
                caption = caption.rstrip() + f" {profile_hashtag}"

            return caption, hashtags

        except json.JSONDecodeError as e:
            self.log_progress(f"Failed to parse AI response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")

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
        profile_hashtag = f"#{context.profile.id.replace('.', '').replace('_', '').title()}" if context.profile.id else ""

        # Template-based caption
        topic_short = context.topic.topic[:50] + "..." if len(context.topic.topic) > 50 else context.topic.topic

        caption = (
            f"This is exactly what you need to know about {topic_short}! "
            f"Save this for later and follow for more tips! "
            f"{profile_hashtag}"
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
