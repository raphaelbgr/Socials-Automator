"""Script planning for video content.

Plans a 1-minute video script with:
- Hook (3 seconds)
- Main segments (each with narration and keywords)
- Call to action (3 seconds)
"""

from typing import Optional

from .base import (
    IScriptPlanner,
    PipelineContext,
    ResearchResult,
    ScriptPlanningError,
    TopicInfo,
    VideoScript,
    VideoSegment,
)


class ScriptPlanner(IScriptPlanner):
    """Plans video scripts from research results."""

    # Words per minute for narration (typical speaking pace)
    WORDS_PER_MINUTE = 150

    # Target video duration
    TARGET_DURATION = 60.0

    # Segment timing
    HOOK_DURATION = 3.0
    CTA_DURATION = 4.0
    MIN_SEGMENT_DURATION = 6.0
    MAX_SEGMENT_DURATION = 12.0

    def __init__(self, ai_client: Optional[object] = None):
        """Initialize script planner.

        Args:
            ai_client: Optional AI client for enhanced script generation.
        """
        super().__init__()
        self.ai_client = ai_client

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute script planning step.

        Args:
            context: Pipeline context with topic and research.

        Returns:
            Updated context with video script.
        """
        if not context.topic:
            raise ScriptPlanningError("No topic available for script planning")
        if not context.research:
            raise ScriptPlanningError("No research available for script planning")

        self.log_start(f"Planning script for: {context.topic.topic}")

        # Get profile info for CTA
        profile_name = context.profile.display_name or context.profile.name or "us"
        profile_handle = f"@{context.profile.id}" if context.profile.id else ""

        try:
            script = await self.plan_script(
                context.topic,
                context.research,
                self.TARGET_DURATION,
                profile_name=profile_name,
                profile_handle=profile_handle,
            )
            context.script = script

            self.log_success(
                f"Script planned: {len(script.segments)} segments, "
                f"{script.total_duration:.1f}s total"
            )
            return context

        except Exception as e:
            self.log_error(f"Script planning failed: {e}")
            raise ScriptPlanningError(f"Failed to plan script: {e}") from e

    async def plan_script(
        self,
        topic: TopicInfo,
        research: ResearchResult,
        duration: float = 60.0,
        profile_name: str = "us",
        profile_handle: str = "",
    ) -> VideoScript:
        """Plan a video script from research.

        Args:
            topic: Topic information.
            research: Research results.
            duration: Target video duration in seconds.
            profile_name: Profile display name for CTA.
            profile_handle: Instagram handle for CTA (e.g., @ai.for.mortals).

        Returns:
            Planned video script.
        """
        self.log_progress("Creating script structure...")

        # Use AI if available for better script generation
        if self.ai_client:
            self.log_progress("Using AI to generate script...")
            try:
                return await self._plan_script_with_ai(
                    topic, research, duration, profile_name, profile_handle
                )
            except Exception as e:
                self.log_progress(f"AI script generation failed: {e}, using templates")

        # Fallback to template-based generation
        return await self._plan_script_with_templates(
            topic, research, duration, profile_name, profile_handle
        )

    async def _plan_script_with_templates(
        self,
        topic: TopicInfo,
        research: ResearchResult,
        duration: float,
        profile_name: str = "us",
        profile_handle: str = "",
    ) -> VideoScript:
        """Plan script using templates (fallback method)."""
        # Calculate available time for main content
        content_duration = duration - self.HOOK_DURATION - self.CTA_DURATION

        # Generate hook
        hook = self._generate_hook(topic, research)
        self.log_progress(f"Hook: {hook[:50]}...")

        # Generate segments from key points
        segments = self._generate_segments(
            topic,
            research,
            content_duration,
        )
        self.log_progress(f"Generated {len(segments)} segments")

        # Generate CTA with profile name
        cta = self._generate_cta(topic, profile_name, profile_handle)
        self.log_progress(f"CTA: {cta[:50]}...")

        # Build full narration
        full_narration = self._build_full_narration(hook, segments, cta)

        # Create script
        script = VideoScript(
            title=self._generate_title(topic),
            hook=hook,
            segments=segments,
            cta=cta,
            total_duration=duration,
            full_narration=full_narration,
        )

        # Calculate timing
        script.calculate_times()

        return script

    async def _plan_script_with_ai(
        self,
        topic: TopicInfo,
        research: ResearchResult,
        duration: float,
        profile_name: str = "us",
        profile_handle: str = "",
    ) -> VideoScript:
        """Plan script using AI for better content."""
        import json

        # Build context from research
        key_points = "\n".join(f"- {p}" for p in research.key_points[:5])

        # Build CTA instruction with profile name (no handle - sounds weird in narration)
        cta_instruction = f'Follow {profile_name} [context-specific ending like "for more AI tips!"]'

        prompt = f"""Write a 60-second video narration script for Instagram Reels about: {topic.topic}

Research findings:
{key_points}

IMPORTANT: The total narration must be approximately 150 words (for 60 seconds at natural speaking pace).

Structure:
- Hook (3 seconds, ~8 words): Attention-grabbing opener
- 5 segments (10 seconds each, ~25 words each): Valuable content
- CTA (4 seconds, ~10 words): Call to action mentioning the creator

Format your response as JSON:
{{
    "hook": "Your hook here - must be exactly 8-10 words, punchy and attention-grabbing",
    "segments": [
        {{"text": "Segment 1 narration - MUST be 20-30 words with specific details and value", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 2 narration - MUST be 20-30 words with specific details and value", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 3 narration - MUST be 20-30 words with specific details and value", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 4 narration - MUST be 20-30 words with specific details and value", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 5 narration - MUST be 20-30 words with specific details and value", "keywords": ["visual1", "visual2"]}}
    ],
    "cta": "{cta_instruction}",
    "cta_context": "A short phrase relevant to this video topic (e.g., 'for more productivity tips!')"
}}

CRITICAL Rules:
- Each segment MUST be 20-30 words (NOT shorter!) - this is essential for timing
- Write in first person, conversational style ("I discovered...", "Here's what works...")
- Include specific examples, numbers, or actionable tips in each segment
- Keywords must be visual concepts for stock video (e.g., "laptop typing", "money cash", "phone screen")
- NO repetition of the same phrases or ideas
- Each segment should deliver UNIQUE value, not repeat the topic
- Write complete sentences that flow naturally when spoken aloud
- The CTA must mention "{profile_name}" and end with a context-relevant phrase

Example of GOOD segment (25 words):
"First, I use ChatGPT to write product descriptions for Etsy sellers. I charge fifty dollars each and complete five per day. That's real money."

Example of BAD segment (too short, 10 words):
"Use AI to make money online easily today."

Respond with ONLY valid JSON, no other text."""

        response = await self.ai_client.generate(prompt)

        # Parse JSON response
        try:
            # Clean response (remove markdown code blocks if present)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = clean_response.split("```")[1]
                if clean_response.startswith("json"):
                    clean_response = clean_response[4:]
            clean_response = clean_response.strip()

            data = json.loads(clean_response)
        except json.JSONDecodeError as e:
            self.log_progress(f"Failed to parse AI response as JSON: {e}")
            raise ValueError(f"Invalid JSON response: {e}")

        # Build segments from AI response
        content_duration = duration - self.HOOK_DURATION - self.CTA_DURATION
        ai_segments = data.get("segments", [])[:5]
        segment_duration = content_duration / max(len(ai_segments), 1)

        segments = []
        for i, seg in enumerate(ai_segments):
            text = self._clean_narration_text(seg.get("text", f"Point {i+1}"))
            keywords = seg.get("keywords", topic.keywords[:3])

            segments.append(VideoSegment(
                index=i + 1,
                text=text,
                duration_seconds=segment_duration,
                keywords=keywords[:5],
            ))

        # Ensure we have at least 3 segments
        if len(segments) < 3:
            self.log_progress("AI returned too few segments, using templates")
            raise ValueError("Too few segments returned")

        hook = data.get("hook", self._generate_hook(topic, research))
        cta = data.get("cta", self._generate_cta(topic, profile_name, profile_handle))

        self.log_progress(f"AI Hook: {hook[:50]}...")
        self.log_progress(f"AI generated {len(segments)} segments")
        self.log_progress(f"AI CTA: {cta[:50]}...")

        # Build full narration
        full_narration = self._build_full_narration(hook, segments, cta)

        # Create script
        script = VideoScript(
            title=self._generate_title(topic),
            hook=hook,
            segments=segments,
            cta=cta,
            total_duration=duration,
            full_narration=full_narration,
        )

        # Calculate timing
        script.calculate_times()

        return script

    def _generate_hook(self, topic: TopicInfo, research: ResearchResult) -> str:
        """Generate attention-grabbing hook.

        Args:
            topic: Topic information.
            research: Research results.

        Returns:
            Hook text (3 seconds worth).
        """
        # Hook templates based on pillar
        templates = {
            "tool_tutorials": [
                f"Stop! Here's the {topic.topic} trick nobody tells you.",
                f"Want to master {topic.topic}? Watch this.",
                f"I tested {topic.topic}. Here's what actually works.",
            ],
            "productivity_hacks": [
                f"Save 2 hours daily with this {topic.topic} hack.",
                f"You're doing {topic.topic} wrong. Here's the fix.",
                f"The {topic.topic} secret that changed everything.",
            ],
            "ai_money_making": [
                f"Here's how people are making money with {topic.topic}.",
                f"Turn {topic.topic} into income. Here's how.",
                f"The {topic.topic} side hustle nobody talks about.",
            ],
            "default": [
                f"Everything you need to know about {topic.topic}.",
                f"Here's the truth about {topic.topic}.",
                f"Let me show you {topic.topic} in 60 seconds.",
            ],
        }

        pillar_templates = templates.get(topic.pillar_id, templates["default"])

        import random
        hook = random.choice(pillar_templates)

        # Ensure hook is short enough (about 8-10 words for 3 seconds)
        words = hook.split()
        if len(words) > 12:
            hook = " ".join(words[:12])

        return hook

    def _generate_segments(
        self,
        topic: TopicInfo,
        research: ResearchResult,
        available_duration: float,
    ) -> list[VideoSegment]:
        """Generate video segments from research.

        Args:
            topic: Topic information.
            research: Research results.
            available_duration: Time available for segments.

        Returns:
            List of video segments.
        """
        segments = []
        key_points = research.key_points[:5]  # Max 5 segments

        if not key_points:
            # Create default segments
            key_points = [
                f"First, understand what {topic.topic} really is",
                f"The key benefit of {topic.topic}",
                f"How to get started with {topic.topic}",
                f"Common mistakes to avoid",
                f"Pro tips for {topic.topic}",
            ]

        # Calculate segment durations
        num_segments = len(key_points)
        base_duration = available_duration / num_segments

        # Clamp duration
        segment_duration = max(
            self.MIN_SEGMENT_DURATION,
            min(base_duration, self.MAX_SEGMENT_DURATION)
        )

        # Adjust if total exceeds available
        total_segment_time = segment_duration * num_segments
        if total_segment_time > available_duration:
            segment_duration = available_duration / num_segments

        for i, point in enumerate(key_points):
            # Clean up the point text
            text = self._clean_narration_text(point)

            # Generate keywords for video search
            keywords = self._extract_video_keywords(text, topic)

            segment = VideoSegment(
                index=i + 1,
                text=text,
                duration_seconds=segment_duration,
                keywords=keywords,
            )
            segments.append(segment)

        return segments

    def _generate_cta(
        self,
        topic: TopicInfo,
        profile_name: str = "us",
        profile_handle: str = "",
    ) -> str:
        """Generate call-to-action with profile name.

        Args:
            topic: Topic information.
            profile_name: Profile display name.
            profile_handle: Instagram handle (not used in narration - sounds weird).

        Returns:
            CTA text with profile mention (display name only, no @ handle).
        """
        # Context-aware CTA endings based on topic/pillar
        cta_contexts = {
            "tool_tutorials": "for more AI tool tutorials!",
            "productivity_hacks": "for more productivity tips that actually work!",
            "ai_money_making": "for more ways to make money with AI!",
            "default": "for more tips like this!",
        }

        context_cta = cta_contexts.get(topic.pillar_id, cta_contexts["default"])

        # Format: "Follow [Profile Name] [context CTA]"
        # Note: Don't include @handle in narration - it sounds unnatural when spoken
        return f"Follow {profile_name} {context_cta}"

    def _generate_title(self, topic: TopicInfo) -> str:
        """Generate video title.

        Args:
            topic: Topic information.

        Returns:
            Video title.
        """
        # Clean up topic for title
        title = topic.topic
        if not title[0].isupper():
            title = title.capitalize()
        return title

    def _clean_narration_text(self, text: str) -> str:
        """Clean text for narration.

        Args:
            text: Raw text.

        Returns:
            Cleaned text suitable for narration.
        """
        # Remove URLs
        import re
        text = re.sub(r'http\S+', '', text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Ensure it ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."

        # Limit length (about 25 words for 10 seconds)
        words = text.split()
        if len(words) > 30:
            text = " ".join(words[:30]) + "..."

        return text

    def _extract_video_keywords(self, text: str, topic: TopicInfo) -> list[str]:
        """Extract keywords for video search.

        Args:
            text: Segment text.
            topic: Topic information.

        Returns:
            List of keywords for Pexels search.
        """
        # Base keywords from topic
        keywords = list(topic.keywords[:3])

        # Add contextual keywords based on text content
        text_lower = text.lower()

        keyword_mappings = {
            "computer": ["computer", "laptop", "technology"],
            "phone": ["smartphone", "mobile", "device"],
            "work": ["office", "business", "professional"],
            "money": ["finance", "business", "success"],
            "learn": ["education", "study", "learning"],
            "create": ["creative", "design", "art"],
            "code": ["programming", "coding", "developer"],
            "write": ["writing", "typing", "content"],
            "automate": ["automation", "robot", "technology"],
            "ai": ["artificial intelligence", "technology", "futuristic"],
            "chat": ["communication", "conversation", "technology"],
            "fast": ["speed", "efficiency", "motion"],
            "save": ["time", "efficiency", "productivity"],
        }

        for word, video_keywords in keyword_mappings.items():
            if word in text_lower:
                keywords.extend(video_keywords[:2])
                break

        # Ensure we have at least some keywords
        if len(keywords) < 2:
            keywords.extend(["technology", "abstract", "digital"])

        # Deduplicate and limit
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)

        return unique[:5]

    def _build_full_narration(
        self,
        hook: str,
        segments: list[VideoSegment],
        cta: str,
    ) -> str:
        """Build complete narration text.

        Args:
            hook: Hook text.
            segments: Video segments.
            cta: Call to action.

        Returns:
            Full narration string.
        """
        parts = [hook]
        for segment in segments:
            parts.append(segment.text)
        parts.append(cta)

        return " ".join(parts)
