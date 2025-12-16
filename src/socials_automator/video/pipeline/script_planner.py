"""Script planning for video content.

Plans a video script with:
- Hook (3 seconds)
- Main segments (each with narration and keywords)
- Call to action (3 seconds)

Duration is configurable via target_duration parameter.
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

    # Segment timing (fixed)
    HOOK_DURATION = 3.0
    CTA_DURATION = 4.0
    MIN_SEGMENT_DURATION = 10.0  # 10 seconds per segment = ~25 words each
    MAX_SEGMENT_DURATION = 12.0

    # Retry settings - UNLIMITED retries until success
    MAX_REGENERATION_ATTEMPTS = 50  # Keep retrying until we get valid content

    def __init__(self, ai_client: Optional[object] = None, target_duration: float = 60.0):
        """Initialize script planner.

        Args:
            ai_client: Optional AI client for enhanced script generation.
            target_duration: Target video duration in seconds (default 60).
        """
        super().__init__()
        self.ai_client = ai_client
        self.target_duration = target_duration

        # Calculate word requirements based on target duration
        # Narration should fill ~92% of video (some silence at start/end)
        self.min_narration_duration = target_duration * 0.92
        self.min_words_required = int(self.min_narration_duration / 60 * self.WORDS_PER_MINUTE)
        self.target_words = int(target_duration / 60 * self.WORDS_PER_MINUTE)

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
                self.target_duration,
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

        # Try generation with validation and regeneration
        last_word_count = 0
        last_script_text = ""  # Keep track of rejected script to show AI

        for attempt in range(self.MAX_REGENERATION_ATTEMPTS):
            script = None

            # Use AI if available for better script generation
            if self.ai_client:
                self.log_progress(f"[Attempt {attempt + 1}/{self.MAX_REGENERATION_ATTEMPTS}] Generating script...")
                try:
                    script = await self._plan_script_with_ai(
                        topic, research, duration, profile_name, profile_handle,
                        attempt_number=attempt + 1,
                        last_word_count=last_word_count,
                        last_script_text=last_script_text,
                    )
                except Exception as e:
                    self.log_progress(f"  -> AI error: {e}, retrying...")
                    continue

            if not script:
                continue

            # Validate script length
            is_valid, estimated_duration, word_count = self._validate_script_length(script)
            last_word_count = word_count
            last_script_text = script.full_narration

            if is_valid:
                self.log_progress(
                    f"  -> SUCCESS: {word_count} words (~{estimated_duration:.1f}s)"
                )
                return script

            # Script too short - log and retry
            self.log_progress(
                f"  -> REJECTED: {word_count} words (~{estimated_duration:.1f}s) - need {self.min_words_required}+ words"
            )

        # FAIL if script is still too short after all attempts
        raise ScriptPlanningError(
            f"FAILED after {self.MAX_REGENERATION_ATTEMPTS} attempts. "
            f"Best: {last_word_count} words (~{last_word_count / self.WORDS_PER_MINUTE * 60:.1f}s), need {self.min_words_required}+. "
            f"Try a different AI model."
        )

    def _validate_script_length(self, script: VideoScript) -> tuple[bool, float, int]:
        """Validate that script is long enough.

        Args:
            script: The video script to validate.

        Returns:
            Tuple of (is_valid, estimated_duration_seconds, word_count).
        """
        # Count total words in narration
        word_count = len(script.full_narration.split())

        # Estimate duration at speaking pace
        estimated_duration = (word_count / self.WORDS_PER_MINUTE) * 60

        # Check if meets minimum
        is_valid = estimated_duration >= self.min_narration_duration

        return is_valid, estimated_duration, word_count

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
        attempt_number: int = 1,
        last_word_count: int = 0,
        last_script_text: str = "",
    ) -> VideoScript:
        """Plan script using AI for better content."""
        import json

        # Build context from research
        key_points = "\n".join(f"- {p}" for p in research.key_points[:5])

        # Show the rejected script so AI knows what was wrong
        rejected_script_section = ""
        if last_script_text and last_word_count > 0:
            rejected_script_section = f"""
========== YOUR LAST SCRIPT WAS REJECTED ==========
Word count: {last_word_count} words (NEED {self.min_words_required}+)
Your rejected script:
\"\"\"{last_script_text}\"\"\"

THIS WAS TOO SHORT! You need to write MORE words per segment!
===================================================
"""

        # Progressive requirements - get stricter on each retry
        words_short = max(0, self.min_words_required - last_word_count)

        if attempt_number == 1:
            word_requirement = f"EXACTLY {self.target_words}-160 words TOTAL"
            segment_requirement = "MUST be 25-30 words each (3-4 sentences)"
            emphasis = f"""
CRITICAL: Total word count must be {self.min_words_required}+ words!
Hook: ~10 words | Each of 8 segments: ~18 words | CTA: ~12 words = ~{self.target_words} words total"""

        elif attempt_number <= 3:
            word_requirement = f"MINIMUM {self.min_words_required} words - you wrote {last_word_count}, need {words_short} MORE!"
            segment_requirement = "EXACTLY 30-35 words each (4-5 sentences per segment)"
            emphasis = f"""{rejected_script_section}
WRITE LONGER! Each segment needs 4-5 complete sentences with details and examples."""

        elif attempt_number <= 6:
            word_requirement = f"MINIMUM {self.min_words_required + 10} words - you are {words_short} words SHORT!"
            segment_requirement = "35-40 words each - MUCH LONGER segments needed"
            emphasis = f"""{rejected_script_section}
Your segments are WAY too short! Write 5-6 sentences per segment.
Include specific tool names, numbers, prices, and step-by-step instructions."""

        else:
            word_requirement = f"WRITE {self.min_words_required + 20}+ WORDS! You keep failing with only {last_word_count}!"
            segment_requirement = "40+ words per segment - WRITE COMPLETE PARAGRAPHS"
            emphasis = f"""{rejected_script_section}
#######################################################
# ATTEMPT {attempt_number}: YOU HAVE FAILED {attempt_number - 1} TIMES!  #
# Write a FULL PARAGRAPH for each segment!            #
#######################################################

Each segment MUST have:
- An opening sentence introducing the point
- 2-3 sentences explaining the details with specific examples
- A closing sentence with actionable advice
- TOTAL: 40+ words per segment"""

        prompt = f"""Write a 60-second video narration script for Instagram Reels about: {topic.topic}

Research findings:
{key_points}
{emphasis}

IMPORTANT: The total narration must be {word_requirement}.

Structure:
- Hook (3 seconds, ~10 words): Attention-grabbing opener that stops the scroll
- 8 segments (7-8 seconds each, ~18 words each): Valuable content with specific tips
- CTA (4 seconds, ~12 words): Call to action mentioning the creator

Format your response as JSON:
{{
    "hook": "Your hook here - must be exactly 8-12 words, punchy and attention-grabbing",
    "segments": [
        {{"text": "Segment 1 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 2 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 3 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 4 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 5 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 6 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 7 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}},
        {{"text": "Segment 8 narration - {segment_requirement}", "keywords": ["visual1", "visual2"]}}
    ],
    "cta_ending": "for more AI tips!"
}}

CRITICAL Rules:
- Each segment MUST be 15-20 words (2-3 sentences) - 8 segments total
- Total narration MUST be at least {self.min_words_required} words - COUNT YOUR WORDS!
- Write in first person, conversational style ("I discovered...", "Here's what works...")
- Include specific examples, numbers, or actionable tips in each segment
- NO repetition of the same phrases or ideas
- Each segment should deliver UNIQUE value, not repeat the topic
- DO NOT include "Follow" or any CTA in the segments! Only provide content in segments.
- The cta_ending should be a SHORT phrase like "for more AI tips!" (I will add "Follow {profile_name}" automatically)

NO COMMERCIAL LANGUAGE - We are NOT selling anything:
- NEVER mention courses, paid content, premium features, subscriptions, or purchases
- NEVER say "buy", "purchase", "enroll", "sign up for our course", "get access"
- NEVER mention prices, discounts, or limited offers
- Focus ONLY on free tips, free tools, and valuable free information
- The goal is to share knowledge, NOT to sell anything

PUNCTUATION Rules (CRITICAL for text-to-speech):
- NEVER use "..." or ".." - these cause awkward pauses in the voice!
- Each segment MUST be COMPLETE sentences ending with "." or "!" or "?"
- Do NOT cut off mid-thought. EVERY sentence must have a complete thought!
- BAD: "It's simple, it's effective, and best of all, it only..."
- GOOD: "It's simple, effective, and takes only 60 seconds to master."
- BAD: "The results are surprisingly." (INCOMPLETE - missing what they are!)
- GOOD: "The results are surprisingly accurate and save hours of work."
- BAD: "That's where Junia AI comes in handy. It takes your rough ideas..."
- GOOD: "That's where Junia AI comes in handy. It transforms your rough ideas into polished prompts instantly."
- For domain names, write them without spaces: "Jasper.ai" not "Jasper. ai"

KEYWORD Rules (for stock video search):
- Keywords must be VISUAL concepts that can be filmed (NOT abstract concepts)
- Use 2-word phrases for better search results (e.g., "person laptop", "cash money", "phone screen")
- Think: "What would I see in a stock video for this?"
- GOOD keywords: "laptop typing", "cash dollars", "office meeting", "phone scrolling", "robot hand"
- BAD keywords: "productivity", "success", "AI" (too abstract - use "person laptop", "celebration happy", "futuristic screen")

Example of GOOD segment (28 words) with GOOD keywords:
{{"text": "First, I use ChatGPT to write product descriptions for Etsy sellers. I charge fifty dollars each and complete about five per day. That's real, consistent money.", "keywords": ["laptop typing", "cash money", "ecommerce shopping"]}}

Example of BAD segment (too short, 10 words) with BAD keywords:
{{"text": "Use AI to make money online easily today.", "keywords": ["AI", "money", "online"]}}

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
        ai_segments = data.get("segments", [])[:8]
        segment_duration = content_duration / max(len(ai_segments), 1)

        segments = []
        for i, seg in enumerate(ai_segments):
            text = self._clean_narration_text(seg.get("text", f"Point {i+1}"))
            # Remove any CTA that AI might have included in segment text
            text = self._strip_cta_from_text(text, profile_name)
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

        # Build CTA from cta_ending (we add "Follow {profile_name}" ourselves)
        cta_ending = data.get("cta_ending", "for more tips like this!")
        # Clean up any placeholder text like "[context-specific ending...]"
        if "[" in cta_ending:
            cta_ending = "for more tips like this!"
        cta = f"Follow {profile_name} {cta_ending}"

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
        key_points = research.key_points[:8]  # Max 8 segments

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
        import re

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove ellipses - they cause awkward pauses in TTS
        # Replace "..." or ".." with proper sentence ending
        text = re.sub(r'\.{2,}', '.', text)

        # Remove extra whitespace
        text = " ".join(text.split())

        # Ensure it ends with proper punctuation (not ellipsis)
        text = text.rstrip('.')
        if text and text[-1] not in ".!?":
            text += "."

        # Limit length (about 30 words for 12 seconds)
        words = text.split()
        if len(words) > 35:
            text = " ".join(words[:35])
            # Ensure proper ending after truncation
            if text[-1] not in ".!?":
                text += "."

        return text

    def _strip_cta_from_text(self, text: str, profile_name: str) -> str:
        """Remove any CTA that AI might have included in segment text.

        Args:
            text: Segment text that might contain CTA.
            profile_name: Profile name to look for.

        Returns:
            Text with CTA removed.
        """
        import re

        # Remove "Follow [profile] ..." patterns
        patterns = [
            rf"Follow\s+{re.escape(profile_name)}[^.!?]*[.!?]?\s*",
            r"Follow\s+\w+\s+for\s+more[^.!?]*[.!?]?\s*",
            r"Follow\s+us\s+for[^.!?]*[.!?]?\s*",
            r"\[context-specific[^\]]*\]\.?\s*",
        ]

        for pattern in patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Clean up any resulting double spaces or trailing punctuation issues
        text = " ".join(text.split())
        text = text.strip()

        # Ensure it still ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."

        return text

    def _extract_video_keywords(self, text: str, topic: TopicInfo) -> list[str]:
        """Extract keywords for video search.

        Generates visual keywords suitable for stock video search (Pexels).

        Args:
            text: Segment text.
            topic: Topic information.

        Returns:
            List of keywords for Pexels search.
        """
        keywords = []
        text_lower = text.lower()

        # Expanded keyword mappings for better video matching
        # These are visual concepts that work well in stock video searches
        keyword_mappings = {
            # Tech-related
            "computer": ["person laptop", "typing computer", "screen code"],
            "laptop": ["laptop work", "laptop coffee", "laptop typing"],
            "phone": ["smartphone hand", "phone scrolling", "mobile app"],
            "code": ["coding screen", "programming", "developer laptop"],
            "software": ["software interface", "computer screen", "digital"],
            "app": ["mobile app", "phone screen", "digital interface"],
            "website": ["web design", "browser screen", "website laptop"],
            "automation": ["robot arm", "automation factory", "futuristic tech"],

            # AI-related
            "ai": ["artificial intelligence", "futuristic technology", "neural network"],
            "chatgpt": ["ai chat", "robot assistant", "futuristic screen"],
            "machine learning": ["data visualization", "neural network", "tech abstract"],
            "robot": ["robot hand", "humanoid robot", "futuristic"],

            # Work-related
            "work": ["office professional", "business meeting", "desk work"],
            "office": ["modern office", "workspace", "business professional"],
            "meeting": ["business meeting", "conference room", "team discussion"],
            "team": ["teamwork collaboration", "office team", "group work"],
            "boss": ["executive office", "business leader", "professional meeting"],

            # Money/Business
            "money": ["cash dollars", "finance success", "wealth business"],
            "dollar": ["dollar bills", "money cash", "finance"],
            "income": ["money growth", "passive income", "financial success"],
            "business": ["business success", "entrepreneur", "startup"],
            "sell": ["ecommerce", "online shopping", "sales success"],
            "client": ["client meeting", "handshake business", "customer service"],
            "freelance": ["freelancer laptop", "remote work", "home office"],

            # Learning/Education
            "learn": ["studying books", "education learning", "online course"],
            "course": ["online learning", "education screen", "e-learning"],
            "study": ["student studying", "books library", "learning focus"],
            "teach": ["teacher classroom", "education online", "learning"],

            # Creative
            "create": ["creative design", "artist work", "digital creation"],
            "design": ["graphic design", "creative workspace", "designer work"],
            "write": ["writing notebook", "typing keyboard", "content creation"],
            "content": ["content creator", "social media", "video creation"],
            "video": ["video editing", "camera recording", "content creation"],
            "image": ["photography camera", "image editing", "creative visual"],

            # Communication
            "email": ["email inbox", "typing keyboard", "business communication"],
            "message": ["messaging phone", "chat communication", "text message"],
            "social media": ["social media phone", "instagram scroll", "online engagement"],

            # Time/Productivity
            "time": ["clock time", "hourglass", "productivity schedule"],
            "fast": ["speed motion", "fast pace", "quick action"],
            "save": ["time saving", "efficiency", "productivity hack"],
            "productivity": ["productive workspace", "efficient work", "focus"],
            "schedule": ["calendar planning", "time management", "organization"],

            # General concepts
            "idea": ["lightbulb idea", "creative thinking", "innovation"],
            "success": ["success celebration", "achievement", "winning"],
            "growth": ["growth chart", "upward arrow", "progress"],
            "strategy": ["chess strategy", "planning board", "business plan"],
            "tip": ["helpful advice", "tutorial screen", "how to"],
            "hack": ["life hack", "productivity tips", "smart solution"],
        }

        # Find matching keywords from text
        for word, video_keywords in keyword_mappings.items():
            if word in text_lower:
                keywords.extend(video_keywords)

        # Add topic-based keywords as fallback
        if not keywords:
            keywords.extend(topic.keywords[:2])

        # Add generic tech keywords if still empty
        if not keywords:
            keywords.extend(["technology business", "digital abstract", "modern office"])

        # Deduplicate and limit
        seen = set()
        unique = []
        for k in keywords:
            k_lower = k.lower()
            if k_lower not in seen:
                seen.add(k_lower)
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
            Full narration string, cleaned for TTS.
        """
        import re

        parts = [hook]
        for segment in segments:
            parts.append(segment.text)
        parts.append(cta)

        narration = " ".join(parts)

        # Clean up for TTS - fix punctuation issues

        # Fix domain names with spaces (e.g., "Jasper. ai" -> "Jasper.ai")
        # Common TLDs that shouldn't have space before them
        tlds = ['ai', 'io', 'com', 'co', 'org', 'net', 'app', 'dev', 'me', 'us', 'uk']
        for tld in tlds:
            narration = re.sub(rf'\.\s+{tld}\b', f'.{tld}', narration, flags=re.IGNORECASE)

        # Replace multiple dots with single period
        narration = re.sub(r'\.{2,}', '.', narration)

        # Remove double spaces
        narration = re.sub(r'\s{2,}', ' ', narration)

        # Fix space before punctuation
        narration = re.sub(r'\s+([.!?,])', r'\1', narration)

        # Ensure space after punctuation (except at end or before TLDs)
        narration = re.sub(r'([.!?,])([A-Za-z])', r'\1 \2', narration)

        # Fix domain names again after the space insertion (in case it broke them)
        for tld in tlds:
            narration = re.sub(rf'\.\s+{tld}\b', f'.{tld}', narration, flags=re.IGNORECASE)

        return narration.strip()
