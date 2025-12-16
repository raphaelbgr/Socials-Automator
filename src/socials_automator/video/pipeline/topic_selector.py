"""Topic selection from profile metadata.

Selects a topic based on:
- Content pillars and their frequency weights
- Trending keywords
- AI-driven topic generation
"""

import random
from typing import Optional

from .base import (
    ITopicSelector,
    PipelineContext,
    ProfileMetadata,
    TopicInfo,
    TopicSelectionError,
)


class TopicSelector(ITopicSelector):
    """Selects topics from profile content pillars and trending keywords."""

    def __init__(self, ai_client: Optional[object] = None):
        """Initialize topic selector.

        Args:
            ai_client: Optional AI client for enhanced topic generation.
        """
        super().__init__()
        self.ai_client = ai_client

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute topic selection step.

        Args:
            context: Pipeline context.

        Returns:
            Updated context with selected topic.
        """
        self.log_start("Selecting topic from profile...")

        try:
            topic = await self.select_topic(context.profile)
            context.topic = topic

            self.log_success(
                f"Selected topic: '{topic.topic}' "
                f"(pillar: {topic.pillar_name})"
            )
            return context

        except Exception as e:
            self.log_error(f"Topic selection failed: {e}")
            raise TopicSelectionError(f"Failed to select topic: {e}") from e

    async def select_topic(self, profile: ProfileMetadata) -> TopicInfo:
        """Select a topic based on profile configuration.

        Args:
            profile: Profile metadata.

        Returns:
            Selected topic information.
        """
        self.log_progress("Analyzing content pillars...")

        # Select a content pillar based on frequency weights
        pillar = self._select_pillar(profile.content_pillars)

        self.log_progress(f"Selected pillar: {pillar.get('name', 'Unknown')}")

        # Generate topic - use AI if available, otherwise use templates
        if self.ai_client:
            self.log_progress("Using AI to generate topic...")
            topic_text = await self._generate_topic_with_ai(pillar, profile)
        else:
            topic_text = self._generate_topic(pillar, profile.trending_keywords)

        self.log_progress(f"Generated topic: {topic_text}")

        # Build search queries for research
        search_queries = self._build_search_queries(topic_text, pillar)

        return TopicInfo(
            topic=topic_text,
            pillar_id=pillar.get("id", ""),
            pillar_name=pillar.get("name", ""),
            keywords=self._extract_keywords(topic_text, pillar),
            search_queries=search_queries,
        )

    async def _generate_topic_with_ai(
        self,
        pillar: dict,
        profile: ProfileMetadata,
    ) -> str:
        """Generate a topic using AI.

        Args:
            pillar: Selected content pillar.
            profile: Profile metadata.

        Returns:
            AI-generated topic string.
        """
        try:
            # Build prompt for AI
            examples = pillar.get("examples", [])
            examples_text = "\n".join(f"- {ex}" for ex in examples[:5]) if examples else "No examples"
            keywords = ", ".join(profile.trending_keywords[:10]) if profile.trending_keywords else "AI, productivity"

            prompt = f"""Generate ONE compelling video topic for a 60-second Instagram Reel.

Profile: {profile.display_name}
Niche: {profile.niche_id}
Content Pillar: {pillar.get('name', 'General')}
Pillar Description: {pillar.get('description', '')}

Example topics from this pillar:
{examples_text}

Trending keywords: {keywords}

Requirements:
- Topic must be specific and actionable (not vague)
- Should hook viewers in the first 3 seconds
- Must be completable in 60 seconds
- Use simple, clear language
- Make it feel urgent or valuable
- Focus on FREE tips, FREE tools, and sharing knowledge
- NO selling, NO courses, NO paid products - just free value!

Respond with ONLY the topic text, nothing else. No quotes, no explanation.
Example good responses:
- 5 ChatGPT prompts that save 2 hours daily
- The AI tool replacing Photoshop (free)
- How I automated my emails with Claude
- 3 free AI tools you need to try today"""

            # Use the TextProvider to generate
            response = await self.ai_client.generate(prompt)

            # Clean up response
            topic = response.strip().strip('"').strip("'")

            # Validate response
            if len(topic) < 10 or len(topic) > 100:
                self.log_progress("AI response too short/long, using template")
                return self._generate_topic(pillar, profile.trending_keywords)

            return topic

        except Exception as e:
            self.log_progress(f"AI generation failed: {e}, using template")
            return self._generate_topic(pillar, profile.trending_keywords)

    def _select_pillar(self, pillars: list[dict]) -> dict:
        """Select a content pillar based on frequency weights.

        Args:
            pillars: List of content pillars with frequency_percent.

        Returns:
            Selected pillar dictionary.
        """
        if not pillars:
            return {
                "id": "general",
                "name": "General",
                "description": "General content",
                "examples": [],
                "frequency_percent": 100,
            }

        # Create weighted selection
        weights = [p.get("frequency_percent", 10) for p in pillars]
        total = sum(weights)

        if total == 0:
            return random.choice(pillars)

        # Weighted random selection
        r = random.uniform(0, total)
        cumulative = 0
        for pillar, weight in zip(pillars, weights):
            cumulative += weight
            if r <= cumulative:
                return pillar

        return pillars[-1]

    def _generate_topic(
        self,
        pillar: dict,
        trending_keywords: list[str],
    ) -> str:
        """Generate a topic from pillar examples and trends.

        Args:
            pillar: Selected content pillar.
            trending_keywords: List of trending keywords.

        Returns:
            Generated topic string.
        """
        examples = pillar.get("examples", [])

        # Combine pillar examples with trending keywords for variety
        if examples and trending_keywords:
            # 70% chance to use example, 30% to use trending keyword
            if random.random() < 0.7 and examples:
                base_topic = random.choice(examples)
            else:
                keyword = random.choice(trending_keywords)
                base_topic = self._topic_from_keyword(keyword, pillar)
        elif examples:
            base_topic = random.choice(examples)
        elif trending_keywords:
            keyword = random.choice(trending_keywords)
            base_topic = self._topic_from_keyword(keyword, pillar)
        else:
            base_topic = f"Tips about {pillar.get('name', 'AI')}"

        return base_topic

    def _topic_from_keyword(self, keyword: str, pillar: dict) -> str:
        """Generate a topic from a trending keyword.

        Args:
            keyword: Trending keyword.
            pillar: Content pillar for context.

        Returns:
            Generated topic string.
        """
        pillar_id = pillar.get("id", "")

        templates = {
            "tool_tutorials": [
                f"How to use {keyword} effectively",
                f"Getting started with {keyword}",
                f"{keyword} tutorial for beginners",
            ],
            "productivity_hacks": [
                f"Boost productivity with {keyword}",
                f"Save time using {keyword}",
                f"5 {keyword} tips for productivity",
            ],
            "ai_money_making": [
                f"Make money with {keyword}",
                f"How to earn income using {keyword}",
                f"{keyword} side hustle ideas",
            ],
            "tool_comparisons": [
                f"Best {keyword} tools compared",
                f"{keyword} vs alternatives",
                f"Top {keyword} options in 2025",
            ],
            "ai_news_simplified": [
                f"What's new with {keyword}",
                f"{keyword} updates explained",
                f"Latest {keyword} features",
            ],
            "prompt_templates": [
                f"Best prompts for {keyword}",
                f"{keyword} prompt templates",
                f"Copy-paste {keyword} prompts",
            ],
        }

        pillar_templates = templates.get(pillar_id, [f"Everything about {keyword}"])
        return random.choice(pillar_templates)

    def _extract_keywords(self, topic: str, pillar: dict) -> list[str]:
        """Extract keywords from topic for video search.

        Args:
            topic: Topic string.
            pillar: Content pillar.

        Returns:
            List of keywords.
        """
        # Start with words from topic
        words = topic.lower().split()
        keywords = [w for w in words if len(w) > 3]

        # Add pillar-specific keywords
        pillar_keywords = {
            "tool_tutorials": ["technology", "software", "tutorial", "learning"],
            "productivity_hacks": ["productivity", "efficiency", "work", "office"],
            "ai_money_making": ["business", "money", "success", "entrepreneur"],
            "tool_comparisons": ["comparison", "technology", "review", "analysis"],
            "ai_news_simplified": ["news", "technology", "innovation", "future"],
            "prompt_templates": ["writing", "creative", "coding", "technology"],
        }

        pillar_id = pillar.get("id", "")
        keywords.extend(pillar_keywords.get(pillar_id, ["technology", "digital"]))

        # Deduplicate and limit
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)

        return unique[:10]

    def _build_search_queries(self, topic: str, pillar: dict) -> list[str]:
        """Build search queries for research phase.

        Args:
            topic: Selected topic.
            pillar: Content pillar.

        Returns:
            List of search queries.
        """
        queries = [topic]

        # Add variations
        queries.append(f"{topic} 2025")
        queries.append(f"{topic} tips")
        queries.append(f"{topic} tutorial")

        # Add pillar-specific queries
        pillar_id = pillar.get("id", "")
        if pillar_id == "ai_money_making":
            queries.append(f"{topic} income")
            queries.append(f"{topic} monetize")
        elif pillar_id == "productivity_hacks":
            queries.append(f"{topic} save time")
            queries.append(f"{topic} workflow")

        return queries[:5]
