"""AI-powered research query generation for topic research.

Generates diverse, targeted search queries in multiple languages
to gather comprehensive information about a topic.

Usage:
    from socials_automator.video.pipeline.research_queries import ResearchQueryGenerator

    generator = ResearchQueryGenerator()
    queries = await generator.generate_queries(
        topic="This FREE AI tool creates perfect meeting notes",
        pillar="productivity_hacks",
    )
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from socials_automator.providers.text import TextProvider

logger = logging.getLogger("ai_calls")


def get_ai_version_context() -> str:
    """Get AI tool version context for accurate research queries.

    Returns:
        Formatted string with current AI tool versions.
    """
    try:
        from socials_automator.knowledge import get_ai_tools_registry
        registry = get_ai_tools_registry()
        return registry.get_version_context()
    except Exception as e:
        logger.debug(f"Could not load AI tools registry: {e}")
        # Fallback to hardcoded versions
        return """Current AI tool versions (use these for accurate queries):
- ChatGPT: GPT-5.2 (December 2025)
- Claude: Opus 4.5 (November 2025)
- Gemini: 3 Flash (December 2025)
- Midjourney: V7 (June 2025)
- Perplexity: Sonar (December 2025)
- ElevenLabs: Eleven v3 (2025)"""


@dataclass
class ResearchQuery:
    """A research query with metadata."""
    query: str
    language: str  # en, es, pt
    category: str  # tutorial, review, comparison, news, howto
    priority: int  # 1-3


QUERY_GENERATION_SYSTEM = """You are a research query optimizer for creating educational video content about AI tools.

Your job is to generate diverse, effective search queries that will find:
- Tutorials and how-to guides
- Reviews and comparisons
- Latest news and updates
- Practical tips and use cases
- User experiences and testimonials

Generate queries in MULTIPLE LANGUAGES to get diverse perspectives:
- English (en): Primary language, most queries
- Spanish (es): 2-3 queries for Latin American perspective
- Portuguese (pt): 1-2 queries for Brazilian perspective

Query categories:
- tutorial: How-to guides, step-by-step instructions
- review: Product reviews, user experiences
- comparison: Tool comparisons, alternatives
- news: Latest updates, announcements
- howto: Practical tips, use cases"""

QUERY_GENERATION_PROMPT = """Generate search queries to research this topic for a 60-second Instagram Reel:

TOPIC: {topic}
CONTENT PILLAR: {pillar}
CURRENT DATE: {current_date}

{version_context}

Generate 12-15 diverse search queries that will help create accurate, engaging content.

Requirements:
1. Mix of categories: tutorials (4), reviews (3), comparisons (2), news (2), howto (3)
2. Include 2-3 Spanish queries and 1-2 Portuguese queries
3. Use current year/month for freshness
4. Be specific - include tool names, features, use cases
5. Vary query structure (questions, phrases, comparisons)
6. Use CORRECT version numbers from the AI tool versions above (e.g., "GPT-5.2" not "GPT-4")

Return a JSON array:
[
  {{"query": "search query here", "language": "en", "category": "tutorial", "priority": 1}},
  {{"query": "mejores herramientas IA para reuniones 2025", "language": "es", "category": "review", "priority": 2}},
  ...
]

Return ONLY valid JSON, no markdown."""


class ResearchQueryGenerator:
    """Generates diverse research queries using AI.

    Creates targeted search queries in multiple languages and categories
    to gather comprehensive information about a topic.
    """

    # Default queries by category (fallback if AI fails)
    DEFAULT_CATEGORIES = {
        "tutorial": [
            "{topic} tutorial 2025",
            "how to use {topic}",
            "{topic} step by step guide",
            "{topic} for beginners",
        ],
        "review": [
            "{topic} review 2025",
            "{topic} honest review",
            "is {topic} worth it",
        ],
        "comparison": [
            "{topic} vs alternatives",
            "best {topic} tools compared",
        ],
        "news": [
            "{topic} latest news {month} {year}",
            "{topic} updates {year}",
        ],
        "howto": [
            "{topic} tips and tricks",
            "{topic} best practices",
            "{topic} productivity hacks",
        ],
    }

    # Spanish translations for common query patterns
    SPANISH_PATTERNS = [
        "mejores herramientas {topic} 2025",
        "{topic} tutorial en espanol",
        "como usar {topic} gratis",
    ]

    # Portuguese translations
    PORTUGUESE_PATTERNS = [
        "melhores ferramentas {topic} 2025",
        "{topic} como usar gratis",
    ]

    def __init__(self, text_provider: Optional[TextProvider] = None):
        """Initialize the generator.

        Args:
            text_provider: AI provider for query generation.
        """
        self._text_provider = text_provider

    @property
    def text_provider(self) -> TextProvider:
        """Lazy-load the text provider."""
        if self._text_provider is None:
            self._text_provider = TextProvider()
        return self._text_provider

    async def generate_queries(
        self,
        topic: str,
        pillar: str = "general",
        count: int = 15,
    ) -> list[ResearchQuery]:
        """Generate diverse research queries using AI.

        Args:
            topic: The topic to research.
            pillar: Content pillar (e.g., productivity_hacks, tool_tutorials).
            count: Target number of queries.

        Returns:
            List of ResearchQuery objects.
        """
        now = datetime.now()
        current_date = now.strftime("%B %d, %Y")

        # Get AI tool version context for accurate queries
        version_context = get_ai_version_context()

        prompt = QUERY_GENERATION_PROMPT.format(
            topic=topic,
            pillar=pillar,
            current_date=current_date,
            version_context=version_context,
        )

        try:
            response = await self.text_provider.generate(
                prompt=prompt,
                system=QUERY_GENERATION_SYSTEM,
                task="research_query_generation",
                temperature=0.7,
                max_tokens=2000,
            )

            queries = self._parse_response(response)
            if queries:
                logger.info(f"RESEARCH_QUERIES | generated {len(queries)} AI queries")
                return queries

        except Exception as e:
            logger.warning(f"RESEARCH_QUERIES | AI generation failed: {e}")

        # Fallback to template-based queries
        return self._generate_fallback_queries(topic, count)

    def _parse_response(self, response: str) -> list[ResearchQuery]:
        """Parse AI response into ResearchQuery objects."""
        # Clean response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"RESEARCH_QUERIES | JSON parse error: {e}")
            return []

        queries = []
        for item in data:
            if isinstance(item, dict) and "query" in item:
                queries.append(ResearchQuery(
                    query=item.get("query", ""),
                    language=item.get("language", "en"),
                    category=item.get("category", "general"),
                    priority=item.get("priority", 2),
                ))

        return queries

    def _generate_fallback_queries(self, topic: str, count: int) -> list[ResearchQuery]:
        """Generate fallback queries using templates.

        Args:
            topic: Topic to research.
            count: Target number of queries.

        Returns:
            List of template-based queries.
        """
        now = datetime.now()
        year = now.year
        month = now.strftime("%B")

        queries = []

        # Extract key terms from topic for query building
        topic_short = self._extract_key_terms(topic)

        # English queries by category
        for category, templates in self.DEFAULT_CATEGORIES.items():
            for template in templates:
                query_text = template.format(
                    topic=topic_short,
                    month=month,
                    year=year,
                )
                queries.append(ResearchQuery(
                    query=query_text,
                    language="en",
                    category=category,
                    priority=1 if category in ["tutorial", "howto"] else 2,
                ))

        # Spanish queries
        for template in self.SPANISH_PATTERNS:
            query_text = template.format(topic=topic_short)
            queries.append(ResearchQuery(
                query=query_text,
                language="es",
                category="review",
                priority=2,
            ))

        # Portuguese queries
        for template in self.PORTUGUESE_PATTERNS:
            query_text = template.format(topic=topic_short)
            queries.append(ResearchQuery(
                query=query_text,
                language="pt",
                category="review",
                priority=3,
            ))

        logger.info(f"RESEARCH_QUERIES | generated {len(queries)} fallback queries")
        return queries[:count]

    def _extract_key_terms(self, topic: str) -> str:
        """Extract key terms from a topic for query building.

        Args:
            topic: Full topic string.

        Returns:
            Shortened key terms.
        """
        # Remove common filler words
        fillers = [
            "this", "that", "the", "a", "an", "in", "on", "at", "to", "for",
            "with", "and", "or", "is", "are", "was", "were", "free", "best",
            "top", "amazing", "incredible", "perfect", "just", "how", "why",
        ]

        words = topic.lower().split()
        key_words = [w for w in words if w not in fillers and len(w) > 2]

        # Return first 4-5 key words
        return " ".join(key_words[:5])


# =============================================================================
# Convenience function
# =============================================================================

async def generate_research_queries(
    topic: str,
    pillar: str = "general",
    count: int = 15,
    text_provider: Optional[TextProvider] = None,
) -> list[str]:
    """Convenience function to generate query strings.

    Args:
        topic: Topic to research.
        pillar: Content pillar.
        count: Target number of queries.
        text_provider: Optional AI provider.

    Returns:
        List of query strings.
    """
    generator = ResearchQueryGenerator(text_provider=text_provider)
    queries = await generator.generate_queries(topic, pillar, count)
    return [q.query for q in queries]
