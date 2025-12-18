"""News curator that uses AI to rank, filter, and summarize articles."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

from socials_automator.news.models import (
    NewsArticle,
    NewsStory,
    NewsBrief,
    NewsEdition,
    NewsCategory,
    AggregationResult,
)
from socials_automator.providers.text import TextProvider

logger = logging.getLogger("ai_calls")


# =============================================================================
# Theme History (prevents repeating the same theme)
# =============================================================================

def get_theme_history_path(profile_name: str) -> Path:
    """Get path to theme history file for a profile."""
    from socials_automator.constants import get_profiles_dir
    return get_profiles_dir() / profile_name / "theme_history.json"


def load_theme_history(profile_name: str, cooldown_hours: int = 24) -> list[str]:
    """Load recent themes that should be avoided.

    Args:
        profile_name: Profile to load history for.
        cooldown_hours: How long themes stay in cooldown.

    Returns:
        List of recently used themes to avoid.
    """
    history_path = get_theme_history_path(profile_name)
    if not history_path.exists():
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            all_history = json.load(f)

        # Filter to recent themes within cooldown
        cutoff = datetime.now() - timedelta(hours=cooldown_hours)
        recent_themes = [
            entry["theme"]
            for entry in all_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]

        return recent_themes
    except Exception as e:
        logger.warning(f"Could not load theme history: {e}")
        return []


def save_theme_to_history(profile_name: str, theme: str) -> None:
    """Save a theme to history.

    Args:
        profile_name: Profile to save history for.
        theme: Theme string to save.
    """
    history_path = get_theme_history_path(profile_name)

    try:
        # Load existing history
        all_history = []
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                all_history = json.load(f)

        # Add new entry
        all_history.append({
            "theme": theme,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last 50 entries
        all_history = all_history[-50:]

        # Save
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(all_history, f, indent=2)

    except Exception as e:
        logger.warning(f"Could not save theme history: {e}")


# =============================================================================
# Prompts
# =============================================================================

CURATION_SYSTEM_PROMPT = """You are the editorial AI for @news.but.quick, an entertainment news account that delivers quick, useful news updates.

Your job is to:
1. Select the most relevant and interesting stories from a batch of news articles
2. Rank them by importance and usefulness to the audience
3. Rewrite headlines to be punchy and engaging
4. Summarize each story in 2-3 casual sentences
5. Add a "why it matters" angle for each story
6. Generate visual keywords for stock video search

IMPORTANT GUIDELINES:
- Prioritize USEFUL news: new releases, tour dates, streaming updates, price changes
- Balance categories: don't pick 5 celebrity gossip stories
- Avoid drama without substance - need facts, not just rumors
- Write casually but accurately - like texting a friend
- Keep summaries SHORT - this is for 60-second videos
- Visual keywords should be 2-word phrases for stock video (e.g., "concert crowd", "movie theater")

TONE: Casual, quick, slightly witty, but always accurate. Never gossipy or mean.

Example good summary:
"Taylor just dropped a surprise album at midnight and fans are losing it. 'The Tortured Poets Department' has 31 tracks including collabs with Post Malone. Why it matters: If you're a Swiftie, clear your weekend."

Example good visual keywords: ["recording studio", "album cover", "excited fans", "music streaming"]"""


CURATION_USER_PROMPT = """Here are today's entertainment news articles. Select the best {story_count} stories for a {edition} video briefing.

Current date: {current_date}
Edition theme: {edition_theme}

ARTICLES:
{articles_text}

Return a JSON object with this EXACT structure:
{{
  "stories": [
    {{
      "original_index": <index of article from list above>,
      "headline": "<punchy rewritten headline, max 12 words>",
      "summary": "<2-3 casual sentences summarizing the news>",
      "why_it_matters": "<1 sentence on why viewers should care>",
      "visual_keywords": ["<2-word phrase>", "<2-word phrase>", "<2-word phrase>"],
      "relevance_score": <0.0-1.0>,
      "virality_score": <0.0-1.0>,
      "usefulness_score": <0.0-1.0>
    }}
  ],
  "theme": "<brief theme for this edition, e.g., 'Today's entertainment buzz'>"
}}

SCORING GUIDE:
- relevance_score: How newsworthy/important is this? (major announcements = high)
- virality_score: How shareable/trending is this? (controversial or exciting = high)
- usefulness_score: How actionable for viewers? (release dates, recommendations = high)

Return ONLY valid JSON, no markdown code blocks or extra text."""


# =============================================================================
# Curator Class
# =============================================================================

@dataclass
class CurationConfig:
    """Configuration for news curation."""

    stories_per_brief: int = 4  # Default number of stories per video
    min_stories: int = 3
    max_stories: int = 5
    prefer_breaking: bool = True  # Prioritize articles < 6 hours old
    balance_categories: bool = True  # Try to include diverse categories
    provider_override: str | None = None  # Force specific AI provider
    profile_name: str | None = None  # Profile name for theme history
    theme_cooldown_hours: int = 24  # Hours before theme can be reused


class NewsCurator:
    """Curates news articles using AI for ranking and summarization.

    Usage:
        curator = NewsCurator()

        # From aggregation result
        aggregation = await aggregator.fetch()
        brief = await curator.curate(aggregation, edition=NewsEdition.MORNING)

        # Or from raw articles
        brief = await curator.curate_articles(
            articles=my_articles,
            edition=NewsEdition.EVENING,
            story_count=4,
        )
    """

    def __init__(
        self,
        text_provider: TextProvider | None = None,
        config: CurationConfig | None = None,
    ):
        """Initialize the curator.

        Args:
            text_provider: AI provider for text generation.
            config: Curation configuration.
        """
        self.config = config or CurationConfig()
        self._text_provider = text_provider

    @property
    def text_provider(self) -> TextProvider:
        """Lazy-load the text provider."""
        if self._text_provider is None:
            self._text_provider = TextProvider(
                provider_override=self.config.provider_override
            )
        return self._text_provider

    async def curate(
        self,
        aggregation: AggregationResult,
        edition: NewsEdition | None = None,
        story_count: int | None = None,
    ) -> NewsBrief:
        """Curate articles from an aggregation result.

        Args:
            aggregation: Result from NewsAggregator.fetch().
            edition: News edition (auto-detected from time if None).
            story_count: Number of stories to select.

        Returns:
            NewsBrief with curated stories.
        """
        return await self.curate_articles(
            articles=aggregation.articles,
            edition=edition,
            story_count=story_count,
            total_fetched=aggregation.total_articles,
        )

    async def curate_articles(
        self,
        articles: list[NewsArticle],
        edition: NewsEdition | None = None,
        story_count: int | None = None,
        total_fetched: int = 0,
    ) -> NewsBrief:
        """Curate a list of articles into a NewsBrief.

        Args:
            articles: List of NewsArticle objects.
            edition: News edition (auto-detected if None).
            story_count: Number of stories to select.
            total_fetched: Total articles fetched (for metadata).

        Returns:
            NewsBrief with curated stories.
        """
        start_time = time.time()

        # =================================================================
        # STEP 1: Determine Edition
        # =================================================================
        logger.info(">>> EDITION")
        if edition is None:
            edition = NewsEdition.from_hour(datetime.utcnow().hour)
            logger.info(f"  [AUTO] Detected edition: {edition.display_name}")
        else:
            logger.info(f"  [SET] Edition: {edition.display_name}")

        # Determine story count
        if story_count is None:
            story_count = self.config.stories_per_brief
        story_count = max(self.config.min_stories, min(self.config.max_stories, story_count))
        logger.info(f"  Target stories: {story_count}")

        # =================================================================
        # STEP 2: Pre-filter Articles
        # =================================================================
        logger.info(">>> FILTERING")
        logger.info(f"  Input: {len(articles)} articles")
        filtered_articles = self._prefilter_articles(articles)
        logger.info(f"  After filter: {len(filtered_articles)} articles")

        if len(filtered_articles) < story_count:
            logger.warning(
                f"  [WARN] Only {len(filtered_articles)} available, requested {story_count}"
            )
            story_count = max(1, len(filtered_articles))

        if not filtered_articles:
            logger.info("  [EMPTY] No articles to curate")
            return NewsBrief(
                edition=edition,
                date=date.today(),
                stories=[],
                theme="No news available",
                total_articles_fetched=total_fetched,
                total_articles_after_dedup=len(articles),
            )

        # Show category distribution
        category_counts = {}
        for a in filtered_articles:
            cat = a.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1
        logger.info(f"  Categories: {category_counts}")

        # =================================================================
        # STEP 3: AI Curation
        # =================================================================
        logger.info(">>> AI CURATION")
        logger.info(f"  Selecting {story_count} stories from {len(filtered_articles)} candidates...")
        stories = await self._ai_curate(
            articles=filtered_articles,
            edition=edition,
            story_count=story_count,
        )
        logger.info(f"  [OK] Selected {len(stories)} stories")

        # =================================================================
        # STEP 4: Generate Theme (with history check)
        # =================================================================
        logger.info(">>> THEME GENERATION")
        theme = self._generate_theme(edition, stories)
        logger.info(f"  [OK] Theme: {theme}")

        duration_ms = int((time.time() - start_time) * 1000)

        # =================================================================
        # COMPLETE
        # =================================================================
        brief = NewsBrief(
            edition=edition,
            date=date.today(),
            stories=stories,
            theme=theme,
            total_articles_fetched=total_fetched,
            total_articles_after_dedup=len(articles),
        )

        logger.info(
            f">>> COMPLETE | edition:{edition.value} | "
            f"stories:{len(stories)} | categories:{brief.categories_covered} | "
            f"{duration_ms}ms"
        )

        return brief

    def _prefilter_articles(self, articles: list[NewsArticle]) -> list[NewsArticle]:
        """Pre-filter and sort articles before AI curation.

        - Remove very old articles (> 48 hours)
        - Prioritize breaking news if configured
        - Ensure category diversity in the pool
        """
        # Filter out old articles
        filtered = [a for a in articles if a.age_hours <= 48]

        # Sort by recency, with breaking news first
        if self.config.prefer_breaking:
            filtered.sort(
                key=lambda a: (not a.is_breaking, a.age_hours)
            )
        else:
            filtered.sort(key=lambda a: a.age_hours)

        # Limit pool size for AI (don't send 100+ articles)
        max_pool = 30
        if len(filtered) > max_pool:
            # Keep top articles but ensure category diversity
            if self.config.balance_categories:
                filtered = self._balance_categories(filtered, max_pool)
            else:
                filtered = filtered[:max_pool]

        return filtered

    def _balance_categories(
        self,
        articles: list[NewsArticle],
        max_count: int,
    ) -> list[NewsArticle]:
        """Select articles with balanced category representation."""
        by_category: dict[NewsCategory, list[NewsArticle]] = {}
        for article in articles:
            if article.category not in by_category:
                by_category[article.category] = []
            by_category[article.category].append(article)

        # Round-robin selection from each category
        result: list[NewsArticle] = []
        categories = list(by_category.keys())
        idx = 0

        while len(result) < max_count and any(by_category.values()):
            cat = categories[idx % len(categories)]
            if by_category[cat]:
                result.append(by_category[cat].pop(0))
            idx += 1

            # Remove empty categories
            categories = [c for c in categories if by_category[c]]

        return result

    async def _ai_curate(
        self,
        articles: list[NewsArticle],
        edition: NewsEdition,
        story_count: int,
    ) -> list[NewsStory]:
        """Use AI to curate articles into stories."""
        # Format articles for the prompt
        articles_text = self._format_articles_for_prompt(articles)

        # Build prompt
        prompt = CURATION_USER_PROMPT.format(
            story_count=story_count,
            edition=edition.display_name,
            edition_theme=edition.theme,
            current_date=date.today().strftime("%B %d, %Y"),
            articles_text=articles_text,
        )

        # Call AI
        try:
            response = await self.text_provider.generate(
                prompt=prompt,
                system=CURATION_SYSTEM_PROMPT,
                task="news_curation",
                temperature=0.7,
                max_tokens=2000,
            )

            # Parse response
            stories = self._parse_curation_response(response, articles)

        except Exception as e:
            logger.error(f"AI curation failed: {e}")
            # Fallback: create basic stories from top articles
            stories = self._fallback_curate(articles[:story_count])

        return stories

    def _format_articles_for_prompt(self, articles: list[NewsArticle]) -> str:
        """Format articles as text for the AI prompt."""
        lines = []
        for i, article in enumerate(articles):
            age_str = f"{article.age_hours:.1f}h ago" if article.age_hours < 24 else f"{article.age_hours/24:.1f}d ago"
            breaking = " [BREAKING]" if article.is_breaking else ""

            lines.append(f"[{i}] {article.title}")
            lines.append(f"    Source: {article.source_name} | Category: {article.category.value} | {age_str}{breaking}")
            if article.summary:
                # Truncate long summaries
                summary = article.summary[:300] + "..." if len(article.summary) > 300 else article.summary
                lines.append(f"    Summary: {summary}")
            lines.append("")

        return "\n".join(lines)

    def _parse_curation_response(
        self,
        response: str,
        articles: list[NewsArticle],
    ) -> list[NewsStory]:
        """Parse AI response into NewsStory objects."""
        # Clean response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```"):
            # Remove code block markers
            lines = response.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return []

        stories = []
        for story_data in data.get("stories", []):
            try:
                # Get original article
                original_idx = story_data.get("original_index", 0)
                if 0 <= original_idx < len(articles):
                    original = articles[original_idx]
                else:
                    original = articles[0] if articles else None

                story = NewsStory(
                    headline=story_data.get("headline", ""),
                    summary=story_data.get("summary", ""),
                    why_it_matters=story_data.get("why_it_matters", ""),
                    source_name=original.source_name if original else "Unknown",
                    category=original.category if original else NewsCategory.GENERAL,
                    visual_keywords=story_data.get("visual_keywords", [])[:4],
                    original_url=original.article_url if original else "",
                    original_title=original.title if original else "",
                    published_at=original.published_at if original else None,
                    relevance_score=float(story_data.get("relevance_score", 0.5)),
                    virality_score=float(story_data.get("virality_score", 0.5)),
                    usefulness_score=float(story_data.get("usefulness_score", 0.5)),
                )

                # Validate story has required fields
                if story.headline and story.summary:
                    stories.append(story)

            except Exception as e:
                logger.warning(f"Failed to parse story: {e}")
                continue

        return stories

    def _fallback_curate(self, articles: list[NewsArticle]) -> list[NewsStory]:
        """Fallback curation when AI fails - create basic stories."""
        stories = []
        for article in articles:
            story = NewsStory(
                headline=article.title[:100],
                summary=article.summary[:200] if article.summary else article.title,
                why_it_matters="Stay informed on the latest entertainment news.",
                source_name=article.source_name,
                category=article.category,
                visual_keywords=["news broadcast", "entertainment", "media"],
                original_url=article.article_url,
                original_title=article.title,
                published_at=article.published_at,
                relevance_score=0.5,
                virality_score=0.5,
                usefulness_score=0.5,
            )
            stories.append(story)
        return stories

    def _generate_theme(
        self,
        edition: NewsEdition,
        stories: list[NewsStory],
    ) -> str:
        """Generate a theme string for the brief, avoiding recently used themes."""
        if not stories:
            return "No stories available"

        # Load recently used themes
        recent_themes: list[str] = []
        if self.config.profile_name:
            recent_themes = load_theme_history(
                self.config.profile_name,
                self.config.theme_cooldown_hours,
            )
            if recent_themes:
                logger.info(f"THEME_HISTORY | avoiding {len(recent_themes)} recent themes")

        # Check if there's a dominant category
        categories = [s.category for s in stories]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}

        # Sort categories by count (most common first)
        sorted_categories = sorted(category_counts.keys(), key=lambda c: -category_counts[c])

        category_themes = {
            NewsCategory.CELEBRITY: "Celebrity buzz",
            NewsCategory.MOVIES: "Movie news",
            NewsCategory.TV: "TV updates",
            NewsCategory.MUSIC: "Music news",
            NewsCategory.STREAMING: "Streaming updates",
            NewsCategory.VIRAL: "Viral moments",
            NewsCategory.GENERAL: "Entertainment news",
        }

        # Try each category theme, avoiding recent ones
        theme = None
        for cat in sorted_categories:
            candidate = f"{edition.display_name}: {category_themes[cat]}"
            if candidate not in recent_themes:
                theme = candidate
                break

        # If all category themes were recently used, add a unique suffix
        if theme is None:
            # Use most common category but add variation
            most_common = sorted_categories[0]
            base = category_themes[most_common]

            # Try different variations
            variations = [
                f"{edition.display_name}: {base} roundup",
                f"{edition.display_name}: Latest {base.lower()}",
                f"{edition.display_name}: {base} highlights",
                f"{edition.display_name}: Today's {base.lower()}",
            ]

            for variation in variations:
                if variation not in recent_themes:
                    theme = variation
                    break

            # Last resort: add timestamp to make unique
            if theme is None:
                theme = f"{edition.display_name}: {base} ({datetime.now().strftime('%H:%M')})"

        # Save theme to history
        if self.config.profile_name and theme:
            save_theme_to_history(self.config.profile_name, theme)
            logger.info(f"THEME_GENERATED | {theme}")

        return theme


# =============================================================================
# Module-level convenience functions
# =============================================================================

_default_curator: NewsCurator | None = None


def get_news_curator() -> NewsCurator:
    """Get or create the default news curator instance."""
    global _default_curator
    if _default_curator is None:
        _default_curator = NewsCurator()
    return _default_curator


async def curate_news(
    aggregation: AggregationResult,
    edition: NewsEdition | None = None,
    story_count: int = 4,
) -> NewsBrief:
    """Convenience function to curate news from aggregation result."""
    return await get_news_curator().curate(
        aggregation=aggregation,
        edition=edition,
        story_count=story_count,
    )
