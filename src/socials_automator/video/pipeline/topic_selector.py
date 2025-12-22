"""Topic selection from profile metadata.

Selects a topic based on:
- Content pillars and their frequency weights
- Trending keywords
- AI-driven topic generation
- Hidden gems from AI tools registry
- Topic history to avoid repetition
"""

import json
import logging
import random
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .base import (
    ITopicSelector,
    PipelineContext,
    ProfileMetadata,
    TopicInfo,
    TopicSelectionError,
)

logger = logging.getLogger("ai_calls")


# =============================================================================
# Topic History (prevents repeating the same topics)
# =============================================================================

def get_topic_history_path(profile_path: Path) -> Path:
    """Get path to topic history file for a profile."""
    return profile_path / "reel_topic_history.json"


def load_topic_history(profile_path: Path, hours: int = 72) -> list[str]:
    """Load recently used topics.

    Args:
        profile_path: Path to profile directory.
        hours: How far back to look (default 72h = 3 days).

    Returns:
        List of recent topic strings.
    """
    history_path = get_topic_history_path(profile_path)
    if not history_path.exists():
        return []

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            all_history = json.load(f)

        # Filter to recent topics
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []

        for entry in all_history:
            # Skip entries with missing or invalid timestamps
            timestamp = entry.get("timestamp")
            if not timestamp or not isinstance(timestamp, str):
                continue

            try:
                entry_time = datetime.fromisoformat(timestamp)
                if entry_time > cutoff:
                    recent.append(entry["topic"])
            except (ValueError, TypeError):
                # Skip entries with malformed timestamps
                continue

        return recent
    except Exception as e:
        logger.warning(f"Could not load topic history: {e}")
        return []


def save_topic_to_history(profile_path: Path, topic: str) -> None:
    """Save a used topic to history.

    Args:
        profile_path: Path to profile directory.
        topic: Topic string that was used.
    """
    history_path = get_topic_history_path(profile_path)

    try:
        # Load existing history
        all_history = []
        if history_path.exists():
            with open(history_path, "r", encoding="utf-8") as f:
                all_history = json.load(f)

        # Add new entry
        all_history.append({
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
        })

        # Keep only last 200 entries
        all_history = all_history[-200:]

        # Save
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(all_history, f, indent=2)

        logger.info(f"TOPIC_HISTORY | saved topic | total={len(all_history)}")

    except Exception as e:
        logger.warning(f"Could not save topic history: {e}")


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
            topic = await self.select_topic(context.profile, context.profile_path)
            context.topic = topic

            self.log_success(
                f"Selected topic: '{topic.topic}' "
                f"(pillar: {topic.pillar_name})"
            )
            return context

        except Exception as e:
            self.log_error(f"Topic selection failed: {e}")
            raise TopicSelectionError(f"Failed to select topic: {e}") from e

    async def select_topic(
        self,
        profile: ProfileMetadata,
        profile_path: Optional[Path] = None,
    ) -> TopicInfo:
        """Select a topic based on profile configuration.

        Args:
            profile: Profile metadata.
            profile_path: Optional path to profile directory for AI tools filtering.

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
            topic_text = await self._generate_topic_with_ai(pillar, profile, profile_path)
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
        profile_path: Optional[Path] = None,
    ) -> str:
        """Generate a topic using AI.

        Integrates with AIToolsRegistry for accurate version info and hidden gems.
        Avoids recently used topics from history.

        Args:
            pillar: Selected content pillar.
            profile: Profile metadata.
            profile_path: Optional path to profile directory for filtering covered tools.

        Returns:
            AI-generated topic string.
        """
        try:
            # Build prompt for AI
            examples = pillar.get("examples", [])
            examples_text = "\n".join(f"- {ex}" for ex in examples[:5]) if examples else "No examples"
            keywords = ", ".join(profile.trending_keywords[:10]) if profile.trending_keywords else "AI, productivity"

            # Get current date for context
            now = datetime.now()
            current_date = now.strftime("%B %d, %Y")  # e.g., "December 16, 2025"
            current_year = now.year

            # Get version context and hidden gems from registry/store
            version_context = self._get_version_context()
            hidden_gems_context = self._get_hidden_gems_context(profile_path)

            # Get topic history to avoid repetition
            topic_history_context = self._get_topic_history_context(profile_path)

            prompt = f"""Generate ONE compelling video topic for a 60-second Instagram Reel.

TODAY'S DATE: {current_date}

{version_context}

Profile: {profile.display_name}
Niche: {profile.niche_id}
Content Pillar: {pillar.get('name', 'General')}
Pillar Description: {pillar.get('description', '')}

Example topics from this pillar:
{examples_text}

Trending keywords: {keywords}
{hidden_gems_context}
{topic_history_context}

Requirements:
- Topic must be specific and actionable (not vague)
- Should hook viewers in the first 3 seconds
- Must be completable in 60 seconds
- Use simple, clear language
- Make it feel urgent or valuable
- Focus on FREE tips, FREE tools, and sharing knowledge
- NO selling, NO courses, NO paid products - just free value!
- When mentioning AI tools, reference their CURRENT versions (listed above)
- Content should feel fresh and up-to-date for {current_year}
- PRIORITIZE hidden gem tools for unique content that stands out!
- CRITICAL: Do NOT repeat or rephrase any topic from the "RECENTLY USED" list above!

Respond with ONLY the topic text, nothing else. No quotes, no explanation.
Example good responses:
- 5 ChatGPT prompts that save 2 hours daily
- Claude Opus 4.5 just changed everything
- This AI tool replaces 5 apps (NotebookLM)
- 3 free AI tools you need to try in {current_year}"""

            # System prompt for topic generation
            system_prompt = f"""You are a creative content strategist for social media video content.
Your role is to generate compelling, specific, and actionable video topics.

Profile: {profile.display_name}
Niche: {profile.niche_id}
Style: Short-form educational content (60-second reels)

Guidelines:
- Be specific and actionable (not vague)
- Use current AI tool versions and terminology
- Focus on free value, not selling
- Make topics urgent and valuable
- Avoid repeating recently used topics"""

            # Use the TextProvider to generate with system context
            response = await self.ai_client.generate(
                prompt=prompt,
                system=system_prompt,
                task="topic_generation",
            )

            # Clean up response
            topic = response.strip().strip('"').strip("'")

            # Validate response
            if len(topic) < 10 or len(topic) > 100:
                self.log_detail("AI response too short/long, using template")
                return self._generate_topic(pillar, profile.trending_keywords)

            # Save topic to history
            if profile_path:
                save_topic_to_history(profile_path, topic)

            return topic

        except Exception as e:
            self.log_detail(f"AI generation failed: {e}, using template")
            return self._generate_topic(pillar, profile.trending_keywords)

    def _get_topic_history_context(self, profile_path: Optional[Path] = None) -> str:
        """Get topic history context to avoid repetition.

        Args:
            profile_path: Optional path to profile directory.

        Returns:
            Formatted string with recently used topics.
        """
        if not profile_path:
            return ""

        recent_topics = load_topic_history(profile_path, hours=72)
        if not recent_topics:
            return ""

        # Format for prompt (show last 20 topics)
        topics_text = "\n".join(f"- {t}" for t in recent_topics[-20:])
        return f"""

RECENTLY USED TOPICS (DO NOT REPEAT OR REPHRASE THESE):
{topics_text}

Generate a COMPLETELY DIFFERENT topic that we haven't covered recently."""

    def _get_version_context(self) -> str:
        """Get AI tool version context from registry.

        Returns:
            Formatted string with current AI tool versions.
        """
        try:
            from socials_automator.knowledge import get_ai_tools_registry
            registry = get_ai_tools_registry()
            return registry.get_version_context()
        except Exception as e:
            logger.debug(f"Could not load AI tools registry for versions: {e}")
            # Fallback to basic versions
            return """AI TOOL VERSIONS (verify before publishing):
- ChatGPT: GPT-4o | Claude: Opus 4.5 | Gemini: Gemini 1.5 Pro
- Midjourney: V7 | DALL-E: 3 | Sora | Runway: Gen-3
NOTE: Always verify current versions before creating content."""

    def _get_hidden_gems_context(self, profile_path: Optional[Path] = None) -> str:
        """Get hidden gems suggestions, filtering out recently covered tools.

        Args:
            profile_path: Optional path to profile directory for filtering.

        Returns:
            Formatted string with hidden gem tool suggestions.
        """
        try:
            # If we have profile_path, use the store to filter recently covered tools
            if profile_path:
                from socials_automator.knowledge import get_ai_tools_store
                store = get_ai_tools_store(profile_path)

                # Get uncovered hidden gems (filters out tools covered in last 14 days)
                gems = store.get_uncovered_gems(days=14, limit=5, high_potential_only=True)
            else:
                # Fallback to registry without filtering
                from socials_automator.knowledge import get_ai_tools_registry
                registry = get_ai_tools_registry()
                gems = registry.get_hidden_gems(limit=5, high_potential_only=True)

            if not gems:
                return ""

            gems_list = []
            for tool in gems:
                gem_info = f"- {tool.name}: {', '.join(tool.features[:2])}"
                if tool.video_ideas:
                    gem_info += f" (Idea: {tool.video_ideas[0]})"
                gems_list.append(gem_info)

            return "\n\nHIDDEN GEM AI TOOLS (lesser-known, great for unique content):\n" + "\n".join(gems_list)

        except Exception as e:
            logger.debug(f"Could not load hidden gems: {e}")
            return ""

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

    def _get_ai_tool_versions(self) -> dict[str, str]:
        """Get AI tool versions from registry.

        Returns:
            Dictionary mapping tool name variations to their current version.
        """
        try:
            from socials_automator.knowledge import get_ai_tools_registry
            registry = get_ai_tools_registry()

            versions = {}
            for tool in registry.get_all_tools():
                # Add primary name
                versions[tool.name.lower()] = f"{tool.name} {tool.current_version}"
                # Add tool ID
                versions[tool.id.lower()] = f"{tool.name} {tool.current_version}"
                # Add company name for major tools
                if tool.company.lower() in ["openai", "anthropic", "google", "meta"]:
                    versions[tool.company.lower()] = f"{tool.name} {tool.current_version}"

            return versions

        except Exception as e:
            logger.debug(f"Could not load tool versions from registry: {e}")
            # Fallback to hardcoded versions
            return {
                "chatgpt": "ChatGPT GPT-4o",
                "claude": "Claude Opus 4.5",
                "gemini": "Gemini 1.5 Pro",
                "midjourney": "Midjourney V7",
                "dall-e": "DALL-E 3",
                "sora": "Sora",
                "runway": "Runway Gen-3",
            }

    def _build_search_queries(self, topic: str, pillar: dict) -> list[str]:
        """Build search queries for research phase.

        Includes version-specific queries for AI tools and current date.

        Args:
            topic: Selected topic.
            pillar: Content pillar.

        Returns:
            List of search queries.
        """
        now = datetime.now()
        current_year = now.year
        current_month = now.strftime("%B")  # e.g., "December"

        queries = [topic]

        # Add current year/month variations
        queries.append(f"{topic} {current_year}")
        queries.append(f"{topic} {current_month} {current_year}")

        # Detect AI tools in topic and add version-specific queries
        topic_lower = topic.lower()
        tool_versions = self._get_ai_tool_versions()
        for tool_name, version in tool_versions.items():
            if tool_name in topic_lower:
                # Add version-specific search
                queries.append(f"{version} features {current_year}")
                queries.append(f"{version} latest updates {current_month} {current_year}")
                break  # Only add for first matched tool

        # Add general variations
        queries.append(f"{topic} tips")
        queries.append(f"{topic} tutorial {current_year}")

        # Add pillar-specific queries
        pillar_id = pillar.get("id", "")
        if pillar_id == "ai_money_making":
            queries.append(f"{topic} income {current_year}")
            queries.append(f"{topic} monetize")
        elif pillar_id == "productivity_hacks":
            queries.append(f"{topic} save time")
            queries.append(f"{topic} workflow {current_year}")
        elif pillar_id == "ai_news_simplified":
            queries.append(f"{topic} news {current_month} {current_year}")
            queries.append(f"{topic} announcement {current_year}")

        # Deduplicate while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        return unique_queries[:6]  # Return up to 6 queries
