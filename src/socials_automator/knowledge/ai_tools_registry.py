"""AI Tools Registry for loading and managing AI tools database.

This module provides the AIToolsRegistry class which loads the YAML configuration
and provides convenient methods for accessing tools based on various criteria
like category, content potential, hidden gems, etc.

Usage:
    from socials_automator.knowledge.ai_tools_registry import get_ai_tools_registry

    registry = get_ai_tools_registry()

    # Get hidden gems for unique content
    gems = registry.get_hidden_gems(limit=10)

    # Get tools by category
    video_tools = registry.get_by_category("video")

    # Search tools
    results = registry.search("presentation")

    # Get video topic suggestions
    topics = registry.suggest_video_topics(limit=20)
"""

from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from socials_automator.knowledge.models import (
    AIToolsConfig,
    AIToolRecord,
    AIToolCategory,
)

logger = logging.getLogger("ai_calls")


class AIToolsRegistry:
    """Registry for managing AI tools database.

    Loads configuration from YAML and provides methods for:
    - Getting tools by category, pricing, content potential
    - Finding hidden gems for unique content
    - Searching tools by name or features
    - Suggesting video topics based on tools
    - Version verification for content generation

    Example:
        registry = AIToolsRegistry.load()

        # Get high-potential hidden gems
        gems = registry.get_hidden_gems(high_potential_only=True)

        # Get tools for a video about research
        research_tools = registry.get_by_category("research")

        # Get version info for a specific tool
        tool = registry.get_by_id("chatgpt")
        print(f"Current version: {tool.current_version}")
    """

    # Default path for config file
    DEFAULT_CONFIG_PATH = Path("config/ai_tools.yaml")

    def __init__(self, config: AIToolsConfig):
        """Initialize with configuration.

        Args:
            config: Parsed configuration object.
        """
        self._config = config
        self._loaded_at = datetime.utcnow()

        # Build category lookup
        self._categories_by_id: dict[str, AIToolCategory] = {
            cat.id: cat for cat in config.categories
        }

        # Build tool lookup
        self._tools_by_id: dict[str, AIToolRecord] = {
            tool.id: tool for tool in config.tools
        }

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AIToolsRegistry":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML config. Uses default if None.

        Returns:
            AIToolsRegistry instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If YAML is invalid.
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH

        # Make path absolute if relative
        if not path.is_absolute():
            # Try relative to current directory
            if not path.exists():
                # Try relative to project root
                project_root = Path(__file__).parents[3]  # src/socials_automator/knowledge -> project root
                path = project_root / path

        if not path.exists():
            raise FileNotFoundError(f"AI tools config not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}")

        try:
            config = AIToolsConfig(**data)
        except Exception as e:
            raise ValueError(f"Invalid config structure in {path}: {e}")

        logger.info(
            f"AI_TOOLS_REGISTRY | loaded={path.name} | "
            f"tools={config.total_tools} | "
            f"hidden_gems={config.hidden_gems_count} | "
            f"high_potential={config.high_potential_count}"
        )

        return cls(config)

    @property
    def config(self) -> AIToolsConfig:
        """Get the underlying configuration."""
        return self._config

    # =========================================================================
    # Tool Access Methods
    # =========================================================================

    def get_all_tools(self) -> list[AIToolRecord]:
        """Get all tools in the database.

        Returns:
            List of all tool records.
        """
        return list(self._config.tools)

    def get_by_id(self, tool_id: str) -> Optional[AIToolRecord]:
        """Get a tool by its ID.

        Args:
            tool_id: Tool ID (e.g., "chatgpt", "midjourney").

        Returns:
            AIToolRecord if found, None otherwise.
        """
        return self._tools_by_id.get(tool_id)

    def get_by_category(self, category: str) -> list[AIToolRecord]:
        """Get all tools in a category.

        Args:
            category: Category ID (e.g., "video", "research", "coding").

        Returns:
            List of tools in that category.
        """
        return [t for t in self._config.tools if t.category == category]

    def get_by_pricing(self, pricing: str) -> list[AIToolRecord]:
        """Get tools by pricing model.

        Args:
            pricing: Pricing type (free, freemium, paid, enterprise).

        Returns:
            List of tools with that pricing.
        """
        return [t for t in self._config.tools if t.pricing == pricing]

    def get_free_tools(self) -> list[AIToolRecord]:
        """Get tools with free tiers (free or freemium).

        Returns:
            List of free/freemium tools.
        """
        return [t for t in self._config.tools if t.is_free]

    # =========================================================================
    # Content Discovery Methods
    # =========================================================================

    def get_hidden_gems(
        self,
        limit: int = 10,
        high_potential_only: bool = False,
        category: Optional[str] = None,
    ) -> list[AIToolRecord]:
        """Get hidden gem tools for unique content.

        Args:
            limit: Maximum tools to return.
            high_potential_only: Only return high content potential gems.
            category: Filter to specific category.

        Returns:
            List of hidden gem tools.
        """
        gems = [t for t in self._config.tools if t.hidden_gem]

        if high_potential_only:
            gems = [t for t in gems if t.content_potential == "high"]

        if category:
            gems = [t for t in gems if t.category == category]

        # Sort by content score (high first), then by name
        gems.sort(key=lambda t: (-t.content_score, t.name))

        return gems[:limit]

    def get_high_potential_tools(
        self,
        limit: int = 20,
        category: Optional[str] = None,
    ) -> list[AIToolRecord]:
        """Get tools with high content potential.

        Args:
            limit: Maximum tools to return.
            category: Filter to specific category.

        Returns:
            List of high potential tools.
        """
        tools = [t for t in self._config.tools if t.content_potential == "high"]

        if category:
            tools = [t for t in tools if t.category == category]

        return tools[:limit]

    def get_tools_with_video_ideas(self) -> list[AIToolRecord]:
        """Get tools that have specific video ideas.

        Returns:
            List of tools with video_ideas populated.
        """
        return [t for t in self._config.tools if t.has_video_ideas]

    def suggest_video_topics(
        self,
        limit: int = 20,
        prioritize_gems: bool = True,
        category: Optional[str] = None,
    ) -> list[str]:
        """Get video topic suggestions from tools database.

        Args:
            limit: Maximum topics to return.
            prioritize_gems: Put hidden gem topics first.
            category: Filter to specific category.

        Returns:
            List of video topic strings.
        """
        topics: list[tuple[str, bool, int]] = []  # (topic, is_gem, content_score)

        for tool in self._config.tools:
            if category and tool.category != category:
                continue

            for idea in tool.video_ideas:
                topics.append((idea, tool.hidden_gem, tool.content_score))

        # Sort: gems first (if prioritized), then by content score
        if prioritize_gems:
            topics.sort(key=lambda x: (-int(x[1]), -x[2]))
        else:
            topics.sort(key=lambda x: -x[2])

        return [t[0] for t in topics[:limit]]

    # =========================================================================
    # Search Methods
    # =========================================================================

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[AIToolRecord]:
        """Search tools by name, features, or best_for.

        Args:
            query: Search query (case-insensitive).
            limit: Maximum results.

        Returns:
            List of matching tools.
        """
        query_lower = query.lower()
        results: list[tuple[int, AIToolRecord]] = []

        for tool in self._config.tools:
            score = 0

            # Name match (highest priority)
            if query_lower in tool.name.lower():
                score += 100
            if query_lower == tool.name.lower():
                score += 50

            # Company match
            if query_lower in tool.company.lower():
                score += 30

            # Category match
            if query_lower in tool.category.lower():
                score += 20

            # Features match
            for feature in tool.features:
                if query_lower in feature.lower():
                    score += 10

            # Best for match
            for use_case in tool.best_for:
                if query_lower in use_case.lower():
                    score += 10

            if score > 0:
                results.append((score, tool))

        # Sort by score (descending)
        results.sort(key=lambda x: -x[0])

        return [tool for _, tool in results[:limit]]

    def find_alternatives(
        self,
        tool_id: str,
        limit: int = 5,
    ) -> list[AIToolRecord]:
        """Find alternative tools in the same category.

        Args:
            tool_id: ID of the tool to find alternatives for.
            limit: Maximum alternatives to return.

        Returns:
            List of alternative tools.
        """
        tool = self.get_by_id(tool_id)
        if not tool:
            return []

        alternatives = [
            t for t in self._config.tools
            if t.category == tool.category and t.id != tool_id
        ]

        # Sort by content potential (high first)
        alternatives.sort(key=lambda t: -t.content_score)

        return alternatives[:limit]

    # =========================================================================
    # Version Methods
    # =========================================================================

    def get_version_context(self, tool_ids: Optional[list[str]] = None) -> str:
        """Get version context string for AI prompts.

        Args:
            tool_ids: Specific tools to include. If None, includes major tools.

        Returns:
            Formatted string with version information.
        """
        if tool_ids is None:
            # Default to major tools
            tool_ids = [
                "chatgpt", "claude", "gemini", "midjourney", "sora",
                "elevenlabs", "runway", "cursor", "perplexity"
            ]

        lines = ["Current AI tool versions (as of " + self._config.last_updated + "):"]

        for tool_id in tool_ids:
            tool = self.get_by_id(tool_id)
            if tool:
                lines.append(
                    f"- {tool.name}: {tool.current_version} ({tool.version_date})"
                )

        lines.append("")
        lines.append("Always verify versions before creating content about specific tools.")

        return "\n".join(lines)

    def get_version_check_notes(self) -> str:
        """Get version checking instructions for content generation.

        Returns:
            Instructions string for AI prompts.
        """
        return self._config.version_check_notes

    # =========================================================================
    # Category Methods
    # =========================================================================

    def get_all_categories(self) -> list[AIToolCategory]:
        """Get all categories.

        Returns:
            List of category records.
        """
        return list(self._config.categories)

    def get_category(self, category_id: str) -> Optional[AIToolCategory]:
        """Get a category by ID.

        Args:
            category_id: Category ID.

        Returns:
            AIToolCategory if found, None otherwise.
        """
        return self._categories_by_id.get(category_id)

    def get_category_stats(self) -> dict[str, int]:
        """Get tool counts by category.

        Returns:
            Dictionary of category_id -> tool count.
        """
        stats: dict[str, int] = {}
        for tool in self._config.tools:
            stats[tool.category] = stats.get(tool.category, 0) + 1
        return stats

    # =========================================================================
    # Random Selection
    # =========================================================================

    def random_hidden_gem(self, category: Optional[str] = None) -> Optional[AIToolRecord]:
        """Get a random hidden gem tool.

        Args:
            category: Filter to specific category.

        Returns:
            Random hidden gem tool or None.
        """
        gems = self.get_hidden_gems(limit=100, category=category)
        return random.choice(gems) if gems else None

    def random_video_topic(self, category: Optional[str] = None) -> Optional[str]:
        """Get a random video topic suggestion.

        Args:
            category: Filter to specific category.

        Returns:
            Random video topic string or None.
        """
        topics = self.suggest_video_topics(limit=100, category=category)
        return random.choice(topics) if topics else None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_summary(self) -> dict:
        """Get a summary of the database.

        Returns:
            Dictionary with counts and statistics.
        """
        category_stats = self.get_category_stats()

        pricing_stats: dict[str, int] = {}
        for tool in self._config.tools:
            pricing_stats[tool.pricing] = pricing_stats.get(tool.pricing, 0) + 1

        return {
            "version": self._config.version,
            "last_updated": self._config.last_updated,
            "loaded_at": self._loaded_at.isoformat(),
            "totals": {
                "tools": self._config.total_tools,
                "categories": len(self._config.categories),
                "hidden_gems": self._config.hidden_gems_count,
                "high_potential": self._config.high_potential_count,
                "with_video_ideas": len(self.get_tools_with_video_ideas()),
            },
            "by_category": category_stats,
            "by_pricing": pricing_stats,
        }


# Module-level cached instance
_registry_instance: Optional[AIToolsRegistry] = None


def get_ai_tools_registry(reload: bool = False) -> AIToolsRegistry:
    """Get or create the default AIToolsRegistry instance.

    Args:
        reload: If True, reload from YAML even if already loaded.

    Returns:
        AIToolsRegistry instance.
    """
    global _registry_instance

    if _registry_instance is None or reload:
        _registry_instance = AIToolsRegistry.load()

    return _registry_instance


def reset_ai_tools_registry() -> None:
    """Reset the cached registry instance."""
    global _registry_instance
    _registry_instance = None
