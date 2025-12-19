"""AI Tools Store with ChromaDB semantic search.

Provides semantic search capabilities for AI tools, allowing queries like:
- "Find tools similar to Notion"
- "Video editing tools for beginners"
- "Free alternatives to Midjourney"

Also tracks which tools have been covered in content to avoid repetition.

Usage:
    from socials_automator.knowledge.ai_tools_store import AIToolsStore

    store = AIToolsStore(profile_path)

    # Semantic search
    results = store.find_similar_tools("video editing like Runway")

    # Track tool coverage
    store.mark_tool_covered("notebooklm", post_id="2025-12-18-001")

    # Get uncovered hidden gems
    uncovered = store.get_uncovered_gems(days=30)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from socials_automator.knowledge.models import (
    AIToolRecord,
    AIToolUsageRecord,
)
from socials_automator.knowledge.ai_tools_registry import get_ai_tools_registry

logger = logging.getLogger("ai_calls")


class AIToolsStore:
    """Store for AI tools with semantic search and usage tracking.

    Provides:
    - ChromaDB semantic search for finding similar tools
    - Usage tracking to avoid covering the same tool repeatedly
    - Suggestions for uncovered tools with high content potential

    The store is profile-scoped, meaning each profile can track
    which tools it has covered independently.
    """

    def __init__(self, profile_path: Path):
        """Initialize the store for a profile.

        Args:
            profile_path: Path to the profile directory.
        """
        self.profile_path = profile_path
        self.knowledge_path = profile_path / "knowledge"
        self.knowledge_path.mkdir(parents=True, exist_ok=True)

        # Usage tracking file
        self.usage_path = self.knowledge_path / "ai_tools_usage.json"

        # ChromaDB (lazy loaded)
        self._chroma_client = None
        self._tools_collection = None
        self._indexed = False

    # =========================================================================
    # ChromaDB Setup
    # =========================================================================

    def _init_chromadb(self) -> bool:
        """Initialize ChromaDB for semantic search.

        Returns:
            True if ChromaDB is available, False otherwise.
        """
        if self._chroma_client is not None:
            return True

        try:
            import chromadb
            from chromadb.config import Settings

            chroma_path = self.knowledge_path / "chromadb_tools"
            chroma_path.mkdir(exist_ok=True)

            self._chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(chroma_path),
                anonymized_telemetry=False,
            ))

            self._tools_collection = self._chroma_client.get_or_create_collection(
                name="ai_tools",
                metadata={"hnsw:space": "cosine"},
            )

            logger.debug("AI_TOOLS_STORE | ChromaDB initialized")
            return True

        except ImportError:
            logger.warning("AI_TOOLS_STORE | ChromaDB not installed, semantic search disabled")
            return False

        except Exception as e:
            logger.error(f"AI_TOOLS_STORE | ChromaDB init failed: {e}")
            return False

    def _ensure_indexed(self) -> None:
        """Ensure tools are indexed in ChromaDB."""
        if self._indexed or not self._init_chromadb():
            return

        try:
            registry = get_ai_tools_registry()
            tools = registry.get_all_tools()

            # Check if already indexed (by checking count)
            existing_count = self._tools_collection.count()
            if existing_count >= len(tools):
                self._indexed = True
                return

            # Index all tools
            documents = []
            ids = []
            metadatas = []

            for tool in tools:
                # Create searchable document from tool info
                doc = self._tool_to_document(tool)
                documents.append(doc)
                ids.append(tool.id)
                metadatas.append({
                    "name": tool.name,
                    "category": tool.category,
                    "pricing": tool.pricing,
                    "hidden_gem": str(tool.hidden_gem),
                    "content_potential": tool.content_potential,
                })

            # Add to collection (upsert style - delete then add)
            try:
                self._tools_collection.delete(ids=ids)
            except Exception:
                pass  # May not exist yet

            self._tools_collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

            self._indexed = True
            logger.info(f"AI_TOOLS_STORE | Indexed {len(tools)} tools in ChromaDB")

        except Exception as e:
            logger.error(f"AI_TOOLS_STORE | Indexing failed: {e}")

    def _tool_to_document(self, tool: AIToolRecord) -> str:
        """Convert a tool to a searchable document string.

        Args:
            tool: Tool record to convert.

        Returns:
            Searchable text document.
        """
        parts = [
            tool.name,
            tool.company,
            tool.category,
            " ".join(tool.features),
            " ".join(tool.best_for),
        ]

        if tool.video_ideas:
            parts.append(" ".join(tool.video_ideas))

        return " ".join(parts)

    # =========================================================================
    # Semantic Search
    # =========================================================================

    def find_similar_tools(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
        min_similarity: float = 0.3,
    ) -> list[AIToolRecord]:
        """Find tools semantically similar to a query.

        Args:
            query: Natural language query (e.g., "video editing like Runway").
            limit: Maximum results to return.
            category: Filter to specific category.
            min_similarity: Minimum similarity threshold (0-1).

        Returns:
            List of matching tools sorted by relevance.
        """
        self._ensure_indexed()

        if self._tools_collection is None:
            # Fallback to registry search
            registry = get_ai_tools_registry()
            return registry.search(query, limit=limit)

        try:
            # Build where filter for category
            where_filter = None
            if category:
                where_filter = {"category": category}

            results = self._tools_collection.query(
                query_texts=[query],
                n_results=limit * 2,  # Get extra for filtering
                where=where_filter,
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            # Convert results to tools
            registry = get_ai_tools_registry()
            tools: list[AIToolRecord] = []

            for i, tool_id in enumerate(results["ids"][0]):
                # Check similarity threshold
                if results["distances"] and results["distances"][0]:
                    distance = results["distances"][0][i]
                    similarity = 1 - distance
                    if similarity < min_similarity:
                        continue

                tool = registry.get_by_id(tool_id)
                if tool:
                    tools.append(tool)

                if len(tools) >= limit:
                    break

            return tools

        except Exception as e:
            logger.error(f"AI_TOOLS_STORE | Semantic search failed: {e}")
            # Fallback
            registry = get_ai_tools_registry()
            return registry.search(query, limit=limit)

    def find_tools_for_topic(
        self,
        topic: str,
        limit: int = 3,
    ) -> list[AIToolRecord]:
        """Find relevant tools for a content topic.

        Args:
            topic: Content topic being created.
            limit: Maximum tools to suggest.

        Returns:
            List of relevant tools.
        """
        return self.find_similar_tools(topic, limit=limit)

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def _load_usage(self) -> list[AIToolUsageRecord]:
        """Load usage records from file."""
        if not self.usage_path.exists():
            return []

        try:
            with open(self.usage_path, encoding="utf-8") as f:
                data = json.load(f)
            return [AIToolUsageRecord(**r) for r in data]
        except Exception as e:
            logger.error(f"AI_TOOLS_STORE | Failed to load usage: {e}")
            return []

    def _save_usage(self, records: list[AIToolUsageRecord]) -> None:
        """Save usage records to file."""
        try:
            data = [r.model_dump() for r in records]
            with open(self.usage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"AI_TOOLS_STORE | Failed to save usage: {e}")

    def mark_tool_covered(
        self,
        tool_id: str,
        post_id: str,
        context: str = "main_topic",
    ) -> None:
        """Mark a tool as covered in content.

        Args:
            tool_id: ID of the tool.
            post_id: ID of the post/reel.
            context: How the tool was mentioned.
        """
        records = self._load_usage()

        record = AIToolUsageRecord(
            tool_id=tool_id,
            post_id=post_id,
            date_used=datetime.now().strftime("%Y-%m-%d"),
            context=context,
        )

        records.append(record)
        self._save_usage(records)

        logger.debug(f"AI_TOOLS_STORE | Marked {tool_id} as covered in {post_id}")

    def get_recently_covered(self, days: int = 30) -> list[str]:
        """Get tool IDs that were covered recently.

        Args:
            days: Number of days to look back.

        Returns:
            List of recently covered tool IDs.
        """
        records = self._load_usage()
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        covered = set()
        for record in records:
            if record.date_used >= cutoff:
                covered.add(record.tool_id)

        return list(covered)

    def get_tool_coverage(self, tool_id: str) -> list[AIToolUsageRecord]:
        """Get coverage history for a specific tool.

        Args:
            tool_id: ID of the tool.

        Returns:
            List of usage records for this tool.
        """
        records = self._load_usage()
        return [r for r in records if r.tool_id == tool_id]

    def was_recently_covered(self, tool_id: str, days: int = 30) -> bool:
        """Check if a tool was covered recently.

        Args:
            tool_id: ID of the tool.
            days: Number of days to consider "recent".

        Returns:
            True if tool was covered within the specified days.
        """
        return tool_id in self.get_recently_covered(days)

    # =========================================================================
    # Content Suggestions
    # =========================================================================

    def get_uncovered_gems(
        self,
        days: int = 30,
        limit: int = 10,
        high_potential_only: bool = True,
    ) -> list[AIToolRecord]:
        """Get hidden gems that haven't been covered recently.

        Args:
            days: Days to look back for coverage.
            limit: Maximum tools to return.
            high_potential_only: Only return high content potential gems.

        Returns:
            List of uncovered hidden gem tools.
        """
        registry = get_ai_tools_registry()
        covered = set(self.get_recently_covered(days))

        gems = registry.get_hidden_gems(
            limit=100,
            high_potential_only=high_potential_only,
        )

        uncovered = [g for g in gems if g.id not in covered]
        return uncovered[:limit]

    def suggest_next_topics(
        self,
        days: int = 30,
        limit: int = 10,
    ) -> list[str]:
        """Suggest video topics for uncovered tools.

        Args:
            days: Days to look back for coverage.
            limit: Maximum topics to return.

        Returns:
            List of video topic suggestions.
        """
        registry = get_ai_tools_registry()
        covered = set(self.get_recently_covered(days))

        topics: list[str] = []
        for tool in registry.get_tools_with_video_ideas():
            if tool.id not in covered:
                topics.extend(tool.video_ideas)
                if len(topics) >= limit:
                    break

        return topics[:limit]

    def get_coverage_stats(self, days: int = 30) -> dict:
        """Get statistics about tool coverage.

        Args:
            days: Days to look back.

        Returns:
            Dictionary with coverage statistics.
        """
        registry = get_ai_tools_registry()
        covered = set(self.get_recently_covered(days))
        all_tools = registry.get_all_tools()
        hidden_gems = registry.get_hidden_gems(limit=100)

        covered_gems = [g for g in hidden_gems if g.id in covered]
        uncovered_gems = [g for g in hidden_gems if g.id not in covered]

        return {
            "period_days": days,
            "total_tools": len(all_tools),
            "covered_tools": len(covered),
            "coverage_percent": round(len(covered) / len(all_tools) * 100, 1) if all_tools else 0,
            "hidden_gems": {
                "total": len(hidden_gems),
                "covered": len(covered_gems),
                "uncovered": len(uncovered_gems),
            },
            "recently_covered": list(covered),
        }


# Convenience functions

def get_ai_tools_store(profile_path: Path) -> AIToolsStore:
    """Create an AIToolsStore for a profile.

    Args:
        profile_path: Path to the profile directory.

    Returns:
        AIToolsStore instance.
    """
    return AIToolsStore(profile_path)
