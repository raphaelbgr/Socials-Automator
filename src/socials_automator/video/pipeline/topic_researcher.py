"""Topic research using web search.

Searches the web for information about the selected topic
and extracts key points for the video script.
"""

from datetime import datetime
from typing import Optional

from .base import (
    ITopicResearcher,
    PipelineContext,
    ResearchError,
    ResearchResult,
    TopicInfo,
)


class TopicResearcher(ITopicResearcher):
    """Researches topics using web search and AI summarization."""

    def __init__(
        self,
        search_client: Optional[object] = None,
        ai_client: Optional[object] = None,
        max_results: int = 5,
    ):
        """Initialize topic researcher.

        Args:
            search_client: Web search client (e.g., DuckDuckGo).
            ai_client: AI client for summarization.
            max_results: Maximum search results to process.
        """
        super().__init__()
        self.search_client = search_client
        self.ai_client = ai_client
        self.max_results = max_results

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute research step.

        Args:
            context: Pipeline context with topic.

        Returns:
            Updated context with research results.
        """
        if not context.topic:
            raise ResearchError("No topic selected for research")

        self.log_start(f"Researching topic: {context.topic.topic}")

        try:
            research = await self.research(context.topic)
            context.research = research

            self.log_success(
                f"Research complete: {len(research.key_points)} key points found"
            )
            return context

        except Exception as e:
            self.log_error(f"Research failed: {e}")
            raise ResearchError(f"Failed to research topic: {e}") from e

    async def research(self, topic: TopicInfo) -> ResearchResult:
        """Research a topic using web search.

        Args:
            topic: Topic information with search queries.

        Returns:
            Research results with summary and key points.
        """
        self.log_progress("Searching the web...")

        all_results = []
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B")

        # Search for each query
        for query in topic.search_queries[:3]:
            self.log_detail(f"Searching: {query}")
            results = await self._web_search(query)
            all_results.extend(results)

        # Always add a general AI tool versions search for context
        version_query = f"ChatGPT Claude Gemini latest version {current_month} {current_year}"
        self.log_detail(f"Version context search: {version_query}")
        results = await self._web_search(version_query)
        all_results.extend(results)

        self.log_progress(f"Found {len(all_results)} results")

        # Extract content from results
        raw_content = self._extract_content(all_results)

        self.log_progress("Summarizing content...")

        # Summarize and extract key points
        summary, key_points = await self._summarize_content(
            topic.topic,
            raw_content,
        )

        return ResearchResult(
            topic=topic.topic,
            summary=summary,
            key_points=key_points,
            sources=[
                {"title": r.get("title", ""), "url": r.get("href", "")}
                for r in all_results[:5]
            ],
            raw_content=raw_content[:5000],  # Limit size
        )

    async def _web_search(self, query: str) -> list[dict]:
        """Perform web search.

        Args:
            query: Search query.

        Returns:
            List of search results.
        """
        # Try DuckDuckGo search
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                return results

        except ImportError:
            self.log_detail("DuckDuckGo not available, using fallback")
            return self._fallback_search(query)

        except Exception as e:
            self.log_detail(f"Search error: {e}, using fallback")
            return self._fallback_search(query)

    def _fallback_search(self, query: str) -> list[dict]:
        """Fallback search when DuckDuckGo is not available.

        Args:
            query: Search query.

        Returns:
            Minimal search results based on query.
        """
        # Return placeholder results based on common AI topics
        return [
            {
                "title": f"Guide to {query}",
                "body": f"Learn about {query} with practical tips and examples.",
                "href": "",
            }
        ]

    def _extract_content(self, results: list[dict]) -> str:
        """Extract text content from search results.

        Args:
            results: List of search results.

        Returns:
            Combined text content.
        """
        content_parts = []

        for result in results:
            title = result.get("title", "")
            body = result.get("body", "")
            if title or body:
                content_parts.append(f"{title}\n{body}")

        return "\n\n".join(content_parts)

    async def _summarize_content(
        self,
        topic: str,
        content: str,
    ) -> tuple[str, list[str]]:
        """Summarize content and extract key points.

        Args:
            topic: Topic being researched.
            content: Raw content to summarize.

        Returns:
            Tuple of (summary, key_points).
        """
        if self.ai_client:
            return await self._ai_summarize(topic, content)

        # Fallback: simple extraction
        return self._simple_summarize(topic, content)

    async def _ai_summarize(
        self,
        topic: str,
        content: str,
    ) -> tuple[str, list[str]]:
        """Use AI to summarize content.

        Args:
            topic: Topic being researched.
            content: Raw content to summarize.

        Returns:
            Tuple of (summary, key_points).
        """
        # This would integrate with the existing AI providers
        # For now, use simple extraction
        return self._simple_summarize(topic, content)

    def _simple_summarize(
        self,
        topic: str,
        content: str,
    ) -> tuple[str, list[str]]:
        """Simple content summarization without AI.

        Args:
            topic: Topic being researched.
            content: Raw content to summarize.

        Returns:
            Tuple of (summary, key_points).
        """
        # Extract sentences that seem important
        sentences = content.replace("\n", " ").split(". ")
        sentences = [s.strip() for s in sentences if len(s) > 20]

        # Create summary from first few sentences
        summary_sentences = sentences[:3]
        summary = ". ".join(summary_sentences) + "." if summary_sentences else f"Information about {topic}."

        # Extract key points (sentences with keywords)
        keywords = ["tip", "best", "how", "why", "important", "should", "can", "will"]
        key_points = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw in sentence_lower for kw in keywords):
                # Clean up and add
                clean = sentence.strip()
                if clean and len(clean) < 200:
                    key_points.append(clean)

                if len(key_points) >= 5:
                    break

        # If no key points found, create generic ones
        if not key_points:
            key_points = [
                f"Understanding {topic} fundamentals",
                f"Practical applications of {topic}",
                f"Getting started with {topic}",
                f"Tips for success with {topic}",
                f"Common mistakes to avoid with {topic}",
            ]

        return summary, key_points
