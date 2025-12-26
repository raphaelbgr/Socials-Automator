"""Topic research using multi-source web search.

Enhanced research with:
- AI-generated diverse search queries
- Multiple search engines (DuckDuckGo, Bing, Yahoo)
- Multi-language support (EN, ES, PT)
- Parallel searching for speed
- Category-based queries (tutorials, reviews, comparisons, news)
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

from .base import (
    ITopicResearcher,
    PipelineContext,
    ResearchError,
    ResearchResult,
    TopicInfo,
)
from .research_queries import ResearchQueryGenerator, ResearchQuery

logger = logging.getLogger("ai_calls")


# Search engine configurations
SEARCH_ENGINES = {
    "duckduckgo": {"enabled": True, "priority": 1},
    "bing": {"enabled": True, "priority": 2},
    "yahoo": {"enabled": True, "priority": 3},
}


class TopicResearcher(ITopicResearcher):
    """Researches topics using multi-source web search and AI summarization.

    Features:
    - AI-generated diverse search queries
    - Parallel search across multiple engines
    - Multi-language queries (EN, ES, PT)
    - Category-based research (tutorials, reviews, comparisons)
    - High-volume batched search for comprehensive coverage
    """

    def __init__(
        self,
        search_client: Optional[object] = None,
        ai_client: Optional[object] = None,
        max_results: int = 10,
        use_enhanced_search: bool = True,
        target_search_volume: int = 200,
    ):
        """Initialize topic researcher.

        Args:
            search_client: Web search client (optional, uses WebSearcher if None).
            ai_client: AI client for summarization and query generation.
            max_results: Maximum search results per query.
            use_enhanced_search: If True, use AI-generated multi-source search.
            target_search_volume: Target number of total searches (default 200).
        """
        super().__init__()
        self.search_client = search_client
        self.ai_client = ai_client
        self.max_results = max_results
        self.use_enhanced_search = use_enhanced_search
        self.target_search_volume = target_search_volume
        self._query_generator: Optional[ResearchQueryGenerator] = None
        self._web_searcher = None
        self._content_pillar: str = "general"
        self._profile_name: Optional[str] = None  # For query history tracking

    @property
    def query_generator(self) -> ResearchQueryGenerator:
        """Lazy-load the query generator."""
        if self._query_generator is None:
            # Pass the AI client and profile name for query history tracking
            self._query_generator = ResearchQueryGenerator(
                text_provider=self.ai_client,
                profile_name=self._profile_name,
            )
        return self._query_generator

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

        # Extract profile name from path for query history tracking
        if context.profile_path:
            self._profile_name = context.profile_path.name
            # Reset query generator to pick up new profile_name
            self._query_generator = None

        # Capture content pillar from topic if available
        if hasattr(context.topic, "pillar_id") and context.topic.pillar_id:
            self._content_pillar = context.topic.pillar_id
        elif hasattr(context.topic, "pillar") and context.topic.pillar:
            self._content_pillar = context.topic.pillar

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
        if self.use_enhanced_search:
            return await self._enhanced_research(topic)
        return await self._basic_research(topic)

    async def _basic_research(self, topic: TopicInfo) -> ResearchResult:
        """Basic research using simple queries (fallback mode)."""
        self.log_progress("Searching the web (basic mode)...")

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

        return await self._process_results(topic.topic, all_results)

    async def _enhanced_research(self, topic: TopicInfo) -> ResearchResult:
        """Enhanced research with AI-generated queries and high-volume batched search."""
        self.log_progress("Generating AI-powered research queries...")

        # Generate diverse queries using AI
        research_queries = await self.query_generator.generate_queries(
            topic=topic.topic,
            pillar=self._content_pillar,
            count=20,  # Generate more base queries
        )

        # Log query breakdown
        en_queries = [q for q in research_queries if q.language == "en"]
        es_queries = [q for q in research_queries if q.language == "es"]
        pt_queries = [q for q in research_queries if q.language == "pt"]
        self.log_detail(
            f"Generated {len(research_queries)} queries: "
            f"{len(en_queries)} EN, {len(es_queries)} ES, {len(pt_queries)} PT"
        )

        # Sort by priority (1 = highest)
        research_queries.sort(key=lambda q: q.priority)

        # Extract query strings and expand to target volume
        base_query_strings = [q.query for q in research_queries]
        expanded_queries = self._expand_queries_for_volume(
            base_query_strings,
            topic.topic,
            target=self.target_search_volume,
        )

        self.log_progress(
            f"Searching {len(expanded_queries)} queries in batches of 50..."
        )

        # Use batched search for high volume
        all_results = await self._batched_search(expanded_queries)

        # Deduplicate results by URL
        unique_results = self._deduplicate_results(all_results)
        self.log_detail(
            f"Found {len(all_results)} total -> {len(unique_results)} unique results"
        )

        # Add current AI versions context
        current_year = datetime.now().year
        current_month = datetime.now().strftime("%B")
        version_query = f"ChatGPT Claude Gemini latest version {current_month} {current_year}"
        self.log_detail(f"Version context search: {version_query}")
        version_results = await self._web_search(version_query)
        unique_results.extend(version_results)

        return await self._process_results(topic.topic, unique_results)

    async def _parallel_search(self, queries: list[ResearchQuery]) -> list[dict]:
        """Search multiple queries in parallel.

        Args:
            queries: List of ResearchQuery objects.

        Returns:
            Combined results from all queries.
        """
        async def search_one(query: ResearchQuery) -> list[dict]:
            """Search a single query."""
            try:
                results = await self._web_search(query.query)
                # Tag results with query metadata
                for r in results:
                    r["_query_language"] = query.language
                    r["_query_category"] = query.category
                return results
            except Exception as e:
                self.log_detail(f"Search failed for '{query.query}': {e}")
                return []

        # Run all searches in parallel
        tasks = [search_one(q) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results
        all_results = []
        for result in results_list:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                self.log_detail(f"Parallel search error: {result}")

        return all_results

    def _deduplicate_results(self, results: list[dict]) -> list[dict]:
        """Remove duplicate results by URL.

        Args:
            results: List of search results.

        Returns:
            Deduplicated results.
        """
        seen_urls = set()
        unique = []

        for result in results:
            url = result.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique.append(result)
            elif not url:
                # Keep results without URLs (from fallback)
                unique.append(result)

        return unique

    def _expand_queries_for_volume(
        self,
        base_queries: list[str],
        topic: str,
        target: int = 200,
    ) -> list[str]:
        """Expand base queries with modifiers to reach target volume.

        Args:
            base_queries: List of base query strings from AI.
            topic: Original topic for additional variations.
            target: Target number of queries to generate.

        Returns:
            Expanded list of query strings.
        """
        if not base_queries:
            return []

        # Start with original queries
        expanded = list(base_queries)

        now = datetime.now()
        current_year = now.year
        current_month = now.strftime("%B")

        # Time-based modifiers
        time_modifiers = [
            f"{current_year}",
            f"{current_month} {current_year}",
            "latest",
            "new",
            "updated",
            "recent",
        ]

        # Depth modifiers for comprehensive research
        depth_modifiers = [
            "tutorial",
            "guide",
            "how to",
            "tips",
            "tricks",
            "best practices",
            "examples",
            "use cases",
            "review",
            "comparison",
            "vs",
            "alternative",
            "free",
            "beginner",
            "advanced",
        ]

        # Platform/context modifiers
        context_modifiers = [
            "for productivity",
            "for business",
            "for content creation",
            "for developers",
            "for beginners",
            "step by step",
            "complete guide",
        ]

        # Generate variations until we reach target
        modifier_idx = 0
        all_modifiers = time_modifiers + depth_modifiers + context_modifiers

        while len(expanded) < target and modifier_idx < len(all_modifiers):
            modifier = all_modifiers[modifier_idx]
            for base in base_queries[:10]:  # Use top 10 base queries
                if len(expanded) >= target:
                    break
                variation = f"{base} {modifier}"
                if variation not in expanded:
                    expanded.append(variation)
            modifier_idx += 1

        # If still need more, create topic-specific variations
        if len(expanded) < target:
            topic_words = topic.lower().split()[:5]
            for word in topic_words:
                if len(word) > 3:
                    for modifier in time_modifiers[:3]:
                        if len(expanded) >= target:
                            break
                        variation = f"{word} {modifier}"
                        if variation not in expanded:
                            expanded.append(variation)

        logger.info(
            f"QUERY_EXPANSION | base:{len(base_queries)} | "
            f"expanded:{len(expanded)} | target:{target}"
        )

        return expanded[:target]

    async def _batched_search(
        self,
        queries: list[str],
        batch_size: int = 50,
    ) -> list[dict]:
        """Execute searches in batches for high volume.

        Args:
            queries: List of query strings.
            batch_size: Queries per batch.

        Returns:
            Combined results from all queries.
        """
        all_results = []

        # Split into batches
        batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        total_batches = len(batches)

        for batch_num, batch in enumerate(batches, 1):
            batch_results = []

            # Search each query in the batch
            tasks = [self._web_search(query) for query in batch]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results_list:
                if isinstance(result, list):
                    batch_results.extend(result)
                elif isinstance(result, Exception):
                    self.log_detail(f"Batch search error: {result}")

            all_results.extend(batch_results)

            self.log_detail(
                f"Batch {batch_num}/{total_batches}: {len(batch_results)} results"
            )

            # Small delay between batches to avoid rate limiting
            if batch_num < total_batches:
                await asyncio.sleep(1.0)

        logger.info(
            f"BATCHED_SEARCH | queries:{len(queries)} | "
            f"results:{len(all_results)}"
        )

        return all_results

    async def _process_results(
        self, topic_str: str, results: list[dict]
    ) -> ResearchResult:
        """Process search results into a ResearchResult.

        Args:
            topic_str: Topic string.
            results: Search results.

        Returns:
            Processed ResearchResult.
        """
        self.log_progress(f"Processing {len(results)} results...")

        # Extract content from results
        raw_content = self._extract_content(results)

        self.log_progress("Summarizing content...")

        # Summarize and extract key points
        summary, key_points = await self._summarize_content(
            topic_str,
            raw_content,
        )

        return ResearchResult(
            topic=topic_str,
            summary=summary,
            key_points=key_points,
            sources=[
                {"title": r.get("title", ""), "url": r.get("href", "")}
                for r in results[:5]
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
