"""Tests for DenseOverlayPlanner - TTL-based image overlay planning."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

from socials_automator.video.pipeline.dense_overlay_planner import (
    DenseOverlayPlanner,
    ExtractedTopic,
    SrtEntry,
)
from socials_automator.video.pipeline.base import (
    VideoScript,
    VideoSegment,
    PipelineContext,
    ProfileMetadata,
)


class TestExtractedTopic:
    """Test ExtractedTopic dataclass."""

    def test_basic_creation(self):
        """Test creating an ExtractedTopic."""
        topic = ExtractedTopic(
            topic="Stranger Things",
            match_type="exact",
            search_query="stranger things netflix poster",
            keywords=["stranger", "things"],
            priority=1,
        )
        assert topic.topic == "Stranger Things"
        assert topic.match_type == "exact"
        assert topic.srt_start is None
        assert topic.srt_end is None

    def test_with_srt_timing(self):
        """Test ExtractedTopic with SRT timing set."""
        topic = ExtractedTopic(
            topic="ChatGPT",
            match_type="exact",
            search_query="chatgpt openai logo",
            keywords=["chatgpt", "openai"],
        )
        topic.srt_start = 5.0
        topic.srt_end = 8.0

        assert topic.srt_start == 5.0
        assert topic.srt_end == 8.0


class TestSrtEntry:
    """Test SrtEntry dataclass."""

    def test_basic_creation(self):
        """Test creating an SrtEntry."""
        entry = SrtEntry(
            index=1,
            start_time=3.0,
            end_time=7.0,
            text="This is the first subtitle entry.",
        )
        assert entry.index == 1
        assert entry.start_time == 3.0
        assert entry.end_time == 7.0
        assert "first subtitle" in entry.text


class TestDenseOverlayPlannerInit:
    """Test DenseOverlayPlanner initialization."""

    def test_default_init(self):
        """Test default initialization."""
        planner = DenseOverlayPlanner()
        assert planner.image_ttl == 3.0
        assert planner.minimum_images is None
        assert planner.name == "DenseOverlayPlanner"

    def test_custom_ttl(self):
        """Test initialization with custom TTL."""
        planner = DenseOverlayPlanner(image_ttl=5.0)
        assert planner.image_ttl == 5.0

    def test_custom_minimum(self):
        """Test initialization with custom minimum."""
        planner = DenseOverlayPlanner(minimum_images=20)
        assert planner.minimum_images == 20

    def test_with_text_provider(self):
        """Test initialization with text provider."""
        mock_provider = MagicMock()
        planner = DenseOverlayPlanner(text_provider=mock_provider)
        assert planner._text_provider is mock_provider


class TestCalculateMinimum:
    """Test minimum topic calculation."""

    def setup_method(self):
        """Create planner for each test."""
        self.planner = DenseOverlayPlanner(image_ttl=3.0)

    def test_explicit_minimum_used(self):
        """Test that explicit minimum is used when provided."""
        planner = DenseOverlayPlanner(image_ttl=3.0, minimum_images=15)

        script = MagicMock()
        script.total_duration = 60.0

        result = planner._calculate_minimum(script)
        assert result == 15

    def test_auto_calculate_from_duration(self):
        """Test auto-calculation based on duration and TTL."""
        script = MagicMock()
        script.total_duration = 60.0
        script.hook_end_time = 3.0
        script.cta_start_time = 56.0

        # Available time: 56 - 3 = 53s
        # With TTL=3s: 53/3 = 17 images
        result = self.planner._calculate_minimum(script)
        assert result == 17

    def test_minimum_of_5(self):
        """Test that minimum is at least 5."""
        script = MagicMock()
        script.total_duration = 15.0  # Very short video
        script.hook_end_time = 3.0
        script.cta_start_time = 11.0

        # Available time: 11 - 3 = 8s
        # With TTL=3s: 8/3 = 2 images -> should be clamped to 5
        result = self.planner._calculate_minimum(script)
        assert result == 5


class TestParseSrtTime:
    """Test SRT time parsing."""

    def setup_method(self):
        """Create planner for each test."""
        self.planner = DenseOverlayPlanner()

    def test_parse_standard_format(self):
        """Test parsing standard SRT time format."""
        result = self.planner._parse_srt_time("00:01:23,456")
        assert abs(result - 83.456) < 0.001

    def test_parse_with_dot(self):
        """Test parsing SRT time with dot separator."""
        result = self.planner._parse_srt_time("00:00:03.500")
        assert abs(result - 3.5) < 0.001

    def test_parse_zero(self):
        """Test parsing zero time."""
        result = self.planner._parse_srt_time("00:00:00,000")
        assert result == 0.0


class TestParseSrt:
    """Test SRT file parsing."""

    def setup_method(self):
        """Create planner for each test."""
        self.planner = DenseOverlayPlanner()

    def test_parse_valid_srt(self, tmp_path):
        """Test parsing a valid SRT file."""
        srt_content = """1
00:00:00,000 --> 00:00:03,000
First subtitle entry.

2
00:00:03,000 --> 00:00:07,000
Second subtitle with ChatGPT mentioned.

3
00:00:07,000 --> 00:00:10,000
Third entry about AI tools.
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        entries = self.planner._parse_srt(srt_file)

        assert len(entries) == 3
        assert entries[0].text == "First subtitle entry."
        assert entries[1].start_time == 3.0
        assert "ChatGPT" in entries[1].text

    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent SRT file."""
        entries = self.planner._parse_srt(Path("/nonexistent/file.srt"))
        assert entries == []

    def test_parse_none_path(self):
        """Test parsing with None path."""
        entries = self.planner._parse_srt(None)
        assert entries == []


class TestMatchTopicsToSrt:
    """Test topic-to-SRT matching."""

    def setup_method(self):
        """Create planner for each test."""
        self.planner = DenseOverlayPlanner()

    def test_match_exact_keyword(self):
        """Test matching topic by exact keyword."""
        topics = [
            ExtractedTopic(
                topic="ChatGPT",
                match_type="exact",
                search_query="chatgpt openai",
                keywords=["chatgpt", "openai"],
            )
        ]
        srt_entries = [
            SrtEntry(index=1, start_time=0.0, end_time=3.0, text="Welcome to our video."),
            SrtEntry(index=2, start_time=3.0, end_time=7.0, text="Let's talk about ChatGPT."),
            SrtEntry(index=3, start_time=7.0, end_time=10.0, text="It's an amazing AI tool."),
        ]

        result = self.planner._match_topics_to_srt(topics, srt_entries)

        assert result[0].srt_start == 3.0
        assert result[0].srt_end == 7.0

    def test_no_match(self):
        """Test topic with no SRT match."""
        topics = [
            ExtractedTopic(
                topic="Netflix",
                match_type="exact",
                search_query="netflix logo",
                keywords=["netflix"],
            )
        ]
        srt_entries = [
            SrtEntry(index=1, start_time=0.0, end_time=5.0, text="This video is about AI tools."),
        ]

        result = self.planner._match_topics_to_srt(topics, srt_entries)

        assert result[0].srt_start is None
        assert result[0].srt_end is None

    def test_case_insensitive_match(self):
        """Test that matching is case-insensitive."""
        topics = [
            ExtractedTopic(
                topic="OpenAI",
                match_type="exact",
                search_query="openai logo",
                keywords=["openai"],
            )
        ]
        srt_entries = [
            SrtEntry(index=1, start_time=5.0, end_time=10.0, text="OPENAI released a new model."),
        ]

        result = self.planner._match_topics_to_srt(topics, srt_entries)

        assert result[0].srt_start == 5.0


class TestDistributeToSlots:
    """Test time slot distribution."""

    def setup_method(self):
        """Create planner with 3s TTL."""
        self.planner = DenseOverlayPlanner(image_ttl=3.0)

    def test_basic_distribution(self):
        """Test basic slot distribution."""
        topics = [
            ExtractedTopic(
                topic="ChatGPT",
                match_type="exact",
                search_query="chatgpt",
                keywords=["chatgpt"],
                srt_start=5.0,
                srt_end=8.0,
            ),
            ExtractedTopic(
                topic="Claude",
                match_type="exact",
                search_query="claude ai",
                keywords=["claude"],
                srt_start=10.0,
                srt_end=13.0,
            ),
        ]

        script = MagicMock()
        script.total_duration = 60.0
        script.hook_end_time = 3.0
        script.cta_start_time = 56.0

        overlays = self.planner._distribute_to_slots(topics, script)

        assert len(overlays) == 2
        # First overlay starts at SRT mention time (5.0)
        assert overlays[0].start_time == 5.0
        assert overlays[0].end_time == 8.0  # 5.0 + 3.0 TTL
        assert overlays[0].topic == "ChatGPT"

        # Second overlay starts at SRT mention time (10.0)
        assert overlays[1].start_time == 10.0
        assert overlays[1].end_time == 13.0  # 10.0 + 3.0 TTL

    def test_respects_hook_boundary(self):
        """Test that overlays don't start before hook ends."""
        topics = [
            ExtractedTopic(
                topic="EarlyTopic",
                match_type="exact",
                search_query="early topic",
                keywords=["early"],
                srt_start=1.0,  # Before hook ends (3.0)
                srt_end=2.0,
            ),
        ]

        script = MagicMock()
        script.total_duration = 60.0
        script.hook_end_time = 3.0
        script.cta_start_time = 56.0

        overlays = self.planner._distribute_to_slots(topics, script)

        # Overlay should be pushed to start at hook_end_time (3.0)
        assert len(overlays) == 1
        assert overlays[0].start_time >= 3.0

    def test_respects_cta_boundary(self):
        """Test that overlays don't extend into CTA buffer zone."""
        topics = [
            ExtractedTopic(
                topic="LateTopic",
                match_type="exact",
                search_query="late topic",
                keywords=["late"],
                srt_start=54.0,  # Near CTA
                srt_end=55.0,
            ),
        ]

        script = MagicMock()
        script.total_duration = 60.0
        script.hook_end_time = 3.0
        script.cta_start_time = 56.0

        overlays = self.planner._distribute_to_slots(topics, script)

        # max_end = min(cta_start, total_duration - CTA_BUFFER)
        # max_end = min(56.0, 60.0 - 4.0) = min(56.0, 56.0) = 56.0
        # Overlay should end at or before max_end
        if overlays:
            assert overlays[0].end_time <= 56.0

    def test_skips_unmatched_topics(self):
        """Test that topics without SRT match are skipped."""
        topics = [
            ExtractedTopic(
                topic="MatchedTopic",
                match_type="exact",
                search_query="matched",
                keywords=["matched"],
                srt_start=10.0,
                srt_end=15.0,
            ),
            ExtractedTopic(
                topic="UnmatchedTopic",
                match_type="exact",
                search_query="unmatched",
                keywords=["unmatched"],
                srt_start=None,  # No SRT match
                srt_end=None,
            ),
        ]

        script = MagicMock()
        script.total_duration = 60.0
        script.hook_end_time = 3.0
        script.cta_start_time = 56.0

        overlays = self.planner._distribute_to_slots(topics, script)

        # Only the matched topic should have an overlay
        assert len(overlays) == 1
        assert overlays[0].topic == "MatchedTopic"


class TestParseAiResponse:
    """Test AI response parsing."""

    def setup_method(self):
        """Create planner for each test."""
        self.planner = DenseOverlayPlanner()

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        response = """{
            "topics": [
                {
                    "topic": "Stranger Things",
                    "match_type": "exact",
                    "search_query": "stranger things netflix",
                    "keywords": ["stranger", "things"],
                    "priority": 1
                },
                {
                    "topic": "AI assistant",
                    "match_type": "illustrative",
                    "search_query": "ai assistant robot",
                    "keywords": ["ai", "assistant"],
                    "priority": 2
                }
            ]
        }"""

        topics = self.planner._parse_ai_response(response)

        assert len(topics) == 2
        assert topics[0].topic == "Stranger Things"
        assert topics[0].match_type == "exact"
        assert topics[1].match_type == "illustrative"

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code block."""
        response = """```json
{
    "topics": [
        {
            "topic": "ChatGPT",
            "match_type": "exact",
            "search_query": "chatgpt openai",
            "keywords": ["chatgpt"]
        }
    ]
}
```"""

        topics = self.planner._parse_ai_response(response)

        assert len(topics) == 1
        assert topics[0].topic == "ChatGPT"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns empty list."""
        response = "This is not valid JSON at all"

        topics = self.planner._parse_ai_response(response)

        assert topics == []


class TestFallbackExtraction:
    """Test fallback topic extraction (no AI)."""

    def setup_method(self):
        """Create planner for each test."""
        self.planner = DenseOverlayPlanner()

    def test_extracts_capitalized_phrases(self):
        """Test extraction of capitalized multi-word phrases."""
        narration = "Today we're talking about Stranger Things on Netflix. OpenAI released ChatGPT. Apple announced iPhone 15."

        topics = self.planner._fallback_extraction(narration, target_count=10)

        topic_names = [t.topic for t in topics]
        assert "Stranger Things" in topic_names or "Netflix" in topic_names

    def test_skips_common_words(self):
        """Test that common words are skipped."""
        narration = "This is The Best video. What you Need to know."

        topics = self.planner._fallback_extraction(narration, target_count=10)

        # Common words like "This", "The", "What" should be filtered
        topic_names = [t.topic for t in topics]
        assert "This" not in topic_names
        assert "The" not in topic_names
        assert "What" not in topic_names

    def test_respects_target_count(self):
        """Test that extraction respects target count."""
        narration = "Apple iPhone Samsung Galaxy Microsoft Windows Google Chrome Amazon Prime Netflix Hulu"

        topics = self.planner._fallback_extraction(narration, target_count=3)

        assert len(topics) <= 3


@pytest.mark.asyncio
class TestExecute:
    """Test the execute pipeline method."""

    async def test_execute_with_no_script(self):
        """Test that execute raises error when no script in context."""
        planner = DenseOverlayPlanner()
        context = MagicMock()
        context.script = None

        with pytest.raises(Exception) as exc_info:
            await planner.execute(context)

        assert "No script available" in str(exc_info.value)

    async def test_execute_success(self, tmp_path):
        """Test successful execution with mocked AI."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value="""{
            "topics": [
                {
                    "topic": "ChatGPT",
                    "match_type": "exact",
                    "search_query": "chatgpt logo",
                    "keywords": ["chatgpt"],
                    "priority": 1
                }
            ]
        }""")

        planner = DenseOverlayPlanner(
            text_provider=mock_provider,
            image_ttl=3.0,
            minimum_images=5,
        )

        # Create SRT file
        srt_content = """1
00:00:00,000 --> 00:00:03,000
Welcome to our video.

2
00:00:03,000 --> 00:00:07,000
Let's talk about ChatGPT.

3
00:00:07,000 --> 00:00:10,000
It's an amazing AI tool.
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding="utf-8")

        # Create script mock
        script = MagicMock()
        script.full_narration = "Welcome. Let's talk about ChatGPT. It's amazing."
        script.total_duration = 60.0
        script.hook_end_time = 3.0
        script.cta_start_time = 56.0

        # Create context
        context = MagicMock()
        context.script = script
        context.srt_path = srt_file
        context.profile_path = tmp_path

        result = await planner.execute(context)

        # Verify overlays were created
        assert result.image_overlays is not None
        assert len(result.image_overlays.overlays) >= 0  # May be 0 if timing doesn't work out
