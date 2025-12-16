"""Tests for video pipeline components."""

import pytest

from socials_automator.video.pipeline import (
    PipelineContext,
    ProfileMetadata,
    ResearchResult,
    ScriptPlanner,
    TopicInfo,
    TopicResearcher,
    TopicSelector,
    VideoPipeline,
    VideoScript,
    VideoSegment,
    VoiceGenerator,
)


class TestProfileMetadata:
    """Tests for ProfileMetadata loading."""

    def test_load_from_file(self, sample_profile_dir):
        """Test loading profile from file."""
        metadata = ProfileMetadata.from_file(sample_profile_dir / "metadata.json")

        assert metadata.id == "test-profile"
        assert metadata.name == "test.profile"
        assert metadata.display_name == "Test Profile"
        assert len(metadata.content_pillars) == 2

    def test_content_pillars(self, sample_profile_metadata):
        """Test content pillars are loaded."""
        pillars = sample_profile_metadata.content_pillars

        assert len(pillars) >= 1
        assert pillars[0]["id"] == "tool_tutorials"


class TestTopicSelector:
    """Tests for TopicSelector."""

    def test_init(self):
        """Test initialization."""
        selector = TopicSelector()
        assert selector.name == "TopicSelector"

    @pytest.mark.asyncio
    async def test_select_topic(self, sample_profile_metadata):
        """Test topic selection."""
        selector = TopicSelector()
        topic = await selector.select_topic(sample_profile_metadata)

        assert isinstance(topic, TopicInfo)
        assert topic.topic  # Has a topic
        assert topic.pillar_id  # Has pillar
        assert len(topic.keywords) > 0  # Has keywords
        assert len(topic.search_queries) > 0  # Has search queries

    @pytest.mark.asyncio
    async def test_execute(self, sample_pipeline_context):
        """Test execute method."""
        selector = TopicSelector()
        context = await selector.execute(sample_pipeline_context)

        assert context.topic is not None
        assert context.topic.topic


class TestTopicResearcher:
    """Tests for TopicResearcher."""

    def test_init(self):
        """Test initialization."""
        researcher = TopicResearcher()
        assert researcher.name == "TopicResearcher"

    @pytest.mark.asyncio
    async def test_research(self, sample_topic_info):
        """Test topic research."""
        researcher = TopicResearcher()
        result = await researcher.research(sample_topic_info)

        assert isinstance(result, ResearchResult)
        assert result.topic == sample_topic_info.topic
        assert result.summary
        assert len(result.key_points) > 0


class TestScriptPlanner:
    """Tests for ScriptPlanner."""

    def test_init(self):
        """Test initialization."""
        planner = ScriptPlanner()
        assert planner.name == "ScriptPlanner"

    @pytest.mark.asyncio
    async def test_plan_script(self, sample_topic_info, sample_research_result):
        """Test script planning."""
        planner = ScriptPlanner()
        script = await planner.plan_script(
            sample_topic_info,
            sample_research_result,
            duration=60.0,
        )

        assert isinstance(script, VideoScript)
        assert script.title
        assert script.hook
        assert len(script.segments) > 0
        assert script.cta
        assert script.total_duration == 60.0

    def test_script_segments_have_keywords(self, sample_video_script):
        """Test that segments have keywords for video search."""
        for segment in sample_video_script.segments:
            assert len(segment.keywords) > 0


class TestVideoScript:
    """Tests for VideoScript model."""

    def test_calculate_times(self, sample_video_script):
        """Test time calculation for segments."""
        sample_video_script.calculate_times()

        assert sample_video_script.segments[0].start_time == 0.0
        assert sample_video_script.segments[0].end_time == 10.0
        assert sample_video_script.segments[1].start_time == 10.0


class TestVoiceGenerator:
    """Tests for VoiceGenerator."""

    def test_init_default(self):
        """Test default initialization."""
        generator = VoiceGenerator()
        assert generator.voice == "en-US-AriaNeural"

    def test_init_with_preset(self):
        """Test initialization with preset."""
        generator = VoiceGenerator(voice="professional_male")
        assert generator.voice == "en-US-GuyNeural"

    def test_init_with_custom_voice(self):
        """Test initialization with custom voice name."""
        generator = VoiceGenerator(voice="en-GB-SoniaNeural")
        assert generator.voice == "en-GB-SoniaNeural"


class TestVideoPipeline:
    """Tests for VideoPipeline orchestrator."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = VideoPipeline()
        assert len(pipeline.steps) == 8

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        pipeline = VideoPipeline(pexels_api_key="test_key")
        assert pipeline is not None

    def test_init_with_text_ai(self):
        """Test initialization with text AI provider."""
        pipeline = VideoPipeline(text_ai="lmstudio")
        assert pipeline.text_ai == "lmstudio"

    def test_init_with_video_matcher(self):
        """Test initialization with video matcher."""
        pipeline = VideoPipeline(video_matcher="pexels")
        assert pipeline.video_matcher == "pexels"

    def test_steps_order(self):
        """Test pipeline steps are in correct order."""
        pipeline = VideoPipeline()

        step_names = [step.name for step in pipeline.steps]
        expected = [
            "TopicSelector",
            "TopicResearcher",
            "ScriptPlanner",
            "VideoSearcher",
            "VideoDownloader",
            "VideoAssembler",
            "VoiceGenerator",
            "SubtitleRenderer",
        ]

        assert step_names == expected


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.skipif(
        True,
        reason="Requires PEXELS_API_KEY and network access",
    )
    @pytest.mark.asyncio
    async def test_full_pipeline(self, sample_profile_dir, temp_dir):
        """Test complete pipeline execution."""
        pipeline = VideoPipeline()

        output_path = await pipeline.generate(
            profile_path=sample_profile_dir,
            output_dir=temp_dir / "output",
        )

        assert output_path.exists()
        assert output_path.suffix == ".mp4"
