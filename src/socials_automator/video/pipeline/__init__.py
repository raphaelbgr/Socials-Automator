"""Video generation pipeline module.

A complete pipeline for generating 1-minute Instagram Reels/TikTok videos:

Pipeline Steps:
1. TopicSelector - Selects topic from profile content pillars
2. TopicResearcher - Searches web for topic information
3. ScriptPlanner - Plans video script with segments
4. VideoSearcher - Searches Pexels for matching clips
5. VideoDownloader - Downloads clips to temp folder
6. VideoAssembler - Assembles clips, crops to 9:16, cuts at 1:00
7. VoiceGenerator - Generates TTS voiceover with timestamps
8. SubtitleRenderer - Adds karaoke-style subtitles

Example usage:
    from socials_automator.video.pipeline import VideoPipeline

    # Create pipeline
    pipeline = VideoPipeline(
        pexels_api_key="your_key",  # Or set PEXELS_API_KEY env var
        voice="rvc_adam",  # Local Adam voice - FREE & UNLIMITED!
    )

    # Generate video
    video_path = await pipeline.generate(
        profile_path=Path("profiles/ai.for.mortals"),
        output_dir=Path("output"),
    )

    print(f"Video saved to: {video_path}")

Or synchronously:
    video_path = pipeline.generate_sync(
        profile_path=Path("profiles/ai.for.mortals"),
    )
"""

from .base import (
    # Data models
    PipelineContext,
    ProfileMetadata,
    ResearchResult,
    TopicInfo,
    VideoClipInfo,
    VideoMetadata,
    VideoScript,
    VideoSegment,
    # Exceptions
    PipelineError,
    ResearchError,
    ScriptPlanningError,
    SubtitleRenderError,
    TopicSelectionError,
    VideoAssemblyError,
    VideoDownloadError,
    VideoSearchError,
    VoiceGenerationError,
    # Interfaces
    ICaptionGenerator,
    IScriptPlanner,
    ISubtitleRenderer,
    ITopicResearcher,
    ITopicSelector,
    IVideoAssembler,
    IVideoDownloader,
    IVideoSearcher,
    IVoiceGenerator,
    PipelineStep,
)
from .caption_generator import CaptionGenerator
from .orchestrator import ProgressCallback, VideoPipeline, setup_logging
from .script_planner import ScriptPlanner
from .subtitle_renderer import SubtitleRenderer
from .topic_researcher import TopicResearcher
from .topic_selector import TopicSelector
from .video_assembler import VideoAssembler
from .video_downloader import VideoDownloader
from .video_searcher import VideoSearcher
from .voice_generator import VOICE_PRESETS, VoiceGenerator

__all__ = [
    # Main class
    "VideoPipeline",
    "setup_logging",
    "ProgressCallback",
    # Pipeline steps
    "TopicSelector",
    "TopicResearcher",
    "ScriptPlanner",
    "VideoSearcher",
    "VideoDownloader",
    "VideoAssembler",
    "VoiceGenerator",
    "SubtitleRenderer",
    "CaptionGenerator",
    # Data models
    "PipelineContext",
    "ProfileMetadata",
    "TopicInfo",
    "ResearchResult",
    "VideoScript",
    "VideoSegment",
    "VideoClipInfo",
    "VideoMetadata",
    # Interfaces
    "PipelineStep",
    "ITopicSelector",
    "ITopicResearcher",
    "IScriptPlanner",
    "IVideoSearcher",
    "IVideoDownloader",
    "IVideoAssembler",
    "IVoiceGenerator",
    "ISubtitleRenderer",
    "ICaptionGenerator",
    # Exceptions
    "PipelineError",
    "TopicSelectionError",
    "ResearchError",
    "ScriptPlanningError",
    "VideoSearchError",
    "VideoDownloadError",
    "VideoAssemblyError",
    "VoiceGenerationError",
    "SubtitleRenderError",
    # Constants
    "VOICE_PRESETS",
]
