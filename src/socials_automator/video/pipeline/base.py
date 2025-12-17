"""Base classes and interfaces for video pipeline components.

SOLID Principles:
- Single Responsibility: Each class has one job
- Open/Closed: Extend via inheritance, don't modify
- Liskov Substitution: Subclasses are interchangeable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Data Models
# =============================================================================


class ProfileMetadata(BaseModel):
    """Profile metadata from metadata.json."""

    id: str
    name: str
    display_name: str
    niche_id: str
    tagline: str
    description: str
    language: str = "en"
    content_pillars: list[dict] = Field(default_factory=list)
    trending_keywords: list[str] = Field(default_factory=list)
    research_sources: dict = Field(default_factory=dict)
    ai_generation: dict = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> "ProfileMetadata":
        """Load profile metadata from JSON file."""
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profile = data.get("profile", {})
        strategy = data.get("content_strategy", {})
        research = data.get("research_sources", {})
        ai_gen = data.get("ai_generation", {})

        return cls(
            id=profile.get("id", ""),
            name=profile.get("name", ""),
            display_name=profile.get("display_name", ""),
            niche_id=profile.get("niche_id", ""),
            tagline=profile.get("tagline", ""),
            description=profile.get("description", ""),
            language=profile.get("language", "en"),
            content_pillars=strategy.get("content_pillars", []),
            trending_keywords=research.get("trending_keywords", []),
            research_sources=research.get("sources", {}),
            ai_generation=ai_gen,
        )


class TopicInfo(BaseModel):
    """Selected topic information."""

    topic: str
    pillar_id: str
    pillar_name: str
    keywords: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)


class ResearchResult(BaseModel):
    """Research results for a topic."""

    topic: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    raw_content: str = ""


class VideoSegment(BaseModel):
    """A single video segment in the script."""

    index: int
    text: str  # Narration text
    duration_seconds: float
    keywords: list[str] = Field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0


class VideoScript(BaseModel):
    """Complete video script with segments."""

    title: str
    hook: str
    segments: list[VideoSegment] = Field(default_factory=list)
    cta: str
    total_duration: float = 60.0
    full_narration: str = ""

    def calculate_times(self) -> None:
        """Calculate start/end times for each segment."""
        current_time = 0.0
        for segment in self.segments:
            segment.start_time = current_time
            segment.end_time = current_time + segment.duration_seconds
            current_time = segment.end_time


class VideoClipInfo(BaseModel):
    """Information about a downloaded video clip."""

    model_config = {"arbitrary_types_allowed": True}

    segment_index: int
    path: Path
    source_url: str
    pexels_id: int
    title: str
    duration_seconds: float
    width: int
    height: int
    keywords_used: list[str] = Field(default_factory=list)


class VideoMetadata(BaseModel):
    """Metadata for the assembled video (SRT-like structure)."""

    post_id: str
    title: str
    topic: str
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float
    segments: list[dict] = Field(default_factory=list)  # With timing info
    clips_used: list[dict] = Field(default_factory=list)
    narration: str = ""


class PipelineContext(BaseModel):
    """Context passed through pipeline steps."""

    model_config = {"arbitrary_types_allowed": True}

    profile: ProfileMetadata
    post_id: str
    output_dir: Path
    temp_dir: Path

    # Populated by pipeline steps
    topic: Optional[TopicInfo] = None
    research: Optional[ResearchResult] = None
    script: Optional[VideoScript] = None
    clips: list[VideoClipInfo] = Field(default_factory=list)
    assembled_video_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    srt_path: Optional[Path] = None
    final_video_path: Optional[Path] = None
    metadata: Optional[VideoMetadata] = None


# =============================================================================
# Abstract Base Classes (Interfaces)
# =============================================================================


class PipelineStep(ABC):
    """Abstract base class for all pipeline steps."""

    def __init__(self, name: str):
        self.name = name
        self._display = None  # Set by orchestrator

    def set_display(self, display) -> None:
        """Set the CLI display instance for this step."""
        self._display = display

    def log_start(self, message: str) -> None:
        """Log step start (shown on console)."""
        if self._display:
            self._display.step(message, self.name)

    def log_progress(self, message: str) -> None:
        """Log important progress (shown on console)."""
        if self._display:
            self._display.info(message, self.name)

    def log_detail(self, message: str) -> None:
        """Log detailed progress (file only, unless verbose)."""
        if self._display:
            self._display.detail(message, self.name)

    def log_success(self, message: str) -> None:
        """Log step success (shown on console)."""
        if self._display:
            self._display.success(message, self.name)

    def log_error(self, message: str) -> None:
        """Log step error (shown on console)."""
        if self._display:
            self._display.error(message, self.name)

    def log_warning(self, message: str) -> None:
        """Log step warning (shown on console)."""
        if self._display:
            self._display.warning(message, self.name)

    def log_debug(self, message: str) -> None:
        """Log debug message (file only)."""
        if self._display:
            self._display.debug(message, self.name)

    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute the pipeline step.

        Args:
            context: Pipeline context with current state.

        Returns:
            Updated pipeline context.
        """
        pass


class ITopicSelector(PipelineStep):
    """Interface for topic selection."""

    def __init__(self):
        super().__init__("TopicSelector")

    @abstractmethod
    async def select_topic(self, profile: ProfileMetadata) -> TopicInfo:
        """Select a topic based on profile."""
        pass


class ITopicResearcher(PipelineStep):
    """Interface for topic research."""

    def __init__(self):
        super().__init__("TopicResearcher")

    @abstractmethod
    async def research(self, topic: TopicInfo) -> ResearchResult:
        """Research a topic and return results."""
        pass


class IScriptPlanner(PipelineStep):
    """Interface for script planning."""

    def __init__(self):
        super().__init__("ScriptPlanner")

    @abstractmethod
    async def plan_script(
        self,
        topic: TopicInfo,
        research: ResearchResult,
        duration: float = 60.0,
    ) -> VideoScript:
        """Plan a video script from research."""
        pass


class IVideoSearcher(PipelineStep):
    """Interface for video search."""

    def __init__(self):
        super().__init__("VideoSearcher")

    @abstractmethod
    async def search_videos(
        self,
        script: VideoScript,
    ) -> list[dict]:
        """Search for videos matching script segments."""
        pass


class IVideoDownloader(PipelineStep):
    """Interface for video download."""

    def __init__(self):
        super().__init__("VideoDownloader")

    @abstractmethod
    async def download_videos(
        self,
        search_results: list[dict],
        output_dir: Path,
    ) -> list[VideoClipInfo]:
        """Download videos to output directory."""
        pass


class IVideoAssembler(PipelineStep):
    """Interface for video assembly."""

    def __init__(self):
        super().__init__("VideoAssembler")

    @abstractmethod
    async def assemble(
        self,
        clips: list[VideoClipInfo],
        script: VideoScript,
        output_path: Path,
    ) -> tuple[Path, VideoMetadata]:
        """Assemble clips into final video with metadata."""
        pass


class ICaptionGenerator(PipelineStep):
    """Interface for caption/narration generation."""

    def __init__(self):
        super().__init__("CaptionGenerator")

    @abstractmethod
    async def generate_captions(
        self,
        script: VideoScript,
        research: ResearchResult,
    ) -> str:
        """Generate full narration text from script."""
        pass


class IVoiceGenerator(PipelineStep):
    """Interface for voice generation."""

    def __init__(self):
        super().__init__("VoiceGenerator")

    @abstractmethod
    async def generate_voice(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Generate voice audio with timestamps.

        Returns:
            Tuple of (audio_path, srt_path, word_timestamps)
        """
        pass


class ISubtitleRenderer(PipelineStep):
    """Interface for subtitle rendering."""

    def __init__(self):
        super().__init__("SubtitleRenderer")

    @abstractmethod
    async def render_subtitles(
        self,
        video_path: Path,
        audio_path: Path,
        srt_path: Path,
        output_path: Path,
    ) -> Path:
        """Render karaoke-style subtitles on video."""
        pass


# =============================================================================
# Exceptions
# =============================================================================


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    pass


class TopicSelectionError(PipelineError):
    """Error during topic selection."""

    pass


class ResearchError(PipelineError):
    """Error during research."""

    pass


class ScriptPlanningError(PipelineError):
    """Error during script planning."""

    pass


class VideoSearchError(PipelineError):
    """Error during video search."""

    pass


class VideoDownloadError(PipelineError):
    """Error during video download."""

    pass


class VideoAssemblyError(PipelineError):
    """Error during video assembly."""

    pass


class CaptionGenerationError(PipelineError):
    """Error during caption generation."""

    pass


class VoiceGenerationError(PipelineError):
    """Error during voice generation."""

    pass


class SubtitleRenderError(PipelineError):
    """Error during subtitle rendering."""

    pass


class ThumbnailGenerationError(PipelineError):
    """Error during thumbnail generation."""

    pass
