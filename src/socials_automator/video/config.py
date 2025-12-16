"""Configuration for video generation."""

import os
from typing import Optional

from pydantic import BaseModel, Field

from socials_automator.constants import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    VIDEO_CODEC,
    AUDIO_CODEC,
    VIDEO_DEFAULT_DURATION_SECONDS,
    PEXELS_VIDEOS_PER_SEARCH,
    CLIP_MIN_DURATION_SECONDS,
    CLIP_MAX_DURATION_SECONDS,
)
from .models import SubtitleAnimation, SubtitlePosition, SubtitleStyle


# Voice presets for different content types
VOICE_PRESETS = {
    # English - US
    "professional_male": "en-US-GuyNeural",
    "professional_female": "en-US-AriaNeural",
    "friendly_male": "en-US-DavisNeural",
    "friendly_female": "en-US-JennyNeural",
    "energetic": "en-US-SaraNeural",
    # English - UK
    "british_male": "en-GB-RyanNeural",
    "british_female": "en-GB-SoniaNeural",
    # Other languages
    "spanish": "es-ES-ElviraNeural",
    "portuguese_br": "pt-BR-FranciscaNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
}


class TTSConfig(BaseModel):
    """Text-to-speech configuration."""

    provider: str = "edge-tts"
    voice: str = "en-US-AriaNeural"
    rate: str = "+0%"  # Speed: -50% to +100%
    pitch: str = "+0Hz"  # Pitch: -50Hz to +50Hz
    volume: str = "+0%"  # Volume: -50% to +50%

    @classmethod
    def from_preset(cls, preset: str) -> "TTSConfig":
        """Create config from a voice preset name."""
        voice = VOICE_PRESETS.get(preset, preset)
        return cls(voice=voice)


class PexelsConfig(BaseModel):
    """Pexels API configuration."""

    api_key_env: str = "PEXELS_API_KEY"
    prefer_orientation: str = "portrait"
    fallback_orientation: str = "landscape"
    quality: str = "hd"
    per_page: int = Field(default=PEXELS_VIDEOS_PER_SEARCH, ge=1, le=80)

    @property
    def api_key(self) -> Optional[str]:
        """Get API key from environment."""
        return os.environ.get(self.api_key_env)


class OutputConfig(BaseModel):
    """Video output configuration."""

    width: int = VIDEO_WIDTH
    height: int = VIDEO_HEIGHT
    fps: int = VIDEO_FPS
    duration: int = VIDEO_DEFAULT_DURATION_SECONDS
    codec: str = VIDEO_CODEC
    audio_codec: str = AUDIO_CODEC
    bitrate: str = "8M"

    @property
    def resolution(self) -> tuple[int, int]:
        """Get resolution as tuple."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.width / self.height


class VideoGeneratorConfig(BaseModel):
    """Complete video generator configuration."""

    tts: TTSConfig = Field(default_factory=TTSConfig)
    pexels: PexelsConfig = Field(default_factory=PexelsConfig)
    subtitles: SubtitleStyle = Field(default_factory=SubtitleStyle)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Generation options
    target_duration: int = Field(default=VIDEO_DEFAULT_DURATION_SECONDS, ge=15, le=180)
    words_per_minute: int = Field(default=150, ge=100, le=200)
    min_scene_duration: float = Field(default=CLIP_MIN_DURATION_SECONDS, ge=1.0)
    max_scene_duration: float = Field(default=CLIP_MAX_DURATION_SECONDS, le=30.0)

    # Paths
    temp_dir: Optional[str] = None
    ffmpeg_path: Optional[str] = None

    @classmethod
    def default(cls) -> "VideoGeneratorConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> "VideoGeneratorConfig":
        """Create configuration from dictionary."""
        return cls(**data)

    def validate_pexels_key(self) -> bool:
        """Check if Pexels API key is available."""
        return self.pexels.api_key is not None


# Keyword fallback mapping for stock footage searches
KEYWORD_FALLBACKS = {
    "technology": ["computer", "digital", "tech office"],
    "ai": ["artificial intelligence", "robot", "futuristic"],
    "productivity": ["working", "office", "laptop"],
    "tips": ["tutorial", "demonstration", "how to"],
    "business": ["corporate", "meeting", "professional"],
    "coding": ["programming", "developer", "software"],
    "marketing": ["advertising", "social media", "digital marketing"],
    "finance": ["money", "banking", "investment"],
    "health": ["wellness", "fitness", "medical"],
    "education": ["learning", "school", "study"],
}


def get_keyword_fallbacks(keyword: str) -> list[str]:
    """Get fallback keywords for a search term."""
    keyword_lower = keyword.lower()
    for category, fallbacks in KEYWORD_FALLBACKS.items():
        if category in keyword_lower:
            return fallbacks
    return ["technology abstract"]
