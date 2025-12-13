"""Pytest fixtures for video module tests."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from socials_automator.video import (
    OutputConfig,
    PexelsConfig,
    SubtitleStyle,
    TTSConfig,
    VideoClip,
    VideoGeneratorConfig,
    VideoScene,
    VideoScript,
    VoiceoverResult,
    WordTimestamp,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_script():
    """Create a sample video script."""
    return VideoScript(
        title="Test Video",
        hook="This is the hook.",
        scenes=[
            VideoScene(
                text="First scene content.",
                duration_seconds=10.0,
                video_keywords=["technology", "abstract"],
            ),
            VideoScene(
                text="Second scene content.",
                duration_seconds=10.0,
                video_keywords=["business", "office"],
            ),
            VideoScene(
                text="Third scene content.",
                duration_seconds=10.0,
                video_keywords=["success", "achievement"],
            ),
        ],
        cta="Follow for more!",
        total_duration=60,
    )


@pytest.fixture
def sample_word_timestamps():
    """Create sample word timestamps."""
    return [
        WordTimestamp(word="This", start_ms=0, end_ms=200),
        WordTimestamp(word="is", start_ms=200, end_ms=400),
        WordTimestamp(word="a", start_ms=400, end_ms=500),
        WordTimestamp(word="test", start_ms=500, end_ms=800),
    ]


@pytest.fixture
def sample_voiceover_result(temp_dir, sample_word_timestamps):
    """Create a sample voiceover result."""
    audio_path = temp_dir / "voiceover.mp3"
    srt_path = temp_dir / "voiceover.srt"

    # Create dummy files
    audio_path.touch()
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:00,200\nThis\n\n"
        "2\n00:00:00,200 --> 00:00:00,400\nis\n\n"
        "3\n00:00:00,400 --> 00:00:00,500\na\n\n"
        "4\n00:00:00,500 --> 00:00:00,800\ntest\n\n"
    )

    return VoiceoverResult(
        audio_path=audio_path,
        srt_path=srt_path,
        duration_seconds=0.8,
        word_timestamps=sample_word_timestamps,
    )


@pytest.fixture
def sample_video_clip(temp_dir):
    """Create a sample video clip."""
    clip_path = temp_dir / "scene_01.mp4"
    clip_path.touch()

    return VideoClip(
        path=clip_path,
        source_url="https://pexels.com/video/123",
        duration_seconds=10.0,
        width=1080,
        height=1920,
        scene_index=1,
        keywords_used=["technology"],
    )


@pytest.fixture
def default_config():
    """Create default video generator configuration."""
    return VideoGeneratorConfig.default()


@pytest.fixture
def tts_config():
    """Create TTS configuration."""
    return TTSConfig(
        voice="en-US-AriaNeural",
        rate="+0%",
        pitch="+0Hz",
        volume="+0%",
    )


@pytest.fixture
def pexels_config():
    """Create Pexels configuration."""
    return PexelsConfig(
        api_key_env="PEXELS_API_KEY",
        prefer_orientation="portrait",
        quality="hd",
    )


@pytest.fixture
def output_config():
    """Create output configuration."""
    return OutputConfig(
        width=1080,
        height=1920,
        fps=30,
        duration=60,
    )


@pytest.fixture
def subtitle_style():
    """Create subtitle style configuration."""
    return SubtitleStyle(
        font="Montserrat-Bold",
        font_size=60,
        color="white",
        highlight_color="#FFD700",
    )


@pytest.fixture
def mock_pexels_response():
    """Create mock Pexels API response."""
    return {
        "total_results": 1,
        "videos": [
            {
                "id": 123456,
                "url": "https://pexels.com/video/123456",
                "duration": 15,
                "video_files": [
                    {
                        "id": 1,
                        "quality": "hd",
                        "width": 1080,
                        "height": 1920,
                        "link": "https://player.vimeo.com/external/123.hd.mp4",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def mock_httpx_client(mock_pexels_response):
    """Create mock httpx client."""
    client = MagicMock()
    client.get = AsyncMock(
        return_value=MagicMock(
            json=MagicMock(return_value=mock_pexels_response),
            raise_for_status=MagicMock(),
        )
    )
    client.stream = MagicMock(
        return_value=AsyncMock(
            __aenter__=AsyncMock(
                return_value=MagicMock(
                    raise_for_status=MagicMock(),
                    aiter_bytes=AsyncMock(return_value=iter([b"video_data"])),
                )
            ),
            __aexit__=AsyncMock(),
        )
    )
    client.aclose = AsyncMock()
    return client
