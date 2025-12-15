"""Video generation module for Instagram Reels and TikTok.

This module provides automated video generation with:
- AI-generated scripts
- Text-to-speech voiceover (edge-tts)
- Stock footage from Pexels
- Karaoke-style animated subtitles
- Final video assembly (MoviePy)

Example usage:
    from socials_automator.video import VideoGenerator, VideoGeneratorConfig, VideoScript

    # Create configuration
    config = VideoGeneratorConfig.default()

    # Create generator with custom script generator
    generator = VideoGenerator(
        config=config,
        script_generator=my_ai_script_generator,
    )

    # Generate video
    output = await generator.generate(
        topic="5 AI Tools for Productivity",
        output_dir=Path("output"),
    )

    print(f"Video saved to: {output.final_path}")

Or with a pre-made script:
    script = VideoScript(
        title="My Video",
        hook="Stop scrolling! This is important.",
        scenes=[
            VideoScene(
                text="First point...",
                duration_seconds=10,
                video_keywords=["technology", "abstract"],
            ),
            # ... more scenes
        ],
        cta="Follow for more!",
    )

    output = await generator.generate_from_script(
        script=script,
        output_dir=Path("output"),
    )
"""

from .assembler import VideoAssembler, get_video_info, select_clip_segment
from .config import (
    KEYWORD_FALLBACKS,
    VOICE_PRESETS,
    OutputConfig,
    PexelsConfig,
    TTSConfig,
    VideoGeneratorConfig,
    get_keyword_fallbacks,
)
from .generator import (
    ProgressCallback,
    VideoGenerator,
    create_output_directory,
    create_sample_script,
)
from .models import (
    GenerationProgress,
    ScriptGenerationError,
    StockFootageError,
    SubtitleAnimation,
    SubtitleError,
    SubtitlePosition,
    SubtitleStyle,
    TTSError,
    VideoAssemblyError,
    VideoClip,
    VideoGenerationError,
    VideoOutput,
    VideoScene,
    VideoScript,
    VoiceoverResult,
    WordTimestamp,
)
from .stock_footage import PexelsClient, StockFootageService, is_9_16
from .subtitles import (
    SRTEntry,
    SubtitleRenderer,
    group_words_into_phrases,
    parse_srt,
    word_timestamps_to_srt,
)
from .tts import (
    TTSGenerator,
    add_background_music,
    calculate_speech_duration,
    normalize_audio,
)

__all__ = [
    # Main classes
    "VideoGenerator",
    "VideoGeneratorConfig",
    "VideoScript",
    "VideoScene",
    "VideoOutput",
    # Config
    "TTSConfig",
    "PexelsConfig",
    "OutputConfig",
    "SubtitleStyle",
    "VOICE_PRESETS",
    "KEYWORD_FALLBACKS",
    # Components
    "TTSGenerator",
    "PexelsClient",
    "StockFootageService",
    "VideoAssembler",
    "SubtitleRenderer",
    # Models
    "VoiceoverResult",
    "VideoClip",
    "WordTimestamp",
    "GenerationProgress",
    "SubtitleAnimation",
    "SubtitlePosition",
    "SRTEntry",
    # Exceptions
    "VideoGenerationError",
    "ScriptGenerationError",
    "TTSError",
    "StockFootageError",
    "VideoAssemblyError",
    "SubtitleError",
    # Utilities
    "calculate_speech_duration",
    "normalize_audio",
    "add_background_music",
    "is_9_16",
    "parse_srt",
    "word_timestamps_to_srt",
    "group_words_into_phrases",
    "get_video_info",
    "select_clip_segment",
    "get_keyword_fallbacks",
    "create_output_directory",
    "create_sample_script",
    "ProgressCallback",
]

__version__ = "0.1.0"
