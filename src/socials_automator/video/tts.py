"""Text-to-speech generation using edge-tts."""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .config import TTSConfig
from .models import TTSError, VoiceoverResult, WordTimestamp

logger = logging.getLogger(__name__)


class TTSGenerator:
    """Generate voiceover audio with word-level timestamps using edge-tts."""

    def __init__(self, config: Optional[TTSConfig] = None):
        """Initialize TTS generator.

        Args:
            config: TTS configuration. Uses defaults if not provided.
        """
        self.config = config or TTSConfig()

    async def generate(
        self,
        text: str,
        output_dir: Path,
        filename: str = "voiceover",
    ) -> VoiceoverResult:
        """Generate voiceover audio and word timestamps.

        Args:
            text: Text to convert to speech.
            output_dir: Directory to save output files.
            filename: Base filename (without extension).

        Returns:
            VoiceoverResult with paths and timing information.

        Raises:
            TTSError: If generation fails.
        """
        try:
            import edge_tts
        except ImportError as e:
            raise TTSError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / f"{filename}.mp3"
        srt_path = output_dir / f"{filename}.srt"

        logger.info(f"Generating voiceover with voice: {self.config.voice}")

        try:
            communicate = edge_tts.Communicate(
                text,
                voice=self.config.voice,
                rate=self.config.rate,
                pitch=self.config.pitch,
                volume=self.config.volume,
            )

            submaker = edge_tts.SubMaker()
            word_timestamps: list[WordTimestamp] = []

            with open(audio_path, "wb") as audio_file:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_file.write(chunk["data"])
                    elif chunk["type"] == "WordBoundary":
                        offset = chunk["offset"]
                        duration = chunk["duration"]
                        word = chunk["text"]

                        # Create subtitle
                        submaker.create_sub((offset, duration), word)

                        # Store word timestamp (convert from 100-nanosecond units to ms)
                        start_ms = offset // 10000
                        end_ms = (offset + duration) // 10000
                        word_timestamps.append(
                            WordTimestamp(
                                word=word,
                                start_ms=start_ms,
                                end_ms=end_ms,
                            )
                        )

            # Save SRT file
            with open(srt_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(submaker.generate_subs())

            # Calculate duration from last word timestamp
            duration_seconds = 0.0
            if word_timestamps:
                duration_seconds = word_timestamps[-1].end_seconds

            logger.info(
                f"Generated voiceover: {duration_seconds:.1f}s, "
                f"{len(word_timestamps)} words"
            )

            return VoiceoverResult(
                audio_path=audio_path,
                srt_path=srt_path,
                duration_seconds=duration_seconds,
                word_timestamps=word_timestamps,
            )

        except Exception as e:
            raise TTSError(f"Failed to generate voiceover: {e}") from e

    def generate_sync(
        self,
        text: str,
        output_dir: Path,
        filename: str = "voiceover",
    ) -> VoiceoverResult:
        """Synchronous wrapper for generate().

        Args:
            text: Text to convert to speech.
            output_dir: Directory to save output files.
            filename: Base filename (without extension).

        Returns:
            VoiceoverResult with paths and timing information.
        """
        return asyncio.run(self.generate(text, output_dir, filename))

    @staticmethod
    async def list_voices(language_filter: Optional[str] = None) -> list[dict]:
        """List available voices.

        Args:
            language_filter: Filter by language code (e.g., "en", "es").

        Returns:
            List of voice information dictionaries.
        """
        try:
            import edge_tts
        except ImportError as e:
            raise TTSError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from e

        voices = await edge_tts.list_voices()

        if language_filter:
            voices = [
                v for v in voices if v["Locale"].startswith(language_filter)
            ]

        return voices


def calculate_speech_duration(text: str, words_per_minute: int = 150) -> float:
    """Estimate speech duration for script validation.

    Args:
        text: Text to estimate duration for.
        words_per_minute: Speaking rate.

    Returns:
        Estimated duration in seconds.
    """
    word_count = len(text.split())
    return (word_count / words_per_minute) * 60


def normalize_audio(
    audio_path: Path,
    target_dbfs: float = -14.0,
    output_path: Optional[Path] = None,
) -> Path:
    """Normalize audio volume for consistent output.

    Args:
        audio_path: Path to input audio file.
        target_dbfs: Target volume level in dBFS.
        output_path: Path for output. If None, adds '_normalized' suffix.

    Returns:
        Path to normalized audio file.

    Raises:
        TTSError: If pydub is not installed or normalization fails.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise TTSError(
            "pydub is not installed. Run: pip install pydub"
        ) from e

    if output_path is None:
        output_path = audio_path.with_stem(f"{audio_path.stem}_normalized")

    try:
        audio = AudioSegment.from_mp3(str(audio_path))
        change_in_dbfs = target_dbfs - audio.dBFS
        normalized = audio.apply_gain(change_in_dbfs)
        normalized.export(str(output_path), format="mp3")
        return output_path
    except Exception as e:
        raise TTSError(f"Failed to normalize audio: {e}") from e


def add_background_music(
    voiceover_path: Path,
    music_path: Path,
    music_volume_db: float = -20,
    output_path: Optional[Path] = None,
) -> Path:
    """Mix voiceover with background music.

    Args:
        voiceover_path: Path to voiceover audio.
        music_path: Path to background music.
        music_volume_db: Volume adjustment for music (negative = quieter).
        output_path: Path for output. If None, adds '_with_music' suffix.

    Returns:
        Path to mixed audio file.

    Raises:
        TTSError: If pydub is not installed or mixing fails.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise TTSError(
            "pydub is not installed. Run: pip install pydub"
        ) from e

    if output_path is None:
        output_path = voiceover_path.with_stem(
            f"{voiceover_path.stem}_with_music"
        )

    try:
        voiceover = AudioSegment.from_mp3(str(voiceover_path))
        music = AudioSegment.from_mp3(str(music_path))

        # Loop music to match voiceover length
        if len(music) < len(voiceover):
            loops_needed = (len(voiceover) // len(music)) + 1
            music = music * loops_needed

        # Trim to voiceover length
        music = music[: len(voiceover)]

        # Lower music volume
        music = music + music_volume_db

        # Mix together
        mixed = voiceover.overlay(music)
        mixed.export(str(output_path), format="mp3")

        return output_path
    except Exception as e:
        raise TTSError(f"Failed to add background music: {e}") from e
