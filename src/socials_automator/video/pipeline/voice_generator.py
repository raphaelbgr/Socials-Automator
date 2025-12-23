"""Voice generation using edge-tts, TikTok TTS, ElevenLabs, or RVC.

Generates voiceover audio with word-level timestamps
for karaoke-style subtitle synchronization.

Supports four backends:
- edge-tts: Free, unlimited usage (Microsoft Edge voices) - has word timestamps
- tiktok: Free, uses TikTok's TTS voices via proxy - estimated timestamps
- elevenlabs: Premium quality (requires API key, free tier: 10k chars/month)
- rvc_adam: Local RVC with ElevenLabs Adam clone - FREE, UNLIMITED, has word timestamps!
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from .base import (
    IVoiceGenerator,
    PipelineContext,
    VoiceGenerationError,
)


def _sanitize_for_tts(text: str) -> str:
    """Sanitize text for TTS - remove hashtags, emojis, and other non-speakable content.

    This is the LAST LINE OF DEFENSE before text goes to the TTS engine.
    Hashtags and emojis should NEVER be spoken aloud.

    Args:
        text: Raw narration text.

    Returns:
        Clean text suitable for TTS.
    """
    if not text:
        return text

    # Remove hashtags (e.g., #news #entertainment)
    text = re.sub(r'#\w+', '', text)

    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental symbols
        "\U00002600-\U000026FF"  # Misc symbols
        "\U0001FA00-\U0001FA6F"  # Chess symbols
        "\U0001FA70-\U0001FAFF"  # Symbols extended
        "\U00002300-\U000023FF"  # Misc technical
        "]+",
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)

    # Clean up multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Edge-TTS voice presets
VOICE_PRESETS = {
    # Standard voices
    "professional_female": "en-US-AriaNeural",
    "professional_male": "en-US-GuyNeural",
    "friendly_female": "en-US-JennyNeural",
    "friendly_male": "en-US-DavisNeural",
    "energetic": "en-US-SaraNeural",
    "british_female": "en-GB-SoniaNeural",
    "british_male": "en-GB-RyanNeural",
    # Excited male voices (best for Reels/TikTok)
    "christopher_excited": "en-US-ChristopherNeural",
    "brian_excited": "en-US-BrianNeural",
    "andrew_excited": "en-US-AndrewNeural",
}

# Preset configurations with rate/pitch adjustments
VOICE_PRESET_CONFIGS = {
    "christopher_excited": {"rate": "+15%", "pitch": "+3Hz"},
    "brian_excited": {"rate": "+15%", "pitch": "+3Hz"},
    "andrew_excited": {"rate": "+15%", "pitch": "+3Hz"},
}

# TikTok TTS voices (free, via proxy API)
TIKTOK_VOICES = {
    # US Male voices
    "tiktok_male_1": "en_us_006",  # Joey - most popular
    "tiktok_male_2": "en_us_007",
    "tiktok_male_3": "en_us_009",
    "tiktok_male_4": "en_us_010",
    # UK Male voices
    "tiktok_uk_male_1": "en_uk_001",
    "tiktok_uk_male_2": "en_uk_003",
    # AU Male
    "tiktok_au_male": "en_au_002",
    # Character voices
    "tiktok_narration": "en_male_narration",  # Story narrator
    "tiktok_funny": "en_male_funny",
    "tiktok_serious": "en_male_cody",  # Cody - serious tone
    "tiktok_jomboy": "en_male_jomboy",  # Game On
    "tiktok_deadpool": "en_male_deadpool",
    "tiktok_jarvis": "en_male_jarvis",  # Alfred/Jarvis
    "tiktok_wizard": "en_male_wizard",
    "tiktok_trevor": "en_male_trevor",  # Marty
    # Female voices
    "tiktok_female_1": "en_us_001",
    "tiktok_female_2": "en_us_002",
    "tiktok_au_female": "en_au_001",
}

# TikTok TTS API endpoint (via proxy)
TIKTOK_TTS_ENDPOINT = "https://tiktok-tts.weilnet.workers.dev/api/generation"

# ElevenLabs voice IDs (requires ELEVENLABS_API_KEY)
ELEVENLABS_VOICES = {
    "adam": "pNInz6obpgDQGcFmaJgB",  # Deep American male - viral voice
    "charlie": "IKne3meq5aSn9XLyUdCD",  # Natural Australian male
    "clyde": "2EiwWnXFnvU5JabPnv8n",  # War veteran character
    "daniel": "onwK4e9ZLuTAKqWW03F9",  # British male
    "james": "ZQe5CZNOzWyzPSCn5a3c",  # Australian male
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # American female
}

# Fish Audio voices (requires FISH_AUDIO_API_KEY)
# Free tier: 1 hour of audio per month
FISH_AUDIO_VOICES = {
    "fish_adam": "728f6ff2240d49308e8137ffe66008e2",  # ElevenLabs Adam clone - THE viral TikTok voice!
}

# RVC Adam - Local voice conversion (FREE, UNLIMITED!)
# Uses sandboxed Python 3.12 environment in rvc_sandbox/
RVC_SANDBOX_DIR = Path(__file__).parent.parent.parent.parent.parent / "rvc_sandbox"
RVC_PYTHON = RVC_SANDBOX_DIR / "python" / "python.exe"
RVC_SCRIPT = RVC_SANDBOX_DIR / "rvc_adam_tts.py"
RVC_VOICES = {
    "rvc_adam": "adam_elevenlabs",  # THE viral TikTok voice - runs locally!
    "rvc_adam_excited": "adam_elevenlabs",  # Excited version with faster rate/higher pitch
    "tiktok-adam": "adam_elevenlabs",  # Alias for --voice tiktok-adam
    "tiktok_adam": "adam_elevenlabs",  # Alias with underscore
    "adam": "adam_elevenlabs",  # Short alias
    "adam_excited": "adam_elevenlabs",  # Excited alias
}

# RVC preset configurations for excitement/tone
RVC_PRESET_CONFIGS = {
    "rvc_adam_excited": {"rate": "+12%", "pitch": "+3Hz"},
    "adam_excited": {"rate": "+12%", "pitch": "+3Hz"},
}


class VoiceGenerator(IVoiceGenerator):
    """Generates voiceover using edge-tts, TikTok, or ElevenLabs."""

    def __init__(
        self,
        voice: str = "rvc_adam",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        backend: str = "auto",
    ):
        """Initialize voice generator.

        Args:
            voice: Voice name or preset key.
                   For edge-tts: preset name or full voice ID
                   For tiktok: "tiktok_male_1", "tiktok_narration", etc.
                   For ElevenLabs: "adam", "charlie", etc.
            rate: Speech rate adjustment (-50% to +100%) - edge-tts only.
            pitch: Pitch adjustment (-50Hz to +50Hz) - edge-tts only.
            backend: "edge-tts", "tiktok", "elevenlabs", or "auto"
                     Auto selects based on voice name prefix
        """
        super().__init__()

        # Determine backend from voice name or explicit setting
        self.backend = backend
        if backend == "auto":
            voice_lower = voice.lower()
            if voice_lower.startswith("rvc_") or voice_lower in RVC_VOICES:
                self.backend = "rvc"
            elif voice_lower.startswith("fish_") or voice_lower in FISH_AUDIO_VOICES:
                if os.environ.get("FISH_AUDIO_API_KEY"):
                    self.backend = "fish"
                else:
                    raise VoiceGenerationError(
                        "FISH_AUDIO_API_KEY not set. Get a free key at https://fish.audio/"
                    )
            elif voice_lower.startswith("tiktok_") or voice_lower in TIKTOK_VOICES:
                self.backend = "tiktok"
            elif voice_lower in ELEVENLABS_VOICES and os.environ.get("ELEVENLABS_API_KEY"):
                self.backend = "elevenlabs"
            else:
                self.backend = "edge-tts"

        # Configure voice based on backend
        if self.backend == "rvc":
            self.voice = RVC_VOICES.get(voice.lower(), voice)
            # Apply excited preset if using an excited voice variant
            preset_config = RVC_PRESET_CONFIGS.get(voice.lower(), {})
            self.rate = preset_config.get("rate", rate)
            self.pitch = preset_config.get("pitch", pitch)
        elif self.backend == "fish":
            self.voice = FISH_AUDIO_VOICES.get(voice.lower(), voice)
            self.rate = rate
            self.pitch = pitch
        elif self.backend == "elevenlabs":
            self.voice = ELEVENLABS_VOICES.get(voice.lower(), voice)
            self.rate = rate
            self.pitch = pitch
        elif self.backend == "tiktok":
            self.voice = TIKTOK_VOICES.get(voice.lower(), voice)
            self.rate = rate
            self.pitch = pitch
        else:
            # edge-tts
            self.voice = VOICE_PRESETS.get(voice, voice)
            preset_config = VOICE_PRESET_CONFIGS.get(voice, {})
            self.rate = preset_config.get("rate", rate)
            self.pitch = preset_config.get("pitch", pitch)

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute voice generation step.

        Args:
            context: Pipeline context with script.

        Returns:
            Updated context with audio and SRT paths.
        """
        if not context.script:
            raise VoiceGenerationError("No script available for voice generation")

        self.log_start("Generating voiceover...")

        try:
            # Sanitize text to remove hashtags, emojis, mentions before TTS
            narration_text = _sanitize_for_tts(context.script.full_narration)

            audio_path, srt_path, timestamps = await self.generate_voice(
                narration_text,
                context.temp_dir,
            )

            context.audio_path = audio_path
            context.srt_path = srt_path

            # Update script segment times based on actual voice timing
            # This is critical for syncing video transitions with speech
            if timestamps:
                self._update_segment_timing(context.script, timestamps)
                total_duration = timestamps[-1]["end_ms"] / 1000 if timestamps else 60.0
                self.log_progress(f"Actual audio duration: {total_duration:.1f}s")

            self.log_success(
                f"Generated voiceover: {audio_path.name}, "
                f"{len(timestamps)} word timestamps"
            )
            return context

        except Exception as e:
            self.log_error(f"Voice generation failed: {e}")
            raise VoiceGenerationError(f"Failed to generate voice: {e}") from e

    def _update_segment_timing(
        self,
        script: "VideoScript",
        timestamps: list[dict],
    ) -> None:
        """Update script segment times based on actual voice timing.

        Uses proportional distribution based on word counts for reliable
        segment boundary detection. Word-level matching is unreliable due
        to TTS pronunciation variations.

        Args:
            script: The video script with segments.
            timestamps: Word timestamps from TTS (each has word, start_ms, end_ms).
        """
        if not timestamps or not script.segments:
            return

        audio_end_ms = timestamps[-1]["end_ms"]

        # Count words in each part of the script
        hook_word_count = len(script.hook.split())
        cta_word_count = len(script.cta.split()) if script.cta else 0
        segment_word_counts = [len(seg.text.split()) for seg in script.segments]
        total_words = hook_word_count + sum(segment_word_counts) + cta_word_count

        if total_words == 0:
            return

        # Distribute timestamps proportionally by word count
        # This is more reliable than word matching which fails on punctuation/pronunciation
        ts_per_word = len(timestamps) / total_words

        # Calculate hook end (in timestamp indices)
        hook_end_idx = int(hook_word_count * ts_per_word)
        hook_end_idx = min(hook_end_idx, len(timestamps) - 1)

        # Calculate segment boundaries (contiguous - no overlaps)
        current_idx = hook_end_idx
        previous_end_time = timestamps[min(hook_end_idx, len(timestamps) - 1)]["start_ms"] / 1000

        for i, segment in enumerate(script.segments):
            segment_start_idx = current_idx
            words_in_segment = segment_word_counts[i]
            segment_end_idx = current_idx + int(words_in_segment * ts_per_word)
            segment_end_idx = min(segment_end_idx, len(timestamps) - 1)

            # Ensure at least 1 timestamp per segment
            if segment_end_idx <= segment_start_idx and segment_start_idx < len(timestamps) - 1:
                segment_end_idx = segment_start_idx + 1

            # Get end time from timestamps
            segment_end_ms = timestamps[min(segment_end_idx, len(timestamps) - 1)]["end_ms"]

            # Update segment timing (start = previous end for contiguous segments)
            segment.start_time = previous_end_time
            segment.end_time = segment_end_ms / 1000
            segment.duration_seconds = segment.end_time - segment.start_time

            # Track for next segment
            previous_end_time = segment.end_time
            current_idx = segment_end_idx

        # CRITICAL: Ensure segments cover the FULL audio duration
        # The hook is narrated at the beginning and CTA at the end,
        # but neither has a dedicated segment. So:
        # - Segment 1's video must start at 0 (to cover hook)
        # - Last segment's video must extend to audio end (to cover CTA)
        if script.segments and timestamps:
            # Extend segment 1 to start at 0 (cover hook)
            first_segment = script.segments[0]
            if first_segment.start_time > 0:
                hook_duration = first_segment.start_time
                first_segment.start_time = 0
                first_segment.duration_seconds = first_segment.end_time - first_segment.start_time
                self.log_detail(f"Extended segment 1 to include hook ({hook_duration:.1f}s)")

            # Extend last segment to cover CTA (end of audio)
            last_segment = script.segments[-1]
            audio_end_time = timestamps[-1]["end_ms"] / 1000
            if last_segment.end_time < audio_end_time:
                cta_duration = audio_end_time - last_segment.end_time
                last_segment.end_time = audio_end_time
                last_segment.duration_seconds = last_segment.end_time - last_segment.start_time
                self.log_detail(f"Extended last segment to include CTA ({cta_duration:.1f}s)")

        # Log the updated timing with clear breakdown
        audio_end_time = timestamps[-1]["end_ms"] / 1000 if timestamps else 0
        total_segment_duration = sum(seg.duration_seconds for seg in script.segments)

        self.log_progress("--- Segment Timing (Actual, After Voice) ---")
        for seg in script.segments:
            self.log_progress(
                f"  Seg {seg.index}:    {seg.start_time:.1f}s - {seg.end_time:.1f}s "
                f"({seg.duration_seconds:.1f}s)"
            )
        self.log_progress(f"  TOTAL:    {total_segment_duration:.1f}s (audio: {audio_end_time:.1f}s)")

        # Warn if there's a mismatch
        if abs(total_segment_duration - audio_end_time) > 0.5:
            self.log_progress(f"  [!] WARNING: Segment coverage ({total_segment_duration:.1f}s) != audio ({audio_end_time:.1f}s)")

    async def generate_voice(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Generate voice audio with timestamps.

        Args:
            text: Text to convert to speech.
            output_dir: Directory for output files.

        Returns:
            Tuple of (audio_path, srt_path, word_timestamps).
        """
        if self.backend == "rvc":
            return await self._generate_voice_rvc(text, output_dir)
        elif self.backend == "fish":
            return await self._generate_voice_fish(text, output_dir)
        elif self.backend == "elevenlabs":
            return await self._generate_voice_elevenlabs(text, output_dir)
        elif self.backend == "tiktok":
            return await self._generate_voice_tiktok(text, output_dir)
        else:
            return await self._generate_voice_edge_tts(text, output_dir)

    async def _generate_voice_rvc(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Generate voice using local RVC with Adam model.

        Uses sandboxed Python 3.12 environment to run RVC.
        This is FREE and UNLIMITED - runs entirely locally!
        Word timestamps are precise (from edge-tts base audio).

        Includes retry logic for concurrent access (GPU/model resource conflicts).
        """
        import asyncio

        # Verify RVC sandbox exists
        if not RVC_PYTHON.exists():
            raise VoiceGenerationError(
                f"RVC sandbox not found at {RVC_SANDBOX_DIR}. "
                "Run the setup script to install RVC."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / "voiceover.mp3"
        srt_path = output_dir / "voiceover.srt"

        self.log_progress("Using RVC Adam voice (local, free, unlimited!)")
        if self.rate != "+0%" or self.pitch != "+0Hz":
            self.log_progress(f"Excitement: rate={self.rate}, pitch={self.pitch}")

        # Call the sandboxed RVC script
        cmd = [
            str(RVC_PYTHON),
            str(RVC_SCRIPT),
            "--text", text,
            "--output", str(audio_path),
            "--srt", str(srt_path),
            "--rate", self.rate,
            "--pitch", self.pitch,
        ]

        # Retry configuration for concurrent access handling
        max_retries = 10
        retry_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                if attempt > 1:
                    self.log_progress(f"Retry attempt {attempt}/{max_retries}...")
                else:
                    self.log_progress("Calling RVC sandbox...")

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip()

                    # Check for errors that indicate resource contention (retry-able)
                    retryable_errors = [
                        "CUDA out of memory",
                        "cuda",
                        "GPU",
                        "memory",
                        "RuntimeError",
                        "torch",
                        "model",
                        "index",
                        "lock",
                        "busy",
                        "in use",
                    ]

                    is_retryable = any(
                        err.lower() in error_msg.lower()
                        for err in retryable_errors
                    )

                    if is_retryable and attempt < max_retries:
                        self.log_progress(
                            f"RVC resource busy, waiting {retry_delay}s before retry..."
                        )
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        raise VoiceGenerationError(f"RVC failed: {error_msg}")

                # Parse timestamps from stdout (JSON)
                word_timestamps = json.loads(result.stdout.strip())

                self.log_progress(
                    f"Generated {len(word_timestamps)} word timestamps"
                )

                return audio_path, srt_path, word_timestamps

            except subprocess.TimeoutExpired:
                if attempt < max_retries:
                    self.log_progress(
                        f"RVC timed out, waiting {retry_delay}s before retry..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                raise VoiceGenerationError("RVC timed out after 5 minutes (max retries)")

            except json.JSONDecodeError as e:
                # JSON decode errors are not retryable - something is fundamentally wrong
                raise VoiceGenerationError(f"Failed to parse RVC output: {e}")

        # Should not reach here, but just in case
        raise VoiceGenerationError("RVC failed after maximum retries")

    async def _generate_voice_elevenlabs(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Generate voice using ElevenLabs API.

        Note: ElevenLabs provides superior quality but limited free tier (10k chars/month).
        Word timestamps are estimated based on word length since ElevenLabs
        doesn't provide word-level timing in the basic API.
        """
        try:
            from elevenlabs.client import ElevenLabs
        except ImportError as e:
            raise VoiceGenerationError(
                "elevenlabs is not installed. Run: pip install elevenlabs"
            ) from e

        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise VoiceGenerationError(
                "ELEVENLABS_API_KEY not set. Get a free key at https://elevenlabs.io/"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / "voiceover.mp3"
        srt_path = output_dir / "voiceover.srt"

        self.log_progress(f"Using ElevenLabs voice: {self.voice}")
        self.log_progress("Generating audio with ElevenLabs...")

        client = ElevenLabs(api_key=api_key)

        # Generate audio
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=self.voice,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        # Write audio to file
        with open(audio_path, "wb") as f:
            for chunk in audio_generator:
                f.write(chunk)

        self.log_progress("Audio generated, estimating word timestamps...")

        # Estimate word timestamps (ElevenLabs basic API doesn't provide these)
        # Average speaking rate is ~150 words/minute = 400ms per word
        word_timestamps = self._estimate_word_timestamps(text)

        # Generate SRT from estimated timestamps
        self._generate_srt_from_timestamps(srt_path, word_timestamps)

        self.log_progress(
            f"Generated {len(word_timestamps)} word timestamps (estimated)"
        )

        return audio_path, srt_path, word_timestamps

    def _estimate_word_timestamps(self, text: str) -> list[dict]:
        """Estimate word timestamps based on word length.

        This is a simple estimation - for precise timestamps, use edge-tts
        or ElevenLabs Speech-to-Speech with alignment.
        """
        words = text.split()
        timestamps = []
        current_ms = 0

        # Average ~400ms per word, adjusted by word length
        base_duration = 350  # ms per word
        char_factor = 30  # additional ms per character

        for word in words:
            # Calculate duration based on word length
            duration = base_duration + (len(word) * char_factor)

            # Add pause after punctuation
            if word.endswith(('.', '!', '?')):
                duration += 200
            elif word.endswith(','):
                duration += 100

            timestamps.append({
                "word": word,
                "start_ms": current_ms,
                "end_ms": current_ms + duration,
            })

            current_ms += duration

        return timestamps

    def _generate_srt_from_timestamps(
        self,
        srt_path: Path,
        word_timestamps: list[dict],
    ) -> None:
        """Generate SRT file from word timestamps."""
        def ms_to_srt_time(ms: int) -> str:
            hours = ms // 3600000
            minutes = (ms % 3600000) // 60000
            seconds = (ms % 60000) // 1000
            millis = ms % 1000
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

        with open(srt_path, "w", encoding="utf-8") as f:
            for i, ts in enumerate(word_timestamps, 1):
                f.write(f"{i}\n")
                f.write(f"{ms_to_srt_time(ts['start_ms'])} --> {ms_to_srt_time(ts['end_ms'])}\n")
                f.write(f"{ts['word']}\n\n")

    async def _generate_voice_tiktok(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Generate voice using TikTok TTS API via proxy.

        Note: TikTok TTS is free but doesn't provide word timestamps.
        Timestamps are estimated based on word length.
        """
        import base64
        import re
        try:
            import requests
        except ImportError as e:
            raise VoiceGenerationError(
                "requests is not installed. Run: pip install requests"
            ) from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / "voiceover.mp3"
        srt_path = output_dir / "voiceover.srt"

        self.log_progress(f"Using TikTok voice: {self.voice}")

        # Split text into chunks (TikTok API limit is 300 chars)
        text_chunks = self._split_text_for_tiktok(text)
        self.log_progress(f"Split into {len(text_chunks)} chunks")

        audio_chunks = []
        for i, chunk in enumerate(text_chunks):
            self.log_progress(f"Generating chunk {i+1}/{len(text_chunks)}...")
            try:
                response = requests.post(
                    TIKTOK_TTS_ENDPOINT,
                    json={"text": chunk, "voice": self.voice},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                audio_b64 = data.get("data")
                if audio_b64:
                    audio_chunks.append(base64.b64decode(audio_b64))
                else:
                    raise VoiceGenerationError(f"No audio data in TikTok response: {data}")
            except requests.RequestException as e:
                raise VoiceGenerationError(f"TikTok TTS API error: {e}") from e

        # Combine audio chunks
        self.log_progress("Combining audio chunks...")
        with open(audio_path, "wb") as f:
            for chunk in audio_chunks:
                f.write(chunk)

        self.log_progress("Audio generated, estimating word timestamps...")

        # Estimate word timestamps
        word_timestamps = self._estimate_word_timestamps(text)

        # Generate SRT from estimated timestamps
        self._generate_srt_from_timestamps(srt_path, word_timestamps)

        self.log_progress(
            f"Generated {len(word_timestamps)} word timestamps (estimated)"
        )

        return audio_path, srt_path, word_timestamps

    def _split_text_for_tiktok(self, text: str, limit: int = 300) -> list[str]:
        """Split text into chunks for TikTok API (max 300 chars)."""
        import re

        # Split by punctuation first
        chunks = re.findall(r'.*?[.,!?:;-]|.+', text)

        # Further split long chunks by spaces
        result = []
        for chunk in chunks:
            if len(chunk.encode("utf-8")) > limit:
                words = chunk.split()
                current = ""
                for word in words:
                    if len((current + " " + word).encode("utf-8")) <= limit:
                        current = (current + " " + word).strip()
                    else:
                        if current:
                            result.append(current)
                        current = word
                if current:
                    result.append(current)
            else:
                result.append(chunk)

        # Merge small chunks
        merged = []
        current = ""
        for chunk in result:
            if len((current + chunk).encode("utf-8")) <= limit:
                current += chunk
            else:
                if current:
                    merged.append(current)
                current = chunk
        if current:
            merged.append(current)

        return merged

    async def _generate_voice_edge_tts(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Generate voice using edge-tts (free, unlimited)."""
        try:
            import edge_tts
        except ImportError as e:
            raise VoiceGenerationError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from e

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / "voiceover.mp3"
        srt_path = output_dir / "voiceover.srt"

        self.log_progress(f"Using edge-tts voice: {self.voice} (rate={self.rate}, pitch={self.pitch})")

        communicate = edge_tts.Communicate(
            text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch,
            boundary="WordBoundary",  # Enable word-level timestamps for subtitles
        )

        submaker = edge_tts.SubMaker()
        word_timestamps = []

        self.log_progress("Generating audio and timestamps...")

        with open(audio_path, "wb") as audio_file:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    offset = chunk["offset"]
                    duration = chunk["duration"]
                    word = chunk["text"]

                    # Create subtitle entry
                    # New edge_tts API: feed takes the entire chunk
                    # Old API: create_sub takes (offset, duration), word
                    try:
                        submaker.feed(chunk)  # New API - pass entire chunk
                    except TypeError:
                        try:
                            submaker.feed(offset, duration, word)  # Alternate new API
                        except TypeError:
                            submaker.create_sub((offset, duration), word)  # Old API

                    # Store timestamp (convert from 100-nanosecond units to ms)
                    start_ms = offset // 10000
                    end_ms = (offset + duration) // 10000

                    word_timestamps.append({
                        "word": word,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                    })

        # Save SRT file
        self.log_progress("Saving SRT file...")

        with open(srt_path, "w", encoding="utf-8") as srt_file:
            # get_srt for new API, generate_subs for old
            if hasattr(submaker, 'get_srt'):
                srt_file.write(submaker.get_srt())
            else:
                srt_file.write(submaker.generate_subs())

        self.log_progress(
            f"Generated {len(word_timestamps)} word timestamps"
        )

        return audio_path, srt_path, word_timestamps

    def generate_voice_sync(
        self,
        text: str,
        output_dir: Path,
    ) -> tuple[Path, Path, list[dict]]:
        """Synchronous wrapper for generate_voice().

        Args:
            text: Text to convert to speech.
            output_dir: Directory for output files.

        Returns:
            Tuple of (audio_path, srt_path, word_timestamps).
        """
        import asyncio
        return asyncio.run(self.generate_voice(text, output_dir))
