"""GPU-accelerated video rendering using NVENC.

Replaces VideoAssembler + SubtitleRenderer with NVENC-accelerated FFmpeg.
Two-pass approach:
1. Assembly pass: clips -> assembled.mp4 (NVENC)
2. Subtitle pass: assembled.mp4 + subtitles + audio -> final.mp4 (NVENC)

This keeps ThumbnailGenerator working between passes (needs video without subtitles).
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from socials_automator.constants import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    SUBTITLE_FONT_SIZE_DEFAULT,
    SUBTITLE_FONT_NAME,
    SUBTITLE_FONT_COLOR,
    SUBTITLE_HIGHLIGHT_COLOR,
    SUBTITLE_STROKE_COLOR,
    SUBTITLE_STROKE_WIDTH,
    SUBTITLE_POSITION_Y_PERCENT,
    get_temp_dir,
)

# FFmpeg ASS subtitle scaling
# ASS format uses a "script resolution" (PlayResY) that's typically ~288
# Font sizes in force_style are relative to this, not the video resolution
# We must scale down our pixel-based sizes to ASS coordinates
ASS_PLAY_RES_Y = 288  # FFmpeg's typical default for ASS scripts
from .base import (
    ArtifactStatus,
    ArtifactsInfo,
    IVideoAssembler,
    ISubtitleRenderer,
    PipelineContext,
    VideoAssemblyError,
    VideoMetadata,
    SubtitleRenderError,
)
from .gpu_utils import GPUInfo, validate_gpu_setup


def _parse_srt(srt_path: Path) -> list[tuple[str, float, float]]:
    """Parse SRT file into word entries.

    Args:
        srt_path: Path to SRT file.

    Returns:
        List of (text, start_seconds, end_seconds) tuples.
    """
    import re

    entries = []

    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"\n\n+", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            times = lines[1]
            text = " ".join(lines[2:])

            match = re.match(
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
                r"(\d{2}):(\d{2}):(\d{2}),(\d{3})",
                times,
            )

            if match:
                start_s = (
                    int(match.group(1)) * 3600
                    + int(match.group(2)) * 60
                    + int(match.group(3))
                    + int(match.group(4)) / 1000
                )
                end_s = (
                    int(match.group(5)) * 3600
                    + int(match.group(6)) * 60
                    + int(match.group(7))
                    + int(match.group(8)) / 1000
                )

                entries.append((text, start_s, end_s))

        except (ValueError, IndexError):
            continue

    return entries


def _generate_karaoke_ass(
    srt_path: Path,
    output_path: Path,
    font_name: str = "Montserrat-Bold",
    font_size: int = 80,
    stroke_width: int = 4,
    position_y_percent: float = 0.75,
) -> Path:
    """Generate ASS subtitle file with karaoke-style word highlighting.

    Creates ASS subtitles where each word is shown with its phrase,
    with the current word highlighted in yellow and others in white.
    This matches the CPU SubtitleRenderer's karaoke effect.

    Args:
        srt_path: Input SRT file with word-level timing.
        output_path: Output ASS file path.
        font_name: Font name for subtitles.
        font_size: Font size in pixels.
        stroke_width: Text outline width.
        position_y_percent: Vertical position (0.0-1.0 from top).

    Returns:
        Path to generated ASS file.
    """
    # Parse SRT
    entries = _parse_srt(srt_path)
    if not entries:
        # Create empty ASS file
        output_path.write_text("")
        return output_path

    # Group words into 3-word phrases (matching CPU renderer)
    phrases = []
    current_words = []
    phrase_start = None

    for text, start, end in entries:
        if not current_words:
            phrase_start = start
        current_words.append((text, start, end))

        if len(current_words) >= 3:
            phrases.append((current_words.copy(), phrase_start, end))
            current_words = []

    # Add remaining words
    if current_words and phrase_start is not None:
        phrases.append((current_words.copy(), phrase_start, entries[-1][2]))

    # ASS header
    # Use video resolution for PlayRes to avoid scaling issues
    ass_content = f"""[Script Info]
Title: Karaoke Subtitles
ScriptType: v4.00+
PlayResX: {VIDEO_WIDTH}
PlayResY: {VIDEO_HEIGHT}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,{stroke_width},0,2,10,10,{int((1-position_y_percent) * VIDEO_HEIGHT)},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Generate dialogue lines for each word highlight
    for phrase_words, phrase_start, phrase_end in phrases:
        for i, (word_text, word_start, word_end) in enumerate(phrase_words):
            # Build the phrase text with current word highlighted
            parts = []
            for j, (w, _, _) in enumerate(phrase_words):
                word_upper = w.upper()
                if j == i:
                    # Highlighted word in yellow (BGR format: 00FFFF = yellow)
                    parts.append(f"{{\\1c&H00FFFF&}}{word_upper}{{\\1c&HFFFFFF&}}")
                else:
                    # Normal word in white
                    parts.append(word_upper)

            text_line = " ".join(parts)

            # Convert times to ASS format (H:MM:SS.cc)
            start_h = int(word_start // 3600)
            start_m = int((word_start % 3600) // 60)
            start_s = word_start % 60
            end_h = int(word_end // 3600)
            end_m = int((word_end % 3600) // 60)
            end_s = word_end % 60

            start_str = f"{start_h}:{start_m:02d}:{start_s:05.2f}"
            end_str = f"{end_h}:{end_m:02d}:{end_s:05.2f}"

            ass_content += f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text_line}\n"

    output_path.write_text(ass_content, encoding="utf-8")
    return output_path


def escape_ffmpeg_filter_path(path: str) -> str:
    """Escape a file path for use in FFmpeg filter arguments.

    FFmpeg filter syntax requires specific escaping for special characters.
    See: https://ffmpeg.org/ffmpeg-utils.html#Quoting-and-escaping

    Escaping rules applied:
    1. Convert Windows backslashes to forward slashes
    2. Escape single quotes: ' -> '\\''  (close quote, escaped quote, reopen)
    3. Escape colons: : -> \\:  (backslash before colon)
    4. The result should be wrapped in single quotes by the caller

    Args:
        path: File path (Windows or Unix style)

    Returns:
        Escaped path string ready to be wrapped in single quotes

    Example:
        >>> path = r"C:\\Users\\test\\John's Video.srt"
        >>> escaped = escape_ffmpeg_filter_path(path)
        >>> filter_arg = f"subtitles='{escaped}'"
    """
    return (
        path
        .replace("\\", "/")           # Windows backslash -> forward slash
        .replace("'", "'\\''")        # Escape single quotes
        .replace(":", "\\:")          # Escape colons (Windows drive letter)
    )


class GPUVideoAssembler(IVideoAssembler):
    """GPU-accelerated video assembler using NVENC.

    Assembles clips into a video without audio (for ThumbnailGenerator).
    """

    WIDTH = VIDEO_WIDTH
    HEIGHT = VIDEO_HEIGHT
    FPS = VIDEO_FPS

    def __init__(self, gpu: Optional[GPUInfo] = None):
        """Initialize GPU video assembler.

        Args:
            gpu: GPU to use for encoding. If None, auto-selects.
        """
        super().__init__()
        self.gpu = gpu

    async def assemble(self, clips, script, output_path):
        """Abstract method implementation - redirects to execute()."""
        # This is called by the base interface but we use execute() instead
        raise NotImplementedError("Use execute() method instead")

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return float(result.stdout.strip())
        except Exception as e:
            raise VideoAssemblyError(f"Failed to get audio duration: {e}")

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute GPU video assembly.

        Args:
            context: Pipeline context with clips and audio.

        Returns:
            Updated context with assembled video path.
        """
        if not context.clips:
            raise VideoAssemblyError("No clips available for assembly")
        if not context.audio_path:
            raise VideoAssemblyError("No audio available for duration reference")
        if not context.script:
            raise VideoAssemblyError("No script available for metadata")

        self.log_start("Assembling video with GPU (NVENC)...")

        # Get target duration from audio
        audio_duration = self._get_audio_duration(context.audio_path)
        self.log_progress(f"Audio duration: {audio_duration:.1f}s (source of truth)")

        # Deduplicate clips by pexels_id BEFORE assembly (matching CPU assembler)
        seen_pexels_ids = set()
        unique_clips = []
        for clip in sorted(context.clips, key=lambda c: c.segment_index):
            if clip.pexels_id not in seen_pexels_ids:
                seen_pexels_ids.add(clip.pexels_id)
                unique_clips.append(clip)
            else:
                self.log_detail(f"Skipping duplicate pexels_id={clip.pexels_id} for segment {clip.segment_index}")

        if len(unique_clips) < len(context.clips):
            self.log_progress(f"Deduplicated: {len(context.clips)} clips -> {len(unique_clips)} unique clips")

        # Get output path
        if context.temp_dir:
            output_path = context.temp_dir / "assembled.mp4"
        else:
            output_path = get_temp_dir() / "assembled.mp4"

        try:
            # Assemble with GPU using deduplicated clips
            result_path = await self._assemble_with_gpu(
                clips=unique_clips,
                output_path=output_path,
                target_duration=audio_duration,
            )

            context.assembled_video_path = result_path

            # Create metadata using deduplicated clips (matching CPU assembler behavior)
            sorted_clips = unique_clips  # Already sorted and deduplicated

            # Calculate segment timing based on clip distribution
            num_clips = len(sorted_clips)
            base_duration = audio_duration / num_clips
            segment_metadata = []
            current_time = 0.0

            for i, clip in enumerate(sorted_clips):
                if i == num_clips - 1:
                    clip_duration = audio_duration - current_time
                else:
                    clip_duration = base_duration

                # Use dict format like CPU assembler (VideoMetadata.segments is list[dict])
                segment_metadata.append({
                    "index": clip.segment_index,
                    "start_time": current_time,
                    "end_time": current_time + clip_duration,
                    "duration": clip_duration,
                    "pexels_id": clip.pexels_id,
                    "source_url": clip.source_url,
                    "keywords": clip.keywords_used,
                })
                current_time += clip_duration

            # Build artifacts info
            artifacts = ArtifactsInfo(
                video=ArtifactStatus(status="ok", file="assembled.mp4"),
                voiceover=ArtifactStatus(status="pending"),
                subtitles=ArtifactStatus(status="pending"),
                thumbnail=ArtifactStatus(status="pending"),
                caption=ArtifactStatus(status="pending"),
                hashtags=ArtifactStatus(status="pending"),
            )

            # Get topic from context (set by TopicSelector earlier in pipeline)
            topic_str = context.script.title
            if context.topic:
                topic_str = context.topic.topic

            # Create metadata
            context.metadata = VideoMetadata(
                post_id=output_path.parent.name,
                title=context.script.title,
                topic=topic_str,
                duration_seconds=audio_duration,
                segments=segment_metadata,
                clips_used=[
                    {
                        "segment_index": c.segment_index,
                        "pexels_id": c.pexels_id,
                        "source_url": c.source_url,
                        "title": c.title,
                    }
                    for c in sorted_clips
                ],
                narration=context.script.full_narration,
                artifacts=artifacts,
            )

            # Save metadata to temp dir
            if context.temp_dir:
                metadata_path = context.temp_dir / "metadata.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(context.metadata.model_dump(), f, indent=2, default=str)

            self.log_success(f"Assembled video: {result_path} ({audio_duration:.1f}s)")
            return context

        except Exception as e:
            self.log_error(f"GPU assembly failed: {e}")
            raise VideoAssemblyError(f"GPU assembly failed: {e}") from e

    def _detect_black_bars(self, clip_path: Path) -> tuple[int, int, int, int]:
        """Detect black bars in a video using FFmpeg cropdetect.

        Runs a quick analysis to find embedded black bars (pillarboxing/letterboxing).

        Args:
            clip_path: Path to video file.

        Returns:
            Tuple of (left, top, right, bottom) pixels to crop.
            Returns (0, 0, 0, 0) if no bars detected or on error.
        """
        try:
            # Run cropdetect on first 2 seconds of video
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-i", str(clip_path),
                    "-t", "2",  # Analyze first 2 seconds
                    "-vf", "cropdetect=limit=24:round=2:reset=0",
                    "-f", "null",
                    "-"
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Parse cropdetect output from stderr
            # Format: [Parsed_cropdetect_0 @ ...] x1:0 x2:1919 y1:140 y2:939 w:1920 h:800 ...
            # Or: crop=W:H:X:Y
            lines = result.stderr.split('\n')

            # Find the last crop= line (most stable detection)
            crop_values = None
            for line in reversed(lines):
                match = re.search(r'crop=(\d+):(\d+):(\d+):(\d+)', line)
                if match:
                    w, h, x, y = map(int, match.groups())
                    crop_values = (w, h, x, y)
                    break

            if crop_values:
                w, h, x, y = crop_values

                # Get original dimensions using ffprobe
                probe_result = subprocess.run(
                    [
                        "ffprobe",
                        "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height",
                        "-of", "csv=p=0",
                        str(clip_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                orig_w, orig_h = map(int, probe_result.stdout.strip().split(','))

                # Calculate crop amounts from each edge
                left = x
                top = y
                right = orig_w - (x + w)
                bottom = orig_h - (y + h)

                # Only return if bars are significant (> 2% of dimension)
                min_bar_h = int(orig_w * 0.02)
                min_bar_v = int(orig_h * 0.02)

                left = left if left >= min_bar_h else 0
                right = right if right >= min_bar_h else 0
                top = top if top >= min_bar_v else 0
                bottom = bottom if bottom >= min_bar_v else 0

                if left > 0 or top > 0 or right > 0 or bottom > 0:
                    self.log_detail(f"  Black bars detected: L={left} T={top} R={right} B={bottom}")

                return (left, top, right, bottom)

        except Exception as e:
            self.log_detail(f"  Black bar detection failed: {e}")

        return (0, 0, 0, 0)

    async def _assemble_with_gpu(
        self,
        clips: list,
        output_path: Path,
        target_duration: float,
    ) -> Path:
        """Assemble clips using FFmpeg with NVENC.

        Uses filter_complex to properly loop clips to fill the target duration,
        matching the behavior of the CPU assembler.

        Args:
            clips: List of VideoClipInfo objects.
            output_path: Output video path.
            target_duration: Target duration in seconds.

        Returns:
            Path to assembled video.
        """
        gpu_index = self.gpu.index if self.gpu else 0

        # Clips should already be deduplicated by execute()
        # Just sort by segment index for consistent ordering
        sorted_clips = sorted(clips, key=lambda c: c.segment_index)
        num_clips = len(sorted_clips)

        # =================================================================
        # STEP 1: Black Bar Detection
        # =================================================================
        self.log_progress(">>> PREPROCESSING")
        clip_crops = []
        for idx, clip in enumerate(sorted_clips):
            clip_path = clip.path if hasattr(clip, 'path') else None
            clip_num = f"[{idx + 1}/{num_clips}]"

            if clip_path:
                self.log_detail(f"  {clip_num} Analyzing {clip_path.name}...")
                crop = self._detect_black_bars(clip_path)
                left, top, right, bottom = crop

                if left > 0 or top > 0 or right > 0 or bottom > 0:
                    self.log_progress(f"  {clip_num} [CROP] Removing bars: L={left} T={top} R={right} B={bottom}")
                else:
                    self.log_detail(f"  {clip_num} [OK] No black bars")

                clip_crops.append(crop)
            else:
                clip_crops.append((0, 0, 0, 0))

        # =================================================================
        # STEP 2: Calculate Segment Durations
        # =================================================================
        self.log_progress(">>> TIMING")
        base_duration = target_duration / num_clips
        segment_durations = []
        remaining = target_duration
        for i in range(num_clips):
            if i == num_clips - 1:
                segment_durations.append(remaining)
            else:
                segment_durations.append(base_duration)
                remaining -= base_duration

        for idx, dur in enumerate(segment_durations):
            self.log_detail(f"  [{idx + 1}/{num_clips}] {dur:.1f}s")

        # =================================================================
        # STEP 3: Build FFmpeg Filter Chain
        # =================================================================
        self.log_progress(">>> FILTER CHAIN")
        self.log_detail(f"  Building {num_clips}-input filter graph...")

        cmd = ["ffmpeg", "-y"]

        # Add hardware acceleration
        cmd.extend(["-hwaccel", "cuda", "-hwaccel_device", str(gpu_index)])

        # Add all input files
        for clip in sorted_clips:
            clip_path = clip.path if hasattr(clip, 'path') else None
            if clip_path:
                cmd.extend(["-i", str(clip_path)])

        # Build filter_complex
        filter_parts = []
        concat_inputs = []

        for i, (clip, seg_dur) in enumerate(zip(sorted_clips, segment_durations)):
            left, top, right, bottom = clip_crops[i]

            # Build filter chain for this clip
            # FILTER CHAIN (matches CPU assembler's _detect_black_bars + _crop_to_9_16):
            # 1. Loop/trim for duration
            # 2. Remove detected black bars (if any)
            # 3. Scale to COVER target resolution
            # 4. Center crop to exact 9:16 dimensions
            filters = [
                f"[{i}:v]",
                f"loop=loop=-1:size=32767:start=0",
                f"trim=0:{seg_dur:.3f}",
                f"setpts=PTS-STARTPTS",
            ]

            # Add black bar removal crop if detected
            if left > 0 or top > 0 or right > 0 or bottom > 0:
                # crop=out_w:out_h:x:y
                filters.append(f"crop=iw-{left}-{right}:ih-{top}-{bottom}:{left}:{top}")

            # Scale to cover and center crop to 9:16
            filters.extend([
                f"scale={self.WIDTH}:{self.HEIGHT}:force_original_aspect_ratio=increase",
                f"crop={self.WIDTH}:{self.HEIGHT}:(iw-{self.WIDTH})/2:(ih-{self.HEIGHT})/2",
                f"setsar=1",
                f"fps={self.FPS}",
                f"[v{i}]",
            ])

            filter_part = ",".join(filters[1:-1])  # Skip first [i:v] and last [vi]
            filter_part = f"{filters[0]}{filter_part}{filters[-1]}"

            filter_parts.append(filter_part)
            concat_inputs.append(f"[v{i}]")

        # Concatenate all processed clips
        concat_filter = f"{''.join(concat_inputs)}concat=n={num_clips}:v=1:a=0[outv]"
        filter_parts.append(concat_filter)

        filter_complex = ";".join(filter_parts)

        cmd.extend(["-filter_complex", filter_complex])
        cmd.extend(["-map", "[outv]"])

        # Log filter summary
        bars_removed = sum(1 for c in clip_crops if any(c))
        self.log_detail(f"  Clips with bars removed: {bars_removed}/{num_clips}")
        self.log_detail(f"  Output: loop -> crop bars -> scale 9:16 -> center crop -> concat")

        # =================================================================
        # STEP 4: GPU Encoding (NVENC)
        # =================================================================
        self.log_progress(">>> ENCODING")
        self.log_progress(f"  GPU {gpu_index} (h264_nvenc) | {target_duration:.1f}s | 1080x1920")

        # Output encoding settings (NVENC)
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",  # Balanced speed/quality
            "-rc", "vbr",
            "-cq", "23",  # Good quality for social media (18-28 range, higher = smaller)
            "-b:v", "4M",  # Target bitrate (Instagram recommends 3.5Mbps for 1080p)
            "-maxrate", "6M",
            "-bufsize", "8M",
            "-gpu", str(gpu_index),
            "-an",  # No audio (added in subtitle pass)
            "-t", str(target_duration),  # Safety trim
            str(output_path),
        ])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for longer videos
        )

        if result.returncode != 0:
            # Log error details
            self.log_progress(f"  [X] FFmpeg failed!")
            self.log_detail(f"FFmpeg stderr: {result.stderr[:1000]}")
            raise VideoAssemblyError(f"FFmpeg failed: {result.stderr[:300]}")

        # =================================================================
        # COMPLETE
        # =================================================================
        self.log_progress(f"  [OK] Assembled: {output_path.name}")
        return output_path


class GPUSubtitleRenderer(ISubtitleRenderer):
    """GPU-accelerated subtitle renderer using NVENC.

    Adds subtitles and audio to assembled video.
    Uses FFmpeg's subtitles filter with ASS styling.

    Note on font sizing:
        FFmpeg's ASS subtitle format uses a "script resolution" (typically 288p)
        that differs from the video resolution. Font sizes in force_style are
        scaled relative to this script resolution, not actual pixels.

        We convert pixel-based font sizes to ASS coordinates using:
            ass_font_size = pixel_font_size * (ASS_PLAY_RES_Y / VIDEO_HEIGHT)

        This ensures the GPU renderer produces the same visual output as the
        CPU renderer which uses PIL/MoviePy with actual pixel sizes.
    """

    def __init__(
        self,
        gpu: Optional[GPUInfo] = None,
        font: str = SUBTITLE_FONT_NAME,
        font_size: int = SUBTITLE_FONT_SIZE_DEFAULT,
        stroke_width: int = SUBTITLE_STROKE_WIDTH,
        position_y_percent: float = SUBTITLE_POSITION_Y_PERCENT,
    ):
        """Initialize GPU subtitle renderer.

        Args:
            gpu: GPU to use for encoding. If None, auto-selects.
            font: Font name for subtitles (e.g., "Montserrat-Bold.ttf").
            font_size: Font size in PIXELS (will be converted to ASS coords).
            stroke_width: Text outline width in pixels.
            position_y_percent: Vertical position as fraction from top (0.0-1.0).
        """
        super().__init__()
        self.gpu = gpu
        self.font = font
        self.font_size = font_size  # In pixels (same as CPU renderer)
        self.stroke_width = stroke_width
        self.position_y_percent = position_y_percent

        # Pre-calculate ASS scaling factor
        self._ass_scale = ASS_PLAY_RES_Y / VIDEO_HEIGHT

    async def render_subtitles(self, video_path, audio_path, srt_path, output_path):
        """Abstract method implementation - redirects to execute()."""
        # This is called by the base interface but we use execute() instead
        return await self._render_subtitles_gpu(video_path, audio_path, srt_path, output_path)

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute GPU subtitle rendering.

        Args:
            context: Pipeline context with assembled video, audio, and SRT.

        Returns:
            Updated context with final video path.
        """
        if not context.assembled_video_path:
            raise SubtitleRenderError("No assembled video available")
        if not context.audio_path:
            raise SubtitleRenderError("No audio available")

        self.log_start("Rendering subtitles with GPU (NVENC)...")

        # Determine output path
        if context.output_dir:
            final_path = context.output_dir / "final.mp4"
        else:
            final_path = context.assembled_video_path.parent / "final.mp4"

        # Get profile handle for watermark
        profile_handle = None
        if context.profile and context.profile.instagram_handle:
            profile_handle = context.profile.instagram_handle

        try:
            result_path = await self._render_subtitles_gpu(
                video_path=context.assembled_video_path,
                audio_path=context.audio_path,
                srt_path=context.srt_path,
                output_path=final_path,
                profile_handle=profile_handle,
            )

            context.final_video_path = result_path

            # Update artifact tracking
            if context.metadata:
                context.metadata.artifacts.subtitles = ArtifactStatus(
                    status="ok",
                    file="final.mp4",
                )
                context.metadata.artifacts.voiceover = ArtifactStatus(
                    status="ok",
                    file=context.audio_path.name if context.audio_path else "voiceover.mp3",
                )
                context.metadata.artifacts.video = ArtifactStatus(
                    status="ok",
                    file="final.mp4",
                )

            self.log_success(f"Final video: {result_path}")
            return context

        except Exception as e:
            self.log_error(f"GPU subtitle rendering failed: {e}")
            if context.metadata:
                context.metadata.artifacts.subtitles = ArtifactStatus(
                    status="failed",
                    error=str(e),
                )
            raise SubtitleRenderError(f"GPU subtitle rendering failed: {e}") from e

    async def _render_subtitles_gpu(
        self,
        video_path: Path,
        audio_path: Path,
        srt_path: Optional[Path],
        output_path: Path,
        profile_handle: Optional[str] = None,
    ) -> Path:
        """Render subtitles using FFmpeg with NVENC.

        Uses FFmpeg's subtitles filter for SRT rendering.

        Args:
            video_path: Input video path.
            audio_path: Audio file path.
            srt_path: SRT subtitle file path (optional).
            output_path: Output video path.
            profile_handle: Instagram handle for watermark (e.g., "@ai.for.mortals").

        Returns:
            Path to final video with subtitles.
        """
        gpu_index = self.gpu.index if self.gpu else 0

        # Find font file
        font_path = self._find_font_path()

        # Build filter chain
        filters = []

        # Add subtitles if SRT exists
        if srt_path and srt_path.exists():
            # Generate karaoke ASS from SRT for word-by-word highlighting
            # This matches the CPU SubtitleRenderer's karaoke effect
            ass_path = srt_path.parent / f"{srt_path.stem}_karaoke.ass"

            self.log_detail(f"Generating karaoke ASS subtitles...")
            _generate_karaoke_ass(
                srt_path=srt_path,
                output_path=ass_path,
                font_name=self.font.replace(".ttf", "").replace(".TTF", ""),
                font_size=self.font_size,
                stroke_width=self.stroke_width,
                position_y_percent=self.position_y_percent,
            )

            # Use ASS filter with our generated karaoke file
            # ASS file has PlayRes matching video resolution, so no scaling needed
            ass_escaped = escape_ffmpeg_filter_path(str(ass_path))

            self.log_detail(
                f"Subtitle settings: font={self.font}, size={self.font_size}px, "
                f"position={self.position_y_percent*100:.0f}% from top"
            )

            # Use ass filter for proper ASS rendering with embedded styles
            filters.append(f"ass='{ass_escaped}'")

        # Add moving watermark text if profile handle is set
        # Matches CPU renderer: moves to different positions every 10 seconds
        # Note: drawtext uses video pixels (not ASS coords), so no scaling needed
        if profile_handle:
            # Font path needs escaping for FFmpeg filter syntax
            font_path_escaped = escape_ffmpeg_filter_path(font_path)

            # Watermark settings to match CPU renderer (subtitle_renderer.py)
            watermark_fontsize = 28
            interval = 10  # Change position every 10 seconds

            # Positions matching CPU renderer (safe zones, avoiding edges)
            # Format: (x_expr, y_expr) - use FFmpeg expressions
            margin_x = int(VIDEO_WIDTH * 0.05)
            margin_top = int(VIDEO_HEIGHT * 0.15)
            margin_mid = int(VIDEO_HEIGHT * 0.40)
            text_width_approx = 220  # Approximate width of watermark text

            positions = [
                (f"{margin_x}", f"{margin_top}"),                    # Top-left
                (f"{VIDEO_WIDTH - text_width_approx}", f"{margin_top}"),  # Top-right
                (f"{margin_x}", f"{margin_mid}"),                    # Middle-left
                (f"{VIDEO_WIDTH - text_width_approx}", f"{margin_mid}"),  # Middle-right
                (f"(w-text_w)/2", f"{margin_top}"),                  # Top-center
            ]

            # Add drawtext filter for each position with enable expression
            # Using between(t, start, end) to show each position for 10 seconds
            # Cycle repeats every 50 seconds (5 positions * 10 seconds)
            num_positions = len(positions)
            cycle_duration = interval * num_positions  # 50 seconds

            for i, (x_pos, y_pos) in enumerate(positions):
                start_time = i * interval
                # Use modular time to repeat the cycle
                # enable expression: show when (t mod cycle_duration) is in this position's window
                enable_expr = f"lt(mod(t\\,{cycle_duration})\\,{start_time + interval})*gte(mod(t\\,{cycle_duration})\\,{start_time})"

                filters.append(
                    f"drawtext=text='{profile_handle}':"
                    f"fontfile='{font_path_escaped}':"
                    f"fontsize={watermark_fontsize}:"
                    f"fontcolor=white@0.5:"
                    f"x={x_pos}:"
                    f"y={y_pos}:"
                    f"enable='{enable_expr}'"
                )

        filter_str = ",".join(filters) if filters else "null"

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel", "cuda",
            "-hwaccel_device", str(gpu_index),
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", f"[0:v]{filter_str}[v]",
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", "23",  # Good quality for social media
            "-b:v", "4M",  # Instagram recommends 3.5Mbps for 1080p
            "-maxrate", "6M",
            "-bufsize", "8M",
            "-gpu", str(gpu_index),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(output_path),
        ]

        self.log_detail(f"Running FFmpeg GPU subtitle rendering on GPU {gpu_index}...")
        self.log_detail(f"Filter: {filter_str[:150]}...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            # FFmpeg prints version info to stderr, find actual error after that
            stderr = result.stderr
            # Look for common error patterns
            error_lines = [line for line in stderr.split('\n') if any(x in line.lower() for x in ['error', 'invalid', 'failed', 'no such', 'cannot'])]
            if error_lines:
                error_msg = '\n'.join(error_lines[:5])
            else:
                # Get last 500 chars which usually has the actual error
                error_msg = stderr[-500:] if len(stderr) > 500 else stderr
            self.log_detail(f"FFmpeg error: {error_msg}")
            raise SubtitleRenderError(f"FFmpeg failed: {error_msg[:300]}")

        return output_path

    def _find_font_path(self) -> str:
        """Find the font file path.

        Returns:
            Path to font file.
        """
        # Check project fonts folder
        project_root = Path(__file__).parent.parent.parent.parent.parent
        fonts_dir = project_root / "fonts"

        font_base = self.font.replace(".ttf", "").replace(".TTF", "")

        # Try different locations
        paths_to_try = [
            fonts_dir / f"{font_base}.ttf",
            fonts_dir / self.font,
            Path(f"C:/Windows/Fonts/{font_base.lower()}.ttf"),
            Path(f"C:/Windows/Fonts/arial.ttf"),  # Fallback
        ]

        for path in paths_to_try:
            if path.exists():
                return str(path).replace("\\", "/")

        # Default fallback
        return "C:/Windows/Fonts/arial.ttf"


# Quick test when run directly
if __name__ == "__main__":
    import asyncio

    async def test():
        print("GPU Video Renderer Test\n" + "=" * 50)

        success, message, gpu = validate_gpu_setup()
        print(f"Validation: {message}")

        if success:
            assembler = GPUVideoAssembler(gpu=gpu)
            renderer = GPUSubtitleRenderer(gpu=gpu)
            print(f"Assembler ready with GPU {gpu.index}")
            print(f"Renderer ready with GPU {gpu.index}")

    asyncio.run(test())
