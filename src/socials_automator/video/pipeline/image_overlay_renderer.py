"""Image overlay rendering with FFmpeg.

Composites images onto video with:
- GPU-accelerated overlay using overlay_cuda filter
- Timed display using enable expressions
- Proper positioning above subtitle area

Uses ffmpeg-python for cleaner filter graph construction.
"""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Optional

try:
    import ffmpeg
    FFMPEG_PYTHON_AVAILABLE = True
except ImportError:
    FFMPEG_PYTHON_AVAILABLE = False

from .base import (
    IImageOverlayRenderer,
    ImageOverlay,
    ImageOverlayScript,
    ImageOverlayError,
    PipelineContext,
)
from socials_automator.constants import (
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    VIDEO_FPS,
    IMAGE_OVERLAY_MARGIN_X,
    IMAGE_OVERLAY_MAX_HEIGHT,
    IMAGE_OVERLAY_MARGIN_BOTTOM,
    IMAGE_OVERLAY_SUBTITLE_Y,
    IMAGE_OVERLAY_POP_IN_DURATION,
    IMAGE_OVERLAY_POP_OUT_DURATION,
)

logger = logging.getLogger("video.pipeline")


# Blur intensity mapping (boxblur radius values)
BLUR_INTENSITY = {
    "light": 8,     # Subtle blur, background still visible
    "medium": 15,   # Balanced blur, draws focus to overlay
    "heavy": 30,    # Strong blur, overlay really pops
}


class ImageOverlayRenderer(IImageOverlayRenderer):
    """Renders image overlays onto video with GPU acceleration."""

    def __init__(
        self,
        use_gpu: bool = False,
        blur: Optional[str] = None,
    ):
        """Initialize renderer.

        Args:
            use_gpu: Whether to use GPU acceleration (overlay_cuda + NVENC).
            blur: Blur background during overlays (light, medium, heavy). None = disabled.
        """
        super().__init__()
        self._use_gpu = use_gpu
        self._blur = blur
        self._blur_radius = BLUR_INTENSITY.get(blur, 0) if blur else 0

    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute image overlay rendering step.

        Args:
            context: Pipeline context with assembled video and overlays.

        Returns:
            Updated context with overlay video path.
        """
        if not context.image_overlays:
            return context

        resolved = context.image_overlays.get_resolved_overlays()
        if not resolved:
            self.log_progress("No resolved overlays to render")
            return context

        if not context.assembled_video_path:
            raise ImageOverlayError("No assembled video available for overlay rendering")

        self.log_start(f"Compositing {len(resolved)} image overlays")

        try:
            # Output path for video with overlays
            output_path = context.temp_dir / "video_with_overlays.mp4"

            await self.render_overlays(
                video_path=context.assembled_video_path,
                overlay_script=context.image_overlays,
                output_path=output_path,
            )

            # Update context - the overlay video becomes the assembled video
            # for subtitle rendering to use
            context.assembled_video_path = output_path

            # Log results
            self._log_results(context.image_overlays, context.required_video_duration)

            self.log_success(f"Rendered {len(resolved)} overlays")

            return context

        except Exception as e:
            self.log_error(f"Image overlay rendering failed: {e}")
            raise ImageOverlayError(f"Failed to render overlays: {e}") from e

    async def render_overlays(
        self,
        video_path: Path,
        overlay_script: ImageOverlayScript,
        output_path: Path,
    ) -> Path:
        """Render image overlays onto video.

        Uses ffmpeg-python for cleaner filter construction.
        GPU mode uses overlay_cuda + scale_cuda for acceleration.

        Args:
            video_path: Path to input video.
            overlay_script: Script with resolved image overlays.
            output_path: Path for output video.

        Returns:
            Path to rendered video.
        """
        resolved = overlay_script.get_resolved_overlays()
        if not resolved:
            shutil.copy2(video_path, output_path)
            return output_path

        if FFMPEG_PYTHON_AVAILABLE:
            await self._render_with_ffmpeg_python(
                video_path, resolved, output_path
            )
        else:
            await self._render_with_subprocess(
                video_path, resolved, output_path
            )

        if not output_path.exists():
            raise ImageOverlayError("FFmpeg did not produce output file")

        return output_path

    async def _render_with_ffmpeg_python(
        self,
        video_path: Path,
        overlays: list[ImageOverlay],
        output_path: Path,
    ) -> None:
        """Render using ffmpeg-python library.

        Args:
            video_path: Input video path.
            overlays: List of resolved overlays.
            output_path: Output path.
        """
        self.log_progress("  [>] Rendering with ffmpeg-python...")

        # Container dimensions
        container_width = VIDEO_WIDTH - (2 * IMAGE_OVERLAY_MARGIN_X)

        # Start with video input
        if self._use_gpu:
            # GPU mode: use hardware acceleration
            video = ffmpeg.input(
                str(video_path),
                hwaccel='cuda',
                hwaccel_output_format='cuda'
            )
            self.log_progress("  [>] Using GPU acceleration (overlay_cuda)")
        else:
            video = ffmpeg.input(str(video_path))

        # Get video stream
        video_stream = video.video

        # Apply blur during overlay times (if enabled)
        # Use split+overlay technique to avoid green screen artifacts with conditional boxblur
        # Reference: https://medium.com/@allanlei/blur-out-videos-with-ffmpeg-92d3dc62d069
        if self._blur_radius > 0 and overlays:
            # Build blur enable expression: blur only when any overlay is showing
            # Example: 'between(t,3.5,12.0)+between(t,15.0,25.0)'
            blur_ranges = []
            for overlay in overlays:
                if overlay.image_path and overlay.image_path.exists():
                    blur_ranges.append(f"between(t,{overlay.start_time},{overlay.end_time})")

            if blur_ranges:
                blur_enable = "+".join(blur_ranges)
                self.log_progress(f"  [>] Background blur enabled ({self._blur}, radius={self._blur_radius})")

                if self._use_gpu:
                    # GPU mode: download, split, blur, overlay, re-upload
                    # 1. Download from CUDA to CPU
                    cpu_stream = video_stream.filter('hwdownload').filter('format', 'yuv420p')

                    # 2. Split into original and copy for blur
                    split = cpu_stream.filter('split')
                    original = split.stream(0)
                    copy = split.stream(1)

                    # 3. Apply blur to copy (permanent, no conditional enable)
                    blurred = copy.filter('boxblur', self._blur_radius)

                    # 4. Overlay blurred onto original with enable expression
                    # When enable is true -> show blurred, when false -> show original
                    merged = ffmpeg.overlay(original, blurred, enable=blur_enable)

                    # 5. Re-upload to CUDA
                    video_stream = merged.filter('format', 'nv12').filter('hwupload_cuda')
                else:
                    # CPU mode: split, blur one copy, overlay with enable
                    # This avoids green screen artifacts from conditional boxblur

                    # 1. Ensure consistent format and split
                    formatted = video_stream.filter('format', 'yuv420p')
                    split = formatted.filter('split')
                    original = split.stream(0)
                    copy = split.stream(1)

                    # 2. Apply blur to copy (permanent, no conditional enable)
                    blurred = copy.filter('boxblur', self._blur_radius)

                    # 3. Overlay blurred onto original with enable expression
                    video_stream = ffmpeg.overlay(original, blurred, enable=blur_enable)

        # Chain overlays one by one
        for i, overlay in enumerate(overlays):
            if not overlay.image_path or not overlay.image_path.exists():
                continue

            # Calculate scaling
            img_width = overlay.width or 1920
            img_height = overlay.height or 1080

            scale_factor = min(
                container_width / img_width,
                IMAGE_OVERLAY_MAX_HEIGHT / img_height
            )
            scaled_width = int(img_width * scale_factor)
            scaled_height = int(img_height * scale_factor)

            # Make dimensions even
            scaled_width = scaled_width - (scaled_width % 2)
            scaled_height = scaled_height - (scaled_height % 2)

            # Calculate position
            container_bottom = IMAGE_OVERLAY_SUBTITLE_Y - IMAGE_OVERLAY_MARGIN_BOTTOM
            img_y = container_bottom - scaled_height
            img_x = (VIDEO_WIDTH - scaled_width) // 2

            # Load image
            if self._use_gpu:
                # GPU: loop the image to create continuous stream for overlay_cuda
                # Without loop=1, still images only provide one frame
                img_input = ffmpeg.input(str(overlay.image_path), loop=1)

                # Convert to NV12, upload to CUDA and scale
                # NV12 format required for overlay_cuda (must match video stream)
                img_stream = (
                    img_input
                    .filter('format', 'nv12')
                    .filter('hwupload_cuda')
                    .filter('scale_cuda', scaled_width, scaled_height)
                )
            else:
                img_input = ffmpeg.input(str(overlay.image_path))
                # CPU: animated scale with pop-in/pop-out effect
                # Scale expression: 0 -> 1.1 -> 1.0 (bounce) for pop-in, 1.0 -> 0 for pop-out
                # Uses eval=frame to evaluate per-frame
                pop_in_end = overlay.start_time + IMAGE_OVERLAY_POP_IN_DURATION
                pop_out_start = overlay.end_time - IMAGE_OVERLAY_POP_OUT_DURATION

                # Build scale factor expression with bounce effect
                # Pop-in: ease from 0 to 1.1 to 1.0 (overshoot bounce)
                # Pop-out: ease from 1.0 to 0
                scale_expr = (
                    f"if(lt(t,{overlay.start_time}),0,"  # Before start: 0
                    f"if(lt(t,{pop_in_end}),"  # During pop-in
                    f"min(1.1,(t-{overlay.start_time})/{IMAGE_OVERLAY_POP_IN_DURATION}*1.15),"  # Scale up with overshoot
                    f"if(gt(t,{pop_out_start}),"  # During pop-out
                    f"max(0,(1-(t-{pop_out_start})/{IMAGE_OVERLAY_POP_OUT_DURATION})),"  # Scale down
                    f"1)))"  # Normal: 1.0
                )

                # Apply animated scale with expression
                # Width and height use the scale expression multiplied by target size
                img_stream = img_input.filter(
                    'scale',
                    w=f"trunc({scaled_width}*({scale_expr})/2)*2",
                    h=f"trunc({scaled_height}*({scale_expr})/2)*2",
                    eval='frame'
                )

            # Build enable expression for timed overlay
            # Use gte()*lte() format which is more compatible
            enable_expr = f"gte(t,{overlay.start_time})*lte(t,{overlay.end_time})"

            # Pop animation timing
            pop_in_dur = IMAGE_OVERLAY_POP_IN_DURATION
            pop_out_dur = IMAGE_OVERLAY_POP_OUT_DURATION
            start = overlay.start_time
            end = overlay.end_time

            # Apply overlay
            if self._use_gpu:
                # GPU: overlay_cuda filter
                video_stream = ffmpeg.filter(
                    [video_stream, img_stream],
                    'overlay_cuda',
                    x=img_x,
                    y=img_y,
                    enable=enable_expr,
                    eof_action='pass'
                )
            else:
                # CPU: overlay with dynamic positioning for animation
                # As image scales, adjust position to keep centered
                # Center X: (VIDEO_WIDTH - current_width) / 2
                # Center Y: container_bottom - current_height
                center_x = VIDEO_WIDTH // 2
                center_y = container_bottom

                # Dynamic position expressions based on overlay size (w, h variables)
                x_expr = f"{center_x}-w/2"
                y_expr = f"{center_y}-h"

                video_stream = ffmpeg.overlay(
                    video_stream,
                    img_stream,
                    x=x_expr,
                    y=y_expr,
                    enable=enable_expr,
                    eof_action='pass',
                    eval='frame'  # Evaluate position per-frame for animation
                )

            self.log_detail(f"  Overlay {i+1}: {overlay.image_path.name} @ {overlay.start_time:.1f}s-{overlay.end_time:.1f}s")

        # Build output with encoding
        # Note: Audio is handled via output kwargs to avoid input ordering issues
        if self._use_gpu:
            output = ffmpeg.output(
                video_stream,
                str(output_path),
                vcodec='h264_nvenc',
                preset='p4',
            )
        else:
            output = ffmpeg.output(
                video_stream,
                str(output_path),
                vcodec='libx264',
                preset='fast',
                crf=18,
            )

        # Overwrite output
        output = output.overwrite_output()

        # Log the command for debugging
        try:
            cmd = output.compile()
            logger.info(f"FFMPEG_CMD | {' '.join(cmd[:20])}...")
            self.log_detail(f"FFmpeg args: {len(cmd)} total")
        except Exception:
            pass

        # Run asynchronously
        await self._run_ffmpeg_python(output)

    async def _run_ffmpeg_python(self, output_stream) -> None:
        """Run ffmpeg-python stream asynchronously.

        Args:
            output_stream: ffmpeg-python output stream.
        """
        loop = asyncio.get_event_loop()

        def run_sync():
            try:
                output_stream.run(capture_stderr=True, quiet=True)
            except ffmpeg.Error as e:
                stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ''
                error_tail = stderr[-1000:] if len(stderr) > 1000 else stderr
                if not error_tail.strip():
                    error_tail = "(stderr was empty)"
                raise ImageOverlayError(f"FFmpeg failed: {error_tail}")

        await loop.run_in_executor(None, run_sync)

    async def _render_with_subprocess(
        self,
        video_path: Path,
        overlays: list[ImageOverlay],
        output_path: Path,
    ) -> None:
        """Fallback: render using subprocess (when ffmpeg-python not available).

        Args:
            video_path: Input video path.
            overlays: List of resolved overlays.
            output_path: Output path.
        """
        self.log_progress("  [>] Rendering with subprocess (ffmpeg-python not available)...")

        # Build filter complex string
        filter_parts = []
        container_width = VIDEO_WIDTH - (2 * IMAGE_OVERLAY_MARGIN_X)

        current_stream = "[0:v]"

        # Add blur filter if enabled (using split+overlay to avoid green screen artifacts)
        if self._blur_radius > 0 and overlays:
            # Build blur enable expression
            blur_ranges = []
            for overlay in overlays:
                if overlay.image_path and overlay.image_path.exists():
                    blur_ranges.append(f"between(t\\,{overlay.start_time}\\,{overlay.end_time})")

            if blur_ranges:
                blur_enable = "+".join(blur_ranges)
                self.log_progress(f"  [>] Background blur enabled ({self._blur}, radius={self._blur_radius})")
                # Use split+overlay technique: split video, blur one copy, overlay with enable
                # This avoids green screen artifacts from conditional boxblur
                filter_parts.append(
                    f"[0:v]format=yuv420p,split[original][copy];"
                    f"[copy]boxblur={self._blur_radius}[blurred];"
                    f"[original][blurred]overlay=enable='{blur_enable}'[merged]"
                )
                current_stream = "[merged]"

        for i, overlay in enumerate(overlays):
            if not overlay.image_path or not overlay.image_path.exists():
                continue

            img_idx = i + 1

            # Calculate dimensions
            img_width = overlay.width or 1920
            img_height = overlay.height or 1080
            scale_factor = min(
                container_width / img_width,
                IMAGE_OVERLAY_MAX_HEIGHT / img_height
            )
            scaled_width = int(img_width * scale_factor) // 2 * 2
            scaled_height = int(img_height * scale_factor) // 2 * 2

            # Calculate position
            container_bottom = IMAGE_OVERLAY_SUBTITLE_Y - IMAGE_OVERLAY_MARGIN_BOTTOM
            img_y = container_bottom - scaled_height
            img_x = (VIDEO_WIDTH - scaled_width) // 2

            # Build filter
            enable_expr = f"gte(t\\,{overlay.start_time})*lte(t\\,{overlay.end_time})"
            output_stream = f"[out{i}]"

            filter_parts.append(
                f"[{img_idx}:v]scale={scaled_width}:{scaled_height}[img{img_idx}];"
                f"{current_stream}[img{img_idx}]overlay=x={img_x}:y={img_y}:"
                f"enable='{enable_expr}'{output_stream}"
            )
            current_stream = output_stream

        filter_complex = ";".join(filter_parts) if filter_parts else ""

        # Build command
        cmd = ["ffmpeg", "-y", "-i", str(video_path)]

        for overlay in overlays:
            if overlay.image_path and overlay.image_path.exists():
                cmd.extend(["-i", str(overlay.image_path)])

        if filter_complex:
            num_overlays = len([o for o in overlays if o.image_path and o.image_path.exists()])
            cmd.extend(["-filter_complex", filter_complex])
            cmd.extend(["-map", f"[out{num_overlays - 1}]"])
        else:
            cmd.extend(["-map", "0:v"])

        cmd.extend(["-map", "0:a?", "-c:a", "copy"])

        if self._use_gpu:
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p4"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "18"])

        cmd.append(str(output_path))

        # Run
        logger.info(f"FFMPEG_CMD | {' '.join(cmd[:15])}...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace")
            error_tail = error_msg[-1000:] if len(error_msg) > 1000 else error_msg
            if not error_tail.strip():
                error_tail = "(stderr was empty)"
            logger.error(f"FFmpeg overlay failed: {error_tail}")
            raise ImageOverlayError(f"FFmpeg failed (returncode={process.returncode}): {error_tail}")

    def _log_results(
        self,
        overlay_script: ImageOverlayScript,
        total_duration: Optional[float],
    ) -> None:
        """Log rendering results in table format.

        Args:
            overlay_script: Overlay script.
            total_duration: Total video duration.
        """
        resolved = overlay_script.get_resolved_overlays()
        if not resolved:
            return

        self.log_progress("")
        self.log_progress("  --- Overlay Timeline ---")
        self.log_progress("  | Time            | Image                     | Mode          |")
        self.log_progress("  |-----------------|---------------------------|---------------|")

        # GPU mode: no pop animation (overlay_cuda limitation)
        # CPU mode: pop-in/pop-out animation with scale expressions
        if self._use_gpu:
            mode = "GPU (static)"
        else:
            mode = f"CPU (pop {IMAGE_OVERLAY_POP_IN_DURATION}s/{IMAGE_OVERLAY_POP_OUT_DURATION}s)"

        for overlay in resolved:
            time_range = f"{overlay.start_time:05.1f}s -> {overlay.end_time:05.1f}s"

            if overlay.source == "local":
                image_name = str(overlay.image_path.name)[:25] if overlay.image_path else "?"
            else:
                # Use actual source (pexels, pixabay, websearch, etc.)
                source = overlay.source or "unknown"
                image_name = f"{source}:{overlay.pexels_id}"[:25] if overlay.pexels_id else "?"

            self.log_progress(
                f"  | {time_range} | {image_name.ljust(25)} | {mode[:13].ljust(13)} |"
            )

        # Calculate coverage
        coverage = overlay_script.total_coverage
        if total_duration and total_duration > 0:
            coverage_pct = (coverage / total_duration) * 100
            blur_info = f", blur={self._blur}" if self._blur else ""
            self.log_progress(
                f"\n  Coverage: {coverage:.1f}s / {total_duration:.1f}s ({coverage_pct:.0f}% of video{blur_info})"
            )

        self.log_progress("")
