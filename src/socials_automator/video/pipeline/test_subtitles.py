"""Test subtitle rendering with placeholder video.

Run: python -m socials_automator.video.pipeline.test_subtitles
"""

import asyncio
import tempfile
from pathlib import Path


async def create_placeholder_video(output_path: Path, duration: float = 60.0):
    """Create a simple placeholder video for testing."""
    from moviepy import ColorClip

    # Create a dark blue background video
    clip = ColorClip(
        size=(1080, 1920),
        color=(20, 30, 50),  # Dark blue
        duration=duration,
    )

    clip.write_videofile(
        str(output_path),
        fps=30,
        codec="libx264",
        audio=False,
        logger=None,
    )
    clip.close()
    print(f"[OK] Created placeholder video: {output_path}")


async def test_voice_and_subtitles():
    """Test voice generation and subtitle rendering."""
    import json
    import shutil
    from .voice_generator import VoiceGenerator
    from .subtitle_renderer import SubtitleRenderer

    # Profile configuration
    profile_handle = "@ai.for.mortals"
    profile_name = "AI For Mortals"

    # Sample narration - realistic 60-second script with custom CTA
    # CTA format: "Follow [Profile Name] [AI-generated context CTA]"
    # Note: Use display name for narration (not @handle which sounds weird when spoken)
    narration = """
    Stop scrolling! Here's something that will change your productivity forever.

    First, I discovered that ChatGPT can write entire emails for me in seconds.
    I just give it the context and tone, and it delivers professional messages instantly.

    Second, I use AI to summarize long documents and articles.
    What used to take me thirty minutes now takes just two minutes with the right prompts.

    Third, here's my secret weapon: I create custom GPTs for repetitive tasks.
    Once set up, they handle the boring stuff while I focus on creative work.

    Fourth, I batch my AI requests. Instead of asking one question at a time,
    I prepare multiple prompts and run them together for maximum efficiency.

    Finally, I review and edit AI outputs quickly rather than writing from scratch.
    This hybrid approach gives me the best of both worlds: speed and quality.

    Follow AI For Mortals for more productivity tips that actually work!
    """

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="subtitle_test_"))
    output_dir = Path("C:/Users/rbgnr/git/Socials-Automator/profiles/ai.for.mortals/reels/2025/12/generated/test")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Temp dir: {temp_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    # Step 1: Create placeholder video
    print("\n[STEP 1] Creating placeholder video...")
    placeholder_video = temp_dir / "placeholder.mp4"
    await create_placeholder_video(placeholder_video, duration=60.0)

    # Step 2: Generate voice
    print("\n[STEP 2] Generating voice...")
    voice_gen = VoiceGenerator(voice="british_male")
    audio_path, srt_path, timestamps = await voice_gen.generate_voice(
        narration.strip(),
        temp_dir,
    )
    print(f"[OK] Generated audio: {audio_path}")
    print(f"[OK] Generated SRT: {srt_path}")
    print(f"[OK] Word timestamps: {len(timestamps)}")

    # Show SRT content
    print("\n[DEBUG] SRT content preview:")
    srt_content = srt_path.read_text(encoding="utf-8")
    print(srt_content[:500] + "..." if len(srt_content) > 500 else srt_content)

    # Step 3: Render subtitles
    print("\n[STEP 3] Rendering subtitles...")
    subtitle_renderer = SubtitleRenderer(
        font_size=60,
        position="bottom",
        profile_handle=profile_handle,
    )

    final_video = await subtitle_renderer.render_subtitles(
        video_path=placeholder_video,
        audio_path=audio_path,
        srt_path=srt_path,
        output_path=output_dir / "test_final.mp4",
    )

    # Step 4: Save metadata and SRT to output directory
    print("\n[STEP 4] Saving metadata and SRT...")

    # Copy SRT file to output
    output_srt = output_dir / "subtitles.srt"
    shutil.copy(srt_path, output_srt)
    print(f"[OK] Saved SRT: {output_srt}")

    # Create metadata with timestamps
    metadata = {
        "post_id": "test_subtitles",
        "title": "Test Subtitle Rendering",
        "profile_name": profile_name,
        "profile_handle": profile_handle,
        "narration": narration.strip(),
        "duration_seconds": 60.0,
        "word_count": len(timestamps),
        "subtitles": [
            {
                "index": i + 1,
                "word": ts["word"],
                "start_ms": ts["start_ms"],
                "end_ms": ts["end_ms"],
                "start_time": f"{ts['start_ms'] // 60000:02d}:{(ts['start_ms'] // 1000) % 60:02d}.{ts['start_ms'] % 1000:03d}",
                "end_time": f"{ts['end_ms'] // 60000:02d}:{(ts['end_ms'] // 1000) % 60:02d}.{ts['end_ms'] % 1000:03d}",
            }
            for i, ts in enumerate(timestamps)
        ],
        "srt_file": "subtitles.srt",
        "video_file": "test_final.mp4",
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved metadata: {metadata_path}")

    print(f"\n[SUCCESS] Final video: {final_video}")
    print(f"[INFO] Check the video to verify subtitles are working!")


if __name__ == "__main__":
    asyncio.run(test_voice_and_subtitles())
