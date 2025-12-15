"""RVC Adam TTS Service.

Converts text to speech using edge-tts + RVC with Adam's voice.
This script runs in an isolated Python 3.12 environment.

Usage:
    python rvc_adam_tts.py --text "Hello world" --output output.mp3 [--srt output.srt]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


# Model paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
MODEL_DIR = SCRIPT_DIR / "models" / "adam_elevenlabs"
MODEL_PATH = MODEL_DIR / "Adam_Elevenlabs_160e_20800s.pth"
INDEX_PATH = MODEL_DIR / "added_Adam_Elevenlabs_v2.index"

# Edge-TTS voice to use as base (will be converted to Adam)
BASE_VOICE = "en-US-ChristopherNeural"
BASE_RATE = "+0%"
BASE_PITCH = "+0Hz"


async def generate_base_audio(text: str, output_path: Path) -> list[dict]:
    """Generate base audio with edge-tts and return word timestamps."""
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice=BASE_VOICE,
        rate=BASE_RATE,
        pitch=BASE_PITCH,
        boundary="WordBoundary",  # Enable word-level timestamps
    )

    submaker = edge_tts.SubMaker()
    word_timestamps = []

    with open(output_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                offset = chunk["offset"]
                duration = chunk["duration"]
                word = chunk["text"]

                # Add to submaker
                try:
                    submaker.feed(chunk)
                except TypeError:
                    try:
                        submaker.feed(offset, duration, word)
                    except TypeError:
                        submaker.create_sub((offset, duration), word)

                # Store timestamp (convert from 100-nanosecond units to ms)
                start_ms = offset // 10000
                end_ms = (offset + duration) // 10000
                word_timestamps.append({
                    "word": word,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                })

    return word_timestamps, submaker


def convert_to_adam(input_path: Path, output_path: Path) -> None:
    """Convert audio to Adam's voice using RVC."""
    from tts_with_rvc import TTS_RVC

    # Initialize RVC with Adam model
    tts_rvc = TTS_RVC(
        model_path=str(MODEL_PATH),
        index_path=str(INDEX_PATH) if INDEX_PATH.exists() else None,
    )

    # Set output directory
    tts_rvc.set_output_directory(str(output_path.parent))

    # Convert audio using voiceover_file method
    result_path = tts_rvc.voiceover_file(
        input_path=str(input_path),
        pitch=0,  # No pitch shift
        filename=output_path.stem,
        index_rate=0.75,
        f0method="rmvpe",  # Best pitch extraction
        verbose=False,
    )

    # Rename if needed (the library might add suffix)
    result = Path(result_path)
    if result != output_path and result.exists():
        if output_path.exists():
            output_path.unlink()
        result.rename(output_path)


def generate_srt(word_timestamps: list[dict], output_path: Path) -> None:
    """Generate SRT file from word timestamps."""
    def ms_to_srt_time(ms: int) -> str:
        hours = ms // 3600000
        minutes = (ms % 3600000) // 60000
        seconds = (ms % 60000) // 1000
        millis = ms % 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    with open(output_path, "w", encoding="utf-8") as f:
        for i, ts in enumerate(word_timestamps, 1):
            f.write(f"{i}\n")
            f.write(f"{ms_to_srt_time(ts['start_ms'])} --> {ms_to_srt_time(ts['end_ms'])}\n")
            f.write(f"{ts['word']}\n\n")


async def main():
    parser = argparse.ArgumentParser(description="Generate Adam TTS audio")
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--srt", help="Output SRT file path (optional)")
    parser.add_argument("--timestamps-json", help="Output timestamps JSON file (optional)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file for base audio
    temp_audio = output_path.parent / "temp_base_audio.mp3"

    print(f"[1/3] Generating base audio with edge-tts...", file=sys.stderr)
    word_timestamps, submaker = await generate_base_audio(args.text, temp_audio)
    print(f"      Generated {len(word_timestamps)} word timestamps", file=sys.stderr)

    print(f"[2/3] Converting to Adam's voice with RVC...", file=sys.stderr)
    convert_to_adam(temp_audio, output_path)
    print(f"      Saved to {output_path}", file=sys.stderr)

    # Clean up temp file
    if temp_audio.exists():
        temp_audio.unlink()

    # Generate SRT if requested
    if args.srt:
        srt_path = Path(args.srt)
        print(f"[3/3] Generating SRT file...", file=sys.stderr)
        generate_srt(word_timestamps, srt_path)
        print(f"      Saved to {srt_path}", file=sys.stderr)

    # Output timestamps JSON if requested
    if args.timestamps_json:
        ts_path = Path(args.timestamps_json)
        with open(ts_path, "w", encoding="utf-8") as f:
            json.dump(word_timestamps, f)
        print(f"      Timestamps saved to {ts_path}", file=sys.stderr)

    # Output timestamps to stdout for the caller to capture
    print(json.dumps(word_timestamps))


if __name__ == "__main__":
    asyncio.run(main())
