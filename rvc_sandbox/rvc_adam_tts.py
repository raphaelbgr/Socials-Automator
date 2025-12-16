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
# Priority: pt-files/ > models/adam_elevenlabs/
SCRIPT_DIR = Path(__file__).parent

# Check pt-files folder first (user's custom models)
PT_FILES_DIR = SCRIPT_DIR / "pt-files"

# Fallback to models folder
MODEL_DIR = SCRIPT_DIR / "models" / "adam_elevenlabs"


def find_model_files() -> tuple[Path, Path | None]:
    """Find the best available model and index files.

    Priority:
    1. pt-files/*.pth (user's custom models)
    2. models/adam_elevenlabs/*.pth (default models)

    Returns:
        Tuple of (model_path, index_path or None)
    """
    # Check pt-files first
    if PT_FILES_DIR.exists():
        pth_files = list(PT_FILES_DIR.glob("*.pth"))
        if pth_files:
            model_path = pth_files[0]  # Use first .pth found
            # Look for matching index file
            index_files = list(PT_FILES_DIR.glob("*.index"))
            index_path = index_files[0] if index_files else None
            return model_path, index_path

    # Fallback to default models folder
    default_model = MODEL_DIR / "Adam_Elevenlabs_160e_20800s.pth"
    default_index = MODEL_DIR / "added_Adam_Elevenlabs_v2.index"

    if default_model.exists():
        return default_model, default_index if default_index.exists() else None

    raise FileNotFoundError(
        f"No RVC models found. Place .pth files in:\n"
        f"  - {PT_FILES_DIR} (preferred)\n"
        f"  - {MODEL_DIR}"
    )


# Resolve model paths at import time
MODEL_PATH, INDEX_PATH = find_model_files()

# Edge-TTS voice to use as base (will be converted to Adam)
BASE_VOICE = "en-US-ChristopherNeural"

# Default: Neutral tone
# For excitement: use --rate "+12%" --pitch "+3Hz"
DEFAULT_RATE = "+0%"
DEFAULT_PITCH = "+0Hz"


async def generate_base_audio(
    text: str, output_path: Path, rate: str = DEFAULT_RATE, pitch: str = DEFAULT_PITCH
) -> list[dict]:
    """Generate base audio with edge-tts and return word timestamps."""
    import edge_tts

    communicate = edge_tts.Communicate(
        text,
        voice=BASE_VOICE,
        rate=rate,
        pitch=pitch,
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
    parser.add_argument(
        "--rate",
        default=DEFAULT_RATE,
        help="Speech rate adjustment (e.g., '+12%%' for excited, '-10%%' for calm)",
    )
    parser.add_argument(
        "--pitch",
        default=DEFAULT_PITCH,
        help="Pitch adjustment (e.g., '+3Hz' for excited, '-2Hz' for calm)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file for base audio
    temp_audio = output_path.parent / "temp_base_audio.mp3"

    print(f"[1/3] Generating base audio with edge-tts...", file=sys.stderr)
    if args.rate != DEFAULT_RATE or args.pitch != DEFAULT_PITCH:
        print(f"      Rate: {args.rate}, Pitch: {args.pitch}", file=sys.stderr)
    word_timestamps, submaker = await generate_base_audio(
        args.text, temp_audio, rate=args.rate, pitch=args.pitch
    )
    print(f"      Generated {len(word_timestamps)} word timestamps", file=sys.stderr)

    print(f"[2/3] Converting to Adam's voice with RVC...", file=sys.stderr)
    print(f"      Model: {MODEL_PATH.name}", file=sys.stderr)
    if INDEX_PATH:
        print(f"      Index: {INDEX_PATH.name}", file=sys.stderr)
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
