"""Enhanced display and logging for TikTok uploads."""

import json
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# JSON Lines logger for TikTok
_json_logger = None


def get_json_logger() -> logging.Logger:
    """Get or create the JSON lines logger for TikTok."""
    global _json_logger
    if _json_logger is None:
        _json_logger = logging.getLogger("tiktok_json")
    return _json_logger


def get_local_timestamp() -> str:
    """Get current local timestamp with GMT offset.

    Returns:
        Formatted string like '16:45:32 GMT-3' or '16:45:32 GMT+5'
    """
    now = datetime.now().astimezone()
    offset_hours = now.utcoffset().total_seconds() / 3600
    offset_sign = '+' if offset_hours >= 0 else ''
    offset_str = f"GMT{offset_sign}{int(offset_hours)}"
    return f"{now.strftime('%H:%M:%S')} {offset_str}"


def get_iso_timestamp() -> str:
    """Get ISO format timestamp with timezone for JSON logs."""
    return datetime.now().astimezone().isoformat()


def log_json(event: str, reel_id: str, **kwargs):
    """Log an event in JSON Lines format.

    Args:
        event: Event name (e.g., 'upload_start', 'upload_complete')
        reel_id: Reel identifier
        **kwargs: Additional fields to include
    """
    logger = get_json_logger()

    log_entry = {
        "ts": get_iso_timestamp(),
        "reel_id": reel_id,
        "event": event,
        **kwargs
    }

    logger.info(json.dumps(log_entry, ensure_ascii=False))


def get_video_info(video_path: Path) -> dict:
    """Get video information using ffprobe.

    Returns:
        Dict with duration, width, height, codec, bitrate
    """
    info = {
        "duration_s": 0,
        "width": 0,
        "height": 0,
        "codec": "unknown",
        "bitrate_kbps": 0,
    }

    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)

            # Get duration from format
            if "format" in data:
                info["duration_s"] = int(float(data["format"].get("duration", 0)))
                info["bitrate_kbps"] = int(int(data["format"].get("bit_rate", 0)) / 1000)

            # Get video stream info
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    info["width"] = stream.get("width", 0)
                    info["height"] = stream.get("height", 0)
                    info["codec"] = stream.get("codec_name", "unknown")
                    break
    except Exception:
        pass

    return info


def get_reel_metadata(reel_path: Path) -> dict:
    """Get reel metadata from metadata.json.

    Returns:
        Dict with instagram info, generation date, etc.
    """
    metadata = {
        "instagram_posted_at": None,
        "instagram_media_id": None,
        "generated_at": None,
        "tiktok_attempts": 0,
    }

    metadata_path = reel_path / "metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Instagram info
            ig = meta.get("platform_status", {}).get("instagram", {})
            if ig.get("uploaded_at"):
                metadata["instagram_posted_at"] = ig["uploaded_at"][:16].replace("T", " ")
            metadata["instagram_media_id"] = ig.get("media_id")

            # TikTok attempts
            tiktok = meta.get("platform_status", {}).get("tiktok", {})
            if tiktok.get("error"):
                metadata["tiktok_attempts"] = 1

            # Generation date
            if "generated_at" in meta:
                metadata["generated_at"] = meta["generated_at"][:16].replace("T", " ")
        except Exception:
            pass

    return metadata


def count_hashtags(text: str) -> tuple[int, list[str]]:
    """Count and extract hashtags from text.

    Returns:
        Tuple of (count, list of hashtags)
    """
    hashtags = re.findall(r'#\w+', text)
    return len(hashtags), hashtags


def format_duration(seconds: int) -> str:
    """Format duration in human readable format."""
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes}m{secs}s"


def format_size(bytes_size: int) -> str:
    """Format file size in human readable format."""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f}MB"


def get_caption_preview(caption: str, max_length: int = 50) -> str:
    """Get a preview of the caption."""
    # Remove newlines and extra spaces
    preview = " ".join(caption.split())
    if len(preview) > max_length:
        return preview[:max_length] + "..."
    return preview


def calculate_eta(current: int, total: int, elapsed_seconds: float) -> str:
    """Calculate estimated time remaining."""
    if current == 0 or elapsed_seconds == 0:
        return "calculating..."

    avg_per_item = elapsed_seconds / current
    remaining = total - current
    eta_seconds = int(remaining * avg_per_item)

    if eta_seconds < 60:
        return f"~{eta_seconds}s"
    elif eta_seconds < 3600:
        return f"~{eta_seconds // 60}m"
    else:
        hours = eta_seconds // 3600
        mins = (eta_seconds % 3600) // 60
        return f"~{hours}h{mins}m"
