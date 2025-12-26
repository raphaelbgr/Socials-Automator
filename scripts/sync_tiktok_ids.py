#!/usr/bin/env python
"""Quick TikTok sync using video IDs extracted from browser console.

This syncs local reels with TikTok by position, using video IDs you already extracted.

Usage:
    python scripts/sync_tiktok_ids.py news.but.quick @news.but.quick "7587450713253383442,7587451447923461384,..."
    python scripts/sync_tiktok_ids.py news.but.quick @news.but.quick --file video_ids.txt
    python scripts/sync_tiktok_ids.py news.but.quick @news.but.quick --dry-run "..."
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def get_all_posted_reels(profile_path: Path) -> list[dict]:
    """Get all posted reels with their metadata, sorted by date."""
    reels = []
    reels_path = profile_path / "reels"

    if not reels_path.exists():
        return reels

    for year_dir in sorted(reels_path.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            posted_dir = month_dir / "posted"
            if not posted_dir.exists():
                continue
            for reel_dir in sorted(posted_dir.iterdir()):
                if not reel_dir.is_dir():
                    continue
                if not (reel_dir / "final.mp4").exists():
                    continue

                reel_info = {
                    "path": reel_dir,
                    "id": reel_dir.name,
                    "tiktok_uploaded": False,
                    "tiktok_video_id": None,
                }

                # Load metadata
                metadata_path = reel_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                        tiktok = meta.get("platform_status", {}).get("tiktok", {})
                        reel_info["tiktok_uploaded"] = tiktok.get("uploaded", False)
                        reel_info["tiktok_video_id"] = tiktok.get("video_id")
                    except Exception:
                        pass

                reels.append(reel_info)

    return reels


def update_tiktok_status(reel_path: Path, video_url: str, video_id: str) -> bool:
    """Update TikTok status in metadata."""
    metadata_path = reel_path / "metadata.json"

    # Load existing metadata
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    # Initialize platform_status if needed
    if "platform_status" not in meta:
        meta["platform_status"] = {}

    # Update TikTok status
    meta["platform_status"]["tiktok"] = {
        "uploaded": True,
        "synced_at": datetime.now().isoformat(),
        "video_url": video_url,
        "video_id": video_id,
    }

    # Write back
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return True


def parse_video_ids(ids_input: str) -> list[str]:
    """Parse video IDs from various formats."""
    # Handle JSON array format
    if ids_input.strip().startswith("["):
        try:
            return json.loads(ids_input)
        except json.JSONDecodeError:
            pass

    # Handle comma-separated or newline-separated
    ids_input = ids_input.replace('"', "").replace("'", "")
    ids_input = ids_input.replace("\n", ",").replace(" ", "")

    ids = [id.strip() for id in ids_input.split(",") if id.strip()]
    return ids


def main():
    parser = argparse.ArgumentParser(description="Quick TikTok sync using video IDs")
    parser.add_argument("profile", help="Profile name (e.g., news.but.quick)")
    parser.add_argument("username", help="TikTok username with @ (e.g., @news.but.quick)")
    parser.add_argument("video_ids", nargs="?", help="Comma-separated video IDs or JSON array")
    parser.add_argument("--file", type=str, help="File containing video IDs (one per line or JSON)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")

    args = parser.parse_args()

    # Normalize username
    username = args.username if args.username.startswith("@") else f"@{args.username}"

    # Find paths
    project_root = Path(__file__).parent.parent
    profile_path = project_root / "profiles" / args.profile

    if not profile_path.exists():
        print(f"[X] Profile not found: {profile_path}")
        return 1

    # Get video IDs
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"[X] File not found: {file_path}")
            return 1
        ids_input = file_path.read_text(encoding="utf-8")
    elif args.video_ids:
        ids_input = args.video_ids
    else:
        print("[X] Provide video IDs as argument or --file")
        return 1

    video_ids = parse_video_ids(ids_input)

    if not video_ids:
        print("[X] No video IDs found!")
        return 1

    print(f"\n{'='*60}")
    print(f"TikTok Quick Sync: {args.profile} <-> {username}")
    print(f"{'='*60}")

    print(f"\n[OK] Parsed {len(video_ids)} video IDs")

    # Build video objects (TikTok returns newest first)
    tiktok_videos = [
        {
            "video_id": vid,
            "url": f"https://www.tiktok.com/{username}/video/{vid}"
        }
        for vid in video_ids
    ]

    # Get local reels (sorted oldest to newest)
    print(f"\n[>] Scanning local reels for {args.profile}...")
    local_reels = get_all_posted_reels(profile_path)
    print(f"[OK] Found {len(local_reels)} posted reels")

    already_synced = sum(1 for r in local_reels if r["tiktok_uploaded"])
    unsynced_reels = [r for r in local_reels if not r["tiktok_uploaded"]]
    print(f"    Already synced: {already_synced}")
    print(f"    Not synced: {len(unsynced_reels)}")

    if not unsynced_reels:
        print("\n[OK] All reels are already synced!")
        return 0

    # TikTok videos are newest-first, reverse to match oldest-first reels
    tiktok_reversed = list(reversed(tiktok_videos))

    # Calculate offset
    # If we have 100 reels and 50 are synced, the next unsynced reel
    # should match TikTok video at index 50 (after reversing)
    print(f"\n[>] Matching by position...")
    print(f"    TikTok videos: {len(tiktok_videos)}")
    print(f"    Offset (already synced): {already_synced}")

    updated = 0
    skipped = 0

    for i, reel in enumerate(unsynced_reels):
        tiktok_index = already_synced + i

        if tiktok_index >= len(tiktok_reversed):
            print(f"    [!] No TikTok video for reel {reel['id']} (index {tiktok_index})")
            skipped += 1
            continue

        video = tiktok_reversed[tiktok_index]

        if args.dry_run:
            print(f"    [DRY] {reel['id']} -> {video['video_id']}")
            updated += 1
        else:
            try:
                update_tiktok_status(
                    reel["path"],
                    video_url=video["url"],
                    video_id=video["video_id"],
                )
                print(f"    [OK] {reel['id']} -> {video['video_id']}")
                updated += 1
            except Exception as e:
                print(f"    [X] {reel['id']}: {e}")
                skipped += 1

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  TikTok videos: {len(tiktok_videos)}")
    print(f"  Local reels: {len(local_reels)}")
    print(f"  Already synced: {already_synced}")
    if args.dry_run:
        print(f"  Would sync: {updated}")
    else:
        print(f"  Newly synced: {updated}")
    if skipped:
        print(f"  Skipped: {skipped}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
