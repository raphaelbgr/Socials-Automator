#!/usr/bin/env python
"""Fully automated TikTok sync using Chrome remote debugging.

This script:
1. Connects to your existing Chrome session (with TikTok logged in)
2. Navigates to TikTok profile page
3. Scrolls to load all videos
4. Extracts video IDs
5. Syncs with local reels (marks them as uploaded on TikTok)

Prerequisites:
- Start Chrome with remote debugging:
  "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\rbgnr\ChromeDebug"
- Be logged into TikTok in that Chrome session

Usage:
    python scripts/sync_tiktok_auto.py news.but.quick @news.but.quick
    python scripts/sync_tiktok_auto.py ai.for.mortals @ai.for.mortals --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("[X] Selenium not installed. Run: pip install selenium")
    sys.exit(1)


def connect_to_chrome(port: int = 9222) -> webdriver.Chrome:
    """Connect to existing Chrome session with remote debugging."""
    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{port}")

    try:
        driver = webdriver.Chrome(options=options)
        print(f"[OK] Connected to Chrome (port {port})")
        return driver
    except Exception as e:
        print(f"[X] Failed to connect to Chrome: {e}")
        print("\nMake sure Chrome is running with remote debugging:")
        print('  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\\Users\\rbgnr\\ChromeDebug"')
        sys.exit(1)


def extract_tiktok_videos(driver: webdriver.Chrome, username: str) -> list[dict]:
    """Navigate to TikTok profile and extract all video IDs."""

    # Navigate to profile
    profile_url = f"https://www.tiktok.com/{username}"
    print(f"\n[>] Navigating to {profile_url}")
    driver.get(profile_url)
    time.sleep(3)

    # Wait for videos to load
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'a[href*="/video/"]'))
        )
    except Exception:
        print("[X] No videos found on page. Check if profile exists and has videos.")
        return []

    # Scroll to load all videos
    print("[>] Scrolling to load all videos...")
    last_count = 0
    no_change_count = 0
    max_no_change = 5

    while no_change_count < max_no_change:
        # Scroll down
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)

        # Count current videos
        video_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/video/"]')
        current_count = len(video_links)

        if current_count == last_count:
            no_change_count += 1
            print(f"    No new videos (attempt {no_change_count}/{max_no_change}), current: {current_count}")
        else:
            no_change_count = 0
            last_count = current_count
            print(f"    Found {current_count} videos, continuing scroll...")

    # Extract video IDs
    print(f"\n[>] Extracting video data...")
    video_links = driver.find_elements(By.CSS_SELECTOR, 'a[href*="/video/"]')

    videos = []
    seen_ids = set()

    for link in video_links:
        href = link.get_attribute("href")
        if "/video/" in href:
            # Extract video ID
            parts = href.split("/video/")
            if len(parts) > 1:
                video_id = parts[1].split("?")[0].split("/")[0]
                if video_id and video_id not in seen_ids:
                    seen_ids.add(video_id)
                    videos.append({
                        "video_id": video_id,
                        "url": f"https://www.tiktok.com/{username}/video/{video_id}"
                    })

    print(f"[OK] Extracted {len(videos)} unique videos")
    return videos


def get_all_posted_reels(profile_path: Path) -> list[dict]:
    """Get all posted reels with their metadata."""
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


def sync_by_position(tiktok_videos: list[dict], local_reels: list[dict], username: str, dry_run: bool = False) -> int:
    """Sync TikTok videos to local reels by position.

    Since both lists are ordered by date (oldest to newest for reels, newest to oldest for TikTok),
    we reverse the TikTok list and match by position.

    This assumes:
    1. All local reels have been uploaded to TikTok in order
    2. No videos were deleted from TikTok
    3. No videos were uploaded to TikTok that aren't in local reels
    """
    # Filter to unsynced reels only
    unsynced_reels = [r for r in local_reels if not r["tiktok_uploaded"]]

    if not unsynced_reels:
        print("[OK] All reels are already synced!")
        return 0

    # TikTok videos are newest-first, reverse to match oldest-first reels
    tiktok_reversed = list(reversed(tiktok_videos))

    print(f"\n[>] Syncing by position...")
    print(f"    TikTok videos: {len(tiktok_videos)}")
    print(f"    Local reels (unsynced): {len(unsynced_reels)}")
    print(f"    Local reels (total): {len(local_reels)}")

    # Calculate offset - how many TikTok videos correspond to already synced reels
    already_synced = len(local_reels) - len(unsynced_reels)

    updated = 0
    skipped = 0

    for i, reel in enumerate(unsynced_reels):
        tiktok_index = already_synced + i

        if tiktok_index >= len(tiktok_reversed):
            print(f"    [!] No TikTok video for reel {reel['id']} (index {tiktok_index} >= {len(tiktok_reversed)})")
            skipped += 1
            continue

        video = tiktok_reversed[tiktok_index]

        if dry_run:
            print(f"    [DRY] {reel['id']} -> {video['video_id']}")
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

    return updated


def main():
    parser = argparse.ArgumentParser(description="Automated TikTok sync via Chrome")
    parser.add_argument("profile", help="Profile name (e.g., news.but.quick)")
    parser.add_argument("username", help="TikTok username with @ (e.g., @news.but.quick)")
    parser.add_argument("--port", type=int, default=9222, help="Chrome remote debugging port")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--save-json", type=str, help="Save extracted videos to JSON file")
    parser.add_argument("--skip-extract", type=str, help="Skip extraction, use this JSON file instead")

    args = parser.parse_args()

    # Normalize username
    username = args.username if args.username.startswith("@") else f"@{args.username}"

    # Find paths
    project_root = Path(__file__).parent.parent
    profile_path = project_root / "profiles" / args.profile

    if not profile_path.exists():
        print(f"[X] Profile not found: {profile_path}")
        return 1

    print(f"\n{'='*60}")
    print(f"TikTok Auto-Sync: {args.profile} <-> {username}")
    print(f"{'='*60}")

    # Get TikTok videos (extract or load from file)
    if args.skip_extract:
        json_path = Path(args.skip_extract)
        if not json_path.exists():
            print(f"[X] JSON file not found: {json_path}")
            return 1
        print(f"\n[>] Loading videos from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            tiktok_videos = json.load(f)
        print(f"[OK] Loaded {len(tiktok_videos)} videos")
    else:
        # Connect to Chrome and extract
        driver = connect_to_chrome(args.port)
        tiktok_videos = extract_tiktok_videos(driver, username)

        if not tiktok_videos:
            print("[X] No videos extracted!")
            return 1

        # Optionally save to JSON
        if args.save_json:
            json_path = Path(args.save_json)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(tiktok_videos, f, indent=2)
            print(f"[OK] Saved to {json_path}")

    # Get local reels
    print(f"\n[>] Scanning local reels for {args.profile}...")
    local_reels = get_all_posted_reels(profile_path)
    print(f"[OK] Found {len(local_reels)} posted reels")

    already_synced = sum(1 for r in local_reels if r["tiktok_uploaded"])
    print(f"    Already synced: {already_synced}")
    print(f"    Not synced: {len(local_reels) - already_synced}")

    if len(local_reels) == 0:
        print("[!] No posted reels found!")
        return 0

    # Sync by position
    updated = sync_by_position(tiktok_videos, local_reels, username, args.dry_run)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  TikTok videos: {len(tiktok_videos)}")
    print(f"  Local reels: {len(local_reels)}")
    print(f"  Already synced: {already_synced}")
    if args.dry_run:
        print(f"  Would sync: {len(local_reels) - already_synced}")
    else:
        print(f"  Newly synced: {updated}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
