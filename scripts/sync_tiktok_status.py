#!/usr/bin/env python
"""Sync TikTok upload status from exported CSV or JSON.

This script reads a CSV/JSON export from TikTok Studio and matches videos
with local reels to sync upload status.

WORKFLOW OPTION 1: Chrome Extension (Easiest)
----------------------------------------------
1. Install "Instant Data Scraper" Chrome extension in your TikTok Chrome session
2. Go to TikTok Studio -> Content (https://www.tiktok.com/tiktokstudio/content)
3. Let the extension auto-scroll to load ALL videos
4. Export to CSV (save as tiktok_videos.csv in project root or specify path)
5. Run this script: python scripts/sync_tiktok_status.py ai.for.mortals

WORKFLOW OPTION 2: Browser Console (No Extension)
--------------------------------------------------
1. Go to TikTok Studio -> Content (Posts tab)
2. Scroll down to load ALL videos
3. Open DevTools (F12) -> Console tab
4. Paste this JavaScript and press Enter:

    // Scroll to load all videos first, then run extraction
    async function extractTikTokVideos() {
        // Auto-scroll to load all videos
        let previousHeight = 0;
        let currentHeight = document.body.scrollHeight;
        while (previousHeight !== currentHeight) {
            previousHeight = currentHeight;
            window.scrollTo(0, document.body.scrollHeight);
            await new Promise(r => setTimeout(r, 1500));
            currentHeight = document.body.scrollHeight;
        }
        window.scrollTo(0, 0);
        await new Promise(r => setTimeout(r, 500));

        // Extract video data
        const videos = [];
        const links = document.querySelectorAll('a[href*="/video/"]');
        links.forEach(link => {
            const href = link.getAttribute('href');
            const title = link.textContent.trim();
            const videoId = href.split('/video/')[1];
            if (title && videoId) {
                let duration = 'N/A';
                let row = link.closest('[class*="Row"]') || link.parentElement?.parentElement?.parentElement;
                if (row) {
                    const match = row.innerText.match(/\\b(\\d{1,2}:\\d{2})\\b/);
                    if (match) duration = match[1];
                }
                videos.push({
                    title: title.substring(0, 100),
                    url: 'https://www.tiktok.com' + href,
                    videoId: videoId,
                    duration: duration
                });
            }
        });
        return videos;
    }

    extractTikTokVideos().then(videos => {
        console.log(`Found ${videos.length} videos`);
        const blob = new Blob([JSON.stringify(videos, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'tiktok_videos.json';
        a.click();
        console.log('Downloaded tiktok_videos.json');
    });

5. Save the downloaded JSON file
6. Run: python scripts/sync_tiktok_status.py ai.for.mortals --json tiktok_videos.json

The script will:
- Parse the CSV/JSON to extract video titles, dates, and URLs
- Match with local reels using fuzzy title/caption matching
- Update metadata.json with TikTok status and video URL
- Generate a report of matched/unmatched videos

Usage:
    python scripts/sync_tiktok_status.py <profile> [--csv <path>] [--json <path>] [--dry-run]
    python scripts/sync_tiktok_status.py ai.for.mortals --csv tiktok_export.csv
    python scripts/sync_tiktok_status.py ai.for.mortals --json tiktok_videos.json
    python scripts/sync_tiktok_status.py ai.for.mortals --dry-run
"""

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    # Normalize strings
    a = re.sub(r'[^\w\s]', '', a.lower())
    b = re.sub(r'[^\w\s]', '', b.lower())
    return SequenceMatcher(None, a, b).ratio()


def extract_keywords(text: str, n: int = 5) -> set:
    """Extract top keywords from text."""
    # Remove hashtags, mentions, and special chars
    text = re.sub(r'[#@]\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    words = text.split()
    # Filter common words
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'and', 'or', 'in', 'on', 'at', 'for', 'with', 'this', 'that', 'you', 'your', 'i', 'my', 'it', 'its'}
    keywords = [w for w in words if w not in stopwords and len(w) > 2]
    return set(keywords[:n])


def parse_tiktok_csv(csv_path: Path) -> list[dict]:
    """Parse TikTok video list from CSV export.

    Tries to auto-detect column names from common export formats.
    """
    videos = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        # Try to detect delimiter
        sample = f.read(2048)
        f.seek(0)

        if '\t' in sample:
            reader = csv.DictReader(f, delimiter='\t')
        else:
            reader = csv.DictReader(f)

        # Get fieldnames and normalize
        fieldnames = [fn.lower().strip() for fn in reader.fieldnames] if reader.fieldnames else []

        # Map common column names
        title_cols = ['title', 'description', 'caption', 'video title', 'name', 'text']
        url_cols = ['url', 'link', 'video url', 'video link', 'href']
        date_cols = ['date', 'created', 'upload date', 'posted', 'time', 'created at', 'post time']
        views_cols = ['views', 'view count', 'plays']

        def find_col(options):
            for opt in options:
                for fn in fieldnames:
                    if opt in fn:
                        return reader.fieldnames[fieldnames.index(fn)]
            return None

        title_col = find_col(title_cols)
        url_col = find_col(url_cols)
        date_col = find_col(date_cols)
        views_col = find_col(views_cols)

        print(f"\nDetected columns:")
        print(f"  Title: {title_col or '[not found]'}")
        print(f"  URL: {url_col or '[not found]'}")
        print(f"  Date: {date_col or '[not found]'}")
        print(f"  Views: {views_col or '[not found]'}")

        if not title_col:
            print(f"\n[!] Could not detect title column. Available columns: {reader.fieldnames}")
            return []

        f.seek(0)
        next(f)  # Skip header
        reader = csv.DictReader(f, fieldnames=reader.fieldnames)
        next(reader)  # Skip header row

        for row in reader:
            video = {
                'title': row.get(title_col, '').strip() if title_col else '',
                'url': row.get(url_col, '').strip() if url_col else '',
                'date': row.get(date_col, '').strip() if date_col else '',
                'views': row.get(views_col, '').strip() if views_col else '',
            }

            # Extract video ID from URL if present
            if video['url']:
                match = re.search(r'/video/(\d+)', video['url'])
                if match:
                    video['video_id'] = match.group(1)

            if video['title']:  # Only include rows with titles
                videos.append(video)

    return videos


def parse_tiktok_json(json_path: Path) -> list[dict]:
    """Parse TikTok video list from JSON export (browser console or API).

    Expected format:
    [
        {"title": "...", "url": "https://www.tiktok.com/@user/video/123", "videoId": "123"},
        ...
    ]
    """
    videos = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[!] JSON must be an array of video objects")
        return []

    print(f"\nParsing JSON with {len(data)} entries...")

    for item in data:
        video = {
            'title': item.get('title', '').strip(),
            'url': item.get('url', '').strip(),
            'date': item.get('date', '').strip() if 'date' in item else '',
            'views': str(item.get('views', '')) if 'views' in item else '',
        }

        # Try different field names for video ID
        video_id = item.get('videoId') or item.get('video_id') or item.get('id')
        if video_id:
            video['video_id'] = str(video_id)
        elif video['url']:
            # Extract from URL
            match = re.search(r'/video/(\d+)', video['url'])
            if match:
                video['video_id'] = match.group(1)

        if video['title'] or video.get('video_id'):
            videos.append(video)

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
                    'path': reel_dir,
                    'id': reel_dir.name,
                    'caption': '',
                    'tiktok_uploaded': False,
                    'tiktok_url': None,
                }

                # Load caption
                caption_path = reel_dir / "caption+hashtags.txt"
                if not caption_path.exists():
                    caption_path = reel_dir / "caption.txt"
                if caption_path.exists():
                    reel_info['caption'] = caption_path.read_text(encoding='utf-8').strip()

                # Load metadata
                metadata_path = reel_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        tiktok = meta.get('platform_status', {}).get('tiktok', {})
                        reel_info['tiktok_uploaded'] = tiktok.get('uploaded', False)
                        reel_info['tiktok_url'] = tiktok.get('video_url')
                        reel_info['generated_at'] = meta.get('generated_at', '')
                    except Exception:
                        pass

                reels.append(reel_info)

    return reels


def match_videos_to_reels(tiktok_videos: list[dict], local_reels: list[dict], threshold: float = 0.4) -> list[tuple]:
    """Match TikTok videos to local reels using fuzzy matching.

    Returns list of (tiktok_video, local_reel, confidence) tuples.
    """
    matches = []
    matched_reels = set()

    for video in tiktok_videos:
        best_match = None
        best_score = 0

        video_title = video['title']
        video_keywords = extract_keywords(video_title, 8)

        for reel in local_reels:
            if reel['id'] in matched_reels:
                continue

            reel_caption = reel['caption']

            # Method 1: Direct similarity
            direct_sim = similarity(video_title, reel_caption[:200])

            # Method 2: Keyword overlap
            reel_keywords = extract_keywords(reel_caption, 8)
            if video_keywords and reel_keywords:
                keyword_overlap = len(video_keywords & reel_keywords) / max(len(video_keywords), len(reel_keywords))
            else:
                keyword_overlap = 0

            # Method 3: Check if video title is substring of caption
            title_in_caption = video_title.lower()[:50] in reel_caption.lower()

            # Combined score
            score = max(direct_sim, keyword_overlap * 0.9)
            if title_in_caption:
                score = max(score, 0.7)

            if score > best_score:
                best_score = score
                best_match = reel

        if best_match and best_score >= threshold:
            matches.append((video, best_match, best_score))
            matched_reels.add(best_match['id'])

    return matches


def update_tiktok_status(reel_path: Path, video_url: str, video_id: str = None) -> bool:
    """Update TikTok status in metadata."""
    metadata_path = reel_path / "metadata.json"

    # Load existing metadata
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    else:
        meta = {}

    # Initialize platform_status if needed
    if 'platform_status' not in meta:
        meta['platform_status'] = {}

    # Update TikTok status
    meta['platform_status']['tiktok'] = {
        'uploaded': True,
        'synced_at': datetime.now().isoformat(),
        'video_url': video_url,
    }
    if video_id:
        meta['platform_status']['tiktok']['video_id'] = video_id

    # Write back
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return True


def main():
    parser = argparse.ArgumentParser(description="Sync TikTok upload status from CSV or JSON export")
    parser.add_argument("profile", help="Profile name (e.g., ai.for.mortals)")
    parser.add_argument("--csv", type=str, help="Path to CSV export")
    parser.add_argument("--json", type=str, help="Path to JSON export (from browser console)")
    parser.add_argument("--threshold", type=float, default=0.4, help="Match confidence threshold (0-1)")
    parser.add_argument("--dry-run", action="store_true", help="Show matches without updating")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed matching info")

    args = parser.parse_args()

    # Find paths
    project_root = Path(__file__).parent.parent
    profile_path = project_root / "profiles" / args.profile

    if not profile_path.exists():
        print(f"[X] Profile not found: {profile_path}")
        return 1

    # Determine input file (prefer JSON if both specified)
    input_path = None
    input_type = None

    if args.json:
        json_path = Path(args.json) if Path(args.json).is_absolute() else project_root / args.json
        if json_path.exists():
            input_path = json_path
            input_type = "json"
        else:
            print(f"[X] JSON file not found: {json_path}")
            return 1
    elif args.csv:
        csv_path = Path(args.csv) if Path(args.csv).is_absolute() else project_root / args.csv
        if csv_path.exists():
            input_path = csv_path
            input_type = "csv"
        else:
            print(f"[X] CSV file not found: {csv_path}")
            return 1
    else:
        # Try to find default files
        for default_file, file_type in [
            ("tiktok_videos.json", "json"),
            ("tiktok_videos.csv", "csv"),
        ]:
            default_path = project_root / default_file
            if default_path.exists():
                input_path = default_path
                input_type = file_type
                break

    if not input_path:
        print("[X] No input file found!")
        print("\nOption 1: Export with Chrome Extension")
        print("  1. Install 'Instant Data Scraper' Chrome extension")
        print("  2. Go to: https://www.tiktok.com/tiktokstudio/content")
        print("  3. Let extension auto-scroll to load all videos")
        print("  4. Click 'Export to CSV'")
        print("  5. Save as: tiktok_videos.csv")
        print("")
        print("Option 2: Export with Browser Console")
        print("  1. Go to: https://www.tiktok.com/tiktokstudio/content")
        print("  2. Open DevTools (F12) -> Console")
        print("  3. Paste the JavaScript from this script's docstring")
        print("  4. Save the downloaded tiktok_videos.json")
        print("")
        print("Then run: python scripts/sync_tiktok_status.py <profile> --json tiktok_videos.json")
        return 1

    print(f"\n=== TikTok Status Sync for {args.profile} ===\n")

    # Parse input file
    print(f"[>] Reading {input_type.upper()}: {input_path}")
    if input_type == "json":
        tiktok_videos = parse_tiktok_json(input_path)
    else:
        tiktok_videos = parse_tiktok_csv(input_path)
    print(f"    Found {len(tiktok_videos)} TikTok videos")

    if not tiktok_videos:
        print(f"[X] No videos found in {input_type.upper()}. Check the file format.")
        return 1

    # Get local reels
    print(f"\n[>] Scanning local reels...")
    local_reels = get_all_posted_reels(profile_path)
    print(f"    Found {len(local_reels)} posted reels")

    already_synced = sum(1 for r in local_reels if r['tiktok_uploaded'])
    not_synced = len(local_reels) - already_synced
    print(f"    Already synced: {already_synced}")
    print(f"    Not synced: {not_synced}")

    # Filter to only unsynced reels for matching
    unsynced_reels = [r for r in local_reels if not r['tiktok_uploaded']]

    # Match videos to reels
    print(f"\n[>] Matching TikTok videos to local reels (threshold: {args.threshold})...")
    matches = match_videos_to_reels(tiktok_videos, unsynced_reels, args.threshold)

    print(f"    Found {len(matches)} matches")

    if not matches:
        print("\n[!] No matches found. Try lowering --threshold or check CSV content.")
        if args.verbose:
            print("\nSample TikTok video titles:")
            for v in tiktok_videos[:5]:
                print(f"  - {v['title'][:80]}...")
            print("\nSample local reel captions:")
            for r in unsynced_reels[:5]:
                print(f"  - {r['caption'][:80]}...")
        return 0

    # Show matches
    print(f"\n{'='*70}")
    print("MATCHES FOUND")
    print(f"{'='*70}\n")

    for i, (video, reel, confidence) in enumerate(matches, 1):
        conf_pct = int(confidence * 100)
        conf_color = "green" if conf_pct >= 70 else "yellow" if conf_pct >= 50 else "red"

        print(f"[{i}] Confidence: {conf_pct}%")
        print(f"    TikTok: {video['title'][:60]}...")
        print(f"    Local:  {reel['id']}")
        print(f"    URL:    {video.get('url', 'N/A')}")
        if args.verbose:
            print(f"    Caption: {reel['caption'][:60]}...")
        print()

    if args.dry_run:
        print(f"[DRY RUN] Would update {len(matches)} reels. Run without --dry-run to apply.")
        return 0

    # Apply updates
    print(f"\n[>] Updating metadata...")
    updated = 0
    for video, reel, confidence in matches:
        try:
            update_tiktok_status(
                reel['path'],
                video_url=video.get('url', ''),
                video_id=video.get('video_id'),
            )
            updated += 1
            print(f"    [OK] {reel['id']}")
        except Exception as e:
            print(f"    [X] {reel['id']}: {e}")

    print(f"\n=== Summary ===")
    print(f"  TikTok videos in CSV: {len(tiktok_videos)}")
    print(f"  Local reels (unsynced): {len(unsynced_reels)}")
    print(f"  Matches found: {len(matches)}")
    print(f"  Metadata updated: {updated}")

    unmatched_videos = len(tiktok_videos) - len(matches)
    unmatched_reels = len(unsynced_reels) - len(matches)

    if unmatched_videos > 0:
        print(f"  Unmatched TikTok videos: {unmatched_videos}")
    if unmatched_reels > 0:
        print(f"  Unmatched local reels: {unmatched_reels}")

    return 0


if __name__ == "__main__":
    exit(main())
