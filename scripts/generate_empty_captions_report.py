"""Generate markdown reports for reels with EMPTY captions only.

Creates clickable reports for GitHub with direct reel URLs and captions to paste.
"""

import json
from datetime import datetime
from pathlib import Path


def find_empty_caption_reels(profile_path: Path) -> list[dict]:
    """Find all reels with empty Instagram captions."""
    reels_dir = profile_path / "reels"
    if not reels_dir.exists():
        return []

    empty_reels = []

    for year_dir in reels_dir.glob("*"):
        if not (year_dir.is_dir() and year_dir.name.isdigit()):
            continue

        for month_dir in year_dir.glob("*"):
            if not (month_dir.is_dir() and month_dir.name.isdigit()):
                continue

            posted_dir = month_dir / "posted"
            if not posted_dir.exists():
                continue

            for reel_dir in posted_dir.glob("*"):
                if not reel_dir.is_dir():
                    continue

                metadata_path = reel_dir / "metadata.json"
                if not metadata_path.exists():
                    continue

                try:
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    continue

                # Check if we have synced caption data
                instagram_data = metadata.get("instagram", {})
                actual_caption = instagram_data.get("actual_caption")

                # Only include if caption was synced AND is empty
                if actual_caption is not None and actual_caption == "":
                    # Get permalink
                    platform_status = metadata.get("platform_status", {})
                    ig_status = platform_status.get("instagram", {})
                    permalink = ig_status.get("permalink", "")
                    media_id = ig_status.get("media_id", "")
                    uploaded_at = ig_status.get("uploaded_at", "")

                    # Load local caption
                    caption_path = reel_dir / "caption+hashtags.txt"
                    local_caption = ""
                    if caption_path.exists():
                        try:
                            with open(caption_path, encoding="utf-8") as f:
                                local_caption = f.read().strip()
                        except Exception:
                            pass

                    if permalink and local_caption:
                        empty_reels.append({
                            "reel_path": reel_dir,
                            "reel_name": reel_dir.name,
                            "permalink": permalink,
                            "media_id": media_id,
                            "uploaded_at": uploaded_at,
                            "local_caption": local_caption,
                        })

    # Sort by reel name (which includes date)
    return sorted(empty_reels, key=lambda x: x["reel_name"])


def generate_markdown_report(profile_name: str, reels: list[dict], output_path: Path) -> None:
    """Generate markdown report for empty caption reels."""
    lines = [
        f"# Empty Captions - {profile_name}",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Total Empty**: {len(reels)}",
        "",
        "---",
        "",
    ]

    if not reels:
        lines.append("No reels with empty captions found.")
    else:
        lines.append("## Reels to Fix")
        lines.append("")
        lines.append("Click each link, edit the reel, and paste the caption below.")
        lines.append("")

        for idx, reel in enumerate(reels, 1):
            lines.append(f"### {idx}. [{reel['reel_name']}]({reel['permalink']})")
            lines.append("")
            lines.append(f"**Reel URL**: {reel['permalink']}")
            lines.append("")
            lines.append("**Caption:**")
            lines.append("")
            lines.append("```")
            lines.append(reel["local_caption"])
            lines.append("```")
            lines.append("")
            lines.append("---")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    """Generate reports for all profiles."""
    profiles_dir = Path("profiles")
    docs_dir = Path("docs/empty_captions")

    if not profiles_dir.exists():
        print("No profiles directory found")
        return

    total_empty = 0

    for profile_dir in profiles_dir.iterdir():
        if not profile_dir.is_dir():
            continue

        profile_name = profile_dir.name
        print(f"Scanning {profile_name}...")

        empty_reels = find_empty_caption_reels(profile_dir)

        if empty_reels:
            output_path = docs_dir / f"{profile_name}.md"
            generate_markdown_report(profile_name, empty_reels, output_path)
            print(f"  Found {len(empty_reels)} empty captions -> {output_path}")
            total_empty += len(empty_reels)
        else:
            print(f"  No empty captions found")

    print(f"\nTotal: {total_empty} reels with empty captions")


if __name__ == "__main__":
    main()
