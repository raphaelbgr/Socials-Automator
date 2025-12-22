# Fix Instagram Reel Captions

Automated tool to fix empty captions on posted Instagram reels using Chrome browser automation.

## Overview

When Instagram's API has issues (rate limits, network errors), reels may be posted with empty captions. This script automates the process of editing those captions through Instagram's web interface.

**Key Features:**
- Automatic Chrome port assignment per profile (no conflicts when running multiple instances)
- Auto-launches Chrome with correct settings
- Interactive profile selection
- Progress tracking and error handling
- Retry logic with/without hashtags

## Prerequisites

### 1. Sync Captions First

Before fixing, you need to identify which reels have empty captions:

```bash
py -m socials_automator.cli sync-captions ai.for.mortals
```

This fetches actual captions from Instagram and stores them in `metadata.json`.

### 2. Install Dependencies

```bash
pip install selenium typer rich
```

### 3. Chrome Browser

Google Chrome must be installed. The script will auto-detect the installation path.

## Usage

### Interactive Mode (Recommended)

```bash
py scripts/fix_captions.py
```

Shows a table of all profiles with empty caption counts and lets you select one.

### Fix Specific Profile

```bash
py scripts/fix_captions.py fix ai.for.mortals
```

### Preview Without Fixing (Dry Run)

```bash
py scripts/fix_captions.py fix ai.for.mortals --dry-run
```

### Limit Number of Reels

```bash
py scripts/fix_captions.py fix ai.for.mortals -n 10
```

### Fix Without Hashtags

Use this if hashtags are causing save errors:

```bash
py scripts/fix_captions.py fix ai.for.mortals --no-hashtags
```

### Check Status

See all profiles with their port assignments and Chrome status:

```bash
py scripts/fix_captions.py status
```

### Launch Chrome Manually

```bash
py scripts/fix_captions.py launch-chrome-cmd ai.for.mortals
```

## How It Works

### Automatic Port Assignment

Each profile gets a unique Chrome debug port calculated from its name:

| Profile | Port | User Data Dir |
|---------|------|---------------|
| ai.for.mortals | 9275 | `~/ChromeDebug/ai.for.mortals` |
| news.but.quick | 9279 | `~/ChromeDebug/news.but.quick` |

This means:
- Same profile always gets same port
- Multiple profiles can run simultaneously
- Each profile has its own Chrome session/cookies

### Fix Process

For each reel:

1. **Load** - Opens the reel's Instagram page
2. **Menu** - Clicks "..." (More options)
3. **Manage** - Clicks "Manage" in the menu
4. **Edit** - Clicks "Edit" to open caption editor
5. **Caption** - Clears existing text and types new caption
6. **Save** - Clicks "Done" to save

If a reel fails with hashtags, it automatically retries without hashtags.

### Error Handling

| Error | Action |
|-------|--------|
| Chrome not running | Auto-launches Chrome |
| Session expired | Prompts to log in |
| Reel not found | Skips with error |
| Save failed | Retries without hashtags |
| Caption field not found | Reports error |

## Running Multiple Accounts Simultaneously

You can fix captions for multiple Instagram accounts at the same time:

**Terminal 1:**
```bash
py scripts/fix_captions.py fix ai.for.mortals
```

**Terminal 2:**
```bash
py scripts/fix_captions.py fix news.but.quick
```

Each profile uses a different Chrome instance with its own:
- Debug port
- User data directory (separate cookies/sessions)
- Instagram login

## Workflow

### Complete Workflow to Fix Empty Captions

```bash
# Step 1: Sync captions to identify empty ones
py -m socials_automator.cli sync-captions ai.for.mortals

# Step 2: Check how many need fixing
py scripts/fix_captions.py status

# Step 3: Fix them (interactive)
py scripts/fix_captions.py

# Or fix specific profile
py scripts/fix_captions.py fix ai.for.mortals

# Step 4: Re-sync to verify they're fixed
py -m socials_automator.cli sync-captions ai.for.mortals --only-empty
```

### Tips

1. **First Run**: Chrome will open fresh - you'll need to log in to Instagram
2. **Keep Chrome Open**: After logging in, keep the Chrome window open for faster subsequent runs
3. **Rate Limits**: Instagram may rate-limit edits - add delays between sessions if needed
4. **Hashtag Issues**: If captions fail to save, try `--no-hashtags` flag

## Troubleshooting

### "Chrome not found"

Ensure Google Chrome is installed in the standard location, or set the path manually.

### "Cannot connect to Chrome"

1. Close all Chrome windows
2. Let the script launch Chrome automatically
3. Or launch manually:
   ```bash
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9275 --user-data-dir="C:\Users\YourName\ChromeDebug\ai.for.mortals"
   ```

### "Session expired"

Log in to Instagram in the Chrome window that opens.

### "Could not find 'More options' button"

Instagram's UI may have changed. The script uses XPath selectors that may need updating.

### "Caption field not found"

The edit dialog may not have opened correctly. Try running again.

## Files

| File | Description |
|------|-------------|
| `scripts/fix_captions.py` | Main script |
| `docs/empty_captions/fixed_captions.json` | Tracking file for fixed reels |
| `~/ChromeDebug/{profile}/` | Chrome user data directories |

## Commands Reference

```
fix_captions.py [COMMAND] [OPTIONS]

Commands:
  fix               Fix empty Instagram reel captions
  status            Show profiles and Chrome port assignments
  launch-chrome-cmd Launch Chrome for a specific profile

fix Options:
  PROFILE           Profile name (interactive if not provided)
  --dry-run, -d     Preview without making changes
  --no-hashtags     Remove hashtags from captions
  --limit, -n INT   Limit number of reels to fix
  --help            Show help
```
