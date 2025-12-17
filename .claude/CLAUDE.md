# Claude Code Instructions for Socials-Automator

## Project Architecture

### Pipeline Flow
```
cli.py -> orchestrator.py -> planner.py -> text.py (AI calls)
                          -> slides/ (image composition)
                          -> output.py (save files)
```

### Key Files
| File | Purpose |
|------|---------|
| `cli.py` | Entry point, commands, logging setup |
| `content/orchestrator.py` | Coordinates generation pipeline |
| `content/planner.py` | 4-phase AI content generation |
| `providers/text.py` | LiteLLM wrapper for all text AI |
| `providers/image.py` | DALL-E, ComfyUI, fal.ai |
| `services/extractor.py` | Instructor-based JSON extraction |
| `tools/executor.py` | AI tool calling (web search) |
| `cli_display.py` | CLI progress display |

### Generation Phases
1. **Phase 1: Planning** - Analyze topic, determine slide count
2. **Phase 2: Structure** - Create hook + slide titles
3. **Phase 3: Content** - Generate each slide (with validation)
4. **Phase 4: CTA** - Create call-to-action

### Logging
- `logs/ai_calls.log` - **Full AI request/response I/O**
- `logs/instagram_api.log` - Instagram API calls

### Config
- `config/providers.yaml` - AI provider settings (priority, models, API keys)
- `profiles/<name>/metadata.json` - Profile config (niches, hashtags, prompts)

### Output Files (per post)
- `caption.txt` - Threads-ready caption (<500 chars)
- `caption+hashtags.txt` - Full Instagram caption (used for posting)
- `metadata.json` - Post metadata
- `slide_01.jpg`, etc. - Slide images

## CRITICAL: No Emojis & ASCII Only

**NEVER use emojis in code or output!** Always use ASCII-compatible characters.

The Windows console (cp1252 encoding) does not support Unicode box-drawing characters or emojis. Use these instead:

| Instead of | Use |
|------------|-----|
| ✓ ✔ | `[OK]` |
| ✗ ✘ | `[X]` |
| ⋯ … | `...` |
| ○ ● | `[ ]` `[*]` |
| → ➔ | `->` |
| ↓ ↑ | `v` `^` |
| ╭╮╰╯│─ | `+` `|` `-` |
| 🔍 📰 🎨 | `@` `#` `*` |

## Tip: Context7 for Documentation

When working with libraries, add `use context7` to get up-to-date docs:

```
How do I use Agno agents? use context7
```

Common libraries:
- Agno: `use library /agno-ai/agno`
- DuckDuckGo Search: `use library /deedy5/duckduckgo_search`
- Rich (CLI): `use library /textualize/rich`
- Pillow: `use library /python-pillow/pillow`
- LiteLLM: `use library /berriai/litellm`

## CRITICAL: Git Commits

**NEVER commit or push automatically!** Only commit when the user explicitly asks.

## CRITICAL: Instagram Posting

**ALWAYS use `--dry-run` when testing upload commands!**

```bash
# CORRECT - for testing carousel posts
python -m socials_automator.cli upload-post ai.for.mortals --dry-run

# CORRECT - for testing reels
python -m socials_automator.cli upload-reel ai.for.mortals --dry-run

# DANGEROUS - only run when user explicitly wants to upload
python -m socials_automator.cli upload-post ai.for.mortals
python -m socials_automator.cli upload-reel ai.for.mortals
```

Never run upload commands without `--dry-run` unless the user explicitly asks to publish to Instagram.

### WARNING: Meta API Rate Limit Bug (Ghost Publish)

**Rate limit errors do NOT mean the post wasn't published!**

Meta's API can:
1. Create all containers successfully
2. Process the carousel
3. Actually publish the post
4. THEN return a rate limit error

This is a known Meta API issue - the publish happens but the response fails.

**Error Codes:**
- `4` = Application rate limit (per-app daily limit)
- `9` = Application-level rate limit (wait 5+ minutes)
- `17` = User-level rate limit (per-account limit)
- `2207032` = Media upload failed (retryable)

**Error Subcodes (more specific):**
- `2207069` = **DAILY POSTING LIMIT** - Content Publishing API limit (~25 posts/day)
  - NOT retryable - must wait until midnight UTC
  - Each carousel counts as multiple actions (1 per image + carousel + publish)
  - Failed retries also count towards the limit
  - The Dashboard rate limits may show 0% used - this is a DIFFERENT limit!

**Ghost Publish Detection:**
The code automatically checks Instagram for recent posts when a rate limit error occurs:
- Fetches last 5 posts from Instagram
- Compares post timestamps within 2-minute window of upload
- If found, marks as success (ghost publish detected)
- If not found AND publish was attempted, does NOT retry (prevents duplicates)

**CRITICAL - Duplicate Prevention:**
Never retry after `publish_attempted=True` flag is set. Meta may have published even if error returned.

### Cloudinary Resume State

When posting is interrupted mid-upload, the state is saved in `metadata.json`:

```json
"_upload_state": {
  "cloudinary_urls": ["https://res.cloudinary.com/..."],
  "uploaded_at": "2025-12-15T10:30:00"
}
```

On resume, the script reuses these URLs instead of re-uploading. After successful post, this state is cleaned up.

### Loop Mode Behavior

The `--loop-each` flag makes the script run continuously. It should NEVER stop on errors:
- Uses exponential backoff: `loop_seconds * 2^consecutive_errors` (max 1 hour)
- Catches ALL exceptions including typer.Exit
- Resets backoff counter on successful iteration
- Rate limit errors (code 9) wait 5 minutes before retry

### Content Generation Rules

**Slide Text Sanitization:**
All slide text (hook, subtext, titles, headings, body) goes through `_sanitize_slide_text()` which removes:
- Hashtags (#word)
- Mentions (@word)
- Emojis (Unicode characters)

Hashtags belong in captions, NOT on image slides.

**Image Prompts:**
Image prompts should describe environmental/lifestyle scenes, NOT tech interfaces:
- Good: "Person working in a cozy coffee shop with warm lighting"
- Bad: "Screenshot of app interface with buttons"

See Phase 2 prompt in `planner.py` for IMAGE STYLE REQUIREMENTS.

**Hook Slide Layout:**
Instagram displays the first slide in a 4:3 container (cropped from 1:1). The hook slide subtext has extra horizontal padding to prevent text being cut off at the edges.

## Commands Overview

**Generation Commands (safe to run):**
- `generate-post` - Generate carousel content
- `generate-reel` - Generate video reel content

**Upload Commands (USE --dry-run FOR TESTING):**
- `upload-post` - Upload carousel to Instagram
- `upload-reel` - Upload reel to Instagram

**Other Commands:**
- `schedule` - Moves posts to pending-post queue (safe to run)
- `queue` - View pending queue

## Folder Structure

```
profiles/<name>/posts/YYYY/MM/
  generated/     - New content from generate-post command
  pending-post/  - Queued for Instagram (from schedule command)
  posted/        - Published to Instagram (from upload-post command)

profiles/<name>/reels/YYYY/MM/
  generated/     - New content from generate-reel command
  pending-post/  - Ready to publish
  posted/        - Published to Instagram (from upload-reel command)
```
