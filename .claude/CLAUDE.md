# Claude Code Instructions for Socials-Automator

## Project Architecture

### Pipeline Flow
```
cli/ -> video/pipeline/orchestrator.py -> script_planner.py -> text.py (AI calls)
     -> content/orchestrator.py -> planner.py -> slides/ (image composition)
                                              -> output.py (save files)
```

### Key Files
| File | Purpose |
|------|---------|
| `cli/app.py` | Typer app, logging setup, command registration |
| `cli/core/` | Shared utilities (types, validators, parsers, paths) |
| `cli/reel/` | Video reel generation and upload feature |
| `cli/post/` | Carousel post generation and upload feature |
| `content/orchestrator.py` | Coordinates carousel generation pipeline |
| `content/planner.py` | 4-phase AI content generation |
| `video/pipeline/orchestrator.py` | Coordinates video reel pipeline |
| `video/pipeline/script_planner.py` | AI script generation for reels |
| `video/pipeline/image_overlay_planner.py` | AI planning for image overlays (segment-based) |
| `video/pipeline/dense_overlay_planner.py` | AI planning for dense overlays (TTL-based) |
| `video/pipeline/image_resolver.py` | Multi-provider image resolution |
| `video/pipeline/image_downloader.py` | Image downloading with caching |
| `video/pipeline/image_overlay_renderer.py` | FFmpeg compositing with pop animations |
| `video/pipeline/image_providers/` | Image provider implementations (pexels, pixabay, websearch, tor) |
| `providers/text.py` | LiteLLM wrapper for all text AI |
| `providers/image.py` | DALL-E, ComfyUI, fal.ai |
| `instagram/client.py` | Instagram Graph API client |
| `instagram/uploader.py` | Cloudinary video/image uploader |
| `hashtag/` | Hashtag validation module (Instagram limit: 5) |
| `hashtag/constants.py` | INSTAGRAM_MAX_HASHTAGS = 5 |
| `hashtag/sanitizer.py` | HashtagSanitizer class for trimming/removal |
| `hashtag/validator.py` | validate_hashtags_in_caption() for pipelines |

## CLI Architecture (Modular, Feature-Based)

The CLI uses a **feature-based vertical slices + functional/stateless** architecture:

```
src/socials_automator/cli/
    __init__.py          # Package exports
    __main__.py          # Entry point for python -m
    app.py               # Typer app, logging, command registration

    core/                # Shared utilities (pure functions)
        types.py         # Result[T], Success, Failure, ProfileConfig
        parsers.py       # parse_interval, parse_length, parse_voice_preset
        validators.py    # validate_profile, validate_voice, validate_length
        paths.py         # get_profile_path, get_output_dir, generate_post_id
        console.py       # Rich console singleton

    reel/                # Video reel feature
        params.py        # ReelGenerationParams, ReelUploadParams (frozen dataclasses)
        validators.py    # validate_reel_generation_params
        display.py       # show_reel_config, show_reel_result (pure functions)
        service.py       # ReelGeneratorService, ReelUploaderService (stateless)
        commands.py      # generate_reel, upload_reel (thin wrappers)

    post/                # Carousel post feature (same pattern as reel/)
    profile/             # Profile management (list_profiles, fix_thumbnails)
    queue/               # Queue management (queue, schedule)
    maintenance/         # Utility commands (init, token, status, new_profile)
```

### Design Patterns

1. **Immutable Parameters**: All params use `@dataclass(frozen=True)`
2. **Result Type**: Explicit error handling with `Result[T] = Success[T] | Failure`
3. **Pure Functions**: Display and validation functions have no side effects
4. **Stateless Services**: Services receive all state via params
5. **Thin Commands**: Commands just orchestrate: params -> validation -> display -> service

### Adding a New Command

1. Create params dataclass in `params.py` with `@dataclass(frozen=True)`
2. Add validation in `validators.py` returning `Result[T]`
3. Add display functions in `display.py` (pure, take Console as arg)
4. Add service logic in `service.py` (stateless class or functions)
5. Add command in `commands.py` (thin wrapper)
6. Register command in `app.py` `register_commands()`

### Generation Phases
1. **Phase 1: Planning** - Analyze topic, determine slide count
2. **Phase 2: Structure** - Create hook + slide titles
3. **Phase 3: Content** - Generate each slide (with validation)
4. **Phase 4: CTA** - Create call-to-action

### Logging
- `logs/ai_calls.log` - **Full AI request/response I/O**
- `logs/instagram_api.log` - Instagram API calls
- `C:\Users\rbgnr\.lmstudio\server-logs` - **LMStudio server logs** (for debugging local AI)

### Real-Time AI Provider Logging

All AI calls show provider/model info in the CLI output:

```
Step 2/10: NewsCurator
Curating and ranking stories with AI
------------------------------------------------------------
  [>] lmstudio/local-model (news_curation)...
  [OK] lmstudio/gemma-the-writer-9b: OK (13071ms)
```

**Components with provider logging:**
| Component | Task Name | Uses |
|-----------|-----------|------|
| NewsCurator | `news_curation` | TextProvider (provider_override) |
| NewsScriptPlanner | `news_script` | TextProvider (preferred_provider) |
| CaptionGenerator | `caption` | LLMFallbackManager (preferred_provider) |
| ScriptPlanner | `script_planning` | LLMFallbackManager (preferred_provider) |
| ResearchQueryGenerator | `research_query_generation` | TextProvider |
| TopicSelector | `topic_selection` | TextProvider |

**All components respect `--text-ai` flag:**
- Passed via `text_ai` parameter to orchestrators
- NewsOrchestrator passes to: NewsCurator (via CurationConfig.provider_override), NewsScriptPlanner, CaptionGenerator
- VideoPipeline passes to: TopicResearcher, ScriptPlanner, CaptionGenerator

### LMStudio Model Selection

**Reasoning models** (GLM-4.5, GLM-4.6v, Qwen3, o1, etc.) "think" first which can cause:
- Slow responses (2+ minutes)
- Empty `content` field (output goes to `reasoning_content`)
- Timeout disconnects

**To disable thinking in LMStudio UI:**
1. Go to **Developer tab** (bottom of sidebar)
2. Navigate to **Inference → Custom Fields**
3. Set **Enable Thinking → OFF**

The code also sends `enable_thinking: false` via API, but the UI setting is most reliable.

**Fast instruction-tuned models** (no thinking):
- `Qwen2.5-7B-Instruct` - fast, excellent for JSON
- `Llama-3.1-8B-Instruct` - reliable, general purpose
- `Mistral-7B-Instruct-v0.3` - very fast
- `Phi-3-medium-instruct` - lightweight

### Config
- `config/providers.yaml` - AI provider settings (priority, models, API keys)
- `config/ai_tools.yaml` - AI tools database (100+ tools with versions, features, video ideas)
- `profiles/<name>/metadata.json` - Profile config (niches, hashtags, prompts, ai_tools_config)

## AI Tools Database

The AI Tools Database keeps content generation up-to-date with current AI tool versions and provides "hidden gem" suggestions to create unique content.

### Key Files

| File | Purpose |
|------|---------|
| `config/ai_tools.yaml` | 100+ AI tools with versions, features, video ideas |
| `knowledge/ai_tools_registry.py` | Singleton registry with tool lookup/search |
| `knowledge/ai_tools_store.py` | ChromaDB store for semantic search + usage tracking |
| `knowledge/models.py` | AITool, VideoIdea, ToolCategory Pydantic models |

### How It Works

1. **Version Context**: Injected into all content generation prompts (carousels + reels)
2. **Hidden Gems**: Lesser-known tools surfaced to AI for unique content ideas
3. **Usage Tracking**: Per-profile tracking avoids repeating tools within cooldown (14 days)
4. **Semantic Search**: ChromaDB enables finding tools by feature description

### Integration Points

| Component | What It Uses |
|-----------|--------------|
| `content/planner.py` | `get_ai_version_context()` in system prompt |
| `content/orchestrator.py` | `_get_ai_tools_suggestions()` + `mark_tools_covered()` |
| `video/pipeline/topic_selector.py` | `_get_version_context()` + `_get_hidden_gems_context()` |
| `video/pipeline/orchestrator.py` | `_mark_tools_covered()` after video generation |

### Updating the Database

Edit `config/ai_tools.yaml` to add or update tools:

```yaml
tools:
  - id: new-tool
    name: "New Tool"
    company: "Company Name"
    current_version: "1.0"
    category: "productivity"  # text, image, video, audio, productivity, coding, research
    is_hidden_gem: true       # Set to surface in hidden gems suggestions
    hidden_gem_score: 85      # 0-100, higher = more likely to be suggested
    features:
      - "Key feature 1"
      - "Key feature 2"
    video_ideas:
      - "Video idea for this tool"
```

### Profile Configuration

Profiles can customize AI tools behavior in `metadata.json`:

```json
"ai_tools_config": {
  "enabled": true,
  "prefer_hidden_gems": true,
  "cooldown_days": 14,
  "categories": ["text", "productivity", "coding"]
}
```

### Usage Tracking

Tools mentioned in generated content are tracked per-profile:
- Stored in `profiles/<name>/data/ai_tools_usage.json`
- ChromaDB index at `profiles/<name>/data/chroma_db/`
- 14-day cooldown prevents repeating same tools

### Output Files (per post)
- `caption.txt` - Threads-ready caption (<500 chars)
- `caption+hashtags.txt` - Full Instagram caption (used for posting)
- `metadata.json` - Post metadata
- `slide_01.jpg`, etc. - Slide images

### Required Artifacts for Reel Upload

All files are REQUIRED for upload to succeed:

| File | Required | Notes |
|------|----------|-------|
| `final.mp4` | Yes | The video file |
| `metadata.json` | Yes | Must have content |
| `caption.txt` | Yes | Min 10 chars |
| `caption+hashtags.txt` | Yes | Min 10 chars, used for Instagram |
| `thumbnail.jpg/png` | Yes | Cover image for Instagram |

The uploader runs a pre-flight checklist and will attempt to regenerate missing files automatically.

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

## Adding New Instagram Profiles

See `docs/INSTAGRAM_API_CHEATSHEET.md` for the complete setup guide including:
- Creating Facebook App
- Getting access tokens (must be EAA... type, NOT IG...)
- Getting Instagram Business Account ID (17841... format)
- Configuring environment variables
- Troubleshooting common errors

Quick steps:

1. **Get Instagram User ID** (quick method - no API needed):
   - Go to [commentpicker.com/instagram-user-id.php](https://commentpicker.com/instagram-user-id.php)
   - Enter username, copy the numeric ID

2. **Add to .env**:
   ```bash
   INSTAGRAM_USER_ID_YOUR_PROFILE=<numeric-id>
   ```

3. **Add platforms section to profile metadata.json**:
   ```json
   "platforms": {
     "instagram": {
       "enabled": true,
       "user_id": "ENV:INSTAGRAM_USER_ID_YOUR_PROFILE",
       "access_token": "ENV:INSTAGRAM_ACCESS_TOKEN"
     }
   }
   ```

4. **Test**: `python -m socials_automator.cli upload-reel your-profile --dry-run`

**Note:** Rate limits are shared at the APP level, not per account. All profiles using the same Facebook App share the same quota.

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

## Hashtag Validation (Instagram Limit: 5)

Instagram reduced the hashtag limit from 30 to **5 maximum** in December 2025. The hashtag module handles this automatically.

### How It Works

1. **Generation** (`generate-reel --hashtags N`):
   - AI generates up to N hashtags (default: 5)
   - `caption_service.py` uses `INSTAGRAM_MAX_HASHTAGS` constant
   - `artifacts.py` limits hashtags when building caption+hashtags.txt

2. **Upload** (`upload-reel`):
   - `validate_hashtags_in_caption()` trims excess hashtags
   - Logs validation result: `[OK] Hashtags: 3/5` or `[!] Trimmed: 8 -> 5`
   - Updates caption+hashtags.txt if trimmed
   - If upload fails with caption error, retries WITHOUT hashtags

### Key Functions

| Function | File | Purpose |
|----------|------|---------|
| `HashtagSanitizer.trim_hashtags()` | sanitizer.py | Remove excess hashtags |
| `HashtagSanitizer.remove_all_hashtags()` | sanitizer.py | Strip all hashtags |
| `validate_hashtags_in_caption()` | validator.py | Validate and auto-trim |
| `remove_hashtags_from_caption()` | validator.py | Fallback for failed uploads |

### CLI Logging

```
Hashtag validation:
  [OK] Hashtags: 3/5
  -- or --
  [!] Trimmed hashtags: 8 -> 5
  Removed: #six, #seven, #eight
  [OK] Updated caption+hashtags.txt
```

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

## Caption Sync & Audit Tools (Reels Only)

When reels are uploaded during Instagram rate limiting, captions may fail to save ("ghost publish" with empty caption). The caption sync tools help detect and fix these issues.

**Note:** These tools work with **reels only**, not carousel posts.

### Key Files

| File | Purpose |
|------|---------|
| `utils/caption_audit.py` | CaptionAuditor (log-based), CaptionSyncer (API-based) |
| `cli/maintenance/commands.py` | sync-captions, audit-captions commands |
| `scripts/fix_empty_captions.py` | Chrome automation to fix captions |
| `docs/empty_captions/` | Generated markdown reports with clickable URLs |

### Workflow: Detecting and Fixing Empty Captions

```bash
# Step 1: Sync actual Instagram captions to local metadata
py -m socials_automator.cli sync-captions ai.for.mortals

# Step 2: Re-sync only empty ones (faster for re-checking)
py -m socials_automator.cli sync-captions ai.for.mortals --only-empty

# Step 3: Fix empty captions via Chrome automation
py scripts/fix_empty_captions.py ai.for.mortals
```

### How sync-captions Works

1. Fetches actual caption from Instagram API for each posted reel
2. Stores in `metadata.json` under `instagram.actual_caption`
3. Compares with local `caption+hashtags.txt`
4. Classifies: `synced` (match), `empty` (IG has no caption), `mismatch` (differs)
5. Generates `fix_captions.md` report with URLs and captions to paste

### Chrome Automation Script (fix_empty_captions.py)

Automates fixing empty captions using your logged-in Chrome browser:

**Setup:**
```bash
# 1. Start Chrome with remote debugging (session persists)
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\Users\rbgnr\ChromeDebug"

# 2. Log in to Instagram (one-time, session persists)

# 3. Run the fix script
py scripts/fix_empty_captions.py ai.for.mortals
```

**What it does (6 steps per reel):**
1. Load reel page
2. Click "..." (more options)
3. Click "Manage"
4. Click "Edit"
5. Set caption text (with Shift+Enter for newlines, 4s wait for autocomplete)
6. Click "Done" (5s wait, check for errors)

**Retry logic:**
- First attempt: with hashtags
- If fails: retry WITHOUT hashtags (starts from step 1)
- Failed reels are removed from tracking, listed at end with errors

**Tracking:**
- Fixed reels saved to `docs/empty_captions/fixed_captions.json`
- Script uses synced `actual_caption` as source of truth
- After re-sync, only truly empty reels remain

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Something went wrong" | Instagram error during save | Script auto-retries without hashtags |
| Empty captions after upload | Rate limiting during ghost publish | Use sync-captions to detect, fix script to repair |
| Newlines not showing | Instagram's Lexical editor | Script uses Shift+Enter for line breaks |

## Image Overlay Feature (--overlay-images)

The `--overlay-images` flag adds contextual images that appear on-screen during narration to illustrate key points (TV shows, products, people, etc.).

### Overlay Modes

| Mode | Flag | Description |
|------|------|-------------|
| Segment-based | `--overlay-images` | One image per narration segment (default) |
| Dense (TTL) | `--overlay-image-ttl 3s` | Fixed display time per image, AI extracts maximum topics |

### How It Works

1. **ImageOverlayPlanner** (segment mode) or **DenseOverlayPlanner** (TTL mode) - AI plans overlays:
   - Identifies moments that benefit from visual illustration
   - Determines timing (start_time, end_time in seconds)
   - Classifies match type: "exact" (specific item) vs "illustrative" (stock OK)
   - Generates search queries for Pexels

2. **ImageResolver** - Resolves images from multiple sources:
   - Local cache lookup first
   - Pexels API search with fallback queries
   - Returns best matching image URL

3. **ImageDownloader** - Downloads and caches images:
   - Profile-scoped cache: `profiles/<name>/data/image_cache/`
   - Avoids re-downloading same images

4. **ImageOverlayRenderer** - Composites images onto video:
   - Uses FFmpeg filter_complex with `enable` expressions
   - Pop-in/pop-out animations (scale + fade)
   - Positioned in corner (default: top-right)
   - GPU acceleration support

### Pipeline Position

```
VideoAssembler -> ThumbnailGenerator -> [ImageOverlay steps] -> SubtitleRenderer -> CaptionGenerator
```

Image overlays are applied AFTER video assembly but BEFORE subtitles to ensure proper layering.

### Key Files

| File | Purpose |
|------|---------|
| `video/pipeline/image_overlay_planner.py` | AI planning for overlay timing/content (segment mode) |
| `video/pipeline/dense_overlay_planner.py` | AI planning for dense overlays (TTL mode) |
| `video/pipeline/image_resolver.py` | Multi-provider image resolution |
| `video/pipeline/image_downloader.py` | Image downloading with profile cache |
| `video/pipeline/image_overlay_renderer.py` | FFmpeg compositing with animations |
| `video/pipeline/image_cache.py` | Profile-scoped image caching |
| `video/pipeline/image_providers/` | Image provider implementations |

### Image Provider Architecture

The image overlay system uses a provider abstraction to support multiple image sources:

```
image_providers/
    __init__.py          # Exports, AVAILABLE_PROVIDERS list
    base.py              # IImageSearchProvider interface, get_image_provider() factory
    pexels.py            # Pexels API provider (default)
    pixabay.py           # Pixabay API provider
    websearch.py         # DuckDuckGo web search provider
    tor_helper.py        # Embedded Tor support (pure Python, no external installation)
```

**IImageSearchProvider Interface:**
```python
class IImageSearchProvider(ABC):
    @property
    def provider_name(self) -> str: ...
    @property
    def cache_folder_name(self) -> str: ...
    async def search(self, query: str, per_page: int, orientation: Optional[str]) -> list[ImageSearchResult]: ...
    async def download(self, image_id: str, url: str, output_path: Path) -> Optional[Path]: ...
    async def close(self) -> None: ...
```

**Available Providers:**

| Provider | API Key | Cache Folder | Notes |
|----------|---------|--------------|-------|
| `pexels` | `PEXELS_API_KEY` | `image-cache` | Default, high quality |
| `pixabay` | `PIXABAY_API_KEY` | `image-cache-pixabay` | Free tier: 100 req/min |
| `websearch` | None | `image-cache-websearch` | DuckDuckGo, unlimited |

**Factory Function:**
```python
from video.pipeline.image_providers import get_image_provider

provider = get_image_provider("websearch", use_tor=True)
results = await provider.search("Netflix logo")
```

### Embedded Tor Support

The `tor_helper.py` module provides embedded Tor using `torpy` (pure Python):

- **No external Tor installation required**
- Thread-safe circuit management
- IP rotation between video generations
- Automatic circuit creation and cleanup

**Key Components:**
```python
from video.pipeline.image_providers.tor_helper import (
    get_tor_helper,      # Get global TorHelper instance
    is_tor_available,    # Check if torpy is installed
    rotate_tor_ip,       # Request new Tor circuit (new IP)
)
```

**IP Rotation Flow:**
```
generate-reel --use-tor
    |
    v
First video generation
    |
    v
[Images downloaded via Tor]
    |
    v
rotate_tor_ip()  <-- Called after each video
    |
    v
New Tor circuit created
    |
    v
Next video generation (different IP)
```

**Dependencies:**
```bash
pip install duckduckgo_search torpy
```

### Usage

```bash
# Generate reel with image overlays
py -m socials_automator.cli generate-reel ai.for.mortals --overlay-images

# Combine with other options
py -m socials_automator.cli generate-reel ai.for.mortals --overlay-images -g --text-ai lmstudio
```

### Image Overlay Script (JSON)

The planner generates a script like:

```json
{
  "overlays": [
    {
      "description": "Netflix logo or interface",
      "search_query": "Netflix streaming service logo",
      "match_type": "exact",
      "start_time": 5.2,
      "end_time": 8.5,
      "position": "top-right"
    }
  ]
}
```

## Commands Overview

**Generation Commands (safe to run):**
- `generate-post <profile>` - Generate carousel content
- `generate-reel <profile>` - Generate video reel content
  - `--length 1m` - Target duration (30s, 1m, 1m30s) - default: 1m
  - `--gpu-accelerate / -g` - Enable GPU acceleration
  - `--gpu <index>` - Specific GPU index
  - `--loop-count / -n <count>` - Generate multiple videos
  - `--loop-each <interval>` - Interval between loops (e.g., 5m, 30m, 1h) - default: 3s
  - `--overlay-images` - Add contextual images that illustrate narration
  - `--image-provider <provider>` - Image source: pexels, pixabay, websearch (default: websearch)
  - `--overlay-image-ttl <time>` - Fixed display time per image (e.g., 3s). Enables dense mode.
  - `--overlay-image-minimum <count>` - Target number of images for dense mode (auto if omitted)
  - `--use-tor` - Route websearch through embedded Tor for anonymity

**Upload Commands (USE --dry-run FOR TESTING):**
- `upload-post <profile> [post_id]` - Upload carousel to Instagram
- `upload-reel <profile> [reel_id]` - Upload reel to Instagram

**Profile Commands:**
- `list-profiles` - List all available profiles
- `new-profile <name> --handle <handle>` - Create a new profile
- `status <profile>` - Show profile status and content counts
- `fix-thumbnails <profile>` - Generate missing thumbnails for reels

**Queue Commands:**
- `schedule <profile>` - Move generated posts to pending-post queue
- `queue <profile>` - View pending queue

**Maintenance Commands:**
- `init` - Initialize project structure
- `token --check` - Check Instagram token validity
- `token --refresh` - Refresh Instagram token
- `list-niches` - List available niches
- `update-artifacts <profile>` - Update artifact metadata for reels

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
