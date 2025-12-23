# Socials Automator

100% automated Instagram carousel content generator. Creates professional carousel posts with AI-generated text and images - from a single command.

**Live Demo:** See it in action at [@ai.for.mortals](https://www.instagram.com/ai.for.mortals/) - every post on this page was generated 100% automatically.

## Run 100% FREE with Local AI

No API costs. No subscriptions. Run everything on your own computer:

- **[LM Studio](https://lmstudio.ai/)** - Run powerful language models (Llama, Mistral, Qwen) locally. Your hardware, your data, zero cost.
- **[ComfyUI](https://www.comfy.org/)** - Node-based Stable Diffusion interface. Generate high-quality images locally with full control.

```bash
# Generate and post with 100% local AI - completely FREE
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui --upload

# Run continuously every 5 minutes
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui --upload --loop-each 5m

# Enable AI web research for up-to-date content
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui --upload --ai-tools
```

## Features

- **100% Local AI Support** - LM Studio for text, ComfyUI for images - completely FREE
- **Cloud AI Providers** - Z.AI, OpenAI, Groq, Gemini, fal.ai, DALL-E with automatic fallback
- **Smart slide count** - AI decides optimal number of slides (3-10) based on topic
- **Square format output** - 1080x1080 Instagram-optimized images
- **Auto topic generation** - AI generates fresh topics based on your niche
- **Post history awareness** - Avoids repeating recent topics
- **Profile-based config** - Manage multiple Instagram accounts
- **AI-driven research** - AI decides when to search the web for facts (--ai-tools)
- **Batch posting** - Post all queued content in one command
- **Loop mode** - Continuously generate posts at intervals (--loop-each)

## Usage Workflows

The tool supports three workflows depending on your needs:

### Workflow 1: Generate Only (Manual Posting)

Generate carousel content, then manually upload to Instagram:

```bash
# Generate a post
python -m socials_automator.cli generate-post ai.for.mortals --topic "5 AI tools for productivity"

# Output goes to: profiles/ai.for.mortals/posts/2025/12/generated/
# Contains: slide images, caption.txt, hashtags.txt

# Manual posting:
# 1. Open the generated folder
# 2. Upload slide_01.jpg to slide_XX.jpg to Instagram
# 3. Copy caption from caption.txt
# 4. Post!
```

**Best for:** Users who want content creation without Meta/Facebook integration.

---

### Workflow 2: One-Command Generate + Post

Generate and immediately publish to Instagram in a single command:

```bash
# Generate AND post in one command
python -m socials_automator.cli generate-post ai.for.mortals --topic "5 AI tools for productivity" --upload
```

**Requires:** Instagram API + Cloudinary setup (see [Instagram Posting Setup](#instagram-posting-setup))

---

### Workflow 3: Full Automation (Generate → Schedule → Post)

Generate content, schedule it, then auto-post to Instagram (for more control):

```bash
# Step 1: Generate content
python -m socials_automator.cli generate-post ai.for.mortals --topic "5 AI tools for productivity"
# → Creates post in: posts/2025/12/generated/

# Step 2: Schedule for posting (moves to pending queue)
python -m socials_automator.cli schedule ai.for.mortals
# → Moves to: posts/2025/12/pending-post/

# Step 3: Publish to Instagram
python -m socials_automator.cli upload-post ai.for.mortals
# → Publishes and moves to: posts/2025/12/posted/
```

**Folder structure:**
```
posts/2025/12/
├── generated/      ← New posts land here
├── pending-post/   ← Scheduled, ready to publish
└── posted/         ← Successfully published
```

**Requires:** Instagram API + Cloudinary setup (see [Instagram Posting Setup](#instagram-posting-setup))

---

### Workflow 4: Batch Generation

Generate multiple posts at once:

```bash
# Generate 5 posts with AI-chosen topics
python -m socials_automator.cli generate-post ai.for.mortals -n 5

# Schedule all generated posts
python -m socials_automator.cli schedule ai.for.mortals --all

# View the queue
python -m socials_automator.cli queue ai.for.mortals

# Post ALL pending posts in order (default behavior)
python -m socials_automator.cli upload-post ai.for.mortals

# Or post just one at a time
python -m socials_automator.cli upload-post ai.for.mortals --one
```

---

### Workflow 5: Continuous Generation (Loop Mode)

Run the generator in a loop for continuous content creation:

```bash
# Generate a new post every 5 minutes
python -m socials_automator.cli generate-post ai.for.mortals --loop-each 5m

# Generate and post every hour
python -m socials_automator.cli generate-post ai.for.mortals --loop-each 1h --upload

# Use specific AI providers in loop mode
python -m socials_automator.cli generate-post ai.for.mortals --loop-each 10m --text-ai lmstudio --image-ai comfy
```

Press Ctrl+C to stop the loop.

---

### Workflow 6: Video Reels (Generate + Post)

Generate video reels and post them to Instagram:

### Workflow 7: News Briefings (Auto-Aggregated News Videos)

Generate news briefing videos that automatically aggregate content from RSS feeds and web search:

```bash
# News profiles are auto-detected via 'news_sources' in metadata.json
# Just run generate-reel on a news profile and it handles everything
python -m socials_automator.cli generate-reel news.but.quick

# Specify a news edition (morning, midday, evening, night)
python -m socials_automator.cli generate-reel news.but.quick --edition morning

# Customize number of stories per video (default 4)
python -m socials_automator.cli generate-reel news.but.quick --stories 5

# Generate multiple news briefings with upload
python -m socials_automator.cli generate-reel news.but.quick -n 10 --upload

# Full news pipeline with GPU acceleration
py -X utf8 -m socials_automator.cli generate-reel news.but.quick --text-ai lmstudio -g --upload
```

**News Pipeline Flow:**
1. Aggregate news from RSS feeds (TMZ, Variety, Rolling Stone, etc.)
2. Search DuckDuckGo for additional entertainment news
3. AI curates and ranks stories by relevance, virality, and usefulness
4. Generate video script with hook, story segments, and CTA
5. Match stock footage from Pexels for each story
6. Generate voiceover and karaoke-style subtitles
7. Output final news briefing video

**Real-Time AI Provider Logging:**

All AI calls show which provider/model is being used in real-time:

```
Step 2/10: NewsCurator
Curating and ranking stories with AI
------------------------------------------------------------
  [>] lmstudio/local-model (news_curation)...
  [OK] lmstudio/gemma-the-writer-9b: OK (13071ms)

Step 3/10: NewsScriptPlanner
Planning video script from curated news
------------------------------------------------------------
  [>] lmstudio/local-model (news_script)...
  [OK] lmstudio/gemma-the-writer-9b: OK (7611ms)
```

This helps verify your `--text-ai` setting is being used correctly across all pipeline steps.

**Note:** Windows users should use `py -X utf8` to handle Unicode characters in news content.

---

```bash
# One-command: Generate AND upload in a single step
python -m socials_automator.cli generate-reel ai.for.mortals --upload

# Or separate steps for more control:
# Step 1: Generate a video reel
python -m socials_automator.cli generate-reel ai.for.mortals
# -> Creates reel in: reels/2025/12/generated/

# Step 2: Post to Instagram Reels
python -m socials_automator.cli upload-reel ai.for.mortals
# -> Auto-moves to pending-post, publishes, then moves to: reels/2025/12/posted/
```

**Or generate multiple and batch post:**

```bash
# Generate 10 reels and upload each immediately after generation
python -m socials_automator.cli generate-reel ai.for.mortals -n 10 --upload

# Or generate first, upload later:
python -m socials_automator.cli generate-reel ai.for.mortals -n 10

# Post all pending reels
python -m socials_automator.cli upload-reel ai.for.mortals

# Or post just one at a time
python -m socials_automator.cli upload-reel ai.for.mortals --one

# Dry run to validate first
python -m socials_automator.cli upload-reel ai.for.mortals --dry-run
```

**Folder structure:**
```
reels/2025/12/
├── generated/      <- New reels land here
├── pending-post/   <- Auto-moved when posting
└── posted/         <- Successfully published
```

**Note:** Video processing on Instagram's servers can take 1-10 minutes. The command will wait automatically.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Socials-Automator.git
cd Socials-Automator

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[full]"
```

### 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
```

**Minimum required keys** (pick one from each category):

| Category | Provider | Key | Cost |
|----------|----------|-----|------|
| **Text AI** | Z.AI | `ZAI_API_KEY` | Cheap |
| | Groq | `GROQ_API_KEY` | Free tier |
| | OpenAI | `OPENAI_API_KEY` | Paid |
| **Image AI** | fal.ai | `FAL_API_KEY` | ~$0.01/image |
| | OpenAI | `OPENAI_API_KEY` | ~$0.08/image |

### 3. Create Your Instagram Profile

The program needs a **profile** that defines your Instagram account's niche, target audience, content style, and brand voice. This helps the AI generate relevant, on-brand content.

```bash
# Create a new profile interactively
python -m socials_automator.cli new-profile

# Or copy and modify the example profile
cp -r profiles/ai.for.mortals profiles/your-profile-name
```

Then edit `profiles/your-profile-name/metadata.json` with your account details.

### 4. Generate Your First Post

```bash
# Generate a post with a specific topic
python -m socials_automator.cli generate-post your-profile-name --topic "5 ChatGPT tricks for productivity"

# Let AI choose the topic automatically
python -m socials_automator.cli generate-post your-profile-name
```

## CLI Reference

Get help for any command with `--help`:

```bash
python -m socials_automator.cli --help
python -m socials_automator.cli generate-post --help
```

### Main Commands

| Command | Description |
|---------|-------------|
| `generate-post` | Generate carousel posts for a profile |
| `generate-reel` | Generate video reels for Instagram/TikTok |
| `upload-post` | Upload pending carousel posts to Instagram |
| `upload-reel` | Upload pending video reels to Instagram/TikTok |
| `cleanup-reels` | Remove video files from posted reels to free disk space |
| `migrate-platform-status` | Mark existing posted reels with platform status |
| `queue` | List all posts in the publishing queue |
| `schedule` | Move generated posts to pending queue |
| `fix-thumbnails` | Generate missing thumbnails for existing reels |
| `update-artifacts` | Update artifact metadata for existing reels |
| `token` | Manage Instagram access tokens |
| `new-profile` | Create a new profile interactively |
| `list-profiles` | List all available profiles |
| `list-niches` | List available niches from niches.json |
| `status` | Show profile status and recent posts |
| `sync-captions` | Sync actual Instagram captions to local metadata (reels only) |
| `audit-captions` | Audit captions for potential issues (reels only) |
| `init` | Initialize project structure |

---

### generate-post

Generate carousel posts for a profile. By default, the AI decides the optimal number of slides (3-10) based on the topic content.

```bash
python -m socials_automator.cli generate-post <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name to generate for (required) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--topic` | `-t` | Topic for the post. If not provided, AI generates a topic | Auto |
| `--pillar` | `-p` | Content pillar (e.g., tool_tutorials, productivity_hacks) | Auto |
| `--count` | `-n` | Number of posts to generate | 1 |
| `--slides` | `-s` | Force specific slide count (overrides AI decision) | AI decides |
| `--min-slides` | | Minimum slides when AI decides | 3 |
| `--max-slides` | | Maximum slides when AI decides | 10 |
| `--upload` | | Upload to Instagram after generating | False |
| `--auto-retry` | | Retry indefinitely until valid content generated | False |
| `--text-ai` | | Override text AI provider (zai, groq, gemini, openai, lmstudio, ollama) | Config |
| `--image-ai` | | Override image AI provider (dalle, fal_flux, comfy) | Config |
| `--loop-each` | | Run continuously with interval (e.g., 5m, 1h, 30s) | None |
| `--ai-tools` | | Enable AI tool calling - AI decides when to search web | False |

**Examples:**
```bash
# Generate 1 post with AI-chosen topic
python -m socials_automator.cli generate-post ai.for.mortals

# Generate 3 posts with AI-chosen topics
python -m socials_automator.cli generate-post ai.for.mortals -n 3

# Generate post with specific topic
python -m socials_automator.cli generate-post ai.for.mortals -t "How to use ChatGPT for email"

# Generate post with exactly 5 slides
python -m socials_automator.cli generate-post ai.for.mortals -t "AI tools for writers" -s 5

# Generate post with 4-8 slides (AI decides within range)
python -m socials_automator.cli generate-post ai.for.mortals --min-slides 4 --max-slides 8

# Generate and immediately post to Instagram
python -m socials_automator.cli generate-post ai.for.mortals -t "AI productivity tips" --upload

# Use local LM Studio for text and ComfyUI for images
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfy

# Enable AI-driven research (AI decides when to search the web)
python -m socials_automator.cli generate-post ai.for.mortals --ai-tools -t "Latest AI trends 2025"

# Retry until valid content is generated
python -m socials_automator.cli generate-post ai.for.mortals --auto-retry

# Run in loop mode - generate new post every 10 minutes
python -m socials_automator.cli generate-post ai.for.mortals --loop-each 10m

# Loop mode with posting
python -m socials_automator.cli generate-post ai.for.mortals --loop-each 1h --upload
```

---

### generate-reel

Generate video reels for Instagram/TikTok. Uses AI for topic selection and script planning, then matches stock footage from Pexels to create a complete video with voiceover and karaoke-style subtitles.

The narration audio is the source of truth for video length - video clips are trimmed to match the narration duration.

```bash
python -m socials_automator.cli generate-reel <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name to generate for (required) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--topic` | `-t` | Topic for the video (auto-generated if not provided) | Auto |
| `--text-ai` | | Text AI provider (zai, groq, gemini, openai, lmstudio, ollama) | Config |
| `--video-matcher` | `-m` | Video source (pexels) | pexels |
| `--voice` | `-v` | Voice preset (see Available Voices below) | rvc_adam |
| `--voice-rate` | | Speech rate adjustment (e.g., "+12%" faster, "-10%" slower) | +0% |
| `--voice-pitch` | | Pitch adjustment (e.g., "+3Hz" higher, "-2Hz" lower) | +0Hz |
| `--subtitle-size` | | Subtitle font size in pixels | 80 |
| `--font` | | Subtitle font from /fonts folder | Montserrat-Bold.ttf |
| `--length` | `-l` | Target video length (e.g., 30s, 1m, 90s) | 1m |
| `--hashtags` | `-H` | Max hashtags to generate (Instagram limit: 5) | 5 |
| `--output` | `-o` | Output directory | Auto |
| `--dry-run` | | Only run first few steps without full video generation | False |
| `--upload` | | Upload to Instagram immediately after generation | False |
| `--loop` | `-L` | Loop continuously until stopped (Ctrl+C) | False |
| `--loop-count` | `-n` | Generate exactly N videos then stop (implies --loop) | None |
| `--loop-each` | | Interval between loops (e.g., 5m, 1h, 30s) | 3s |
| `--gpu-accelerate` | `-g` | Enable GPU acceleration with NVENC (requires NVIDIA GPU) | False |
| `--gpu` | | GPU index to use (0, 1, etc.). Auto-selects if not specified | Auto |
| `--news` | | Force news mode (auto-detected for profiles with news_sources) | False |
| `--edition` | `-e` | News edition: morning, midday, evening, night | Auto |
| `--stories` | `-s` | Number of news stories per video | 4 |
| `--news-age` | | Max age of news articles in hours | 24 |
| `--overlay-images` | | Add contextual images that illustrate narration (pop-in/pop-out) | False |
| `--image-provider` | | Image provider for overlays: websearch, pexels, pixabay | websearch |
| `--use-tor` | | Route websearch requests through embedded Tor for anonymity | False |

**Available Voices:**
| Voice | Description |
|-------|-------------|
| `rvc_adam` | THE viral TikTok voice - FREE, runs locally (default) |
| `rvc_adam_excited` | Same voice with faster rate and higher pitch |
| `adam_excited` | Alias for rvc_adam_excited |
| `tiktok-adam` | Alias for rvc_adam |
| `adam` | Short alias for rvc_adam |
| `professional_female` | Professional female voice |
| `professional_male` | Professional male voice |
| `friendly_female` | Friendly female voice |
| `friendly_male` | Friendly male voice |
| `energetic` | High-energy voice |
| `british_female` | British accent female |
| `british_male` | British accent male |

**Examples:**
```bash
# Generate a 1-minute video reel with auto-generated topic
python -m socials_automator.cli generate-reel ai.for.mortals

# Generate with specific topic and AI provider
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai lmstudio --topic "5 AI productivity tips"

# Generate a 30-second video
python -m socials_automator.cli generate-reel ai.for.mortals --length 30s

# Generate a 90-second video
python -m socials_automator.cli generate-reel ai.for.mortals --length 90s

# Use a different voice
python -m socials_automator.cli generate-reel ai.for.mortals --voice british_female

# Use excited preset (faster + higher pitch)
python -m socials_automator.cli generate-reel ai.for.mortals --voice adam_excited

# Custom voice adjustments (make it more energetic)
python -m socials_automator.cli generate-reel ai.for.mortals --voice-rate "+12%" --voice-pitch "+3Hz"

# Slower, calmer voice
python -m socials_automator.cli generate-reel ai.for.mortals --voice-rate "-10%" --voice-pitch "-2Hz"

# Larger subtitles
python -m socials_automator.cli generate-reel ai.for.mortals --subtitle-size 100

# Use a different font (from /fonts folder)
python -m socials_automator.cli generate-reel ai.for.mortals --font Poppins-Bold.ttf

# Custom font and size
python -m socials_automator.cli generate-reel ai.for.mortals --font BebasNeue-Regular.ttf --subtitle-size 90

# Test without full video generation
python -m socials_automator.cli generate-reel ai.for.mortals --dry-run

# Generate videos continuously (infinite loop)
python -m socials_automator.cli generate-reel ai.for.mortals --loop

# Generate exactly 10 videos then stop
python -m socials_automator.cli generate-reel ai.for.mortals -n 10

# Generate videos every 5 minutes
python -m socials_automator.cli generate-reel ai.for.mortals --loop-each 5m

# Generate and upload every 30 minutes
python -m socials_automator.cli generate-reel ai.for.mortals --loop-each 30m --upload

# Generate 10 videos with 1 hour interval between each
python -m socials_automator.cli generate-reel ai.for.mortals -n 10 --loop-each 1h --upload

# Generate 50 videos with custom length
python -m socials_automator.cli generate-reel ai.for.mortals -n 50 --length 30s

# Full example: excited voice, large subtitles, 45 seconds
python -m socials_automator.cli generate-reel ai.for.mortals --voice adam_excited --subtitle-size 90 --length 45s

# GPU acceleration (faster rendering with NVIDIA GPU)
python -m socials_automator.cli generate-reel ai.for.mortals --gpu-accelerate
python -m socials_automator.cli generate-reel ai.for.mortals -g

# Use specific GPU (for multi-GPU systems)
python -m socials_automator.cli generate-reel ai.for.mortals -g --gpu 0

# Full example: GPU acceleration, 30-second video, local AI
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai lmstudio --length 30s -g

# Generate and immediately upload to Instagram
python -m socials_automator.cli generate-reel ai.for.mortals --upload

# Generate 5 videos and upload each one after generation
python -m socials_automator.cli generate-reel ai.for.mortals -n 5 --upload

# Full pipeline: GPU acceleration, custom voice, auto-upload
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai lmstudio -g --voice adam_excited --upload

# Add contextual image overlays (TV shows, products, etc.)
# Uses DuckDuckGo websearch by default - no API key needed!
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images

# Full pipeline with image overlays, GPU, and upload
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images -g --text-ai lmstudio --upload

# Use Tor for anonymous image scraping (websearch is default)
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images --use-tor

# Use alternative image providers (require API keys)
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images --image-provider pexels
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images --image-provider pixabay

# Control hashtag count (Instagram limit is 5 as of Dec 2025)
python -m socials_automator.cli generate-reel ai.for.mortals --hashtags 3

# Generate with maximum allowed hashtags
python -m socials_automator.cli generate-reel ai.for.mortals --hashtags 5 --upload

# --- News Briefing Examples ---

# Generate news briefing (auto-detected for news profiles)
python -m socials_automator.cli generate-reel news.but.quick

# Force morning edition
python -m socials_automator.cli generate-reel news.but.quick --edition morning

# Include 5 stories instead of default 4
python -m socials_automator.cli generate-reel news.but.quick --stories 5

# Only use news from the last 12 hours
python -m socials_automator.cli generate-reel news.but.quick --news-age 12

# Full news pipeline with GPU and upload (Windows)
py -X utf8 -m socials_automator.cli generate-reel news.but.quick --text-ai lmstudio -g --upload

# Generate 25 news videos in a loop
py -X utf8 -m socials_automator.cli generate-reel news.but.quick -n 25 -g --upload
```

**Standard Pipeline:**
1. [AI] Select topic from profile content pillars
2. Research topic via web search
3. [AI] Plan video script (targeting --length duration)
4. Generate voiceover (determines actual video duration)
5. **[AI] Validate duration** - regenerates script if too long (up to 10 retries)
6. Search Pexels for stock footage (with local cache)
7. Download video clips
8. Assemble into 9:16 vertical video (matches narration length)
9. Generate thumbnail with auto-fitted text (truncates long text to fit 3:4 container)
10. **(Optional) Add image overlays** - with `--overlay-images` flag:
    - [AI] Plan overlay timing and content from script
    - Search Pexels for contextual images (TV shows, products, etc.)
    - Composite images with pop-in/pop-out animations
11. Add karaoke-style subtitles with moving watermark
12. [AI] Generate caption and hashtags (with AI validation and retry)
13. Output final.mp4 + thumbnail.jpg

**News Pipeline (for news profiles):**
1. Aggregate news from RSS feeds (TMZ, Variety, E!, Rolling Stone, etc.)
2. Search DuckDuckGo for additional entertainment news
3. Deduplicate and filter by age (--news-age hours)
4. [AI] Curate and rank stories (relevance, virality, usefulness scores)
5. Select top N stories (--stories count)
6. [AI] Generate news briefing script with hook and story segments
7. Generate voiceover with news narration
8. Search Pexels for stock footage matching story keywords
9. Assemble 9:16 vertical video with all clips
10. Generate thumbnail with teaser list (auto-fitted: "JUST IN -> Story 1 -> Story 2...")
11. Add karaoke-style subtitles
12. [AI] Generate caption with hashtags
13. Output final.mp4 + thumbnail.jpg

**Duration Validation:**
- The AI generates scripts targeting your `--length` duration
- After voice generation, actual audio duration is checked
- If duration exceeds 1.5x target (e.g., >90s for 1m target), script is regenerated
- AI receives feedback to write shorter content
- Up to 10 regeneration attempts to hit target duration

**Hashtag Validation (Instagram Limit: 5):**
- Instagram reduced hashtag limit from 30 to 5 in December 2025
- Hashtags are validated and trimmed during both generation and upload
- `--hashtags` flag controls max hashtags generated (default: 5)
- During upload, excess hashtags are automatically trimmed
- If upload fails due to caption issues, system retries without hashtags

**GPU Acceleration (`-g`):**
- Uses NVIDIA NVENC for hardware-accelerated video encoding
- Significantly faster than CPU rendering (especially for longer videos)
- Requires NVIDIA GPU with NVENC support (GTX 600+, most modern GPUs)
- Falls back to CPU if GPU unavailable

**Image Providers (`--image-provider`):**

Three providers are available for image overlays:

| Provider | API Key | Description |
|----------|---------|-------------|
| `websearch` (default) | Not needed | DuckDuckGo image search - finds exact matches for shows, products, etc. |
| `pexels` | Required (`PEXELS_API_KEY`) | High-quality stock photos, 200 req/month free |
| `pixabay` | Required (`PIXABAY_API_KEY`) | Free stock photos, 100 req/min |

The default `websearch` provider is ideal for finding exact images of specific content (TV shows, products, apps) mentioned in the narration - no API key required!

**Tor Support (`--use-tor`):**

The `websearch` provider supports embedded Tor for anonymous image scraping:
- Uses pure Python Tor (`torpy`) - no external Tor installation required
- Automatically closes connection after each video (fresh IP for next loop)
- Helps avoid rate limiting when generating many videos
- Requires: `pip install ddgs torpy`

```bash
# Anonymous web scraping with Tor (websearch is already default)
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images --use-tor
```

**Output:**
```
profiles/<profile>/reels/YYYY/MM/generated/<post-id>/
  ├── final.mp4           # Final video with audio and subtitles
  ├── caption.txt         # Instagram caption
  ├── caption+hashtags.txt # Caption with hashtags
  └── metadata.json       # Video metadata
```

**Requirements:**
- `PEXELS_API_KEY` environment variable for stock footage
- `ddgs` package for image overlays (default websearch provider) - `pip install ddgs`
- `torpy` package (optional, for `--use-tor`) - `pip install torpy`
- `PIXABAY_API_KEY` environment variable (optional, for `--image-provider pixabay`)
- RVC models for custom voice (rvc_adam) or use built-in voices (alloy, shimmer)

**Available Fonts** (in `/fonts` folder):
| Font | Style | Best For |
|------|-------|----------|
| `Montserrat-Bold.ttf` | Bold sans-serif | Default - #1 for TikTok/Reels (61% of viral videos) |
| `Montserrat-ExtraBold.ttf` | Extra bold | Maximum impact |
| `Poppins-Bold.ttf` | Clean bold | Educational content |
| `BebasNeue-Regular.ttf` | Condensed | Modern, stylish look |
| `Impact.ttf` | Classic bold | Traditional style |

---

### upload-post

Upload pending carousel posts to Instagram. **By default, posts ALL pending posts** in chronological order. Requires Instagram API and Cloudinary credentials (see [Instagram Posting Setup](#instagram-posting-setup)).

```bash
python -m socials_automator.cli upload-post <profile> [post-id] [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |
| `post-id` | Post ID to publish (optional, posts only this specific post) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--one` | `-1` | Post only the oldest pending (instead of all) | False |
| `--dry-run` | | Validate the post without actually publishing | False |

**Examples:**
```bash
# Post ALL pending posts (default behavior)
python -m socials_automator.cli upload-post ai.for.mortals

# Post only the oldest pending post
python -m socials_automator.cli upload-post ai.for.mortals --one
python -m socials_automator.cli upload-post ai.for.mortals -1

# Post a specific post by ID
python -m socials_automator.cli upload-post ai.for.mortals 20251211-001

# Validate without posting
python -m socials_automator.cli upload-post ai.for.mortals --dry-run
```

**Workflow:**
1. Collects all pending posts from `pending-post/` folder
2. For each post (oldest first):
   - Uploads slide images to Cloudinary (Instagram requires public URLs)
   - Creates Instagram media containers for each image
   - Creates carousel container combining all slides
   - Publishes to Instagram
   - Cleans up temporary Cloudinary uploads
   - Moves post to `posted/` folder
3. Shows summary of published/failed posts

---

### upload-reel

Upload pending video reels to Instagram and/or TikTok. **By default, posts ALL pending reels** in chronological order to Instagram. Supports multi-platform publishing with platform status tracking.

```bash
python -m socials_automator.cli upload-reel <profile> [reel-id] [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |
| `reel-id` | Reel ID to publish (optional, posts only this specific reel) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--one` | `-1` | Post only the oldest pending (instead of all) | False |
| `--dry-run` | | Validate the reel without actually publishing | False |
| `--instagram` | | Upload to Instagram only | True (default) |
| `--tiktok` | | Upload to TikTok only | False |
| `--all` | | Upload to all configured platforms | False |

**Examples:**
```bash
# Post ALL pending reels to Instagram (default behavior)
python -m socials_automator.cli upload-reel ai.for.mortals

# Post only the oldest pending reel
python -m socials_automator.cli upload-reel ai.for.mortals --one
python -m socials_automator.cli upload-reel ai.for.mortals -1

# Post a specific reel by ID
python -m socials_automator.cli upload-reel ai.for.mortals 16-001

# Validate without posting
python -m socials_automator.cli upload-reel ai.for.mortals --dry-run

# Post to TikTok only
python -m socials_automator.cli upload-reel ai.for.mortals --tiktok

# Post to all platforms (Instagram + TikTok)
python -m socials_automator.cli upload-reel ai.for.mortals --all

# Post existing Instagram reels to TikTok (uploads from posted/ folder)
python -m socials_automator.cli upload-reel ai.for.mortals --tiktok
```

**Workflow:**
1. Auto-moves reels from `reels/YYYY/MM/generated/` to `reels/YYYY/MM/pending-post/`
2. For each reel (oldest first):
   - Uploads video to Cloudinary (Instagram requires public URLs)
   - Creates Instagram Reels container
   - Waits for video processing (can take 1-10 minutes)
   - Publishes to Instagram Reels
   - Cleans up temporary Cloudinary upload
   - Moves reel to `reels/YYYY/MM/posted/` folder
3. Shows summary of published/failed reels

**Folder Structure:**
```
profiles/<profile>/reels/YYYY/MM/
├── generated/      <- New reels from `generate-reel` command
├── pending-post/   <- Ready to publish
└── posted/         <- Successfully published
```

**Resume Capability:**
If the Instagram publish fails after Cloudinary upload succeeds, the Cloudinary URL is saved. Running the command again will reuse the existing upload instead of re-uploading.

**Full Upload Flow (Preflight -> Upload -> Postflight):**

When you run `upload-reel` or `generate-reel --upload`, the system runs a comprehensive validation and repair flow:

```
>>> PRE-FLIGHT SCAN
  Scanning generated/pending-post/posted folders...
  Found: 28 reels across folders

>>> DUPLICATE DETECTION
  [OK] No duplicates found
  (or merges duplicates by Instagram media_id/permalink)

>>> FOLDER NORMALIZATION
  [OK] All folder names valid
  (or renames "18-003-reel" -> "18-003-evening-wrap-dec-18")

>>> VALIDATION
  18-004-my-topic-slug
    [OK] All artifacts valid
  (or [REPAIR] Regenerates missing caption/hashtags/thumbnail via AI)
  (or [INVALID] Deletes folders missing final.mp4)

>>> PRE-FLIGHT SUMMARY
  Ready to upload: 1
  Already posted:  27
  Repaired:        0
  Renamed:         0

>>> UPLOAD
  [1/1] 18-004-my-topic-slug
    Uploading to Cloudinary...
    Creating Instagram container...
    Waiting for processing... (1-10 min)
    Publishing...
    [OK] instagram: https://instagram.com/reel/XXX

>>> POST-FLIGHT VERIFICATION
  Checking posted/ folders...
  [FIX] 18-003-reel:
        -> updated topic: Evening Wrap - 2025-12-18...
        -> renamed: 18-003-reel -> 18-003-evening-wrap-2025-12-18
        [OK] All issues fixed

  Verified: 28 | Issues: 0 | Fixed: 1
```

**Preflight Checks:**
| Check | Action |
|-------|--------|
| Duplicate detection | Merges folders with same Instagram media_id |
| Folder normalization | Renames generic names (e.g., "reel") to topic slugs |
| Artifact validation | Checks for final.mp4, caption.txt, thumbnail.jpg |
| Artifact repair | AI regenerates missing captions/hashtags/thumbnails |
| Invalid cleanup | Deletes folders without recoverable video |

**Postflight Checks (on posted/ folder):**
| Check | Action |
|-------|--------|
| Instagram metadata | Verifies media_id, permalink, uploaded_at |
| Folder name | Auto-fixes generic names using metadata/script |
| Topic extraction | Pulls topic from script.json, news_brief, or caption |
| Missing artifacts | Regenerates caption/hashtags if missing |

**Targeted Mode (--upload flag or specific reel_id):**
When uploading a single reel (via `--upload` or `upload-reel <id>`), the system runs a lightweight preflight on just that reel, then full postflight on all posted folders.

**Video Requirements:**
- Format: MP4 (H.264 codec recommended)
- Duration: 15-90 seconds via API
- Aspect Ratio: 9:16 (portrait/vertical)
- Resolution: 1080x1920 recommended

**Platform Status Tracking:**

The system tracks which platforms each reel has been uploaded to via `platform_status` in `metadata.json`:

```json
{
  "platform_status": {
    "instagram": {
      "uploaded": true,
      "uploaded_at": "2025-12-17T10:30:00",
      "media_id": "17841234567890",
      "permalink": "https://www.instagram.com/reel/..."
    },
    "tiktok": {
      "uploaded": false
    }
  }
}
```

This allows you to:
- Upload existing Instagram reels to TikTok later
- See exactly which platforms each reel was posted to
- Avoid duplicate uploads to the same platform

---

### cleanup-reels

Clean up posted reels by removing video files to free disk space. Keeps metadata, thumbnails, and captions for reference. Before deletion, fetches and saves the Instagram video URL to metadata.

```bash
python -m socials_automator.cli cleanup-reels <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--older-than` | `-o` | Only clean reels older than N days | All |
| `--dry-run` | | Preview what would be cleaned without deleting | False |
| `--no-fetch-urls` | | Skip fetching Instagram video URLs (faster) | False |

**Examples:**
```bash
# Preview what would be cleaned (no deletion)
python -m socials_automator.cli cleanup-reels ai.for.mortals --dry-run

# Clean all posted reels
python -m socials_automator.cli cleanup-reels ai.for.mortals

# Clean only reels older than 7 days
python -m socials_automator.cli cleanup-reels ai.for.mortals --older-than 7

# Skip fetching Instagram URLs (faster, but no video_url in metadata)
python -m socials_automator.cli cleanup-reels ai.for.mortals --no-fetch-urls
```

**What gets deleted:**
- `final.mp4` - The video file (20-55 MB each)
- `debug_log.txt` - Debug logs (if present)

**What gets kept:**
- `metadata.json` - Enhanced with cleanup record and Instagram video URL
- `thumbnail.jpg` - Visual reference
- `caption.txt` - Original caption
- `caption+hashtags.txt` - Full posted caption

**Metadata after cleanup:**
```json
{
  "cleanup": {
    "cleaned_at": "2025-12-18T14:00:00",
    "video_deleted": true,
    "space_freed_mb": 45.2,
    "files_removed": ["final.mp4", "debug_log.txt"]
  },
  "platform_status": {
    "instagram": {
      "permalink": "https://www.instagram.com/reel/XXX/",
      "video_url": "https://scontent-xxx.cdninstagram.com/..."
    }
  }
}
```

**Space Savings:**
Typical cleanup results:
- 20 reels = ~500 MB freed
- 100 reels = ~4 GB freed
- 150 reels = ~6 GB freed

---

### migrate-platform-status

Migrate existing posted reels to the new platform status tracking format. This marks all reels in the `posted/` folder as uploaded to Instagram.

```bash
python -m socials_automator.cli migrate-platform-status <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Show what would be done without making changes | False |

**Examples:**
```bash
# Migrate all posted reels to new format
python -m socials_automator.cli migrate-platform-status ai.for.mortals

# Preview what would be done
python -m socials_automator.cli migrate-platform-status ai.for.mortals --dry-run
```

**When to use:**
- After upgrading to support multi-platform uploads
- Before uploading existing Instagram reels to TikTok
- The command is idempotent (safe to run multiple times)

---

### queue

List all posts in the publishing queue (from `generated/` and `pending-post/` folders).

```bash
python -m socials_automator.cli queue <profile>
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Examples:**
```bash
# View the publishing queue
python -m socials_automator.cli queue ai.for.mortals
```

**Output:**
```
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Status  ┃ Post ID           ┃ Slides  ┃ Topic                   ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ pending │ 12-001-ai-tools   │ 6       │ 5 AI tools for email    │
│ pending │ 12-002-chatgpt    │ 5       │ ChatGPT productivity    │
│ generated│ 12-003-automation│ 7       │ Automation tips         │
└─────────┴───────────────────┴─────────┴─────────────────────────┘

Total: 3 post(s) in queue
```

---

### schedule

Move generated posts to the pending-post queue for publishing.

```bash
python -m socials_automator.cli schedule <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Options:**
| Option | Description |
|--------|-------------|
| `--all` | Schedule all generated posts |

**Examples:**
```bash
# Schedule all generated posts
python -m socials_automator.cli schedule ai.for.mortals --all
```

---

### fix-thumbnails

Generate missing thumbnails for existing reels, or regenerate all thumbnails. Uses the raw Pexels source video (from cache) to avoid text-over-text issues.

```bash
python -m socials_automator.cli fix-thumbnails <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--dry-run` | | Show what would be done without making changes | False |
| `--force` | | Regenerate ALL thumbnails, not just missing ones | False |
| `--font-size` | `-s` | Font size in pixels | 54 |

**Examples:**
```bash
# Generate only missing thumbnails
python -m socials_automator.cli fix-thumbnails ai.for.mortals

# Preview what would be done
python -m socials_automator.cli fix-thumbnails ai.for.mortals --dry-run

# Regenerate ALL thumbnails (useful to fix incorrectly generated ones)
python -m socials_automator.cli fix-thumbnails ai.for.mortals --force

# Regenerate with larger text (80px)
python -m socials_automator.cli fix-thumbnails ai.for.mortals --force --font-size 80

# Regenerate with smaller text (40px)
python -m socials_automator.cli fix-thumbnails ai.for.mortals --force -s 40

# Preview full regeneration
python -m socials_automator.cli fix-thumbnails ai.for.mortals --force --dry-run
```

**Thumbnail Specifications:**
- **Font size**: 54px default (customizable with `--font-size`)
- **Max words**: 10 (longer titles are truncated)
- **Max lines**: 3 (text wraps to fit)
- **Source**: Raw Pexels video from cache (no subtitles)
- **Fallback**: Uses final.mp4 if cache unavailable

**Note:** Instagram doesn't support updating cover images for already-posted reels via API. For posted reels, thumbnails are generated locally for reference only. To update them on Instagram, you must do it manually through the Instagram app.

---

### update-artifacts

Update artifact metadata for all existing reels. Scans reel folders and populates the `artifacts` section in `metadata.json` based on what files exist.

```bash
python -m socials_automator.cli update-artifacts <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Show what would be done without making changes | False |

**Examples:**
```bash
# Update all metadata files
python -m socials_automator.cli update-artifacts ai.for.mortals

# Preview what would be done
python -m socials_automator.cli update-artifacts ai.for.mortals --dry-run
```

**Artifact Tracking:**

The command checks for these artifacts and sets their status in `metadata.json`:

| Artifact | File | Status |
|----------|------|--------|
| `video` | final.mp4 | ok/missing |
| `voiceover` | voiceover.mp3 | ok/missing |
| `subtitles` | final.mp4 (burned in) | ok/missing |
| `thumbnail` | thumbnail.jpg | ok/missing |
| `caption` | caption.txt | ok/missing |
| `hashtags` | caption+hashtags.txt | ok/missing |

**Example metadata.json artifacts section:**
```json
{
  "artifacts": {
    "video": {"status": "ok", "file": "final.mp4"},
    "voiceover": {"status": "ok", "file": "voiceover.mp3"},
    "subtitles": {"status": "ok", "file": "final.mp4"},
    "thumbnail": {"status": "ok", "file": "thumbnail.jpg"},
    "caption": {"status": "ok", "file": "caption.txt"},
    "hashtags": {"status": "ok", "file": "caption+hashtags.txt"}
  }
}
```

---

### token

Manage Instagram access tokens. Tokens expire and need periodic refresh.

```bash
python -m socials_automator.cli token <profile> [OPTIONS]
```

**Options:**
| Option | Description |
|--------|-------------|
| `--refresh` | Refresh the access token |
| `--check` | Check token validity |

**Examples:**
```bash
# Check if token is valid
python -m socials_automator.cli token ai.for.mortals --check

# Refresh the token
python -m socials_automator.cli token ai.for.mortals --refresh
```

---

### new-profile

Create a new profile for your Instagram account.

```bash
python -m socials_automator.cli new-profile <name> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `name` | Profile folder name (required) |

**Options:**
| Option | Short | Description | Required |
|--------|-------|-------------|----------|
| `--handle` | `-h` | Instagram handle (without @) | Yes |
| `--niche` | `-n` | Niche ID from niches.json | No |

**Examples:**
```bash
# Create a new profile
python -m socials_automator.cli new-profile my-brand -h mybrand

# Create with specific niche
python -m socials_automator.cli new-profile tech-tips -h techtips -n ai_tools
```

---

### status

Show profile status and recent posts.

```bash
python -m socials_automator.cli status <profile>
```

**Examples:**
```bash
python -m socials_automator.cli status ai.for.mortals
```

---

### sync-captions

Sync actual Instagram captions to local metadata for **reels only**. Fetches the current caption from Instagram for each posted reel and stores it in `metadata.json`. This helps detect reels with empty captions (caused by rate limit errors during upload).

**Note:** This command works with reels only, not carousel posts.

```bash
python -m socials_automator.cli sync-captions <profile> [OPTIONS]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Options:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--no-report` | | Skip generating markdown report | False |
| `--only-empty` | `-e` | Only sync reels with previously synced empty captions (faster) | False |

**Examples:**
```bash
# Full sync - fetch all captions from Instagram
python -m socials_automator.cli sync-captions ai.for.mortals

# Quick re-check - only sync previously empty captions
python -m socials_automator.cli sync-captions ai.for.mortals --only-empty

# Sync without generating a markdown report
python -m socials_automator.cli sync-captions ai.for.mortals --no-report
```

**Output:**
```
Syncing Instagram captions for ai.for.mortals...

  [1/50] 16-001-ai-productivity-tips
         [MATCHED] Caption matches local

  [2/50] 16-002-chatgpt-secrets
         [EMPTY] Caption is empty on Instagram!

  ...

Sync complete!
  Synced: 50
  Empty captions: 2
  Mismatched: 0
  Errors: 0

Report saved to: docs/empty_captions/ai.for.mortals.md
```

**What gets stored in metadata.json:**
```json
{
  "instagram": {
    "actual_caption": "The caption from Instagram...",
    "synced_at": "2025-12-19T10:30:00"
  }
}
```

**Use with fix_captions.py:**
After syncing, run the Chrome automation script to fix empty captions (see [Caption Fixing Scripts](#caption-fixing-scripts) below).

---

### audit-captions

Audit captions for potential issues like missing hashtags, too short/long, or missing CTA. Works with **reels only**.

```bash
python -m socials_automator.cli audit-captions <profile>
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `profile` | Profile name (required) |

**Examples:**
```bash
# Audit all reels for caption issues
python -m socials_automator.cli audit-captions ai.for.mortals
```

**Checks performed:**
- Empty caption
- Missing hashtags
- Caption too short (<50 chars)
- Caption too long (>2000 chars)
- Missing CTA (no "follow", "save", or "share")

---

### list-profiles

List all available profiles.

```bash
python -m socials_automator.cli list-profiles
```

---

### list-niches

List available niches from niches.json with their content pillars.

```bash
python -m socials_automator.cli list-niches
```

---

### init

Initialize project structure (creates config and profiles directories).

```bash
python -m socials_automator.cli init
```

---

## Caption Fixing Scripts

When reels are uploaded during Instagram rate limiting, they may be published with empty captions ("ghost publish"). These scripts help detect and fix this issue.

### Workflow

```
1. sync-captions             Fetch actual captions from Instagram
                             -> Stores in metadata.json

2. fix_captions.py           Chrome automation to edit captions
                             -> Auto-launches Chrome with correct settings
                             -> Each profile has unique port (no conflicts)
                             -> Retries without hashtags on failure

3. sync-captions --only-empty   Quick re-check after fixing
```

### fix_captions.py

Chrome automation script that edits reel captions using Instagram's web interface.

**Key Features:**
- Auto-launches Chrome with correct port/profile settings
- Each profile gets a unique Chrome instance (run multiple accounts simultaneously)
- Interactive profile selection with status table
- Rich CLI with progress display and error handling

**Installation:**
```bash
pip install selenium typer rich
```

**Usage:**
```bash
# Interactive mode (shows profile table, lets you pick)
py scripts/fix_captions.py

# Fix specific profile
py scripts/fix_captions.py fix ai.for.mortals

# Preview what would be fixed (dry run)
py scripts/fix_captions.py fix ai.for.mortals --dry-run

# Limit to 5 reels (good for testing)
py scripts/fix_captions.py fix ai.for.mortals -n 5

# Check status of all profiles
py scripts/fix_captions.py status
```

**What happens when you run it:**
```
+--------------------------- >>> FIX CAPTIONS ----------------------------+
| Instagram Reel Caption Fixer                                            |
+-------------------------------------------------------------------------+

                    Available Profiles
+---------------------------------------------------------+
| #   | Profile        | Empty Captions | Port | Chrome   |
|-----+----------------+----------------+------+----------|
| 1   | ai.for.mortals |             41 | 9275 | Stopped  |
| 2   | news.but.quick |             10 | 9279 | Stopped  |
+---------------------------------------------------------+

Select profile number: 1

Launching Chrome on port 9275...
Chrome launched successfully

+------------------------------ >>> LOGIN --------------------------------+
| ACTION REQUIRED                                                         |
|                                                                         |
| 1. A Chrome window should now be open                                   |
| 2. Log in to Instagram with the account for ai.for.mortals              |
| 3. Make sure you're on instagram.com and logged in                      |
| 4. Come back here and confirm when ready                                |
+-------------------------------------------------------------------------+

Are you logged in to Instagram as ai.for.mortals? [y/n]: y

Fixing 41 reels...
  [  1/41] [OK] 18-003-some-reel-topic
  [  2/41] [OK] 18-004-another-reel
  [  3/41] [RETRY] 18-005-hashtag-issue - without hashtags
  [  3/41] [OK] 18-005-hashtag-issue
  ...
```

**Multi-Account Support:**

Each profile automatically gets a unique Chrome port and user-data directory:

| Profile | Port | Chrome Data Dir |
|---------|------|-----------------|
| ai.for.mortals | 9275 | ~/ChromeDebug/ai.for.mortals |
| news.but.quick | 9279 | ~/ChromeDebug/news.but.quick |

Run simultaneously in separate terminals:
```bash
# Terminal 1
py scripts/fix_captions.py fix ai.for.mortals

# Terminal 2
py scripts/fix_captions.py fix news.but.quick
```

**After fixing, verify:**
```bash
# Re-sync only previously empty captions
py -m socials_automator.cli sync-captions ai.for.mortals --only-empty
```

**Commands Reference:**
```
fix_captions.py [COMMAND]

Commands:
  fix               Fix empty captions (default)
  status            Show all profiles and Chrome port assignments
  launch-chrome-cmd Launch Chrome for a specific profile

fix Options:
  PROFILE           Profile name (interactive if omitted)
  --dry-run, -d     Preview without making changes
  --no-hashtags     Remove hashtags from captions
  --limit, -n INT   Limit number of reels to fix
```

### generate_empty_captions_report.py

Generate markdown reports with clickable links and captions for manual fixing.

```bash
py scripts/generate_empty_captions_report.py
```

Creates reports in `docs/empty_captions/<profile>.md` with:
- Direct reel URLs (clickable)
- Full caption text in code blocks (easy to copy)
- Organized by profile

Useful if you prefer to fix captions manually via Instagram's web interface.

## Output

Generated posts are saved to:
```
profiles/<profile-name>/posts/YYYY/MM/<post-folder>/
  ├── slide_01.jpg      # Hook slide with AI-generated background
  ├── slide_02.jpg      # Content slide 1
  ├── slide_03.jpg      # Content slide 2
  ├── ...
  ├── slide_XX.jpg      # CTA slide
  ├── caption.txt       # Instagram caption
  ├── hashtags.txt      # Hashtags
  ├── metadata.json     # Full post metadata
  └── alt_texts.json    # Accessibility alt texts
```

### Sample Output

After running the generate command, you'll see:

```
╭───────────────────── Socials Automator ─────────────────────╮
│ Generating 1 post(s) for ai.for.mortals                     │
│ Slide count: AI decides (4-10 slides)                       │
╰─────────────────────────────────────────────────────────────╯

╭───────────────────── Post: 20251210-001 ────────────────────╮
│ Status: GENERATING                                          │
│ Step: Generating slide content                              │
│ Progress: 5/8 (62%)                                         │
│                                                             │
│ Slide: 4/6                                                  │
│                                                             │
│ Text AI:                                                    │
│   Provider: zai                                             │
│   Model: GLM-4.5-Air                                        │
│                                                             │
│ Image AI:                                                   │
│   Provider: dalle                                           │
│   Model: dall-e-3                                           │
│                                                             │
│ Session Stats:                                              │
│   Text API calls: 2                                         │
│   Image API calls: 1                                        │
│   Total cost: $0.0820                                       │
╰─────────────────────────────────────────────────────────────╯

Generated 1 post(s)
  - 5 AI Tools for Email Management (6 slides)

Output: profiles/ai.for.mortals/posts/2025/12/
```

## Configuration

### Provider Configuration

Edit `config/providers.yaml` to customize:
- Provider priorities (which AI to try first)
- Model selection
- Timeouts and retry settings
- Task-specific overrides

```yaml
# Example: Make Groq the primary text provider
text_providers:
  groq:
    priority: 1
    enabled: true
    litellm_model: "groq/llama-3.3-70b-versatile"
    api_key_env: "GROQ_API_KEY"
```

### Profile Configuration

Each profile has a `metadata.json` with:
- Profile info (name, handle, niche)
- Target audience demographics
- Content pillars and topics
- Brand voice and style guidelines
- Carousel settings (min/max slides)

### AI Tools Database

The project includes a database of 100+ AI tools (`config/ai_tools.yaml`) that keeps generated content up-to-date with current versions and surfaces "hidden gem" tools for unique content ideas.

**Features:**
- **Version Context**: AI prompts automatically include current tool versions (ChatGPT GPT-4o, Claude Opus 4.5, etc.)
- **Hidden Gems**: Lesser-known tools are surfaced to AI to create unique, differentiated content
- **Usage Tracking**: Per-profile tracking prevents repeating the same tools within 14 days
- **Semantic Search**: Find tools by feature description using ChromaDB

**Updating the Database:**

Edit `config/ai_tools.yaml` to add or update tools:

```yaml
tools:
  - id: new-tool
    name: "New Tool"
    company: "Company Name"
    current_version: "1.0"
    category: "productivity"  # text, image, video, audio, productivity, coding, research
    is_hidden_gem: true       # Surface in hidden gems suggestions
    hidden_gem_score: 85      # 0-100, higher = more likely to be suggested
    features:
      - "Key feature 1"
      - "Key feature 2"
    video_ideas:
      - "Video idea for this tool"
```

**Profile Configuration:**

Optionally customize AI tools behavior per profile in `metadata.json`:

```json
"ai_tools_config": {
  "enabled": true,
  "prefer_hidden_gems": true,
  "cooldown_days": 14,
  "categories": ["text", "productivity", "coding"]
}
```

## API Keys Setup

You need **at least one text provider** and **one image provider** to use Socials Automator. The cheapest combination is Z.AI + fal.ai (~$0.02 per post).

### Minimum Setup (Recommended)

For the cheapest setup, get these two keys:

1. **Z.AI** (text generation) - ~$0.001 per request
2. **fal.ai** (image generation) - ~$0.01 per image

---

### Text Providers

#### Z.AI (Recommended - Cheapest)
Uses the GLM-4.5-Air model, a powerful and affordable option.

1. Go to [https://z.ai](https://z.ai)
2. Click "Sign Up" and create an account
3. Navigate to "API Keys" in your dashboard
4. Click "Create API Key" and copy the key
5. Add to your `.env` file:
   ```
   ZAI_API_KEY=your-api-key-here
   ZAI_API_URL=https://api.z.ai/v1
   ```

**Cost**: ~$0.001 per 1K tokens (very cheap)

#### Groq (Free Tier Available)
Fast inference with Llama 3.3 70B model. Has a generous free tier.

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up with Google, GitHub, or email
3. Go to "API Keys" in the left sidebar
4. Click "Create API Key"
5. Copy the key (starts with `gsk_`)
6. Add to your `.env` file:
   ```
   GROQ_API_KEY=gsk_your-api-key-here
   ```

**Cost**: Free tier includes ~14,400 requests/day

#### Google Gemini
Google's Gemini 2.0 Flash model.

1. Go to [https://aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Click "Get API Key" in the top right
4. Click "Create API Key" and select a project
5. Copy the API key
6. Add to your `.env` file:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

**Cost**: Free tier available, then pay-as-you-go

#### OpenAI (GPT-4o)
Premium option with GPT-4o model.

1. Go to [https://platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to "API Keys" in the left sidebar
4. Click "Create new secret key"
5. Copy the key (starts with `sk-`)
6. Add to your `.env` file:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

**Cost**: ~$0.005 per 1K tokens (more expensive)

---

### Image Providers

#### fal.ai (Recommended - Cheapest)
Uses Flux models for high-quality image generation.

1. Go to [https://fal.ai](https://fal.ai)
2. Click "Sign Up" (top right)
3. Sign up with Google, GitHub, or email
4. Go to "Keys" in your dashboard ([https://fal.ai/dashboard/keys](https://fal.ai/dashboard/keys))
5. Click "Create Key"
6. Copy the API key
7. Add to your `.env` file:
   ```
   FAL_API_KEY=your-api-key-here
   ```

**Cost**: ~$0.01 per image (Flux Schnell) or ~$0.025 (Flux Dev)

#### Replicate (SDXL)
Access to Stable Diffusion XL and other models.

1. Go to [https://replicate.com](https://replicate.com)
2. Sign up with GitHub
3. Go to Account Settings > API tokens
4. Copy your API token
5. Add to your `.env` file:
   ```
   REPLICATE_API_TOKEN=r8_your-token-here
   ```

**Cost**: ~$0.03 per image

#### OpenAI DALL-E 3
Premium image generation from OpenAI.

1. Use the same OpenAI API key from text setup
2. DALL-E 3 access is included with OpenAI API
3. Add to your `.env` file (if not already):
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

**Cost**: ~$0.08 per image (most expensive)

---

### Stock Image Providers (for Image Overlays)

#### Pexels (Default)
High-quality stock photos. Required for video generation.

1. Go to [pexels.com/api](https://www.pexels.com/api/)
2. Sign up for a free account
3. Copy your API key from the dashboard
4. Add to your `.env` file:
   ```
   PEXELS_API_KEY=your-pexels-key
   ```

**Cost**: Free (200 requests/month, then rate limited)

#### Pixabay (Optional)
Alternative stock photo provider.

1. Go to [pixabay.com/api/docs](https://pixabay.com/api/docs/)
2. Sign up for a free account
3. Copy your API key
4. Add to your `.env` file:
   ```
   PIXABAY_API_KEY=your-pixabay-key
   ```

**Cost**: Free (100 requests/minute)

#### Web Search (Default - No API Key)
DuckDuckGo image search with optional Tor support. This is now the default provider for `--overlay-images`.

```bash
# Install dependencies
pip install ddgs torpy

# Use image overlays (websearch is default - no API key needed!)
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images

# With Tor for anonymity (closes connection after each video for fresh IP)
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images --use-tor
```

**Cost**: Free, unlimited, no API key needed

---

### Example .env File

```bash
# Text Providers (pick at least one)
ZAI_API_KEY=your-zai-key
ZAI_API_URL=https://api.z.ai/v1
GROQ_API_KEY=gsk_your-groq-key
GOOGLE_API_KEY=your-google-key
OPENAI_API_KEY=sk-your-openai-key

# Image AI Providers (pick at least one for carousel generation)
FAL_API_KEY=your-fal-key
REPLICATE_API_TOKEN=r8_your-replicate-token
# OPENAI_API_KEY is also used for DALL-E 3

# Stock Image/Video Providers (for reels)
PEXELS_API_KEY=your-pexels-key           # Required for video generation
PIXABAY_API_KEY=your-pixabay-key         # Optional, for --image-provider pixabay
# websearch provider needs no API key (uses DuckDuckGo)
```

### Provider Priority

The system tries providers in order of priority (cheapest first by default):

**Text**: Z.AI → Groq → Gemini → OpenAI
**Image**: fal.ai Flux Dev → fal.ai Flux Schnell → Replicate → DALL-E 3

If a provider fails (rate limit, API error), it automatically falls back to the next one.

## Local AI Setup (100% FREE)

Run completely FREE using local AI models on your own computer. No API keys needed, no costs, no data leaving your machine.

### LM Studio (Text Generation)

[LM Studio](https://lmstudio.ai/) lets you run powerful language models locally.

1. Download and install LM Studio from [lmstudio.ai](https://lmstudio.ai/)
2. Download a model (recommended: `Qwen2.5-7B-Instruct` or `Llama-3.1-8B-Instruct`)
3. Start the local server in LM Studio (default: `http://localhost:1234`)
4. Use with the `--text-ai lmstudio` flag

**Requirements:**
- 8GB+ RAM for 7B models
- 16GB+ RAM for 13B+ models
- GPU recommended but not required

### ComfyUI (Image Generation)

[ComfyUI](https://www.comfy.org/) is a node-based interface for Stable Diffusion.

1. Download ComfyUI from [comfy.org](https://www.comfy.org/) or [GitHub](https://github.com/comfyanonymous/ComfyUI)
2. Download SDXL model (recommended: `sd_xl_base_1.0.safetensors`)
3. Place model in `ComfyUI/models/checkpoints/`
4. Start ComfyUI (default: `http://127.0.0.1:8188`)
5. Use with the `--image-ai comfyui` flag

**Requirements:**
- NVIDIA GPU with 8GB+ VRAM (for SDXL)
- Or use SD 1.5 models with 4GB+ VRAM

### Full Local Setup Example

```bash
# Start LM Studio server (in LM Studio app)
# Start ComfyUI server: python main.py

# Generate with 100% local AI
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui

# Generate, post to Instagram, and loop every 5 minutes
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui --upload --loop-each 5m

# With AI-powered web research
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui --upload --ai-tools
```

**Cost: $0.00** - Everything runs on your hardware.

## Instagram Posting Setup

To use the `upload-post` command to publish directly to Instagram, you need to set up both Instagram API access and Cloudinary for image hosting.

### Requirements

1. **Instagram Business or Creator Account** - Personal accounts don't have API access
2. **Facebook Page** - Your Instagram account must be connected to a Facebook Page
3. **Facebook Developer App** - To get API access tokens
4. **Cloudinary Account** - For temporary image hosting (Instagram requires public URLs)

### Step 1: Set Up Instagram Business Account

1. Open Instagram app and go to Settings > Account
2. Tap "Switch to Professional Account"
3. Choose "Business" or "Creator"
4. Connect to a Facebook Page (create one if needed)

### Step 2: Create Facebook App

1. Go to [developers.facebook.com](https://developers.facebook.com)
2. Click "My Apps" > "Create App"
3. Choose "Business" type
4. Add "Instagram Graph API" product to your app
5. In Instagram Graph API settings, add your Instagram account as a tester

### Step 3: Generate Access Token

1. Go to [Facebook Graph API Explorer](https://developers.facebook.com/tools/explorer/)
2. Select your app from the dropdown
3. Click "Generate Access Token"
4. Select these permissions:
   - `instagram_basic`
   - `instagram_content_publish`
   - `pages_read_engagement`
5. Copy the access token

**Note:** Access tokens expire. For production use, generate a long-lived token using the [Token Debugger](https://developers.facebook.com/tools/debug/accesstoken/).

### Step 4: Get Your Instagram User ID

1. In the Graph API Explorer, make this request:
   ```
   GET /me/accounts
   ```
2. Find your Facebook Page and note the `id`
3. Make this request with your Page ID:
   ```
   GET /{page-id}?fields=instagram_business_account
   ```
4. The `instagram_business_account.id` is your Instagram User ID

### Step 5: Set Up Cloudinary

1. Go to [cloudinary.com](https://cloudinary.com) and create a free account
2. In your Cloudinary Dashboard, find:
   - Cloud Name
   - API Key
   - API Secret

### Step 6: Add Credentials to .env

Add these to your `.env` file:

```bash
# Instagram API
INSTAGRAM_USER_ID=17841405793187218  # Your Instagram User ID
INSTAGRAM_ACCESS_TOKEN=EAAxxxxx...    # Your access token

# Cloudinary (for image hosting)
CLOUDINARY_CLOUD_NAME=your-cloud-name
CLOUDINARY_API_KEY=123456789012345
CLOUDINARY_API_SECRET=abcdefghijklmnop
```

### Step 7: Test Your Setup

```bash
# Dry run to validate credentials
python -m socials_automator.cli upload-post ai.for.mortals --dry-run

# Publish for real
python -m socials_automator.cli upload-post ai.for.mortals
```

### Rate Limits

Instagram has these posting limits:
- **25-50 posts per day** (varies by account age/standing)
- **Carousels**: Up to 10 images per carousel
- **Captions**: Up to 2,200 characters
- **Hashtags**: Maximum 5 per post (reduced from 30 in December 2025)

### Known Meta API Quirks

**Ghost Publish Issue:** Meta's API can return a rate limit error AFTER successfully publishing your post. The system automatically detects this by checking your recent Instagram posts. If you see a rate limit error, don't panic - check Instagram first before retrying.

**Empty Caption Issue:** During ghost publish, the caption may be lost. Use `sync-captions` to detect empty captions and `fix_captions.py` to fix them. See [Caption Fixing Scripts](#caption-fixing-scripts).

**Rate Limit Error Codes:**
- Error 4: Application daily limit reached
- Error 9: Application-level throttling (wait 5+ minutes)
- Error 17: User-level rate limit

The system handles these automatically with exponential backoff and ghost publish detection.

## Project Structure

```
Socials-Automator/
├── config/
│   ├── providers.yaml      # AI provider configuration
│   └── ai_tools.yaml       # AI tools database (100+ tools with versions)
├── profiles/
│   └── ai.for.mortals/     # Example profile
│       ├── metadata.json   # Profile configuration
│       ├── brand/          # Brand assets (logo, fonts)
│       ├── knowledge/      # Knowledge base
│       └── posts/          # Generated posts
│           └── 2025/12/
│               ├── generated/     # New posts
│               ├── pending-post/  # Ready to publish
│               └── posted/        # Published posts
├── logs/
│   └── ai_calls.log        # Detailed AI execution logs
├── src/socials_automator/
│   ├── cli/                # Modular CLI (feature-based architecture)
│   │   ├── app.py          # Typer app, logging, command registration
│   │   ├── core/           # Shared utilities (types, validators, parsers)
│   │   ├── reel/           # Video reel commands (generate, upload)
│   │   ├── post/           # Carousel post commands (generate, upload)
│   │   ├── profile/        # Profile management commands
│   │   ├── queue/          # Queue and schedule commands
│   │   └── maintenance/    # Utility commands (init, token, status)
│   ├── content/            # Content generation
│   │   ├── orchestrator.py # Carousel generation pipeline
│   │   ├── planner.py      # 4-phase AI content planning
│   │   └── models.py       # Data models
│   ├── video/              # Video reel generation
│   │   └── pipeline/       # Video generation pipeline
│   │       └── image_providers/  # Multi-provider image support (websearch, pexels, pixabay, tor)
│   ├── design/             # Slide design
│   │   ├── composer.py     # Image composition
│   │   └── templates.py    # Slide templates
│   ├── providers/          # AI providers
│   │   ├── text.py         # Text generation (with tool calling)
│   │   ├── image.py        # Image generation
│   │   └── config.py       # Provider config
│   ├── hashtag/            # Hashtag validation (Instagram limit: 5)
│   │   ├── constants.py    # INSTAGRAM_MAX_HASHTAGS = 5
│   │   ├── sanitizer.py    # HashtagSanitizer class
│   │   └── validator.py    # Validation for upload pipeline
│   ├── instagram/          # Instagram posting
│   │   ├── client.py       # Instagram Graph API client
│   │   ├── uploader.py     # Cloudinary image uploader
│   │   └── models.py       # Instagram data models
│   ├── tools/              # AI tool calling
│   │   ├── definitions.py  # Tool schemas (web_search, news_search)
│   │   └── executor.py     # Tool execution
│   ├── research/           # Web research
│   │   └── web_search.py   # DuckDuckGo parallel search
│   └── knowledge/          # AI tools database & knowledge base
│       ├── ai_tools_registry.py  # Singleton registry for tool lookup
│       ├── ai_tools_store.py     # ChromaDB store + usage tracking
│       └── models.py             # AITool, VideoIdea, ToolCategory models
├── scripts/
│   ├── fix_captions.py              # Chrome automation to fix empty captions
│   └── generate_empty_captions_report.py  # Generate markdown reports
├── docs/
│   └── empty_captions/              # Markdown reports of empty captions
├── .env.example            # Example environment file
├── pyproject.toml          # Python package config
└── README.md
```

## Internal Architecture (For Developers)

This section documents the internal flow for developers and AI assistants working on the codebase.

### CLI Architecture (Feature-Based)

```
src/socials_automator/cli/
    app.py               # Typer app, logging, command registration
    core/                # Shared utilities (pure functions)
        types.py         # Result[T], Success, Failure, ProfileConfig
        parsers.py       # parse_interval, parse_length, parse_voice_preset
        validators.py    # validate_profile, validate_voice, validate_length
        paths.py         # get_profile_path, get_output_dir, generate_post_id
        console.py       # Rich console singleton
    reel/                # Video reel feature (vertical slice)
        params.py        # ReelGenerationParams, ReelUploadParams (frozen dataclasses)
        validators.py    # validate_reel_generation_params
        display.py       # show_reel_config, show_reel_result (pure functions)
        service.py       # ReelGeneratorService, ReelUploaderService (stateless)
        commands.py      # generate_reel, upload_reel (thin wrappers)
        artifacts.py     # Artifact validation and regeneration
        duplicates.py    # Duplicate detection and merging
        validator.py     # ReelValidator with repair capabilities
    post/                # Carousel post feature (same pattern)
```

### Key Design Patterns

| Pattern | Implementation |
|---------|---------------|
| Immutable params | `@dataclass(frozen=True)` for all parameters |
| Result type | `Result[T] = Success[T] \| Failure` for error handling |
| Stateless services | All state passed via params, no instance variables |
| Thin commands | Commands only orchestrate: params -> validate -> display -> service |

### Upload Flow Architecture

```
upload-reel / generate-reel --upload
         |
         v
    upload_all(params)                    # Single entry point
         |
         +-- params.reel_id set? ------+
         |                              |
         v                              v
  _run_preflight()              _run_preflight_single()
  (full scan/dedupe)            (targeted validation)
         |                              |
         +-------------+----------------+
                       |
                       v
              _upload_single()           # Actual upload logic
                       |
                       v
              _run_postflight()          # Verify & fix posted/
```

### Key Service Methods (reel/service.py)

| Method | Purpose |
|--------|---------|
| `upload_all()` | Main entry point, runs full preflight/upload/postflight |
| `upload_single()` | Delegates to upload_all() with reel_id filter |
| `_run_preflight()` | Full scan, dedupe, normalize, validate all reels |
| `_run_preflight_single()` | Targeted validation for one reel |
| `_run_postflight()` | Verify posted/, auto-fix folder names and artifacts |
| `_normalize_folder_name()` | Rename folders to DD-NNN-topic-slug format |
| `_extract_better_topic()` | Find topic from script.json/news_brief/caption |

### Utility Modules

| Module | Purpose |
|--------|---------|
| `utils/text_fitting.py` | Auto-fit text in image containers (thumbnails) |
| `video/pipeline/thumbnail_generator.py` | Generate thumbnails with fitted text |
| `services/caption_service.py` | AI-powered caption/hashtag generation |
| `news/aggregator.py` | RSS feed + web search news collection |
| `news/curator.py` | AI-powered news ranking and selection |

### Artifact Flow

```
generate-reel
    |
    v
[generated/DD-NNN-topic-slug/]
    final.mp4           # Required - video file
    metadata.json       # Required - topic, duration, platform_status
    caption.txt         # Required - plain caption
    caption+hashtags.txt # Required - caption with hashtags
    thumbnail.jpg       # Required - cover image (auto-fitted text)
    script.json         # Optional - full script for reference
    voiceover.mp3       # Optional - audio file
```

### Cloudinary Upload Architecture

When uploading to Instagram, videos and thumbnails must be hosted at publicly accessible URLs. The system uses Cloudinary for temporary hosting with profile/post-scoped folders to prevent conflicts during concurrent uploads.

#### Scoped Folder Structure

```
socials-automator/                    # Base folder
    ai.for.mortals/                   # Profile name
        18-003-chatgpt-tips/          # Reel folder name (post_id)
            final                     # Video file
            thumbnail                 # Thumbnail image
    news.but.quick/                   # Another profile
        18-004-morning-news/
            final
            thumbnail
```

This ensures concurrent uploads from different profiles (or even different reels from the same profile) never interfere with each other.

#### Upload Flow

```
upload-reel
    |
    v
CloudinaryUploader.upload_video(video_path, profile="ai.for.mortals", post_id="18-003-chatgpt-tips")
    |
    v
Cloudinary folder: socials-automator/ai.for.mortals/18-003-chatgpt-tips/final
    |
    v
Instagram API receives public URL
    |
    v
On success: CloudinaryUploader.cleanup_async() deletes uploaded files
On failure: Files kept in Cloudinary for retry
```

#### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `profile` | `str` | Profile name (e.g., "ai.for.mortals") |
| `post_id` | `str` | Reel folder name (e.g., "18-003-chatgpt-tips") |

These parameters are passed through the upload chain:
1. `ReelUploaderService._upload_to_platform()` extracts profile name from `profile_path.name`
2. `InstagramPublisher.publish_reel()` receives `profile` and `post_id` kwargs
3. `CloudinaryUploader.upload_video()` and `upload_image()` build the scoped folder path

#### Cleanup Behavior

| Scenario | Cloudinary Files |
|----------|------------------|
| Upload succeeds | Deleted immediately after Instagram confirms post |
| Upload fails | Kept in Cloudinary (allows retry without re-upload) |
| Exception occurs | Kept in Cloudinary (same as failure) |

**Note:** Currently, retries re-upload files even if they exist in Cloudinary. Future optimization could check for existing uploads and reuse them.

### Validation States (ReelStatus)

| Status | Meaning | Action |
|--------|---------|--------|
| `VALID` | All artifacts present | Ready to upload |
| `ALREADY_POSTED` | Has Instagram media_id | Skip (move to posted/) |
| `REPAIRABLE` | Missing caption/hashtags/thumbnail | AI regeneration |
| `INVALID` | Missing final.mp4 | Delete folder |

### Folder Naming Convention

```
DD-NNN-topic-slug
 |  |      |
 |  |      +-- Slugified topic (lowercase, hyphens, max 50 chars)
 |  +--------- Sequential number (001-999)
 +------------ Day of month (01-31)

Examples:
  18-001-5-ai-tools-for-productivity
  18-002-evening-wrap-2025-12-18
  18-003-netflix-acquires-warner
```

### News Pipeline Context

```python
class NewsPipelineContext(PipelineContext):
    news_brief: Optional[NewsBrief]      # Curated stories
    thumbnail_text: Optional[str]         # Teaser list for thumbnail
```

The `thumbnail_text` is auto-generated from `news_brief.get_thumbnail_text()`:
```
JUST IN
-> Netflix acquires...
-> GNR announces tour...
-> Emily in Paris S5...
```

### Global News Sources Strategy

The news aggregation system uses a sophisticated multi-source strategy to ensure diverse, fresh, and non-repetitive content.

#### Source Configuration (`config/news_sources.yaml`)

All news sources are defined in a single YAML file with:
- **57 RSS feeds** from 8 global regions
- **36 search queries** in 4 languages (English, Spanish, Portuguese, French)
- Time-based weighting for regional relevance
- Query batch rotation to avoid API rate limits

#### How Source Rotation Works

```
+------------------+     +------------------+     +------------------+
|   Feed Rotator   |     |  Query Rotator   |     |  Time Weighting  |
+------------------+     +------------------+     +------------------+
        |                        |                        |
        v                        v                        v
  Select feeds by         Rotate through           Weight regions by
  current UTC time        query batches            time of day (UTC)
        |                        |                        |
        +------------------------+------------------------+
                                 |
                                 v
                    +------------------------+
                    |   NewsAggregator       |
                    |   fetch_with_rotation()|
                    +------------------------+
                                 |
                                 v
                    Articles with region/language metadata
```

#### Regional Coverage (8 Regions)

| Region | Feeds | Focus | Timezone |
|--------|-------|-------|----------|
| US | 25 | Celebrity, Movies, Music, Streaming | EST |
| UK | 6 | BBC, Guardian, NME, Digital Spy | GMT |
| Korea | 8 | K-Pop, K-Drama (Soompi, AllKPop) | KST |
| Japan | 5 | Anime, J-Pop (ANN, Crunchyroll) | JST |
| LatAm | 7 | Latin music, Telenovelas | BRT |
| Europe | 5 | Euronews, France24, DW | CET |
| India | 5 | Bollywood (Hungama, Pinkvilla) | IST |
| Australia | 1 | Pacific entertainment | AEST |

#### Time-Weighted Selection

Feeds are weighted by time of day (UTC) to prioritize active regions:

| Period | US | UK | Korea | Japan | LatAm |
|--------|----|----|-------|-------|-------|
| Morning (6-12 UTC) | 0.6 | 1.0 | 0.9 | 0.8 | 0.5 |
| Afternoon (12-18 UTC) | 0.9 | 0.9 | 0.6 | 0.6 | 0.8 |
| Evening (18-24 UTC) | 1.0 | 0.7 | 0.5 | 0.5 | 1.0 |
| Night (0-6 UTC) | 0.7 | 0.5 | 0.8 | 0.9 | 0.9 |

Higher weight = more feeds selected from that region.

#### Query Batch Rotation

To avoid API rate limits and ensure diverse results:

```
Batch 1 (12 queries)  -->  Batch 2 (12 queries)  -->  Batch 3 (12 queries)
        ^                                                      |
        |                                                      |
        +------------------- 6-hour cooldown ------------------+
```

- Queries are divided into 3 batches
- Each batch runs for 6 hours before rotating
- State is persisted in `data/query_rotation_state.json`

#### UTC Time Normalization

All times in news content are converted to UTC for consistency:

```
Original: "Concert starts at 7pm EST"
   |
   v
Normalized: "Concert starts at 12:00 AM UTC (next day)"
```

The `TimeNormalizer` class handles:
- Multiple time formats (12h, 24h, with/without timezone)
- 50+ timezone abbreviations (EST, PST, KST, JST, etc.)
- Day rollover handling ("next day", "prev day")

#### Article Metadata

Each article includes enhanced metadata for tracking:

```python
@dataclass
class NewsArticle:
    title: str
    summary: str
    # ... standard fields ...

    # Location/language metadata
    region: str          # "us", "korea", "uk", etc.
    source_language: str # "en", "es", "ko", etc.
    source_timezone: str # "EST", "KST", etc.
    was_translated: bool # True if translated to English
    country_code: str    # "US", "KR", "GB", etc.
```

#### CLI Logging

The aggregator displays source diversity in the CLI:

```
>>> STEP 1: NewsAggregator
    Fetching global news...
    Fetched 45 articles (RSS: 38, Search: 7)
    Regions: australia, europe, india, japan, korea, latam, uk, us
    Languages: en, es, pt
    Query batch: 2
```

#### Key Files

| File | Purpose |
|------|---------|
| `config/news_sources.yaml` | All feeds, queries, rotation config |
| `news/sources/registry.py` | Loads YAML, provides query methods |
| `news/sources/feed_rotator.py` | Time-weighted feed selection |
| `news/sources/query_rotator.py` | Round-robin query batch rotation |
| `news/aggregator.py` | `fetch_with_rotation()` method |
| `utils/time_normalizer.py` | UTC time conversion for content |
| `data/query_rotation_state.json` | Persisted rotation state |

## Troubleshooting

### "No providers available"
- Check that at least one API key is set in `.env`
- Verify the key is valid and has credits
- Check `config/providers.yaml` has providers enabled

### "Failed to parse AI response"
- The AI returned invalid JSON (happens occasionally)
- The system will automatically retry with a simpler prompt
- If it persists, try a different text provider

### Images look stretched
- Ensure you're using the latest version
- Images should be generated in 1080x1080 square format
- Check `config/providers.yaml` doesn't have hardcoded sizes

### Placeholder content ("Point 1", "Content to be generated")
- JSON parsing failed and fallback content was used
- Usually resolved by automatic retry
- Try running the command again

### Debugging LMStudio (Local AI)

When using LMStudio for local AI, check these log locations:

| Log Location | Purpose |
|--------------|---------|
| `logs/ai_calls.log` | Application AI request/response logs |
| `C:\Users\rbgnr\.lmstudio\server-logs` | **LMStudio server logs** - detailed inference logs, GPU usage, token stats |

The LMStudio server logs show:
- Request/response JSON payloads
- Model loading and GPU memory allocation
- Token processing speed (tokens/second)
- Sampling parameters and generation stats

## Cost Estimation

| Provider | Type | Cost per Post (6 slides) |
|----------|------|--------------------------|
| **LM Studio + ComfyUI** | **Local AI** | **$0.00 (FREE)** |
| Z.AI + fal.ai | Cloud API | ~$0.02 |
| Groq + fal.ai | Cloud API | ~$0.01 (Groq free) |
| OpenAI (GPT-4o + DALL-E) | Cloud API | ~$0.15 |

## Known Limitations

Features that are **NOT included** in this project:

| Feature | Status | Notes |
|---------|--------|-------|
| **Video/Reels** | **Included!** | Use `generate-reel` command to generate video reels |
| **Stories** | Not included | Instagram Stories not supported |
| **Scheduled posting** | Not included | No built-in scheduler (use cron/Task Scheduler with `--loop-each`) |
| **Multiple platforms** | Partial | Instagram + TikTok for reels (no Twitter/X, LinkedIn) |
| **Analytics** | Not included | No engagement tracking or analytics |
| **Comment management** | Not included | No auto-replies or comment moderation |
| **Direct messages** | Not included | No DM automation |
| **User interactions** | Not included | No auto-follow/like/comment on other posts |

**Content Limitations:**
- Video reels use stock footage from Pexels (no AI video generation)
- English content only (other languages not tested)
- Carousel format only for images (no single-image posts)
- Text overlays use fixed templates (customization requires code changes)

**Technical Limitations:**
- Instagram Business/Creator account required for posting
- Facebook Page connection required
- Cloudinary account required for image hosting
- Local AI (LM Studio/ComfyUI) requires significant hardware resources

## Roadmap

- [x] Instagram API integration (`upload-post` command)
- [x] Batch posting - post all queued content
- [x] Queue management (`queue` command)
- [x] AI tool calling - AI decides when to search (`--ai-tools`)
- [x] Loop mode for continuous generation (`--loop-each`)
- [x] Provider override flags (`--text-ai`, `--image-ai`)
- [x] Auto-retry for robust generation (`--auto-retry`)
- [x] 100% Local AI support - LM Studio + ComfyUI (FREE)
- [x] Video/Reels generation (`generate-reel` command with stock footage)
- [x] Instagram Reels posting (`upload-reel` command)
- [ ] Scheduled posting (`--schedule "2025-12-11 10:00"`)
- [x] TikTok reel uploads (`upload-reel --tiktok`)
- [ ] Multiple social platforms (Twitter/X, LinkedIn)
- [ ] A/B testing for hooks

## License

MIT License - see LICENSE file for details.
