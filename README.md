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

```bash
# Step 1: Generate a video reel
python -m socials_automator.cli generate-reel ai.for.mortals
# -> Creates reel in: reels/2025/12/generated/

# Step 2: Post to Instagram Reels
python -m socials_automator.cli upload-reel ai.for.mortals
# -> Auto-moves to pending-post, publishes, then moves to: reels/2025/12/posted/
```

**Or generate multiple and batch post:**

```bash
# Generate 10 reels then stop
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
python -m socials_automator.cli generate-post--help
```

### Main Commands

| Command | Description |
|---------|-------------|
| `generate-post` | Generate carousel posts for a profile |
| `generate-reel` | Generate video reels for Instagram/TikTok |
| `upload-post` | Upload pending carousel posts to Instagram |
| `upload-reel` | Upload pending video reels to Instagram |
| `queue` | List all posts in the publishing queue |
| `schedule` | Move generated posts to pending queue |
| `fix-thumbnails` | Generate missing thumbnails for existing reels |
| `update-artifacts` | Update artifact metadata for existing reels |
| `token` | Manage Instagram access tokens |
| `new-profile` | Create a new profile interactively |
| `list-profiles` | List all available profiles |
| `list-niches` | List available niches from niches.json |
| `status` | Show profile status and recent posts |
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
| `--subtitle-size` | `-s` | Subtitle font size in pixels | 80 |
| `--font` | | Subtitle font from /fonts folder | Montserrat-Bold.ttf |
| `--length` | `-l` | Target video length (e.g., 30s, 1m, 90s) | 1m |
| `--output` | `-o` | Output directory | Auto |
| `--dry-run` | | Only run first few steps without full video generation | False |
| `--loop` | `-L` | Loop continuously until stopped (Ctrl+C) | False |
| `--loop-count` | `-n` | Generate exactly N videos then stop (implies --loop) | None |
| `--gpu-accelerate` | `-g` | Enable GPU acceleration with NVENC (requires NVIDIA GPU) | False |
| `--gpu` | | GPU index to use (0, 1, etc.). Auto-selects if not specified | Auto |

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
```

**Pipeline:**
1. [AI] Select topic from profile content pillars
2. Research topic via web search
3. [AI] Plan video script (targeting --length duration)
4. Generate voiceover (determines actual video duration)
5. **[AI] Validate duration** - regenerates script if too long (up to 10 retries)
6. Search Pexels for stock footage (with local cache)
7. Download video clips
8. Assemble into 9:16 vertical video (matches narration length)
9. Add karaoke-style subtitles with moving watermark
10. [AI] Generate caption and hashtags (with AI validation and retry)
11. Output final.mp4

**Duration Validation:**
- The AI generates scripts targeting your `--length` duration
- After voice generation, actual audio duration is checked
- If duration exceeds 1.5x target (e.g., >90s for 1m target), script is regenerated
- AI receives feedback to write shorter content
- Up to 10 regeneration attempts to hit target duration

**GPU Acceleration (`-g`):**
- Uses NVIDIA NVENC for hardware-accelerated video encoding
- Significantly faster than CPU rendering (especially for longer videos)
- Requires NVIDIA GPU with NVENC support (GTX 600+, most modern GPUs)
- Falls back to CPU if GPU unavailable

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

Upload pending video reels to Instagram. **By default, posts ALL pending reels** in chronological order. Requires Instagram API and Cloudinary credentials (see [Instagram Posting Setup](#instagram-posting-setup)).

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

**Examples:**
```bash
# Post ALL pending reels (default behavior)
python -m socials_automator.cli upload-reel ai.for.mortals

# Post only the oldest pending reel
python -m socials_automator.cli upload-reel ai.for.mortals --one
python -m socials_automator.cli upload-reel ai.for.mortals -1

# Post a specific reel by ID
python -m socials_automator.cli upload-reel ai.for.mortals 16-001

# Validate without posting
python -m socials_automator.cli upload-reel ai.for.mortals --dry-run
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

**Video Requirements:**
- Format: MP4 (H.264 codec recommended)
- Duration: 15-90 seconds via API
- Aspect Ratio: 9:16 (portrait/vertical)
- Resolution: 1080x1920 recommended

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

### Example .env File

```bash
# Text Providers (pick at least one)
ZAI_API_KEY=your-zai-key
ZAI_API_URL=https://api.z.ai/v1
GROQ_API_KEY=gsk_your-groq-key
GOOGLE_API_KEY=your-google-key
OPENAI_API_KEY=sk-your-openai-key

# Image Providers (pick at least one)
FAL_API_KEY=your-fal-key
REPLICATE_API_TOKEN=r8_your-replicate-token
# OPENAI_API_KEY is also used for DALL-E 3
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

### Known Meta API Quirks

**Ghost Publish Issue:** Meta's API can return a rate limit error AFTER successfully publishing your post. The system automatically detects this by checking your recent Instagram posts. If you see a rate limit error, don't panic - check Instagram first before retrying.

**Rate Limit Error Codes:**
- Error 4: Application daily limit reached
- Error 9: Application-level throttling (wait 5+ minutes)
- Error 17: User-level rate limit

The system handles these automatically with exponential backoff and ghost publish detection.

## Project Structure

```
Socials-Automator/
├── config/
│   └── providers.yaml      # AI provider configuration
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
│   ├── cli.py              # Command-line interface
│   ├── cli_display.py      # Rich CLI progress displays
│   ├── content/            # Content generation
│   │   ├── generator.py    # Main generator
│   │   ├── planner.py      # Content planning
│   │   └── models.py       # Data models
│   ├── design/             # Slide design
│   │   ├── composer.py     # Image composition
│   │   └── templates.py    # Slide templates
│   ├── providers/          # AI providers
│   │   ├── text.py         # Text generation (with tool calling)
│   │   ├── image.py        # Image generation
│   │   └── config.py       # Provider config
│   ├── instagram/          # Instagram posting
│   │   ├── client.py       # Instagram Graph API client
│   │   ├── uploader.py     # Cloudinary image uploader
│   │   └── models.py       # Instagram data models
│   ├── tools/              # AI tool calling
│   │   ├── definitions.py  # Tool schemas (web_search, news_search)
│   │   └── executor.py     # Tool execution
│   ├── research/           # Web research
│   │   └── web_search.py   # DuckDuckGo parallel search
│   └── knowledge/          # Knowledge base
├── .env.example            # Example environment file
├── pyproject.toml          # Python package config
└── README.md
```

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
| **Multiple platforms** | Not included | Instagram only (no Twitter/X, LinkedIn, TikTok) |
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
- [ ] Multiple social platforms (Twitter/X, LinkedIn, TikTok)
- [ ] A/B testing for hooks

## License

MIT License - see LICENSE file for details.
