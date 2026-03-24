# Socials Automator

100% automated Instagram/TikTok content generator. Creates professional carousel posts and video reels with AI-generated text and images.

**Live Demo:** [@ai.for.mortals](https://www.instagram.com/ai.for.mortals/) - every post generated 100% automatically.

## Run 100% FREE with Local AI

```bash
# Generate and post with local AI - completely FREE
python -m socials_automator.cli generate-post ai.for.mortals --text-ai lmstudio --image-ai comfyui --upload

# Video reels with local AI
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai lmstudio -g --upload
```

- **[LM Studio](https://lmstudio.ai/)** - Local language models (Llama, Mistral, Qwen)
- **[ComfyUI](https://www.comfy.org/)** - Local Stable Diffusion image generation

## Features

| Feature | Description |
|---------|-------------|
| Local AI | LM Studio + ComfyUI - completely FREE |
| Cloud AI | Z.AI, OpenAI, Groq, Gemini, fal.ai, DALL-E with fallback |
| Carousel Posts | 1080x1080 Instagram-optimized, AI-driven slide count (3-10) |
| Video Reels | Stock footage + TTS voiceover + karaoke subtitles |
| News Briefings | Auto-aggregate from RSS + web search |
| TikTok Upload | Browser automation with rate limiting |
| Loop Mode | Continuous generation at intervals |
| Multi-profile | Manage multiple accounts |

## Quick Start

```bash
# 1. Install
git clone https://github.com/yourusername/Socials-Automator.git
cd Socials-Automator
pip install -r requirements.txt

# 2. Configure .env (see API Keys Setup section)

# 3. Create profile
python -m socials_automator.cli new-profile ai.for.mortals --handle @ai.for.mortals

# 4. Generate content
python -m socials_automator.cli generate-post ai.for.mortals
python -m socials_automator.cli generate-reel ai.for.mortals
```

## CLI Reference

### Commands Overview

| Command | Description |
|---------|-------------|
| `generate-post` | Generate carousel posts |
| `generate-reel` | Generate video reels |
| `upload-post` | Upload posts to Instagram |
| `upload-reel` | Upload reels to Instagram/TikTok |
| `upload-tiktok-browser` | Upload to TikTok via browser (with rate limiting) |
| `queue` | View pending queue |
| `schedule` | Move generated to pending |
| `token` | Manage Instagram tokens |
| `new-profile` | Create new profile |
| `list-profiles` | List profiles |
| `status` | Show profile status |
| `cleanup-reels` | Remove video files to free space |
| `sync-captions` | Sync Instagram captions to local |
| `fix-thumbnails` | Generate missing thumbnails |

---

### generate-post

Generate carousel posts for Instagram.

```bash
python -m socials_automator.cli generate-post <profile> [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--topic` | `-t` | Specific topic | AI generates |
| `--slides` | `-s` | Number of slides | AI decides (3-10) |
| `--text-ai` | | Text provider (lmstudio, zai, openai, groq) | Config priority |
| `--model` | `-m` | Model override (see Model Selection) | Provider default |
| `--image-ai` | | Image provider (comfyui, dalle, fal) | Config priority |
| `--upload` | | Upload after generating | False |
| `--loop-each` | | Loop interval (5m, 1h) | - |
| `-n` | | Number of posts to generate | 1 |
| `--dry-run` | | Preview without generating | False |
| `--ai-tools` | | Enable AI web research | False |

```bash
# Examples
python -m socials_automator.cli generate-post ai.for.mortals
python -m socials_automator.cli generate-post ai.for.mortals --topic "5 AI tools" --upload
python -m socials_automator.cli generate-post ai.for.mortals -n 5 --text-ai lmstudio
python -m socials_automator.cli generate-post ai.for.mortals --loop-each 30m --upload
```

---

### generate-reel

Generate video reels with AI script, stock footage, TTS voiceover, and karaoke subtitles.

```bash
python -m socials_automator.cli generate-reel <profile> [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--topic` | `-t` | Specific topic | AI generates |
| `--length` | `-l` | Duration (30s, 1m, 1m30s) | 1m |
| `--voice` | `-v` | TTS voice | rvc_adam |
| `--text-ai` | | Text provider | Config priority |
| `--model` | `-m` | Model override (see Model Selection) | Provider default |
| `--video-matcher` | | Video source (pexels) | pexels |
| `--upload` | | Upload after generating | False |
| `--loop-count` | `-n` | Number of reels | 1 |
| `--loop-each` | | Interval between loops | 3s |
| `-g` | | GPU acceleration | False |
| `--hashtags` | `-H` | Max hashtags | 5 |

**News Reels** (auto-detected via `news_sources` in profile):
| Option | Description | Default |
|--------|-------------|---------|
| `--edition` | morning, midday, evening, night | Auto |
| `--stories` | Number of news stories | auto |
| `--news-age` | Max article age in hours | 24 |

**Image Overlays:**
| Option | Description | Default |
|--------|-------------|---------|
| `--overlay-images` | Add contextual images | False |
| `--image-provider` | websearch, pexels, pixabay | websearch |
| `--blur` | Dim background: light, medium, heavy | - |
| `--smart-pick` | AI vision to select best images | False |

```bash
# Examples
python -m socials_automator.cli generate-reel ai.for.mortals --upload
python -m socials_automator.cli generate-reel ai.for.mortals --topic "AI tips" --length 30s
python -m socials_automator.cli generate-reel ai.for.mortals -n 10 -g --upload
python -m socials_automator.cli generate-reel news.but.quick --edition morning --stories 5
python -m socials_automator.cli generate-reel ai.for.mortals --overlay-images --blur medium

# Use specific model within a provider
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai zai --model glm-4.7
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai groq --model llama-3.1-8b
```

**Available Voices:**
| Voice | Description |
|-------|-------------|
| `rvc_adam` | Viral TikTok voice - FREE, local (default) |
| `edge_*` | Microsoft Edge TTS (free) |
| `elevenlabs_*` | ElevenLabs (paid, high quality) |

---

### generate-reel-v2

**Cinematic Script Generation** - Enhanced video reels with multi-hook scoring and pattern breaks.

```bash
python -m socials_automator.cli generate-reel-v2 <profile> [OPTIONS]
```

**V2 Features:**
- **Multi-hook generation** - Generates 5-8 hook candidates with different angles
- **6-metric scoring** - Scores each hook on: curiosity, clarity, specificity, credibility, retention, shareability
- **Pattern breaks** - Planned every 2-4s (punch_in, hard_cut, zoom_keyword, whip_pan)
- **Timed beats** - Hook → Promise → Payoff → Steps → Recap → CTA structure
- **Shot hints** - Enhanced Pexels keywords with shot type guidance
- **Role-based AI** - Separate AI for creative vs high-volume tasks
- **Batched video selection** - Single AI call for all segments (faster)
- **Token usage tracking** - Per-call and summary token/cost display

| Option | Description | Default |
|--------|-------------|---------|
| `--hooks` | Number of hook candidates (2-10) | 5 |
| `--show-scores/--hide-scores` | Display hook score table | show |
| `--smart-video` | Vision AI to select best Pexels videos | False |
| `--smart-video-provider` | Vision AI provider (lmstudio, openai, zai) | lmstudio |
| `--smart-video-model` | Vision model override (e.g., glm-4.6v-flash) | Provider default |
| `--model` / `-m` | Text AI model override (see Model Selection) | Provider default |

**Role-Based AI Configuration:**

Different pipeline steps need different AI capabilities. Use these flags to optimize cost and speed:

| Option | Description | Default |
|--------|-------------|---------|
| `--smart-ai` | AI for creative tasks (topic, script, caption) | from --text-ai |
| `--smart-model` | Model override for smart AI | Provider default |
| `--fast-ai` | AI for high-volume tasks (video search, keywords) | lmstudio |
| `--fast-model` | Model override for fast AI | Provider default |

**Pipeline Step AI Routing:**

| Step | AI Used | Why |
|------|---------|-----|
| TopicSelector | Smart | Creative topic generation |
| TopicResearcher | Smart | Web search synthesis |
| ScriptPlannerV2 | Smart | Creative script with hooks |
| CaptionGenerator | Smart | Engaging captions |
| VideoSearcher (keywords) | Fast | High-volume, simple task |
| VideoSearcher (selection) | Fast | Batched video selection |
| ImageOverlayPlanner | Smart | Creative image planning |

**CLI Output Shows AI Configuration:**
```
AI PROVIDERS
  Smart:      zai / glm-4.7 (topic, script, caption)
  Fast:       lmstudio (video search, keywords)
  Vision:     lmstudio (video analysis)
```

**Token Usage Display:**
```
>>> AI CALLS
  [AI] lmstudio/local-model (script_planning)... [OK] 2.3s | in:1250 out:890
  [AI] zai/glm-4.7 (caption)... [OK] 1.5s | in:450 out:120

>>> AI SUMMARY
  Providers: lmstudio, zai
  Calls: 4
  Total AI time: 12.5s
  Tokens: in:5.2K out:2.1K (7.3K total)
  Est. cost: $0.0012
```

**Smart Video Selection:** Analyzes Pexels video frames with vision AI to select the best match for your topic. Uses a global cache at `data/pexels/cache/` shared across profiles.

All other options from `generate-reel` are supported (--text-ai, --length, --upload, --loop-each, etc.)

```bash
# Examples - Basic
python -m socials_automator.cli generate-reel-v2 ai.for.mortals --hooks 8
python -m socials_automator.cli generate-reel-v2 ai.for.mortals -g --upload --loop-each 30m

# Role-based AI - Use expensive model for creative, cheap for volume
python -m socials_automator.cli generate-reel-v2 ai.for.mortals \
  --smart-ai zai --smart-model glm-4.7 \
  --fast-ai lmstudio

# All local (100% FREE)
python -m socials_automator.cli generate-reel-v2 ai.for.mortals \
  --smart-ai lmstudio --fast-ai lmstudio -g

# With smart video selection (vision AI picks best Pexels videos)
python -m socials_automator.cli generate-reel-v2 ai.for.mortals --smart-video -g --upload

# Production setup: Z.AI for creative, local for volume
python -m socials_automator.cli generate-reel-v2 ai.for.mortals \
  --text-ai zai --model glm-4.7 \
  --fast-ai lmstudio \
  --smart-video --smart-video-provider lmstudio \
  -g --upload -n 10 --loop-each 5m
```

**Note:** V2 is designed for educational/tips content (ai.for.mortals style). For news profiles, use `generate-reel` which has the NewsOrchestrator pipeline.

---

### upload-post

Upload carousel posts to Instagram.

```bash
python -m socials_automator.cli upload-post <profile> [post-id] [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--one` | `-1` | Upload only oldest pending | False |
| `--dry-run` | | Validate without posting | False |

```bash
python -m socials_automator.cli upload-post ai.for.mortals          # All pending
python -m socials_automator.cli upload-post ai.for.mortals --one    # Just one
python -m socials_automator.cli upload-post ai.for.mortals 16-001   # Specific post
```

---

### upload-reel

Upload video reels to Instagram and/or TikTok.

```bash
python -m socials_automator.cli upload-reel <profile> [reel-id] [OPTIONS]
```

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--one` | `-1` | Upload only oldest pending | False |
| `--instagram` | `-I` | Instagram only | True |
| `--tiktok` | `-T` | TikTok only | False |
| `--all` | `-a` | All platforms | False |
| `--dry-run` | | Validate without posting | False |

```bash
python -m socials_automator.cli upload-reel ai.for.mortals          # All to Instagram
python -m socials_automator.cli upload-reel ai.for.mortals --one    # Just one
python -m socials_automator.cli upload-reel ai.for.mortals --tiktok # TikTok only
python -m socials_automator.cli upload-reel ai.for.mortals --all    # All platforms
```

---

### upload-tiktok-browser

Upload reels to TikTok via browser automation with **rate limiting** to avoid shadowbans.

```bash
python -m socials_automator.cli upload-tiktok-browser --source <profile> [OPTIONS]
```

**TikTok Shadowban Risk:**
- **Safe**: 1-3 videos/day, 2+ hours apart
- **Risky**: 6-10 videos/day
- **Shadowban**: 10+ videos/day

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--source` | `-s` | Source profile (required) | - |
| `--posted-only` | `-p` | Only from posted/ (cross-posting) | False |
| `--one` | `-1` | Upload only one | False |
| `--dry-run` | | Simulate | False |

**Rate Limiting:**
| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--sort` | | `most-recent` or `oldest` | most-recent |
| `--daily-limit` | `-L` | Max uploads per day | 3 |
| `--timing` | | `random`, `spaced`, `immediate` | random |
| `--time-window` | `-w` | Hours to post (e.g., 8-23) | 8-23 |
| `--min-gap` | `-g` | Min time between uploads | 2h |

```bash
# Cross-post Instagram reels to TikTok (3/day, random timing)
python -m socials_automator.cli upload-tiktok-browser --source ai.for.mortals --posted-only

# Custom rate limiting
python -m socials_automator.cli upload-tiktok-browser --source ai.for.mortals --daily-limit 5 --time-window "9-21"

# Bypass rate limiting (testing only)
python -m socials_automator.cli upload-tiktok-browser --source ai.for.mortals --timing immediate
```

**First Time Setup:**
1. `pip install selenium`
2. Start Chrome with remote debugging:
   ```bash
   # Windows
   "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9333 --user-data-dir="%USERPROFILE%\TikTokProfile"
   ```
3. Log in to TikTok in that Chrome window
4. Run the upload command

---

### queue / schedule

```bash
# View pending queue
python -m socials_automator.cli queue ai.for.mortals

# Move generated to pending
python -m socials_automator.cli schedule ai.for.mortals
```

---

### token

```bash
python -m socials_automator.cli token --check    # Check validity
python -m socials_automator.cli token --refresh  # Refresh token
```

---

### Other Commands

```bash
# Profile management
python -m socials_automator.cli new-profile my.profile --handle @my.profile
python -m socials_automator.cli list-profiles
python -m socials_automator.cli status ai.for.mortals

# Maintenance
python -m socials_automator.cli fix-thumbnails ai.for.mortals
python -m socials_automator.cli cleanup-reels ai.for.mortals --older-than 7
python -m socials_automator.cli sync-captions ai.for.mortals
```

---

## Folder Structure

```
profiles/<profile>/
├── posts/YYYY/MM/
│   ├── generated/      <- New posts
│   ├── pending-post/   <- Scheduled
│   └── posted/         <- Published
├── reels/YYYY/MM/
│   ├── generated/      <- New reels
│   ├── pending-post/   <- Scheduled
│   └── posted/         <- Published
└── metadata.json       <- Profile config
```

---

## Configuration

### Provider Priority (config/providers.yaml)

```yaml
text_providers:
  - name: lmstudio
    priority: 1
    base_url: http://localhost:1234/v1
  - name: zai
    priority: 2
    api_key: ENV:ZAI_API_KEY

image_providers:
  - name: comfyui
    priority: 1
    base_url: http://localhost:8188
  - name: fal
    priority: 2
    api_key: ENV:FAL_API_KEY
```

### Profile Configuration (profiles/*/metadata.json)

```json
{
  "handle": "@ai.for.mortals",
  "niches": ["AI Tools", "Productivity"],
  "platforms": {
    "instagram": {
      "enabled": true,
      "user_id": "ENV:INSTAGRAM_USER_ID",
      "access_token": "ENV:INSTAGRAM_ACCESS_TOKEN"
    }
  }
}
```

---

## API Keys Setup

### Minimum Setup (Text + Image)

```bash
# Option A: 100% Free (Local)
# Just install LM Studio + ComfyUI, no API keys needed

# Option B: Cloud APIs
ZAI_API_KEY=your_key              # z.ai - cheap, fast
FAL_API_KEY=your_key              # fal.ai - image generation
```

### All Providers

**Text:**
| Provider | Env Variable | Notes |
|----------|--------------|-------|
| LM Studio | - | Local, FREE |
| Z.AI | `ZAI_API_KEY` | Cheap, fast |
| OpenAI | `OPENAI_API_KEY` | GPT-4 |
| Groq | `GROQ_API_KEY` | Fast, free tier |
| Gemini | `GEMINI_API_KEY` | Google |

**Image:**
| Provider | Env Variable | Notes |
|----------|--------------|-------|
| ComfyUI | - | Local, FREE |
| fal.ai | `FAL_API_KEY` | Fast, cheap |
| DALL-E | `OPENAI_API_KEY` | OpenAI |

**Stock Video/Images:**
| Provider | Env Variable | Notes |
|----------|--------------|-------|
| Pexels | `PEXELS_API_KEY` | FREE, required for reels |
| Pixabay | `PIXABAY_API_KEY` | FREE, optional |

**Instagram Posting:**
```bash
INSTAGRAM_USER_ID=17841...
INSTAGRAM_ACCESS_TOKEN=EAA...
CLOUDINARY_CLOUD_NAME=your_cloud
CLOUDINARY_API_KEY=your_key
CLOUDINARY_API_SECRET=your_secret
```

---

## Instagram Posting Setup

1. **Instagram Business Account** - Convert personal to business via Facebook Page
2. **Facebook App** - Create at [developers.facebook.com](https://developers.facebook.com)
3. **Access Token** - Generate via Graph API Explorer with permissions:
   - `instagram_basic`, `instagram_content_publish`, `pages_read_engagement`
4. **Instagram User ID** - Get from [commentpicker.com/instagram-user-id.php](https://commentpicker.com/instagram-user-id.php)
5. **Cloudinary** - Free account at [cloudinary.com](https://cloudinary.com)
6. **Add to .env** - See API Keys Setup above

**Test:** `python -m socials_automator.cli upload-post ai.for.mortals --dry-run`

---

## Local AI Setup

### LM Studio (Text)

1. Download from [lmstudio.ai](https://lmstudio.ai)
2. Load a model (Llama 3.1, Mistral, Qwen recommended)
3. Start local server (default: http://localhost:1234)
4. Use: `--text-ai lmstudio`

### ComfyUI (Images)

1. Install from [comfy.org](https://www.comfy.org)
2. Download SDXL checkpoint
3. Start server (default: http://localhost:8188)
4. Use: `--image-ai comfyui`

---

## Model Selection

Use `--model` / `-m` to select a specific model within a provider. Models are configured in `config/providers.yaml`.

```bash
# Use Z.AI with GLM-4.7 (flagship model)
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai zai --model glm-4.7

# Use Groq with faster Llama 3.1 8B
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai groq --model llama-3.1-8b

# Use Gemini with Pro model
python -m socials_automator.cli generate-reel ai.for.mortals --text-ai gemini --model gemini-1.5-pro
```

### Available Models

**Z.AI** (`--text-ai zai`):
| Model | Flag | Price | Notes |
|-------|------|-------|-------|
| GLM-4.5-Air | `--model glm-4.5-air` | $0.20/$1.10 | Default, cheap |
| GLM-4.7 | `--model glm-4.7` | $0.60/$2.20 | Latest flagship |
| GLM-4.5-Flash | `--model glm-4.5-flash` | FREE | Free tier |
| GLM-4.6V-Flash | `--model glm-4.6v-flash` | FREE | Vision, free |

**Groq** (`--text-ai groq`):
| Model | Flag | Notes |
|-------|------|-------|
| Llama 3.3 70B | `--model llama-3.3-70b` | Default |
| Llama 3.1 8B | `--model llama-3.1-8b` | Faster |
| Mixtral 8x7B | `--model mixtral-8x7b` | - |

**Gemini** (`--text-ai gemini`):
| Model | Flag | Notes |
|-------|------|-------|
| Gemini 2.0 Flash | `--model gemini-2.0-flash` | Default |
| Gemini 1.5 Pro | `--model gemini-1.5-pro` | More capable |
| Gemini 1.5 Flash | `--model gemini-1.5-flash` | Faster |

---

## Platform Status Tracking

Each reel tracks upload status per platform in `metadata.json`:

```json
{
  "platform_status": {
    "instagram": {
      "uploaded": true,
      "uploaded_at": "2025-12-26T10:30:00",
      "media_id": "17841234567890",
      "permalink": "https://www.instagram.com/reel/..."
    },
    "tiktok": {
      "uploaded": true,
      "video_url": "https://www.tiktok.com/@user/video/123",
      "video_id": "123"
    }
  }
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No providers available" | Check API keys in .env, verify provider is running |
| "Failed to parse AI response" | Try different model, check LM Studio logs |
| Rate limit errors | Wait and retry, use `--dry-run` to test |
| TikTok shadowban | Reduce to 3 videos/day, use `--timing random` |

---

## Cost Estimation

| Provider | Cost | Notes |
|----------|------|-------|
| LM Studio | FREE | Local, your hardware |
| ComfyUI | FREE | Local, your hardware |
| Z.AI | ~$0.001/post | Very cheap |
| Pexels | FREE | Stock video |
| OpenAI | ~$0.05/post | GPT-4 + DALL-E |

**100% FREE setup:** LM Studio + ComfyUI + Pexels

---

## License

MIT License - see LICENSE file.
