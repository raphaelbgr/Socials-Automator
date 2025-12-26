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
```

**Available Voices:**
| Voice | Description |
|-------|-------------|
| `rvc_adam` | Viral TikTok voice - FREE, local (default) |
| `edge_*` | Microsoft Edge TTS (free) |
| `elevenlabs_*` | ElevenLabs (paid, high quality) |

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
