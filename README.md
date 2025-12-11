# Socials Automator

AI-powered Instagram carousel content generator. Automatically creates professional carousel posts with AI-generated content and images.

## Features

- **Multi-provider AI text generation** - Z.AI, OpenAI, Groq, Gemini with automatic fallback
- **Multi-provider image generation** - DALL-E, fal.ai Flux, Replicate SDXL
- **Smart slide count** - AI decides optimal number of slides (3-10) based on topic
- **Square format output** - 1080x1080 Instagram-optimized images
- **Auto topic generation** - AI generates fresh topics based on your niche
- **Post history awareness** - Avoids repeating recent topics
- **Profile-based config** - Manage multiple Instagram accounts

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
python -m socials_automator.cli generate your-profile-name --topic "5 ChatGPT tricks for productivity"

# Let AI choose the topic automatically
python -m socials_automator.cli generate your-profile-name
```

## CLI Reference

Get help for any command with `--help`:

```bash
python -m socials_automator.cli --help
python -m socials_automator.cli generate --help
```

### Main Commands

| Command | Description |
|---------|-------------|
| `generate` | Generate carousel posts for a profile |
| `new-profile` | Create a new profile interactively |
| `list-profiles` | List all available profiles |
| `list-niches` | List available niches from niches.json |
| `status` | Show profile status and recent posts |
| `init` | Initialize project structure |

---

### generate

Generate carousel posts for a profile. By default, the AI decides the optimal number of slides (3-10) based on the topic content.

```bash
python -m socials_automator.cli generate <profile> [OPTIONS]
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

**Examples:**
```bash
# Generate 1 post with AI-chosen topic
python -m socials_automator.cli generate ai.for.mortals

# Generate 3 posts with AI-chosen topics
python -m socials_automator.cli generate ai.for.mortals -n 3

# Generate post with specific topic
python -m socials_automator.cli generate ai.for.mortals -t "How to use ChatGPT for email"

# Generate post with exactly 5 slides
python -m socials_automator.cli generate ai.for.mortals -t "AI tools for writers" -s 5

# Generate post with 4-8 slides (AI decides within range)
python -m socials_automator.cli generate ai.for.mortals --min-slides 4 --max-slides 8
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
├── src/socials_automator/
│   ├── cli.py              # Command-line interface
│   ├── content/            # Content generation
│   │   ├── generator.py    # Main generator
│   │   ├── planner.py      # Content planning
│   │   └── models.py       # Data models
│   ├── design/             # Slide design
│   │   ├── composer.py     # Image composition
│   │   └── templates.py    # Slide templates
│   ├── providers/          # AI providers
│   │   ├── text.py         # Text generation
│   │   ├── image.py        # Image generation
│   │   └── config.py       # Provider config
│   ├── knowledge/          # Knowledge base
│   └── research/           # Topic research
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
| Z.AI + fal.ai | Text + Image | ~$0.02 |
| Groq + fal.ai | Text + Image | ~$0.01 (Groq free) |
| OpenAI (GPT-4o + DALL-E) | Text + Image | ~$0.15 |

## Roadmap

- [ ] Instagram API integration (`--post-to-instagram`)
- [ ] Scheduled posting (`--schedule "2025-12-11 10:00"`)
- [ ] Multiple social platforms (Twitter/X, LinkedIn)
- [ ] Video/Reels generation
- [ ] A/B testing for hooks

## License

MIT License - see LICENSE file for details.
