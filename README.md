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

### 3. Generate Your First Post

```bash
# Generate a post with a specific topic
python -m socials_automator.cli generate ai.for.mortals --topic "5 ChatGPT tricks for productivity"

# Let AI choose the topic automatically
python -m socials_automator.cli generate ai.for.mortals
```

## Usage

### Commands

```bash
# List all available profiles
python -m socials_automator.cli list-profiles

# Generate posts
python -m socials_automator.cli generate <profile> [options]

# Create a new profile
python -m socials_automator.cli new-profile

# Show profile status and recent posts
python -m socials_automator.cli status <profile>

# List available niches
python -m socials_automator.cli list-niches
```

### Generate Command Options

```bash
python -m socials_automator.cli generate <profile> [OPTIONS]

Options:
  -t, --topic TEXT       Topic for the post (optional - AI will generate if not provided)
  -p, --pillar TEXT      Content pillar (e.g., tool_tutorials, productivity_hacks)
  -n, --count INTEGER    Number of posts to generate [default: 1]
  -s, --slides INTEGER   Force specific slide count (default: AI decides)
  --min-slides INTEGER   Minimum slides when AI decides [default: 3]
  --max-slides INTEGER   Maximum slides when AI decides [default: 10]
```

### Examples

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

### Z.AI (Recommended for Text - Cheap)
1. Go to [z.ai](https://z.ai)
2. Create an account and get API key
3. Add to `.env`:
   ```
   ZAI_API_KEY=your-key
   ZAI_API_URL=https://api.z.ai/v1
   ```

### Groq (Free Tier)
1. Go to [console.groq.com](https://console.groq.com)
2. Create account and generate API key
3. Add to `.env`:
   ```
   GROQ_API_KEY=gsk_...
   ```

### OpenAI (Text + Images)
1. Go to [platform.openai.com](https://platform.openai.com)
2. Create API key
3. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

### fal.ai (Cheap Images)
1. Go to [fal.ai](https://fal.ai)
2. Create account and get API key
3. Add to `.env`:
   ```
   FAL_API_KEY=...
   ```

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
