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

**ALWAYS use `--dry-run` when testing the `post` command!**

```bash
# CORRECT - for testing
python -m socials_automator.cli post ai.for.mortals --dry-run

# DANGEROUS - only run when user explicitly wants to post
python -m socials_automator.cli post ai.for.mortals
```

Never run the post command without `--dry-run` unless the user explicitly asks to publish to Instagram.

### WARNING: Meta API Rate Limit Bug

**Rate limit errors do NOT mean the post wasn't published!**

Meta's API can:
1. Create all containers successfully
2. Process the carousel
3. Actually publish the post
4. THEN return a rate limit error

This is a known Meta API issue - the publish happens but the response fails. Always check Instagram manually after a "failed" publish before retrying.

## Commands Overview

- `generate` - Creates carousel content (safe to run)
- `schedule` - Moves posts to pending-post queue (safe to run)
- `post` - Publishes to Instagram (USE --dry-run FOR TESTING)
- `post --dry-run` - Validates without posting (SAFE)

## Folder Structure

```
profiles/<name>/posts/YYYY/MM/
  generated/     - New content from generate command
  pending-post/  - Queued for Instagram (from schedule command)
  posted/        - Published to Instagram (from post command)
```
