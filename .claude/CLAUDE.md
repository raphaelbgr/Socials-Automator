# Claude Code Instructions for Socials-Automator

## CRITICAL: Instagram Posting

**ALWAYS use `--dry-run` when testing the `post` command!**

```bash
# CORRECT - for testing
python -m socials_automator.cli post ai.for.mortals --dry-run

# DANGEROUS - only run when user explicitly wants to post
python -m socials_automator.cli post ai.for.mortals
```

Never run the post command without `--dry-run` unless the user explicitly asks to publish to Instagram.

Failed API calls can still result in posts being published. Rate limit errors do NOT mean the post wasn't created.

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
