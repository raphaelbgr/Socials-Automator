# Claude Code Instructions for Socials-Automator

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
