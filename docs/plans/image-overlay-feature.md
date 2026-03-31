# Image Overlay Feature Implementation Plan

## Overview

Add `--overlay-images` flag to `generate-reel` that overlays contextual images during narration. Images appear with pop animations, positioned above subtitles in a frosted glass container.

## Requirements

1. **AI-Driven Timing** - AI decides when images appear based on narration content
2. **Exact Match Priority** - For specific content (TV shows, products), must show real images
3. **Local Image Library** - Profile-local images with metadata (aliases, tags)
4. **Pexels Fallback** - For illustrative/generic content
5. **Skip on No Match** - Don't show image if no good match found
6. **Pop Animation** - Bouncy scale animation (in/out)
7. **Frosted Glass Container** - Blur backdrop, but subtitles remain sharp

---

## Architecture

### Data Flow

```
ScriptPlanner (existing)
    |
    v
ImageOverlayPlanner (NEW) -----> AI generates image timing + queries
    |
    v
ImageResolver (NEW) -----------> Resolves local library vs Pexels
    |
    v
ImageDownloader (NEW) ---------> Downloads Pexels images to cache
    |
    v
VideoAssembler (existing)
    |
    v
ImageOverlayRenderer (NEW) ----> FFmpeg composites images with animations
    |
    v
SubtitleRenderer (existing) ---> Subtitles rendered AFTER (on top, sharp)
```

### New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `ImageOverlayPlanner` | `video/pipeline/image_overlay_planner.py` | AI plans image timing |
| `ImageResolver` | `video/pipeline/image_resolver.py` | Resolves local vs Pexels |
| `ImageDownloader` | `video/pipeline/image_downloader.py` | Downloads + caches images |
| `ImageOverlayRenderer` | `video/pipeline/image_overlay_renderer.py` | FFmpeg compositing |
| `PexelsImageClient` | `video/pipeline/pexels_image.py` | Pexels Image API |
| `ImageCache` | `video/pipeline/image_cache.py` | Image cache system |

---

## Data Models

### ImageOverlay (in `base.py`)

```python
class ImageOverlay(BaseModel):
    """Single image overlay with timing."""

    segment_index: int              # Which segment this belongs to
    start_time: float               # Seconds from video start
    end_time: float                 # Seconds from video start
    topic: str                      # What's being discussed (e.g., "Stranger Things")
    match_type: str                 # "exact" or "illustrative"

    # Resolution results (filled by ImageResolver)
    source: Optional[str] = None    # "local", "pexels", or None (skip)
    local_hint: Optional[str] = None  # Folder name in local library
    pexels_query: Optional[str] = None  # Fallback search query
    pexels_id: Optional[int] = None  # Resolved Pexels image ID
    image_path: Optional[Path] = None  # Final local path to image

    # Metadata
    alt_text: Optional[str] = None  # Accessibility text
    confidence: float = 0.0         # AI confidence this image is needed (0-1)


class ImageOverlayScript(BaseModel):
    """Collection of image overlays for a video."""

    overlays: list[ImageOverlay] = Field(default_factory=list)

    def get_overlays_at_time(self, time_seconds: float) -> list[ImageOverlay]:
        """Get all overlays active at a given time."""
        return [
            o for o in self.overlays
            if o.start_time <= time_seconds < o.end_time and o.image_path
        ]
```

### LocalImageMetadata (for profile assets)

```python
class LocalImageMetadata(BaseModel):
    """Metadata for a local image in profile assets."""

    aliases: list[str] = Field(default_factory=list)  # ["stranger things", "ST", "netflix stranger things"]
    tags: list[str] = Field(default_factory=list)     # ["netflix", "tv", "show", "horror"]
    source: Optional[str] = None   # "official", "screenshot", "fan-art"
    attribution: Optional[str] = None  # Credit if needed
```

### Extended PipelineContext

```python
# Add to PipelineContext in base.py:
class PipelineContext(BaseModel):
    # ... existing fields ...

    # Image overlay system
    image_overlays: Optional[ImageOverlayScript] = None
    overlay_images_enabled: bool = False
```

---

## Local Image Library Structure

```
profiles/{name}/assets/images/
    stranger-things/
        image.jpg           # The actual image (jpg, png, webp)
        metadata.json       # LocalImageMetadata
    the-witcher/
        image.png
        metadata.json
    chatgpt/
        image.png
        metadata.json
```

### Example `metadata.json`

```json
{
  "aliases": ["stranger things", "ST", "stranger things netflix", "upside down show"],
  "tags": ["netflix", "tv", "show", "sci-fi", "horror", "80s"],
  "source": "official",
  "attribution": "Netflix promotional material"
}
```

---

## Pexels Image Cache Structure

```
/pexels/image-cache/
    index.json              # Quick lookup by pexels_id
    12345678.jpg            # Images named by pexels_id
    87654321.jpg
    ...
```

### Cache Index Entry

```json
{
  "12345678": {
    "pexels_id": 12345678,
    "filename": "12345678.jpg",
    "width": 1920,
    "height": 1280,
    "photographer": "John Doe",
    "photographer_url": "https://pexels.com/@johndoe",
    "pexels_url": "https://pexels.com/photo/12345678",
    "alt": "Person working on laptop in coffee shop",
    "keywords_matched": ["coffee shop", "laptop", "working"],
    "cached_at": "2025-12-23T10:00:00",
    "last_used": "2025-12-23T10:00:00",
    "hit_count": 5
  }
}
```

---

## Component Details

### 1. ImageOverlayPlanner

**Purpose:** AI analyzes script and determines which segments need images.

**Input:** `VideoScript` with segments, timing, and full narration

**Output:** `ImageOverlayScript` with timing and queries

**AI Prompt Strategy:**

```
You are analyzing a video script to determine where images should appear.

SCRIPT:
{full_narration}

SEGMENTS WITH TIMING:
1. [0.0s - 5.2s] "Have you heard about these Netflix shows?"
2. [5.2s - 15.8s] "First, Stranger Things captivated millions with its 80s nostalgia..."
3. [15.8s - 26.4s] "Then The Witcher brought fantasy fans a new hero..."

TASK:
For each segment, determine if an image should appear.

IMAGE CRITERIA:
- EXACT MATCH: When discussing specific content (TV shows, movies, products, apps, people)
  - The image MUST be the actual thing being discussed
  - Example: Discussing "Stranger Things" -> show the Stranger Things poster

- ILLUSTRATIVE: When discussing concepts or abstract ideas
  - A relevant stock photo is acceptable
  - Example: Discussing "productivity tips" -> person working at desk

OUTPUT FORMAT (JSON):
{
  "overlays": [
    {
      "segment_index": 2,
      "start_time": 5.2,
      "end_time": 15.8,
      "topic": "Stranger Things",
      "match_type": "exact",
      "local_hint": "stranger-things",
      "pexels_query": "stranger things netflix poster",
      "confidence": 0.95,
      "alt_text": "Stranger Things TV show poster"
    }
  ]
}

RULES:
1. Only add images for content that benefits from visualization
2. For EXACT match, set local_hint to a likely folder name (lowercase, hyphenated)
3. For ILLUSTRATIVE, set pexels_query to a good stock photo search
4. Confidence should reflect how much this image adds value (0.0-1.0)
5. Skip segments that don't need images (opening hooks, CTAs, transitions)
```

### 2. ImageResolver

**Purpose:** Resolves each ImageOverlay to an actual image file.

**Resolution Order:**
1. Check `local_hint` in profile assets → if found, use local image
2. Search Pexels with `pexels_query` → if good match, queue for download
3. If `match_type == "exact"` and no good match → skip (don't show image)
4. If `match_type == "illustrative"` → use best Pexels result

**Local Library Search:**
```python
def find_local_image(profile_path: Path, hint: str, topic: str) -> Optional[Path]:
    """Find image in profile's local library.

    Search order:
    1. Exact folder name match (hint)
    2. Alias match in metadata.json
    3. Tag match in metadata.json
    """
    assets_dir = profile_path / "assets" / "images"
    if not assets_dir.exists():
        return None

    # Try exact folder match first
    folder = assets_dir / hint
    if folder.exists():
        return _get_image_from_folder(folder)

    # Search all folders for alias/tag match
    for folder in assets_dir.iterdir():
        if not folder.is_dir():
            continue
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            aliases = [a.lower() for a in metadata.get("aliases", [])]
            tags = [t.lower() for t in metadata.get("tags", [])]

            if hint.lower() in aliases or topic.lower() in aliases:
                return _get_image_from_folder(folder)
            if any(t in topic.lower() for t in tags):
                return _get_image_from_folder(folder)

    return None
```

### 3. ImageDownloader

**Purpose:** Downloads Pexels images to cache.

**Features:**
- Cache-first (check before download)
- Download best quality (original or large)
- Store metadata in cache index
- Copy to temp for pipeline use

**Pexels Image API:**
```python
# GET https://api.pexels.com/v1/search?query=...&per_page=15
# Response:
{
  "photos": [
    {
      "id": 12345678,
      "width": 1920,
      "height": 1280,
      "url": "https://pexels.com/photo/...",
      "photographer": "John Doe",
      "photographer_url": "https://pexels.com/@johndoe",
      "alt": "Person working on laptop",
      "src": {
        "original": "https://images.pexels.com/photos/.../original.jpeg",
        "large": "https://images.pexels.com/photos/.../large.jpeg",
        "medium": "https://images.pexels.com/photos/.../medium.jpeg",
        ...
      }
    }
  ]
}
```

### 4. ImageOverlayRenderer

**Purpose:** Composites images onto video with animations.

**FFmpeg Filter Strategy:**

```
Input Video
    |
    v
[Split into segments where overlay is active]
    |
    v
For each overlay segment:
    1. Create frosted glass backdrop (blur only in container area)
    2. Scale image to fit container (maintain aspect ratio)
    3. Apply pop-in animation (scale 0->1.1->1.0 with easing)
    4. Composite onto video
    5. Apply pop-out animation at end
    |
    v
[Concatenate all segments]
    |
    v
Output Video (ready for subtitle rendering)
```

**Animation Keyframes:**

```python
# Pop-in (300ms = 9 frames at 30fps)
POP_IN_KEYFRAMES = [
    (0, 0.0, 0.0),      # Frame 0: scale=0, opacity=0
    (4, 1.1, 1.0),      # Frame 4: scale=1.1, opacity=1 (overshoot)
    (6, 0.95, 1.0),     # Frame 6: scale=0.95 (bounce back)
    (8, 1.02, 1.0),     # Frame 8: scale=1.02 (small bounce)
    (9, 1.0, 1.0),      # Frame 9: scale=1.0 (settle)
]

# Pop-out (200ms = 6 frames at 30fps)
POP_OUT_KEYFRAMES = [
    (0, 1.0, 1.0),      # Frame 0: scale=1.0, opacity=1
    (3, 1.1, 0.8),      # Frame 3: scale=1.1, opacity=0.8 (slight grow)
    (6, 0.0, 0.0),      # Frame 6: scale=0, opacity=0
]
```

**Blur Backdrop (Frosted Glass):**

```python
# FFmpeg filter for frosted glass effect (blur only the container area)
def create_frosted_glass_filter(x, y, width, height, blur_strength=20):
    """Create FFmpeg filter for frosted glass container.

    Only blurs the area where the container will be placed,
    leaving the rest of the video (including subtitle area) sharp.
    """
    return f"""
    [0:v]split=2[bg][fg];
    [bg]crop={width}:{height}:{x}:{y},boxblur={blur_strength}:1,
       scale={width}:{height}[blurred];
    [fg][blurred]overlay={x}:{y}[with_blur]
    """
```

**Positioning:**

```
Video: 1080x1920 (9:16)

Container:
  - Width: 1080 - 80 = 1000px (40px margin each side)
  - Height: Calculated from image aspect ratio, max 600px
  - X: 40px (centered)
  - Y: Positioned so bottom is 20px above subtitle area top

Subtitle area (approximate):
  - Starts at Y=1400 (bottom 520px for subtitles)

So container bottom = 1400 - 20 = 1380
Container top = 1380 - height
```

---

## Pipeline Integration

### Modified Orchestrator Flow

```python
# In orchestrator.py generate() method:

# After ScriptPlanner, before VoiceGenerator
if self.overlay_images_enabled:
    # Plan image overlays
    image_planner = ImageOverlayPlanner(ai_client=self.ai_client)
    context = await image_planner.execute(context)

    # Resolve images (local + Pexels)
    image_resolver = ImageResolver(profile_path=profile_path)
    context = await image_resolver.execute(context)

    # Download Pexels images
    image_downloader = ImageDownloader()
    context = await image_downloader.execute(context)

# ... voice generation, video search, download, assembly ...

# After VideoAssembler, before SubtitleRenderer
if self.overlay_images_enabled and context.image_overlays:
    image_renderer = ImageOverlayRenderer()
    context = await image_renderer.execute(context)

# SubtitleRenderer runs last (subtitles on top, always sharp)
```

### CLI Flag

```python
# In cli/reel/commands.py:

@app.command()
def generate_reel(
    # ... existing params ...
    overlay_images: bool = typer.Option(
        False,
        "--overlay-images",
        help="Add contextual images that illustrate the narration",
    ),
):
    # ...
    pipeline = VideoPipeline(
        # ... existing params ...
        overlay_images=overlay_images,
    )
```

---

## Output Files

When `--overlay-images` is used, additional files are saved:

```
reels/generated/XX-XXX-reel/
    final.mp4
    metadata.json           # Includes image_overlays section
    caption.txt
    caption+hashtags.txt
    thumbnail.jpg
    voiceover.mp3
    voiceover.srt
    image_overlays.json     # NEW: Image timing and sources
```

### image_overlays.json Example

```json
{
  "overlays": [
    {
      "segment_index": 2,
      "start_time": 5.2,
      "end_time": 15.8,
      "topic": "Stranger Things",
      "match_type": "exact",
      "source": "local",
      "local_hint": "stranger-things",
      "image_path": "assets/images/stranger-things/image.jpg",
      "alt_text": "Stranger Things TV show poster",
      "confidence": 0.95
    },
    {
      "segment_index": 3,
      "start_time": 15.8,
      "end_time": 26.4,
      "topic": "The Witcher",
      "match_type": "exact",
      "source": "pexels",
      "pexels_id": 12345678,
      "image_path": "pexels/image-cache/12345678.jpg",
      "alt_text": "The Witcher fantasy series image",
      "confidence": 0.9
    },
    {
      "segment_index": 4,
      "start_time": 26.4,
      "end_time": 35.0,
      "topic": "streaming habits",
      "match_type": "illustrative",
      "source": "pexels",
      "pexels_id": 87654321,
      "image_path": "pexels/image-cache/87654321.jpg",
      "alt_text": "Person watching TV on couch",
      "confidence": 0.7
    }
  ],
  "skipped": [
    {
      "segment_index": 1,
      "reason": "Opening hook - no image needed"
    },
    {
      "segment_index": 5,
      "reason": "CTA segment - no image needed"
    }
  ]
}
```

---

## Implementation Order

### Phase 1: Core Infrastructure
1. Add data models to `base.py` (ImageOverlay, ImageOverlayScript)
2. Create `image_cache.py` (copy pattern from pexels_cache.py)
3. Create `pexels_image.py` (Pexels Image API client)
4. Add `--overlay-images` flag to CLI

### Phase 2: Planning & Resolution
5. Create `image_overlay_planner.py` (AI planning step)
6. Create `image_resolver.py` (local + Pexels resolution)
7. Create `image_downloader.py` (cache-first download)

### Phase 3: Rendering
8. Create `image_overlay_renderer.py` (FFmpeg compositing)
9. Integrate into orchestrator pipeline
10. Add GPU-accelerated version (optional)

### Phase 4: Testing & Polish
11. Test with ai.for.mortals profile (ChatGPT, Midjourney images)
12. Test with news.but.quick profile (TV shows, movies)
13. Add local image library examples
14. Performance optimization

---

## Constants

```python
# In constants.py:

# Image overlay positioning
IMAGE_OVERLAY_MARGIN_X = 40        # Pixels from left/right edge
IMAGE_OVERLAY_MAX_HEIGHT = 600     # Max image container height
IMAGE_OVERLAY_MARGIN_BOTTOM = 20   # Pixels above subtitle area
IMAGE_OVERLAY_SUBTITLE_Y = 1400    # Where subtitles start (approx)

# Animation durations (in seconds)
IMAGE_OVERLAY_POP_IN_DURATION = 0.3
IMAGE_OVERLAY_POP_OUT_DURATION = 0.2

# Blur strength for frosted glass
IMAGE_OVERLAY_BLUR_STRENGTH = 20

# Pexels image cache
PEXELS_IMAGE_CACHE_DIR = "pexels/image-cache"
```

---

## Questions/Decisions

1. **Should images fade between segments or pop in/out independently?**
   - Recommendation: Pop out previous, then pop in next (with small gap)

2. **Max simultaneous images?**
   - Recommendation: 1 at a time (simpler, cleaner)

3. **Should we support video clips as overlays too?**
   - Future enhancement, not in MVP

4. **GPU acceleration for blur?**
   - Can use NVENC for encoding, but blur is CPU-based
   - Consider using `zscale` or simpler blur for performance
