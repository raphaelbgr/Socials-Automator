# Dense Image Overlay System - Implementation Plan

## Overview

New image overlay mode that provides:
- **Fixed TTL per image** (e.g., 3s display time)
- **High density** (e.g., 20 images per 60s video)
- **No repeated topics** (each image represents unique content)
- **No generic fallbacks** (gaps left empty if no match found)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DenseOverlayPlanner                        │
│  (replaces ImageOverlayPlanner when dense mode enabled)         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Topic Extraction (AI)                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Input: full_narration, target_count (e.g., 20)          │   │
│  │ Output: List of unique visual topics                     │   │
│  │   - topic: "Stranger Things"                             │   │
│  │   - match_type: "exact" | "illustrative"                 │   │
│  │   - search_query: "stranger things netflix poster"       │   │
│  │   - keywords: ["stranger", "things", "netflix"]          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  Phase 2: Time Slot Distribution (Code)                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Input: topics[], srt_timestamps, ttl, duration          │   │
│  │ Process:                                                 │   │
│  │   1. Parse SRT for word-level timing                     │   │
│  │   2. Match topics to SRT entries (when mentioned)        │   │
│  │   3. Assign each topic a TTL-sized slot                  │   │
│  │   4. Resolve overlaps (shift later topics)               │   │
│  │   5. Skip topics without SRT match (no forced placement) │   │
│  │ Output: ImageOverlayScript with TTL-sized overlays       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              Existing Pipeline (unchanged)                       │
├─────────────────────────────────────────────────────────────────┤
│  ImageResolver → ImageDownloader → ImageOverlayRenderer          │
│                                                                  │
│  - If topic not found: skip overlay (leave gap)                  │
│  - No generic/illustrative fallbacks                             │
└─────────────────────────────────────────────────────────────────┘
```

## CLI Parameters

```bash
# Enable dense overlay mode with TTL and minimum count
python -m socials_automator.cli generate-reel ai.for.mortals \
    --overlay-images \
    --overlay-image-ttl 3s \          # Each image displays for 3 seconds
    --overlay-image-minimum 20         # Target 20 images (AI extracts 20+ topics)

# Without these flags, uses existing ImageOverlayPlanner (segment-based)
```

### Parameter Behavior

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--overlay-image-ttl` | None (disabled) | Fixed display time per image. Enables dense mode. |
| `--overlay-image-minimum` | Auto-calculated | Target topic count. Default: `(duration - hook - cta) / ttl` |

When `--overlay-image-ttl` is set:
- Dense mode is enabled
- `DenseOverlayPlanner` is used instead of `ImageOverlayPlanner`
- Each overlay gets exactly TTL duration (or less if at boundaries)

## Data Models

### New: ExtractedTopic

```python
@dataclass
class ExtractedTopic:
    """A visual topic extracted from narration by AI."""
    topic: str                      # "Stranger Things"
    match_type: str                 # "exact" or "illustrative"
    search_query: str               # "stranger things netflix poster"
    keywords: list[str]             # ["stranger", "things", "season", "5"]
    priority: int = 1               # 1=must-have, 2=nice-to-have

    # Filled by TimeSlotDistributor
    srt_start: Optional[float] = None   # When first mentioned in SRT
    srt_end: Optional[float] = None     # When mention ends
    assigned_slot: Optional[int] = None # Which time slot assigned to
```

### Existing: ImageOverlay (unchanged)

The output is still `ImageOverlayScript` with `ImageOverlay` objects.
The only difference is that `start_time` and `end_time` are TTL-spaced.

## Phase 1: Topic Extraction

### AI Prompt Strategy

```
DENSE TOPIC EXTRACTION MODE

Analyze this video narration and extract EVERY distinct visual subject.
Target: {minimum} unique topics (extract more if available).

NARRATION:
{full_narration}

RULES:
1. Extract SPECIFIC subjects that can be visually represented
2. NO DUPLICATES - each topic must be unique
3. Prioritize EXACT matches (products, shows, brands, people)
4. Include ILLUSTRATIVE topics only if they add visual variety
5. For each topic, provide search keywords for image lookup

OUTPUT JSON:
{
  "topics": [
    {
      "topic": "Stranger Things",
      "match_type": "exact",
      "search_query": "stranger things netflix poster logo",
      "keywords": ["stranger", "things", "netflix", "season"],
      "priority": 1
    },
    {
      "topic": "person using smartphone",
      "match_type": "illustrative",
      "search_query": "person holding smartphone screen",
      "keywords": ["phone", "smartphone", "mobile", "screen"],
      "priority": 2
    }
  ]
}

IMPORTANT:
- Extract at least {minimum} topics
- Each topic MUST be distinct (no "Netflix" and "Netflix logo" separately)
- Keywords should include variations for SRT matching
```

### Topic Extraction Logic

```python
class DenseOverlayPlanner(PipelineStep):
    def __init__(
        self,
        text_provider: TextProvider,
        image_ttl: float = 3.0,          # seconds
        minimum_images: Optional[int] = None,  # auto-calculate if None
    ):
        self.ttl = image_ttl
        self.minimum = minimum_images
        self.text_provider = text_provider

    async def execute(self, context: PipelineContext) -> PipelineContext:
        # Calculate minimum if not provided
        if self.minimum is None:
            available = context.script.total_duration - 3.0 - 4.0  # hook, cta
            self.minimum = int(available / self.ttl)

        # Phase 1: Extract topics with AI
        topics = await self._extract_topics(
            context.script.full_narration,
            target_count=self.minimum + 5,  # Ask for extra
        )

        # Phase 2: Distribute into time slots
        overlay_script = self._distribute_to_slots(
            topics=topics,
            script=context.script,
            srt_path=context.srt_path,
            ttl=self.ttl,
        )

        context.image_overlays = overlay_script
        return context
```

## Phase 2: Time Slot Distribution

### Algorithm

```python
def _distribute_to_slots(
    self,
    topics: list[ExtractedTopic],
    script: VideoScript,
    srt_path: Optional[Path],
    ttl: float,
) -> ImageOverlayScript:
    """Distribute topics into TTL-sized time slots."""

    # 1. Parse SRT for word-level timing
    srt_entries = parse_srt(srt_path) if srt_path else []

    # 2. Match topics to SRT entries
    for topic in topics:
        match = find_srt_match(topic.keywords, srt_entries)
        if match:
            topic.srt_start = match.start_time
            topic.srt_end = match.end_time

    # 3. Sort topics by SRT start time (unmatched go to end)
    matched = [t for t in topics if t.srt_start is not None]
    matched.sort(key=lambda t: t.srt_start)

    # 4. Assign time slots with TTL duration
    overlays = []
    min_start = max(3.0, script.hook_end_time)  # After hook
    max_end = min(script.cta_start_time, script.total_duration - 4.0)

    current_time = min_start

    for topic in matched:
        # Start at SRT mention time (or current_time if earlier)
        start = max(topic.srt_start, current_time)

        # Ensure within boundaries
        if start >= max_end:
            break  # No more room

        end = min(start + ttl, max_end)

        # Skip if slot too short
        if end - start < 1.5:  # Minimum visible duration
            continue

        overlays.append(ImageOverlay(
            segment_index=0,  # Not segment-based
            start_time=start,
            end_time=end,
            topic=topic.topic,
            match_type=topic.match_type,
            pexels_query=topic.search_query,
            confidence=0.9 if topic.match_type == "exact" else 0.7,
        ))

        current_time = end + 0.2  # Small gap between overlays

    return ImageOverlayScript(overlays=overlays, skipped=[])
```

### SRT Matching Logic

```python
def find_srt_match(
    keywords: list[str],
    srt_entries: list[SrtEntry],
) -> Optional[SrtEntry]:
    """Find first SRT entry containing any keyword."""

    keywords_lower = [k.lower() for k in keywords]

    for entry in srt_entries:
        text_lower = entry.text.lower()
        for keyword in keywords_lower:
            if keyword in text_lower:
                return entry

    return None  # No match - topic won't get a slot
```

## Integration with Orchestrator

### news_orchestrator.py changes

```python
class NewsOrchestrator:
    def __init__(
        self,
        # ... existing params ...
        overlay_images: bool = False,
        overlay_image_ttl: Optional[float] = None,    # NEW
        overlay_image_minimum: Optional[int] = None,  # NEW
        # ...
    ):
        self.overlay_image_ttl = overlay_image_ttl
        self.overlay_image_minimum = overlay_image_minimum

        # Build overlay steps
        if overlay_images:
            if overlay_image_ttl is not None:
                # Dense mode
                self.image_overlay_steps = [
                    DenseOverlayPlanner(
                        text_provider=ai_client,
                        image_ttl=overlay_image_ttl,
                        minimum_images=overlay_image_minimum,
                    ),
                    ImageResolver(...),      # unchanged
                    ImageDownloader(...),    # unchanged
                    ImageOverlayRenderer(...),  # unchanged
                ]
            else:
                # Existing segment-based mode
                self.image_overlay_steps = [
                    ImageOverlayPlanner(...),  # existing
                    ImageResolver(...),
                    ImageDownloader(...),
                    ImageOverlayRenderer(...),
                ]
```

## CLI Integration

### cli/reel/params.py

```python
@dataclass(frozen=True)
class ReelGenerationParams:
    # ... existing fields ...
    overlay_images: bool = False
    overlay_image_ttl: Optional[float] = None      # NEW: seconds
    overlay_image_minimum: Optional[int] = None    # NEW: target count
```

### cli/reel/commands.py

```python
@app.command()
def generate_reel(
    # ... existing params ...
    overlay_images: bool = typer.Option(False, "--overlay-images"),
    overlay_image_ttl: Optional[str] = typer.Option(
        None,
        "--overlay-image-ttl",
        help="Fixed display time per image (e.g., '3s'). Enables dense mode.",
    ),
    overlay_image_minimum: Optional[int] = typer.Option(
        None,
        "--overlay-image-minimum",
        help="Target number of images. Default: auto-calculated from TTL.",
    ),
):
    # Parse TTL string to float
    ttl_seconds = None
    if overlay_image_ttl:
        ttl_seconds = parse_duration(overlay_image_ttl)  # "3s" -> 3.0

    params = ReelGenerationParams(
        # ...
        overlay_images=overlay_images,
        overlay_image_ttl=ttl_seconds,
        overlay_image_minimum=overlay_image_minimum,
    )
```

## File Structure

```
src/socials_automator/video/pipeline/
    image_overlay_planner.py      # Existing (segment-based)
    dense_overlay_planner.py      # NEW (TTL-based)
    image_resolver.py             # Unchanged
    image_downloader.py           # Unchanged
    image_overlay_renderer.py     # Unchanged

tests/ai_test/
    test_dense_overlay.py         # NEW
```

## Example Output

### Input
```
Narration: "Tonight's top stories. Stranger Things Season 5 releases December 26th.
OpenAI announces GPT-5. Apple unveils new iPhone features. Follow for more!"

TTL: 3s
Minimum: 6
Duration: 30s (hook: 3s, cta: 4s, available: 23s)
```

### Phase 1: AI Extraction
```json
{
  "topics": [
    {"topic": "Stranger Things", "match_type": "exact", "keywords": ["stranger", "things"]},
    {"topic": "Netflix", "match_type": "exact", "keywords": ["netflix", "streaming"]},
    {"topic": "December calendar", "match_type": "illustrative", "keywords": ["december", "26"]},
    {"topic": "OpenAI", "match_type": "exact", "keywords": ["openai", "gpt"]},
    {"topic": "GPT-5", "match_type": "exact", "keywords": ["gpt", "gpt-5"]},
    {"topic": "Apple", "match_type": "exact", "keywords": ["apple"]},
    {"topic": "iPhone", "match_type": "exact", "keywords": ["iphone"]}
  ]
}
```

### Phase 2: SRT Matching & Distribution
```
SRT entries:
  0.0-3.0: "Tonight's top stories."
  3.0-7.0: "Stranger Things Season 5 releases"
  7.0-10.0: "December 26th."
  10.0-15.0: "OpenAI announces GPT-5."
  15.0-20.0: "Apple unveils new iPhone features."
  20.0-26.0: "Follow for more!"

Matched topics:
  - Stranger Things → 3.0s (SRT match)
  - December calendar → 7.0s (SRT match)
  - OpenAI → 10.0s (SRT match)
  - GPT-5 → 10.0s (SRT match, but OpenAI takes slot)
  - Apple → 15.0s (SRT match)
  - iPhone → 15.0s (SRT match, but Apple takes slot)
  - Netflix → No SRT match, skipped
```

### Final Overlays
```
Overlay 1: Stranger Things  [3.0s - 6.0s]   TTL=3s
Overlay 2: December calendar [7.0s - 10.0s]  TTL=3s
Overlay 3: OpenAI           [10.0s - 13.0s] TTL=3s
Overlay 4: Apple            [15.0s - 18.0s] TTL=3s
           (GPT-5, iPhone skipped - slots taken)
           (Netflix skipped - no SRT match)

Total: 4 overlays (not 6 - some couldn't fit or match)
Gaps: 6.0-7.0s, 13.0-15.0s, 18.0-26.0s (no image, just video)
```

## Testing Strategy

1. **Unit tests** for topic extraction parsing
2. **Unit tests** for SRT matching algorithm
3. **Unit tests** for slot distribution logic
4. **Integration test** with mock AI response
5. **End-to-end test** with LMStudio

## Implementation Order

1. Add CLI parameters and pass through to orchestrator
2. Create `DenseOverlayPlanner` class with Phase 1 (AI extraction)
3. Implement Phase 2 (time slot distribution)
4. Wire up to orchestrator (select planner based on TTL flag)
5. Add tests
6. Test with real video generation
