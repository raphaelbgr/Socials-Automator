# Video Generation Feature Specification

## Overview

Add automated video generation for Instagram Reels and TikTok. The AI will generate scripts, select stock footage, add text-to-speech narration with karaoke-style subtitles, and export 60-second vertical videos.

## Target Output

- **Format**: 1080x1920 (9:16 vertical)
- **Duration**: 60 seconds exactly
- **Platforms**: Instagram Reels, TikTok
- **Audio**: AI-generated voiceover with word-level timestamps
- **Subtitles**: Karaoke-style (word highlighted as spoken)

---

## Architecture

```
1. AI generates 60-sec script + scene breakdown
        |
        v
2. edge-tts -> generates audio + word timestamps (.srt)
        |
        v
3. Pexels API -> search & download stock clips for each scene
        |
        v
4. MoviePy -> assemble clips to match audio timing
        |
        v
5. pycaps -> add karaoke-style animated subtitles
        |
        v
6. Export 1080x1920 MP4 (9:16)
```

---

## Components & Libraries

### 1. Text-to-Speech: edge-tts

- **GitHub**: https://github.com/rany2/edge-tts
- **Purpose**: Generate voiceover audio with word-level timestamps
- **Cost**: Free (uses Microsoft Edge TTS)
- **Install**: `pip install edge-tts`

```python
import edge_tts
import asyncio

async def generate_voiceover(text: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")
    await communicate.save(output_path)

    # Also generates .srt with word timestamps
    submaker = edge_tts.SubMaker()
    async for chunk in communicate.stream():
        if chunk["type"] == "WordBoundary":
            submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(output_path.replace(".mp3", ".srt"), "w") as f:
        f.write(submaker.generate_subs())
```

**Key Features**:
- 100+ high-quality voices
- Multiple languages
- Word-level timestamps in SRT format
- No API key required

---

## Audio/TTS Strategy

### Voice Selection

```python
# Recommended voices for different content types
VOICE_PRESETS = {
    # English - US
    "professional_male": "en-US-GuyNeural",
    "professional_female": "en-US-AriaNeural",
    "friendly_male": "en-US-DavisNeural",
    "friendly_female": "en-US-JennyNeural",
    "energetic": "en-US-SaraNeural",

    # English - UK
    "british_male": "en-GB-RyanNeural",
    "british_female": "en-GB-SoniaNeural",

    # Other languages
    "spanish": "es-ES-ElviraNeural",
    "portuguese_br": "pt-BR-FranciscaNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
}
```

### Speed and Pacing Control

```python
async def generate_voiceover(
    text: str,
    voice: str = "en-US-AriaNeural",
    rate: str = "+0%",      # -50% to +100%
    pitch: str = "+0Hz",    # -50Hz to +50Hz
    volume: str = "+0%"     # -50% to +50%
) -> tuple[str, str]:
    """Generate voiceover with precise control"""

    communicate = edge_tts.Communicate(
        text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume
    )

    audio_path = "voiceover.mp3"
    srt_path = "voiceover.srt"

    # Generate audio and word timestamps
    submaker = edge_tts.SubMaker()

    with open(audio_path, "wb") as audio_file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub(
                    (chunk["offset"], chunk["duration"]),
                    chunk["text"]
                )

    # Save word-level timestamps
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write(submaker.generate_subs())

    return audio_path, srt_path
```

### Word Timing for Karaoke

The SRT output from edge-tts contains word-level timing:

```srt
1
00:00:00,000 --> 00:00:00,300
AI

2
00:00:00,300 --> 00:00:00,500
is

3
00:00:00,500 --> 00:00:00,900
changing

4
00:00:00,900 --> 00:00:01,100
the

5
00:00:01,100 --> 00:00:01,400
world
```

### Duration Calculation

```python
def calculate_speech_duration(text: str, words_per_minute: int = 150) -> float:
    """Estimate speech duration for script validation"""
    word_count = len(text.split())
    return (word_count / words_per_minute) * 60  # seconds

def validate_script_duration(script: VideoScript, target: int = 60) -> bool:
    """Ensure script fits target duration"""
    total_text = script.hook + ' '.join(s.text for s in script.scenes) + script.cta
    estimated = calculate_speech_duration(total_text)

    # Allow 5 second tolerance
    return abs(estimated - target) <= 5
```

### Audio Post-Processing

```python
from pydub import AudioSegment

def normalize_audio(audio_path: str, target_dbfs: float = -14.0) -> str:
    """Normalize audio volume for consistent output"""
    audio = AudioSegment.from_mp3(audio_path)

    # Calculate adjustment needed
    change_in_dbfs = target_dbfs - audio.dBFS

    # Apply normalization
    normalized = audio.apply_gain(change_in_dbfs)

    output_path = audio_path.replace(".mp3", "_normalized.mp3")
    normalized.export(output_path, format="mp3")

    return output_path

def add_background_music(
    voiceover_path: str,
    music_path: str,
    music_volume: float = -20  # dB below voiceover
) -> str:
    """Mix voiceover with background music"""
    voiceover = AudioSegment.from_mp3(voiceover_path)
    music = AudioSegment.from_mp3(music_path)

    # Loop music to match voiceover length
    if len(music) < len(voiceover):
        loops_needed = (len(voiceover) // len(music)) + 1
        music = music * loops_needed

    # Trim to voiceover length
    music = music[:len(voiceover)]

    # Lower music volume
    music = music + music_volume

    # Mix together
    mixed = voiceover.overlay(music)

    output_path = voiceover_path.replace(".mp3", "_with_music.mp3")
    mixed.export(output_path, format="mp3")

    return output_path
```

---

### 2. Stock Videos: Pexels API

- **Docs**: https://www.pexels.com/api/documentation/
- **Purpose**: Search and download royalty-free stock videos
- **Cost**: Free
- **Install**: `pip install pexels-api-py`

```python
from pexelsapi.pexels import Pexels
import requests

pexel = Pexels('PEXELS_API_KEY')

def search_videos(query: str, count: int = 5):
    """Search for vertical videos matching query"""
    results = pexel.search_videos(
        query=query,
        orientation='portrait',  # Prioritize 9:16 videos
        size='medium',           # 1080p quality
        per_page=count
    )
    return results['videos']

def is_9_16(video: dict) -> bool:
    """Check if video is close to 9:16 aspect ratio"""
    width = video['video_files'][0]['width']
    height = video['video_files'][0]['height']
    ratio = width / height
    target = 9 / 16  # 0.5625
    return abs(ratio - target) < 0.1

def download_video(video: dict, output_path: str):
    """Download the best quality video file"""
    # Get HD quality file
    video_file = next(
        (f for f in video['video_files'] if f['quality'] == 'hd'),
        video['video_files'][0]
    )
    response = requests.get(video_file['link'])
    with open(output_path, 'wb') as f:
        f.write(response.content)
```

**Search Strategy**:
1. First search with `orientation='portrait'` (native vertical)
2. Filter results by exact 9:16 ratio using `is_9_16()`
3. Fallback: search `orientation='landscape'` and crop to 9:16

---

## Video Selection Strategy

### Priority Order

```
Priority 1: Native 9:16 portrait videos (no cropping needed)
    |
    v (if not enough results)
Priority 2: Other portrait videos (minor crop/adjust)
    |
    v (if still not enough)
Priority 3: Landscape videos (center crop to 9:16)
```

### Smart Video Selection Algorithm

```python
async def find_best_video(keywords: list[str], duration: float) -> str:
    """Find best matching video with orientation priority"""

    # Priority 1: Search portrait orientation first
    results = pexel.search_videos(
        query=' '.join(keywords),
        orientation='portrait',
        size='medium',
        per_page=15
    )

    # Filter for exact 9:16 ratio
    perfect_matches = [v for v in results['videos'] if is_9_16(v)]

    if perfect_matches:
        # Pick video with duration closest to needed
        best = min(perfect_matches, key=lambda v: abs(v['duration'] - duration))
        return await download_video(best)

    # Priority 2: Any portrait video (will need minor adjustment)
    if results['videos']:
        best = min(results['videos'], key=lambda v: abs(v['duration'] - duration))
        return await download_video(best)

    # Priority 3: Fallback to landscape (will crop to 9:16)
    results = pexel.search_videos(
        query=' '.join(keywords),
        orientation='landscape',
        size='medium',
        per_page=10
    )

    if results['videos']:
        best = min(results['videos'], key=lambda v: abs(v['duration'] - duration))
        return await download_video(best)

    raise VideoNotFoundError(f"No videos found for: {keywords}")
```

### Duration Matching

```python
def select_clip_segment(clip_path: str, needed_duration: float) -> VideoFileClip:
    """Select best segment from a longer clip"""
    clip = VideoFileClip(clip_path)

    if clip.duration <= needed_duration:
        # Clip is shorter, loop or use as-is
        return clip.loop(duration=needed_duration) if clip.duration < needed_duration * 0.5 else clip

    # Clip is longer, select best segment
    # Prefer middle section (usually more interesting than start/end)
    start_time = (clip.duration - needed_duration) / 2
    return clip.subclip(start_time, start_time + needed_duration)
```

### Keyword Fallback Strategy

```python
async def search_with_fallback(keywords: list[str]) -> dict:
    """Try progressively broader searches"""

    # Try 1: All keywords combined
    results = await search_videos(' '.join(keywords))
    if results['total_results'] > 0:
        return results

    # Try 2: First keyword only (most relevant)
    results = await search_videos(keywords[0])
    if results['total_results'] > 0:
        return results

    # Try 3: Generic fallback based on content type
    fallback_keywords = {
        'technology': ['computer', 'digital', 'tech office'],
        'ai': ['artificial intelligence', 'robot', 'futuristic'],
        'productivity': ['working', 'office', 'laptop'],
        'tips': ['tutorial', 'demonstration', 'how to'],
    }

    for keyword in keywords:
        for category, fallbacks in fallback_keywords.items():
            if category in keyword.lower():
                for fb in fallbacks:
                    results = await search_videos(fb)
                    if results['total_results'] > 0:
                        return results

    # Last resort: generic tech footage
    return await search_videos('technology abstract')
```

---

### 3. Animated Subtitles: pycaps

- **GitHub**: https://github.com/francozanardi/pycaps
- **Purpose**: Add TikTok-style karaoke subtitles with word highlighting
- **Cost**: Free
- **Install**: `pip install pycaps`

```python
from pycaps import render_video

def add_karaoke_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str
):
    """Add animated word-by-word subtitles"""
    render_video(
        input_video=video_path,
        subtitles=srt_path,
        output_video=output_path,
        style={
            "font": "Montserrat-Bold",
            "font_size": 60,
            "color": "white",
            "highlight_color": "#FFD700",  # Gold highlight
            "stroke_color": "black",
            "stroke_width": 3,
            "position": "center",
            "animation": "pop"  # Word pop effect
        }
    )
```

**Style Options**:
- Font, size, color customization
- Highlight color for active word
- Animations: pop, fade, slide
- Position: top, center, bottom

---

### 4. Video Assembly: MoviePy

- **Docs**: https://zulko.github.io/moviepy/
- **Purpose**: Assemble video clips, add audio, crop/resize
- **Cost**: Free
- **Install**: `pip install moviepy`

```python
from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeVideoClip,
    concatenate_videoclips
)

def assemble_video(
    clips: list[str],
    audio_path: str,
    output_path: str,
    target_duration: int = 60
):
    """Assemble clips with audio into final video"""

    # Load audio to get duration
    audio = AudioFileClip(audio_path)
    total_duration = min(audio.duration, target_duration)

    # Load and process video clips
    video_clips = []
    clip_duration = total_duration / len(clips)

    for clip_path in clips:
        clip = VideoFileClip(clip_path)

        # Crop to 9:16 if needed
        clip = crop_to_9_16(clip)

        # Resize to 1080x1920
        clip = clip.resize((1080, 1920))

        # Set duration
        clip = clip.subclip(0, min(clip.duration, clip_duration))

        video_clips.append(clip)

    # Concatenate all clips
    final_video = concatenate_videoclips(video_clips)

    # Add audio
    final_video = final_video.set_audio(audio)

    # Export
    final_video.write_videofile(
        output_path,
        fps=30,
        codec='libx264',
        audio_codec='aac'
    )

def crop_to_9_16(clip):
    """Center crop video to 9:16 aspect ratio"""
    w, h = clip.size
    current_ratio = w / h
    target_ratio = 9 / 16

    if current_ratio > target_ratio:
        # Video is too wide, crop width
        new_width = int(h * target_ratio)
        x_center = w // 2
        clip = clip.crop(
            x1=x_center - new_width // 2,
            x2=x_center + new_width // 2
        )
    elif current_ratio < target_ratio:
        # Video is too tall, crop height
        new_height = int(w / target_ratio)
        y_center = h // 2
        clip = clip.crop(
            y1=y_center - new_height // 2,
            y2=y_center + new_height // 2
        )

    return clip
```

---

## AI Script Generation

The AI generates a structured video plan:

```python
from pydantic import BaseModel

class VideoScene(BaseModel):
    """Single scene in the video"""
    text: str                    # Narration text for this scene
    duration_seconds: float      # Target duration
    video_keywords: list[str]    # Keywords to search Pexels

class VideoScript(BaseModel):
    """Complete video script"""
    title: str
    hook: str                    # Opening hook (first 3 seconds)
    scenes: list[VideoScene]
    cta: str                     # Call to action (last scene)
    total_duration: int = 60
```

**Prompt for AI**:

```
Generate a 60-second video script for Instagram Reels/TikTok.

Topic: {topic}

Requirements:
1. Total duration: exactly 60 seconds
2. Hook in first 3 seconds to grab attention
3. 5-7 scenes, each with:
   - Narration text (what the voiceover says)
   - Duration in seconds
   - Video keywords for stock footage search
4. End with a call-to-action

Output as JSON matching this structure:
{
  "title": "...",
  "hook": "...",
  "scenes": [
    {
      "text": "narration for scene 1",
      "duration_seconds": 8,
      "video_keywords": ["keyword1", "keyword2"]
    }
  ],
  "cta": "Follow for more tips!"
}

Important:
- Keep narration concise (spoken word ~150 words/minute)
- Use engaging, conversational tone
- Video keywords should be generic enough to find stock footage
```

---

## CLI Command

```bash
# Generate video for profile
python -m socials_automator.cli generate-reel ai.for.mortals --topic "5 AI Tools for Productivity"

# With auto-generated topic
python -m socials_automator.cli generate-reel ai.for.mortals

# Dry run (generate but don't post)
python -m socials_automator.cli generate-reel ai.for.mortals --dry-run
```

---

## Output Structure

```
profiles/ai.for.mortals/reels/YYYY/MM/
  generated/
    {post_id}/
      script.json          # AI-generated script
      voiceover.mp3        # TTS audio
      voiceover.srt        # Word timestamps
      clips/               # Downloaded Pexels clips
        scene_01.mp4
        scene_02.mp4
        ...
      assembled.mp4        # Video without subtitles
      final.mp4            # Final video with karaoke subs
      thumbnail.jpg        # Auto-generated thumbnail
```

---

## Configuration (providers.yaml)

```yaml
video_generation:
  # Pexels API for stock footage
  pexels:
    api_key_env: "PEXELS_API_KEY"
    prefer_orientation: "portrait"
    fallback_orientation: "landscape"
    quality: "hd"

  # Text-to-speech
  tts:
    provider: "edge-tts"
    voice: "en-US-AriaNeural"  # Or other voices
    rate: "+0%"                 # Speed adjustment

  # Subtitle styling
  subtitles:
    font: "Montserrat-Bold"
    font_size: 60
    color: "white"
    highlight_color: "#FFD700"
    stroke_color: "black"
    stroke_width: 3
    position: "center"
    animation: "pop"

  # Output settings
  output:
    width: 1080
    height: 1920
    fps: 30
    duration: 60
    codec: "libx264"
```

---

## Environment Variables

```bash
PEXELS_API_KEY=your_pexels_api_key
```

Get free API key at: https://www.pexels.com/api/

---

## Dependencies

Add to `requirements.txt`:

```
edge-tts>=6.1.0
pexels-api-py>=1.0.0
pycaps>=0.1.0
moviepy>=1.0.3
```

Also requires **FFmpeg** installed and in PATH.

---

## Implementation Steps

1. [ ] Create `src/socials_automator/video/` module
2. [ ] Implement `script_generator.py` - AI script generation
3. [ ] Implement `tts.py` - edge-tts wrapper with word timestamps
4. [ ] Implement `stock_footage.py` - Pexels API client
5. [ ] Implement `assembler.py` - MoviePy video assembly
6. [ ] Implement `subtitles.py` - pycaps karaoke subtitles
7. [ ] Implement `generator.py` - Main orchestrator
8. [ ] Add `reel` command to CLI
9. [ ] Add video configuration to `providers.yaml`
10. [ ] Add tests

---

## Example Complete Flow

```python
async def generate_reel(profile: str, topic: str):
    """Complete video generation pipeline"""

    # 1. Generate script with AI
    script = await generate_video_script(topic)

    # 2. Generate voiceover + timestamps
    audio_path, srt_path = await generate_voiceover(script)

    # 3. Download stock footage for each scene
    clips = []
    for scene in script.scenes:
        video = await search_and_download_video(
            keywords=scene.video_keywords,
            duration=scene.duration_seconds
        )
        clips.append(video)

    # 4. Assemble video with audio
    assembled_path = assemble_video(
        clips=clips,
        audio_path=audio_path,
        duration=60
    )

    # 5. Add karaoke subtitles
    final_path = add_karaoke_subtitles(
        video_path=assembled_path,
        srt_path=srt_path
    )

    return final_path
```

---

## References

- edge-tts: https://github.com/rany2/edge-tts
- pycaps: https://github.com/francozanardi/pycaps
- Pexels API: https://www.pexels.com/api/documentation/
- MoviePy: https://zulko.github.io/moviepy/
- pexels-api-py: https://pypi.org/project/pexels-api-py/
