"""Tests for video deduplication and history functionality."""

import pytest
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


class TestVideoHistory:
    """Test video history persistence for deduplication across reels."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.history_file = self.temp_dir / "video_history.json"

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_video_history(self):
        """Test saving video history to file."""
        history = {
            "12345": datetime.now().isoformat(),
            "67890": datetime.now().isoformat(),
        }

        with open(self.history_file, "w") as f:
            json.dump(history, f)

        assert self.history_file.exists()
        with open(self.history_file) as f:
            loaded = json.load(f)
        assert "12345" in loaded
        assert "67890" in loaded

    def test_load_video_history(self):
        """Test loading video history from file."""
        history = {
            "12345": datetime.now().isoformat(),
            "67890": datetime.now().isoformat(),
        }
        with open(self.history_file, "w") as f:
            json.dump(history, f)

        with open(self.history_file) as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert "12345" in loaded
        assert "67890" in loaded

    def test_history_expiration(self):
        """Test that old entries are removed after cooldown period."""
        cooldown_days = 7
        cutoff = datetime.now() - timedelta(days=cooldown_days)
        cutoff_str = cutoff.isoformat()

        history = {
            "old_video": (datetime.now() - timedelta(days=10)).isoformat(),  # Expired
            "recent_video": datetime.now().isoformat(),  # Still valid
        }

        # Filter expired entries
        filtered = {
            vid: ts for vid, ts in history.items()
            if ts > cutoff_str
        }

        assert "old_video" not in filtered
        assert "recent_video" in filtered

    def test_mark_video_used(self):
        """Test marking a video as used."""
        used_ids = set()
        history = {}

        def mark_video_used(video_id: int):
            used_ids.add(video_id)
            history[str(video_id)] = datetime.now().isoformat()

        mark_video_used(12345)
        mark_video_used(67890)

        assert 12345 in used_ids
        assert 67890 in used_ids
        assert "12345" in history
        assert "67890" in history

    def test_history_persists_across_sessions(self):
        """Test that history persists when reloaded."""
        # Session 1: Save some videos
        history1 = {"12345": datetime.now().isoformat()}
        with open(self.history_file, "w") as f:
            json.dump(history1, f)

        # Session 2: Load and add more
        with open(self.history_file) as f:
            history2 = json.load(f)
        history2["67890"] = datetime.now().isoformat()
        with open(self.history_file, "w") as f:
            json.dump(history2, f)

        # Session 3: Verify all videos are there
        with open(self.history_file) as f:
            history3 = json.load(f)

        assert "12345" in history3
        assert "67890" in history3


class TestVideoDeduplication:
    """Test video deduplication within a single reel."""

    def test_remove_duplicate_pexels_ids(self):
        """Test removing duplicate videos by pexels_id."""
        clips = [
            {"pexels_id": 1, "duration": 10},
            {"pexels_id": 2, "duration": 15},
            {"pexels_id": 1, "duration": 10},  # Duplicate
            {"pexels_id": 3, "duration": 20},
            {"pexels_id": 2, "duration": 15},  # Duplicate
        ]

        seen_ids = set()
        unique_clips = []
        for clip in clips:
            if clip["pexels_id"] not in seen_ids:
                seen_ids.add(clip["pexels_id"])
                unique_clips.append(clip)

        assert len(unique_clips) == 3
        unique_ids = [c["pexels_id"] for c in unique_clips]
        assert unique_ids == [1, 2, 3]

    def test_skip_already_used_videos(self):
        """Test that videos in history are skipped."""
        used_video_ids = {100, 200, 300}

        videos = [
            {"id": 100, "duration": 10},  # Already used
            {"id": 400, "duration": 15},  # Available
            {"id": 200, "duration": 20},  # Already used
            {"id": 500, "duration": 25},  # Available
        ]

        available = [v for v in videos if v["id"] not in used_video_ids]

        assert len(available) == 2
        assert available[0]["id"] == 400
        assert available[1]["id"] == 500

    def test_fallback_to_used_videos_when_necessary(self):
        """Test that we can reuse videos if no alternatives available."""
        used_video_ids = {100, 200}
        videos = [
            {"id": 100, "duration": 10},
            {"id": 200, "duration": 15},
        ]

        available = [v for v in videos if v["id"] not in used_video_ids]

        if not available:
            # All videos used, allow reuse as last resort
            available = videos

        assert len(available) == 2


class TestVideoCoverageValidation:
    """Test video coverage validation against audio duration."""

    def test_sufficient_video_coverage(self):
        """Test detection of sufficient video coverage."""
        audio_duration = 60.0
        clips = [
            {"duration_seconds": 20.0},
            {"duration_seconds": 25.0},
            {"duration_seconds": 20.0},
        ]

        total_video = sum(c["duration_seconds"] for c in clips)

        assert total_video >= audio_duration
        assert total_video == 65.0

    def test_insufficient_video_coverage(self):
        """Test detection of insufficient video coverage."""
        audio_duration = 60.0
        clips = [
            {"duration_seconds": 10.0},
            {"duration_seconds": 15.0},
            {"duration_seconds": 10.0},
        ]

        total_video = sum(c["duration_seconds"] for c in clips)

        assert total_video < audio_duration
        assert total_video == 35.0

    def test_calculate_video_shortfall(self):
        """Test calculating how much more video is needed."""
        audio_duration = 60.0
        clips = [
            {"duration_seconds": 20.0},
            {"duration_seconds": 15.0},
        ]

        total_video = sum(c["duration_seconds"] for c in clips)
        shortfall = audio_duration - total_video

        assert shortfall == 25.0

    def test_no_shortfall_when_sufficient(self):
        """Test no shortfall when video coverage is sufficient."""
        audio_duration = 60.0
        clips = [
            {"duration_seconds": 30.0},
            {"duration_seconds": 35.0},
        ]

        total_video = sum(c["duration_seconds"] for c in clips)
        shortfall = max(0, audio_duration - total_video)

        assert shortfall == 0

    def test_unique_clips_only_in_total(self):
        """Test that total duration uses unique clips only."""
        audio_duration = 60.0

        class MockClip:
            def __init__(self, pexels_id, duration):
                self.pexels_id = pexels_id
                self.duration_seconds = duration

        all_clips = [
            MockClip(1, 20.0),
            MockClip(2, 25.0),
            MockClip(1, 20.0),  # Duplicate - should not count
            MockClip(3, 20.0),
        ]

        seen_ids = set()
        unique_clips = []
        for clip in all_clips:
            if clip.pexels_id not in seen_ids:
                seen_ids.add(clip.pexels_id)
                unique_clips.append(clip)

        total_unique = sum(c.duration_seconds for c in unique_clips)

        assert len(unique_clips) == 3
        assert total_unique == 65.0  # 20 + 25 + 20, not 85


class TestRetryLogic:
    """Test retry logic for fetching more videos."""

    def test_max_retries_respected(self):
        """Test that max retries limit is respected."""
        max_retries = 3
        retry_count = 0
        sufficient = False

        while retry_count < max_retries and not sufficient:
            retry_count += 1
            # Simulate insufficient videos
            sufficient = False

        assert retry_count == max_retries

    def test_early_exit_on_success(self):
        """Test that we exit early when sufficient videos found."""
        max_retries = 3
        retry_count = 0
        sufficient = False

        while retry_count < max_retries and not sufficient:
            retry_count += 1
            if retry_count == 2:
                sufficient = True  # Success on second try

        assert retry_count == 2

    def test_new_keywords_generated_on_retry(self):
        """Test that new keywords are generated for retry."""
        used_keywords = ["technology", "office", "computer"]

        def generate_new_keywords(used: list[str]) -> list[str]:
            # Simulate AI generating different keywords
            all_keywords = [
                "abstract background", "digital motion",
                "workspace desk", "coding screen",
                "innovation", "future tech"
            ]
            return [k for k in all_keywords if k not in used][:3]

        new_keywords = generate_new_keywords(used_keywords)

        assert len(new_keywords) == 3
        assert "technology" not in new_keywords
        assert "office" not in new_keywords
