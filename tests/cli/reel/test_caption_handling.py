"""Unit tests for caption handling in the upload flow.

These tests verify that captions are properly:
- Read from files
- Validated for content
- Passed through the upload pipeline
- Not lost during async operations
"""

import pytest
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

from socials_automator.cli.reel.service import ReelUploaderService
from socials_automator.cli.reel.params import ReelUploadParams
from socials_automator.cli.reel.artifacts import (
    audit_reel_artifacts,
    AuditResult,
    ArtifactStatus,
)


class TestCaptionFileReading:
    """Tests for caption file reading behavior."""

    @pytest.fixture
    def temp_reel_dir(self):
        """Create a temporary reel directory."""
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)
        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def uploader(self):
        """Create uploader service."""
        return ReelUploaderService()

    def test_caption_read_prefers_hashtags_file(self, temp_reel_dir):
        """Test that caption+hashtags.txt is preferred over caption.txt."""
        # Create both files with different content
        (temp_reel_dir / "caption.txt").write_text("Caption only", encoding="utf-8")
        (temp_reel_dir / "caption+hashtags.txt").write_text(
            "Caption with hashtags #test", encoding="utf-8"
        )
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")

        # Simulate the caption reading logic from _upload_single
        caption_hashtags_path = temp_reel_dir / "caption+hashtags.txt"
        caption_path = temp_reel_dir / "caption.txt"

        if caption_hashtags_path.exists():
            caption = caption_hashtags_path.read_text(encoding="utf-8")
        elif caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8")
        else:
            caption = ""

        assert caption == "Caption with hashtags #test"

    def test_caption_read_falls_back_to_caption_txt(self, temp_reel_dir):
        """Test fallback to caption.txt when hashtags file missing."""
        (temp_reel_dir / "caption.txt").write_text("Caption only", encoding="utf-8")
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")

        caption_hashtags_path = temp_reel_dir / "caption+hashtags.txt"
        caption_path = temp_reel_dir / "caption.txt"

        if caption_hashtags_path.exists():
            caption = caption_hashtags_path.read_text(encoding="utf-8")
        elif caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8")
        else:
            caption = ""

        assert caption == "Caption only"

    def test_caption_read_falls_back_to_metadata(self, temp_reel_dir):
        """Test fallback to metadata when no caption files exist."""
        metadata = {"caption": "Metadata caption"}
        (temp_reel_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")

        caption_hashtags_path = temp_reel_dir / "caption+hashtags.txt"
        caption_path = temp_reel_dir / "caption.txt"
        metadata_path = temp_reel_dir / "metadata.json"

        if caption_hashtags_path.exists():
            caption = caption_hashtags_path.read_text(encoding="utf-8")
        elif caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8")
        else:
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
            caption = meta.get("caption", "")

        assert caption == "Metadata caption"


class TestEmptyCaptionDetection:
    """Tests for detecting and handling empty captions.

    These tests verify the bug where empty captions were uploaded to Instagram.
    """

    @pytest.fixture
    def temp_reel_dir(self):
        """Create a temporary reel directory."""
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)
        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_empty_caption_file_detected(self, temp_reel_dir):
        """Test that empty caption files are detected."""
        # Create empty caption file
        (temp_reel_dir / "caption+hashtags.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (temp_reel_dir / "thumbnail.jpg").write_bytes(b"fake image")

        audit = audit_reel_artifacts(temp_reel_dir)

        # Both caption files should be marked as missing content
        caption_artifact = next(
            (a for a in audit.artifacts if a.name == "caption.txt"), None
        )
        hashtags_artifact = next(
            (a for a in audit.artifacts if a.name == "caption+hashtags.txt"), None
        )

        assert caption_artifact is not None
        assert caption_artifact.exists is True
        assert caption_artifact.has_content is False  # Empty file detected

        assert hashtags_artifact is not None
        assert hashtags_artifact.exists is True
        assert hashtags_artifact.has_content is False  # Empty file detected

    def test_whitespace_only_caption_detected(self, temp_reel_dir):
        """Test that whitespace-only captions are detected as empty."""
        # Create caption with only whitespace
        (temp_reel_dir / "caption+hashtags.txt").write_text(
            "   \n\n   \t  ", encoding="utf-8"
        )
        (temp_reel_dir / "caption.txt").write_text("  \n  ", encoding="utf-8")
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (temp_reel_dir / "thumbnail.jpg").write_bytes(b"fake image")

        audit = audit_reel_artifacts(temp_reel_dir)

        caption_artifact = next(
            (a for a in audit.artifacts if a.name == "caption.txt"), None
        )
        hashtags_artifact = next(
            (a for a in audit.artifacts if a.name == "caption+hashtags.txt"), None
        )

        # After stripping, content length < 10 chars = no content
        assert caption_artifact.has_content is False
        assert hashtags_artifact.has_content is False

    def test_short_caption_detected(self, temp_reel_dir):
        """Test that very short captions (< 10 chars) are flagged."""
        # Create very short caption
        (temp_reel_dir / "caption+hashtags.txt").write_text("Hi!", encoding="utf-8")
        (temp_reel_dir / "caption.txt").write_text("Hi!", encoding="utf-8")
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (temp_reel_dir / "thumbnail.jpg").write_bytes(b"fake image")

        audit = audit_reel_artifacts(temp_reel_dir)

        caption_artifact = next(
            (a for a in audit.artifacts if a.name == "caption.txt"), None
        )

        # "Hi!" is only 3 chars, should be flagged as no content
        assert caption_artifact.has_content is False

    def test_valid_caption_passes(self, temp_reel_dir):
        """Test that valid captions pass the audit."""
        valid_caption = "Check out these amazing AI tools for productivity! #AI #Tech"
        (temp_reel_dir / "caption+hashtags.txt").write_text(
            valid_caption, encoding="utf-8"
        )
        (temp_reel_dir / "caption.txt").write_text(
            "Check out these amazing AI tools!", encoding="utf-8"
        )
        (temp_reel_dir / "final.mp4").write_bytes(b"fake video")
        # metadata.json needs content to pass validation
        (temp_reel_dir / "metadata.json").write_text(
            '{"topic": "AI Tools", "duration_seconds": 60}', encoding="utf-8"
        )
        (temp_reel_dir / "thumbnail.jpg").write_bytes(b"fake image")

        audit = audit_reel_artifacts(temp_reel_dir)

        caption_artifact = next(
            (a for a in audit.artifacts if a.name == "caption.txt"), None
        )
        hashtags_artifact = next(
            (a for a in audit.artifacts if a.name == "caption+hashtags.txt"), None
        )

        assert caption_artifact.has_content is True
        assert hashtags_artifact.has_content is True
        assert audit.is_valid is True


class TestCaptionValidationBug:
    """Tests specifically for the caption validation bug.

    The bug: Caption files exist but are empty/whitespace, and the upload
    proceeds with an empty caption string.
    """

    @pytest.fixture
    def temp_reel_dir(self):
        """Create a temporary reel directory with minimal valid structure."""
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)

        # Create required files
        (reel_dir / "final.mp4").write_bytes(b"fake video content")
        (reel_dir / "metadata.json").write_text(
            json.dumps({"topic": "Test Topic", "duration_seconds": 60}),
            encoding="utf-8",
        )
        (reel_dir / "thumbnail.jpg").write_bytes(b"fake image")

        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_empty_caption_should_fail_audit(self, temp_reel_dir):
        """Test that empty caption files cause audit to fail."""
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "caption+hashtags.txt").write_text("", encoding="utf-8")

        audit = audit_reel_artifacts(temp_reel_dir)

        # Audit should NOT be valid because captions are empty
        assert audit.is_valid is False

        # Check which artifacts failed
        failed_required = [a.name for a in audit.missing_required]
        assert "caption.txt" in failed_required or "caption+hashtags.txt" in failed_required

    def test_caption_read_returns_empty_string(self, temp_reel_dir):
        """Test that reading empty caption returns empty string."""
        (temp_reel_dir / "caption+hashtags.txt").write_text("", encoding="utf-8")

        caption_path = temp_reel_dir / "caption+hashtags.txt"
        caption = caption_path.read_text(encoding="utf-8")

        # This is the bug - we need to validate this is not empty
        assert caption == ""

    def test_caption_should_be_validated_before_upload(self, temp_reel_dir):
        """Test that caption is validated before upload.

        This test documents the expected behavior that should prevent
        empty captions from being uploaded.
        """
        (temp_reel_dir / "caption+hashtags.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")

        # Read caption as done in _upload_single
        caption_hashtags_path = temp_reel_dir / "caption+hashtags.txt"
        caption_path = temp_reel_dir / "caption.txt"
        metadata_path = temp_reel_dir / "metadata.json"

        if caption_hashtags_path.exists():
            caption = caption_hashtags_path.read_text(encoding="utf-8")
        elif caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8")
        else:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            caption = metadata.get("caption", "")

        # Validate caption (THIS IS WHAT SHOULD HAPPEN)
        caption = caption.strip()
        is_valid_caption = len(caption) >= 10

        assert is_valid_caption is False, "Empty caption should not be valid"


class TestCaptionEncoding:
    """Tests for caption encoding handling."""

    @pytest.fixture
    def temp_reel_dir(self):
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)
        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_utf8_caption_preserved(self, temp_reel_dir):
        """Test that UTF-8 characters in captions are preserved."""
        caption = "Check out these AI tools! cafe Munchen emoji test"
        (temp_reel_dir / "caption+hashtags.txt").write_text(caption, encoding="utf-8")

        read_caption = (temp_reel_dir / "caption+hashtags.txt").read_text(
            encoding="utf-8"
        )

        assert read_caption == caption

    def test_emoji_in_caption(self, temp_reel_dir):
        """Test that emojis in captions are handled."""
        # Note: The codebase removes emojis in some places
        caption = "Amazing AI tools! Check them out!"
        (temp_reel_dir / "caption+hashtags.txt").write_text(caption, encoding="utf-8")

        read_caption = (temp_reel_dir / "caption+hashtags.txt").read_text(
            encoding="utf-8"
        )

        assert "Amazing" in read_caption
        assert "AI tools" in read_caption


class TestCaptionInUploadPipeline:
    """Tests for caption handling throughout the upload pipeline."""

    @pytest.fixture
    def mock_publisher(self):
        """Create a mock platform publisher."""
        publisher = MagicMock()
        publisher.publish_reel = AsyncMock()
        publisher.check_credentials = AsyncMock(return_value=(True, "Connected"))
        return publisher

    @pytest.fixture
    def temp_reel_dir(self):
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "profiles" / "test" / "reels" / "2025" / "12" / "generated" / "19-001-test"
        reel_dir.mkdir(parents=True)

        # Create all required files
        (reel_dir / "final.mp4").write_bytes(b"fake video")
        (reel_dir / "metadata.json").write_text(
            json.dumps({"topic": "Test", "duration_seconds": 60}),
            encoding="utf-8",
        )
        (reel_dir / "thumbnail.jpg").write_bytes(b"fake image")
        (reel_dir / "caption.txt").write_text(
            "This is a test caption for the video", encoding="utf-8"
        )
        (reel_dir / "caption+hashtags.txt").write_text(
            "This is a test caption for the video #AI #Tech", encoding="utf-8"
        )

        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_caption_passed_to_publisher(self, temp_reel_dir, mock_publisher):
        """Test that caption is correctly passed to the platform publisher."""
        expected_caption = "This is a test caption for the video #AI #Tech"

        # Simulate caption reading from _upload_single
        caption_hashtags_path = temp_reel_dir / "caption+hashtags.txt"
        caption = caption_hashtags_path.read_text(encoding="utf-8")

        # Verify caption is correct before "upload"
        assert caption == expected_caption

        # Simulate the upload call
        mock_publisher.publish_reel.return_value = MagicMock(
            success=True, media_id="123", permalink="https://instagram.com/p/123"
        )

        await mock_publisher.publish_reel(
            video_path=temp_reel_dir / "final.mp4",
            caption=caption,
            thumbnail_path=temp_reel_dir / "thumbnail.jpg",
        )

        # Verify caption was passed correctly
        mock_publisher.publish_reel.assert_called_once()
        call_args = mock_publisher.publish_reel.call_args
        assert call_args.kwargs["caption"] == expected_caption

    @pytest.mark.asyncio
    async def test_empty_caption_not_uploaded(self, temp_reel_dir, mock_publisher):
        """Test that empty captions are caught before upload.

        This is the expected behavior after the fix.
        """
        # Overwrite with empty caption
        (temp_reel_dir / "caption+hashtags.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")

        # Run artifact audit - should fail
        audit = audit_reel_artifacts(temp_reel_dir)

        # Empty captions should cause audit to fail
        assert audit.is_valid is False

        # Publisher should not be called for invalid reels
        mock_publisher.publish_reel.assert_not_called()


class TestCaptionRegeneration:
    """Tests for caption regeneration when missing or empty."""

    @pytest.fixture
    def temp_reel_dir(self):
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)
        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_regenerate_caption_from_metadata(self, temp_reel_dir):
        """Test caption regeneration from metadata."""
        from socials_automator.cli.reel.artifacts import _regenerate_caption

        metadata = {
            "topic": "5 AI Tools for Productivity",
            "hook": "Want to save hours every week?",
        }
        (temp_reel_dir / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )

        success = _regenerate_caption(temp_reel_dir, use_ai=False)

        assert success is True
        caption = (temp_reel_dir / "caption.txt").read_text(encoding="utf-8")
        assert len(caption) > 10

    def test_regenerate_caption_from_script(self, temp_reel_dir):
        """Test caption regeneration from script.json."""
        from socials_automator.cli.reel.artifacts import _regenerate_caption

        script = {
            "title": "5 Amazing AI Tools",
            "hook": {"text": "Stop wasting time on repetitive tasks!"},
        }
        (temp_reel_dir / "script.json").write_text(json.dumps(script), encoding="utf-8")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")

        success = _regenerate_caption(temp_reel_dir, use_ai=False)

        assert success is True
        caption = (temp_reel_dir / "caption.txt").read_text(encoding="utf-8")
        assert "Stop wasting time" in caption or "AI Tools" in caption


class TestCaptionRaceConditions:
    """Tests for race conditions in caption handling.

    Verifies that async operations don't cause caption loss.
    """

    @pytest.fixture
    def temp_reel_dir(self):
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)

        (reel_dir / "final.mp4").write_bytes(b"fake video")
        (reel_dir / "metadata.json").write_text("{}", encoding="utf-8")
        (reel_dir / "thumbnail.jpg").write_bytes(b"fake image")
        (reel_dir / "caption.txt").write_text("Test caption", encoding="utf-8")
        (reel_dir / "caption+hashtags.txt").write_text(
            "Test caption #test", encoding="utf-8"
        )

        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_caption_not_lost_in_async_read(self, temp_reel_dir):
        """Test that caption is not lost during async operations."""
        import asyncio

        async def read_caption():
            """Simulate async caption reading."""
            await asyncio.sleep(0.01)  # Simulate async delay
            return (temp_reel_dir / "caption+hashtags.txt").read_text(encoding="utf-8")

        # Read caption multiple times concurrently
        results = await asyncio.gather(*[read_caption() for _ in range(10)])

        # All reads should return the same caption
        assert all(r == "Test caption #test" for r in results)

    @pytest.mark.asyncio
    async def test_caption_local_variable_isolation(self, temp_reel_dir):
        """Test that caption as local variable is not affected by other operations."""
        import asyncio

        caption = (temp_reel_dir / "caption+hashtags.txt").read_text(encoding="utf-8")

        # Simulate modifying the file while upload is in progress
        async def modify_file():
            await asyncio.sleep(0.01)
            (temp_reel_dir / "caption+hashtags.txt").write_text(
                "Modified caption", encoding="utf-8"
            )

        async def use_caption():
            await asyncio.sleep(0.02)
            return caption  # Should still be original value

        # Run both concurrently
        _, result = await asyncio.gather(modify_file(), use_caption())

        # Caption variable should still have original value
        assert result == "Test caption #test"


class TestCaptionRegenerationDuringUpload:
    """Tests for automatic caption regeneration during upload when caption is empty."""

    @pytest.fixture
    def temp_reel_dir(self):
        temp_dir = tempfile.mkdtemp()
        reel_dir = Path(temp_dir) / "19-001-test-reel"
        reel_dir.mkdir(parents=True)

        # Create required files
        (reel_dir / "final.mp4").write_bytes(b"fake video")
        (reel_dir / "thumbnail.jpg").write_bytes(b"fake image")

        yield reel_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_empty_caption_triggers_regeneration(self, temp_reel_dir):
        """Test that empty caption attempts to regenerate from metadata."""
        from socials_automator.cli.reel.artifacts import (
            _regenerate_caption_with_hashtags,
            _regenerate_caption,
        )

        # Create empty caption files but valid metadata
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "caption+hashtags.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "metadata.json").write_text(
            json.dumps({
                "topic": "5 AI Tools That Will Change Your Workflow",
                "hook": "Ready to boost your productivity?",
                "caption": "Discover these amazing AI tools #AI #Productivity",
            }),
            encoding="utf-8",
        )

        # Simulate the regeneration logic from service.py
        caption_hashtags_path = temp_reel_dir / "caption+hashtags.txt"
        caption_path = temp_reel_dir / "caption.txt"

        # Read caption (will be empty)
        caption = caption_hashtags_path.read_text(encoding="utf-8").strip()
        assert len(caption) < 10  # Empty

        # Attempt regeneration
        if _regenerate_caption_with_hashtags(temp_reel_dir, None):
            caption = caption_hashtags_path.read_text(encoding="utf-8").strip()
        elif _regenerate_caption(temp_reel_dir, use_ai=False):
            caption = caption_path.read_text(encoding="utf-8").strip()

        # Caption should now be populated
        assert len(caption) >= 10

    def test_regeneration_from_script_json(self, temp_reel_dir):
        """Test caption regeneration from script.json when metadata is insufficient."""
        from socials_automator.cli.reel.artifacts import _regenerate_caption

        # Create empty caption and minimal metadata
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")

        # Create script.json with content
        (temp_reel_dir / "script.json").write_text(
            json.dumps({
                "title": "How to Use ChatGPT for Email Automation",
                "hook": {"text": "Stop spending hours on email!"},
            }),
            encoding="utf-8",
        )

        success = _regenerate_caption(temp_reel_dir, use_ai=False)

        assert success is True
        caption = (temp_reel_dir / "caption.txt").read_text(encoding="utf-8")
        assert len(caption) >= 10
        assert "email" in caption.lower() or "ChatGPT" in caption

    def test_regeneration_fails_gracefully(self, temp_reel_dir):
        """Test that regeneration fails gracefully when no source is available."""
        from socials_automator.cli.reel.artifacts import _regenerate_caption

        # Create empty caption and empty metadata
        (temp_reel_dir / "caption.txt").write_text("", encoding="utf-8")
        (temp_reel_dir / "metadata.json").write_text("{}", encoding="utf-8")

        success = _regenerate_caption(temp_reel_dir, use_ai=False)

        # Should fail - no content source
        assert success is False

    def test_whitespace_caption_triggers_regeneration(self, temp_reel_dir):
        """Test that whitespace-only captions trigger regeneration."""
        from socials_automator.cli.reel.artifacts import _regenerate_caption

        # Create whitespace-only caption files
        (temp_reel_dir / "caption.txt").write_text("   \n\n\t  ", encoding="utf-8")
        (temp_reel_dir / "caption+hashtags.txt").write_text("   ", encoding="utf-8")
        # Use caption field in metadata for reliable regeneration
        (temp_reel_dir / "metadata.json").write_text(
            json.dumps({
                "topic": "Top 3 Free AI Tools",
                "caption": "These free AI tools are amazing! Check them out now!",
            }),
            encoding="utf-8",
        )

        caption_path = temp_reel_dir / "caption.txt"
        caption = caption_path.read_text(encoding="utf-8").strip()

        # Should be considered empty after strip
        assert len(caption) < 10

        # Regeneration should work (uses metadata.caption field)
        success = _regenerate_caption(temp_reel_dir, use_ai=False)
        assert success is True

        caption = caption_path.read_text(encoding="utf-8").strip()
        assert len(caption) >= 10
