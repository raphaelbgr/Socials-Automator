"""Extensive tests for hashtag sanitization module.

Tests cover:
- HashtagSanitizer class methods
- Edge cases (empty, unicode, mixed content)
- SanitizeResult dataclass
- validate_hashtags_in_caption function
- validate_reel_hashtags function
- remove_hashtags_from_caption function
"""

import pytest
from pathlib import Path
import tempfile

from socials_automator.hashtag import (
    INSTAGRAM_MAX_HASHTAGS,
    DEFAULT_FALLBACK_HASHTAGS,
    HashtagSanitizer,
    HashtagValidationResult,
    validate_hashtags_in_caption,
)
from socials_automator.hashtag.sanitizer import SanitizeResult
from socials_automator.hashtag.validator import (
    validate_reel_hashtags,
    remove_hashtags_from_caption,
)


class TestHashtagConstants:
    """Test hashtag constants."""

    def test_instagram_max_hashtags_is_5(self):
        """Instagram limit is 5 as of Dec 2025."""
        assert INSTAGRAM_MAX_HASHTAGS == 5

    def test_default_fallback_hashtags_exist(self):
        """Default fallback hashtags are defined."""
        assert len(DEFAULT_FALLBACK_HASHTAGS) > 0
        assert all(isinstance(tag, str) for tag in DEFAULT_FALLBACK_HASHTAGS)

    def test_fallback_hashtags_are_valid(self):
        """Fallback hashtags don't include # prefix."""
        for tag in DEFAULT_FALLBACK_HASHTAGS:
            assert not tag.startswith("#"), f"Fallback tag '{tag}' should not start with #"


class TestSanitizeResult:
    """Test SanitizeResult dataclass."""

    def test_was_modified_true_when_different(self):
        """was_modified is True when text changed."""
        result = SanitizeResult(
            original_text="Caption #one #two",
            sanitized_text="Caption #one",
            original_count=2,
            final_count=1,
            removed_count=1,
            removed_hashtags=["#two"],
            kept_hashtags=["#one"],
        )
        assert result.was_modified is True

    def test_was_modified_false_when_same(self):
        """was_modified is False when text unchanged."""
        result = SanitizeResult(
            original_text="Caption #one",
            sanitized_text="Caption #one",
            original_count=1,
            final_count=1,
            removed_count=0,
            removed_hashtags=[],
            kept_hashtags=["#one"],
        )
        assert result.was_modified is False

    def test_exceeded_limit_true_when_removed(self):
        """exceeded_limit is True when hashtags were removed."""
        result = SanitizeResult(
            original_text="text",
            sanitized_text="text",
            original_count=8,
            final_count=5,
            removed_count=3,
            removed_hashtags=["#a", "#b", "#c"],
            kept_hashtags=["#1", "#2", "#3", "#4", "#5"],
        )
        assert result.exceeded_limit is True

    def test_exceeded_limit_false_when_no_removal(self):
        """exceeded_limit is False when nothing removed."""
        result = SanitizeResult(
            original_text="text",
            sanitized_text="text",
            original_count=3,
            final_count=3,
            removed_count=0,
            removed_hashtags=[],
            kept_hashtags=["#1", "#2", "#3"],
        )
        assert result.exceeded_limit is False


class TestHashtagSanitizerInit:
    """Test HashtagSanitizer initialization."""

    def test_default_max_hashtags(self):
        """Default max_hashtags is INSTAGRAM_MAX_HASHTAGS."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.max_hashtags == INSTAGRAM_MAX_HASHTAGS

    def test_custom_max_hashtags(self):
        """Can set custom max_hashtags."""
        sanitizer = HashtagSanitizer(max_hashtags=10)
        assert sanitizer.max_hashtags == 10

    def test_zero_max_hashtags(self):
        """Can set max_hashtags to zero."""
        sanitizer = HashtagSanitizer(max_hashtags=0)
        assert sanitizer.max_hashtags == 0


class TestHashtagSanitizerCountHashtags:
    """Test HashtagSanitizer.count_hashtags method."""

    def test_count_no_hashtags(self):
        """Count zero when no hashtags."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("No hashtags here") == 0

    def test_count_single_hashtag(self):
        """Count single hashtag."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("Text #hashtag") == 1

    def test_count_multiple_hashtags(self):
        """Count multiple hashtags."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("#one #two #three") == 3

    def test_count_hashtags_in_middle(self):
        """Count hashtags mixed with text."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("Text #with #hashtags mixed in") == 2

    def test_count_empty_string(self):
        """Count zero for empty string."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("") == 0

    def test_count_none_returns_zero(self):
        """Count zero for None (empty check)."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags(None) == 0

    def test_count_hashtags_with_numbers(self):
        """Count hashtags containing numbers."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("#tag1 #2024 #ai4life") == 3

    def test_count_hashtags_with_underscores(self):
        """Count hashtags containing underscores."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("#hello_world #ai_tools") == 2

    def test_count_unicode_hashtags(self):
        """Count hashtags with unicode characters."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("#cafe #technology") == 2

    def test_count_consecutive_hashtags(self):
        """Count consecutive hashtags without spaces."""
        sanitizer = HashtagSanitizer()
        # Each hashtag needs at least one char after #
        assert sanitizer.count_hashtags("#one#two#three") == 3

    def test_count_hashtag_at_start(self):
        """Count hashtag at start of text."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("#start of text") == 1

    def test_count_hashtag_at_end(self):
        """Count hashtag at end of text."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("end of text #hashtag") == 1

    def test_count_only_hash_symbol(self):
        """Single # symbol is not a hashtag."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.count_hashtags("Price is $100 or # something") == 0


class TestHashtagSanitizerExtractHashtags:
    """Test HashtagSanitizer.extract_hashtags method."""

    def test_extract_no_hashtags(self):
        """Extract empty list when no hashtags."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.extract_hashtags("No hashtags here") == []

    def test_extract_single_hashtag(self):
        """Extract single hashtag."""
        sanitizer = HashtagSanitizer()
        hashtags = sanitizer.extract_hashtags("Caption #one")
        assert hashtags == ["#one"]

    def test_extract_multiple_hashtags(self):
        """Extract multiple hashtags in order."""
        sanitizer = HashtagSanitizer()
        hashtags = sanitizer.extract_hashtags("Caption #one #two #three")
        assert hashtags == ["#one", "#two", "#three"]

    def test_extract_preserves_order(self):
        """Extract hashtags in original order."""
        sanitizer = HashtagSanitizer()
        hashtags = sanitizer.extract_hashtags("Start #alpha middle #beta end #gamma")
        assert hashtags == ["#alpha", "#beta", "#gamma"]

    def test_extract_empty_string(self):
        """Extract empty list for empty string."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.extract_hashtags("") == []

    def test_extract_none_returns_empty(self):
        """Extract empty list for None."""
        sanitizer = HashtagSanitizer()
        assert sanitizer.extract_hashtags(None) == []

    def test_extract_includes_hash_prefix(self):
        """Extracted hashtags include # prefix."""
        sanitizer = HashtagSanitizer()
        hashtags = sanitizer.extract_hashtags("#test")
        assert hashtags[0].startswith("#")


class TestHashtagSanitizerTrimHashtags:
    """Test HashtagSanitizer.trim_hashtags method."""

    def test_trim_under_limit_no_change(self):
        """No change when under limit."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        result = sanitizer.trim_hashtags("Caption #one #two #three")
        assert not result.was_modified
        assert result.final_count == 3
        assert result.removed_count == 0
        assert result.removed_hashtags == []

    def test_trim_at_limit_no_change(self):
        """No change when exactly at limit."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        result = sanitizer.trim_hashtags("Caption #1 #2 #3 #4 #5")
        assert not result.was_modified
        assert result.final_count == 5
        assert result.removed_count == 0

    def test_trim_over_limit_removes_excess(self):
        """Remove excess hashtags when over limit."""
        sanitizer = HashtagSanitizer(max_hashtags=3)
        text = "Caption #one #two #three #four #five"
        result = sanitizer.trim_hashtags(text)

        assert result.was_modified
        assert result.original_count == 5
        assert result.final_count == 3
        assert result.removed_count == 2
        assert result.removed_hashtags == ["#four", "#five"]
        assert "#one" in result.sanitized_text
        assert "#two" in result.sanitized_text
        assert "#three" in result.sanitized_text
        assert "#four" not in result.sanitized_text
        assert "#five" not in result.sanitized_text

    def test_trim_keeps_first_n_hashtags(self):
        """Keeps first N hashtags, removes rest."""
        sanitizer = HashtagSanitizer(max_hashtags=2)
        result = sanitizer.trim_hashtags("#first #second #third #fourth")
        assert result.kept_hashtags == ["#first", "#second"]
        assert result.removed_hashtags == ["#third", "#fourth"]

    def test_trim_empty_string(self):
        """Handle empty string."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.trim_hashtags("")
        assert result.original_text == ""
        assert result.sanitized_text == ""
        assert result.original_count == 0
        assert not result.was_modified

    def test_trim_no_hashtags(self):
        """Handle text with no hashtags."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.trim_hashtags("Just plain text")
        assert result.original_text == "Just plain text"
        assert result.sanitized_text == "Just plain text"
        assert result.original_count == 0
        assert not result.was_modified

    def test_trim_custom_max_count(self):
        """Override max_count in trim_hashtags call."""
        sanitizer = HashtagSanitizer(max_hashtags=10)  # Default 10
        result = sanitizer.trim_hashtags("#1 #2 #3 #4 #5", max_count=2)  # Override to 2
        assert result.final_count == 2
        assert result.removed_count == 3

    def test_trim_cleans_whitespace(self):
        """Trimming cleans up extra whitespace."""
        sanitizer = HashtagSanitizer(max_hashtags=1)
        result = sanitizer.trim_hashtags("Caption  #one  #two  #three")
        # Should not have double spaces after removal
        assert "  " not in result.sanitized_text

    def test_trim_preserves_caption_text(self):
        """Caption text is preserved when trimming."""
        sanitizer = HashtagSanitizer(max_hashtags=1)
        result = sanitizer.trim_hashtags("My great caption with emojis #one #two #three")
        assert "My great caption with emojis" in result.sanitized_text
        assert "#one" in result.sanitized_text


class TestHashtagSanitizerRemoveAllHashtags:
    """Test HashtagSanitizer.remove_all_hashtags method."""

    def test_remove_all_single_hashtag(self):
        """Remove single hashtag."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.remove_all_hashtags("Caption #hashtag")
        assert result.sanitized_text == "Caption"
        assert result.removed_count == 1

    def test_remove_all_multiple_hashtags(self):
        """Remove all multiple hashtags."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.remove_all_hashtags("Caption #one #two #three")
        assert result.sanitized_text == "Caption"
        assert result.final_count == 0
        assert result.removed_count == 3

    def test_remove_all_preserves_caption(self):
        """Caption text is preserved."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.remove_all_hashtags("My caption here #tag1 #tag2")
        assert result.sanitized_text == "My caption here"

    def test_remove_all_empty_string(self):
        """Handle empty string."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.remove_all_hashtags("")
        assert result.sanitized_text == ""
        assert not result.was_modified

    def test_remove_all_no_hashtags(self):
        """Handle text without hashtags."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.remove_all_hashtags("Just text no hashtags")
        assert result.sanitized_text == "Just text no hashtags"
        assert not result.was_modified

    def test_remove_all_only_hashtags(self):
        """Handle text that is only hashtags."""
        sanitizer = HashtagSanitizer()
        result = sanitizer.remove_all_hashtags("#one #two #three")
        assert result.sanitized_text == ""


class TestHashtagSanitizerValidate:
    """Test HashtagSanitizer.validate method."""

    def test_validate_under_limit(self):
        """Validate passes when under limit."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        is_valid, count, message = sanitizer.validate("#one #two #three")
        assert is_valid is True
        assert count == 3
        assert "OK" in message

    def test_validate_at_limit(self):
        """Validate passes when at limit."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        is_valid, count, message = sanitizer.validate("#1 #2 #3 #4 #5")
        assert is_valid is True
        assert count == 5

    def test_validate_over_limit(self):
        """Validate fails when over limit."""
        sanitizer = HashtagSanitizer(max_hashtags=3)
        is_valid, count, message = sanitizer.validate("#one #two #three #four #five")
        assert is_valid is False
        assert count == 5
        assert "Too many" in message

    def test_validate_no_hashtags(self):
        """Validate passes with no hashtags."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        is_valid, count, message = sanitizer.validate("No hashtags")
        assert is_valid is True
        assert count == 0


class TestHashtagValidationResult:
    """Test HashtagValidationResult dataclass."""

    def test_message_when_valid(self):
        """Message shows OK when valid."""
        result = HashtagValidationResult(
            is_valid=True,
            original_count=3,
            final_count=3,
            was_trimmed=False,
            removed_hashtags=[],
            caption_before="text",
            caption_after="text",
        )
        assert "[OK]" in result.message

    def test_message_when_trimmed(self):
        """Message shows trimming info."""
        result = HashtagValidationResult(
            is_valid=True,
            original_count=8,
            final_count=5,
            was_trimmed=True,
            removed_hashtags=["#a", "#b", "#c"],
            caption_before="text",
            caption_after="text",
        )
        assert "Trimmed" in result.message or "8" in result.message

    def test_message_when_error(self):
        """Message shows error."""
        result = HashtagValidationResult(
            is_valid=False,
            original_count=0,
            final_count=0,
            was_trimmed=False,
            removed_hashtags=[],
            caption_before="",
            caption_after="",
            error="Some error",
        )
        assert "error" in result.message.lower()

    def test_message_no_hashtags(self):
        """Message when no hashtags present."""
        result = HashtagValidationResult(
            is_valid=True,
            original_count=0,
            final_count=0,
            was_trimmed=False,
            removed_hashtags=[],
            caption_before="text",
            caption_after="text",
        )
        assert "[OK]" in result.message


class TestValidateHashtagsInCaption:
    """Test the validate_hashtags_in_caption function."""

    def test_valid_caption_returns_valid(self):
        """Test valid caption with few hashtags."""
        result = validate_hashtags_in_caption("Caption #one #two #three")
        assert result.is_valid
        assert not result.was_trimmed
        assert result.original_count == 3

    def test_over_limit_auto_trims(self):
        """Test over-limit caption is auto-trimmed."""
        caption = "Caption #1 #2 #3 #4 #5 #6 #7 #8"
        result = validate_hashtags_in_caption(caption, auto_trim=True)

        assert result.is_valid
        assert result.was_trimmed
        assert result.final_count == 5  # Instagram limit
        assert result.original_count == 8
        assert len(result.removed_hashtags) == 3

    def test_over_limit_no_trim_returns_invalid(self):
        """Test over-limit without auto_trim returns invalid."""
        caption = "Caption #1 #2 #3 #4 #5 #6 #7 #8"
        result = validate_hashtags_in_caption(caption, auto_trim=False)

        assert not result.is_valid
        assert not result.was_trimmed
        assert result.error is not None

    def test_custom_max_hashtags(self):
        """Test custom max_hashtags parameter."""
        caption = "Caption #1 #2 #3 #4 #5"
        result = validate_hashtags_in_caption(caption, max_hashtags=3, auto_trim=True)

        assert result.is_valid
        assert result.was_trimmed
        assert result.final_count == 3
        assert len(result.removed_hashtags) == 2

    def test_no_hashtags_is_valid(self):
        """Test caption with no hashtags is valid."""
        result = validate_hashtags_in_caption("Just a caption with no hashtags")
        assert result.is_valid
        assert result.original_count == 0
        assert not result.was_trimmed

    def test_empty_caption(self):
        """Test empty caption."""
        result = validate_hashtags_in_caption("")
        assert result.is_valid
        assert result.original_count == 0

    def test_caption_unchanged_when_valid(self):
        """Caption text unchanged when under limit."""
        caption = "My caption #one #two"
        result = validate_hashtags_in_caption(caption)
        assert result.caption_after == caption

    def test_caption_trimmed_text_is_correct(self):
        """Trimmed caption has correct content."""
        caption = "My caption #keep1 #keep2 #remove1 #remove2"
        result = validate_hashtags_in_caption(caption, max_hashtags=2, auto_trim=True)
        assert "#keep1" in result.caption_after
        assert "#keep2" in result.caption_after
        assert "#remove1" not in result.caption_after
        assert "#remove2" not in result.caption_after


class TestValidateReelHashtags:
    """Test the validate_reel_hashtags function."""

    def test_missing_caption_file(self):
        """Test with missing caption file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reel_path = Path(tmpdir)
            result = validate_reel_hashtags(reel_path)
            assert not result.is_valid
            assert "not found" in result.error

    def test_valid_caption_file(self):
        """Test with valid caption file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reel_path = Path(tmpdir)
            caption_path = reel_path / "caption+hashtags.txt"
            caption_path.write_text("Caption #one #two #three", encoding="utf-8")

            result = validate_reel_hashtags(reel_path)
            assert result.is_valid
            assert not result.was_trimmed

    def test_trims_and_updates_file(self):
        """Test trimming updates the file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reel_path = Path(tmpdir)
            caption_path = reel_path / "caption+hashtags.txt"
            original = "Caption #1 #2 #3 #4 #5 #6 #7 #8"
            caption_path.write_text(original, encoding="utf-8")

            result = validate_reel_hashtags(reel_path, auto_trim=True, update_file=True)

            assert result.is_valid
            assert result.was_trimmed
            # File should be updated
            updated_content = caption_path.read_text(encoding="utf-8")
            assert updated_content != original
            # Should have 5 hashtags now
            from socials_automator.hashtag import HashtagSanitizer
            sanitizer = HashtagSanitizer()
            assert sanitizer.count_hashtags(updated_content) == 5

    def test_no_update_when_update_file_false(self):
        """Test file not updated when update_file=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reel_path = Path(tmpdir)
            caption_path = reel_path / "caption+hashtags.txt"
            original = "Caption #1 #2 #3 #4 #5 #6 #7 #8"
            caption_path.write_text(original, encoding="utf-8")

            result = validate_reel_hashtags(reel_path, auto_trim=True, update_file=False)

            assert result.is_valid
            assert result.was_trimmed
            # File should NOT be updated
            assert caption_path.read_text(encoding="utf-8") == original


class TestRemoveHashtagsFromCaption:
    """Test the remove_hashtags_from_caption function."""

    def test_remove_all(self):
        """Remove all hashtags."""
        caption = "My caption #one #two #three"
        result = remove_hashtags_from_caption(caption)
        assert result == "My caption"

    def test_remove_preserves_text(self):
        """Text is preserved."""
        caption = "Great video about AI! #ai #tech #viral"
        result = remove_hashtags_from_caption(caption)
        assert result == "Great video about AI!"

    def test_remove_empty_string(self):
        """Handle empty string."""
        result = remove_hashtags_from_caption("")
        assert result == ""

    def test_remove_no_hashtags(self):
        """Handle text without hashtags."""
        caption = "Just plain text"
        result = remove_hashtags_from_caption(caption)
        assert result == "Just plain text"

    def test_remove_only_hashtags(self):
        """Handle text that is only hashtags."""
        caption = "#one #two #three"
        result = remove_hashtags_from_caption(caption)
        assert result == ""


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_multiline_caption(self):
        """Handle multiline captions."""
        sanitizer = HashtagSanitizer(max_hashtags=2)
        caption = """My awesome caption.

Line 2 here.

#hashtag1 #hashtag2 #hashtag3 #hashtag4"""
        result = sanitizer.trim_hashtags(caption)
        assert result.was_modified
        assert result.final_count == 2
        assert "My awesome caption" in result.sanitized_text

    def test_hashtags_with_newlines(self):
        """Handle hashtags on separate lines."""
        sanitizer = HashtagSanitizer(max_hashtags=2)
        caption = """Caption text

#one
#two
#three
#four"""
        result = sanitizer.trim_hashtags(caption)
        assert result.was_modified
        assert result.final_count == 2

    def test_very_long_caption(self):
        """Handle very long captions."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        caption = "A" * 2000 + " #hashtag1 #hashtag2"
        result = sanitizer.trim_hashtags(caption)
        assert not result.was_modified
        assert "A" * 100 in result.sanitized_text

    def test_unicode_in_caption(self):
        """Handle unicode characters in caption."""
        sanitizer = HashtagSanitizer(max_hashtags=5)
        caption = "Cafe with coffee #cafe #morning"
        result = sanitizer.trim_hashtags(caption)
        assert "Cafe" in result.sanitized_text

    def test_special_characters_near_hashtags(self):
        """Handle special characters near hashtags."""
        sanitizer = HashtagSanitizer()
        # Parentheses, quotes, etc.
        assert sanitizer.count_hashtags("(#hashtag)") == 1
        assert sanitizer.count_hashtags('"#hashtag"') == 1
        assert sanitizer.count_hashtags("#hashtag!") == 1
        assert sanitizer.count_hashtags("#hashtag?") == 1

    def test_hashtag_like_but_not_hashtag(self):
        """Things that look like hashtags but aren't."""
        sanitizer = HashtagSanitizer()
        # Email-like format - #domain is still matched as hashtag
        assert sanitizer.count_hashtags("test@#domain.com") == 1  # #domain matches
        # Color codes - #FF0000 matches (alphanumeric after #)
        assert sanitizer.count_hashtags("Color is #FF0000") == 1
        # C# (no char after #) - not a hashtag
        assert sanitizer.count_hashtags("I love C#") == 0
        # Just a # symbol
        assert sanitizer.count_hashtags("Price is $100 #") == 0

    def test_duplicate_hashtags(self):
        """Handle duplicate hashtags."""
        sanitizer = HashtagSanitizer(max_hashtags=3)
        caption = "#same #same #same #different #another"
        result = sanitizer.trim_hashtags(caption)
        # All #same are the same hashtag but appear 3 times
        assert result.final_count == 3
        # First 3 are kept
        assert result.kept_hashtags == ["#same", "#same", "#same"]

    def test_hashtag_at_very_end(self):
        """Handle hashtag at very end without trailing space."""
        sanitizer = HashtagSanitizer(max_hashtags=1)
        caption = "Caption #one #two"
        result = sanitizer.trim_hashtags(caption)
        assert result.was_modified
        assert result.sanitized_text.endswith("#one") or result.sanitized_text.strip().endswith("#one")


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_validation_and_trim_flow(self):
        """Test complete flow: validate -> trim -> verify."""
        caption = "Great content! Check this out #viral #fyp #trending #ai #tech #reels #instagram #follow"

        # Step 1: Validate shows over limit
        result1 = validate_hashtags_in_caption(caption, auto_trim=False)
        assert not result1.is_valid
        assert result1.original_count == 8

        # Step 2: Auto-trim
        result2 = validate_hashtags_in_caption(caption, auto_trim=True)
        assert result2.is_valid
        assert result2.final_count == 5
        assert result2.was_trimmed

        # Step 3: Validate trimmed caption
        result3 = validate_hashtags_in_caption(result2.caption_after)
        assert result3.is_valid
        assert result3.original_count == 5
        assert not result3.was_trimmed

    def test_fallback_remove_all_flow(self):
        """Test fallback flow: try with hashtags, fail, remove all."""
        caption = "My post #one #two #three #four #five"

        # Simulate: first upload failed, now remove all hashtags
        no_hashtags = remove_hashtags_from_caption(caption)
        assert no_hashtags == "My post"

        # Validate no hashtags version
        result = validate_hashtags_in_caption(no_hashtags)
        assert result.is_valid
        assert result.original_count == 0
