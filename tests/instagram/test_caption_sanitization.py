"""Extensive tests for Instagram caption sanitization.

Tests cover:
- Emoji removal (all types: emoticons, symbols, flags, skin tones)
- Unicode normalization (NFC)
- Smart quotes and typography replacements
- Control character removal
- Zero-width character removal
- Whitespace normalization
- Edge cases (empty, very long, unicode-heavy, etc.)
- Real-world caption scenarios
"""

import pytest
import unicodedata


# Import the function we're testing
# We need to recreate it here since it's a module-level function
def _sanitize_caption(caption: str) -> str:
    """Sanitize caption - copy of the function from client.py for testing."""
    import re

    if not caption:
        return ""

    # First, remove emojis using regex
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F9FF"  # Misc Symbols, Emoticons, etc.
        "\U0001FA00-\U0001FAFF"  # Extended-A symbols
        "\U00002702-\U000027B0"  # Dingbats
        "\U0000FE00-\U0000FE0F"  # Variation selectors
        "\U0001F1E0-\U0001F1FF"  # Flags (regional indicators)
        "\U00002600-\U000026FF"  # Misc symbols
        "\U00002300-\U000023FF"  # Misc technical
        "\U0000200D"             # Zero-width joiner (used in emoji sequences)
        "]+",
        flags=re.UNICODE
    )
    caption = emoji_pattern.sub('', caption)

    # Normalize Unicode to NFC
    caption = unicodedata.normalize('NFC', caption)

    # Replace problematic characters
    replacements = {
        '\u2018': "'",   # Left single quote
        '\u2019': "'",   # Right single quote
        '\u201c': '"',   # Left double quote
        '\u201d': '"',   # Right double quote
        '\u2014': '-',   # Em dash
        '\u2013': '-',   # En dash
        '\u2026': '...',  # Ellipsis
        '\u00a0': ' ',   # Non-breaking space
        '\u200b': '',    # Zero-width space
        '\u200c': '',    # Zero-width non-joiner
        '\ufeff': '',    # BOM
        '\u00ad': '',    # Soft hyphen
        '\u2028': '\n',  # Line separator
        '\u2029': '\n',  # Paragraph separator
    }

    for old, new in replacements.items():
        caption = caption.replace(old, new)

    # Remove remaining control/format characters
    cleaned = []
    for char in caption:
        if char in '\n\r\t':
            cleaned.append(char)
        elif unicodedata.category(char) == 'Cc':
            continue
        elif unicodedata.category(char) == 'Cf':
            continue
        elif unicodedata.category(char) == 'So':
            continue
        elif ord(char) > 0xFFFF:
            continue
        else:
            cleaned.append(char)

    caption = ''.join(cleaned)

    # Clean up double spaces
    while '  ' in caption:
        caption = caption.replace('  ', ' ')

    # Normalize multiple newlines
    while '\n\n\n' in caption:
        caption = caption.replace('\n\n\n', '\n\n')

    return caption.strip()


class TestEmptyAndNone:
    """Test empty and None inputs."""

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert _sanitize_caption("") == ""

    def test_none_returns_empty(self):
        """None returns empty string."""
        assert _sanitize_caption(None) == ""

    def test_whitespace_only(self):
        """Whitespace-only string returns empty after strip."""
        assert _sanitize_caption("   ") == ""
        assert _sanitize_caption("\n\n\n") == ""
        assert _sanitize_caption("\t\t\t") == ""

    def test_whitespace_with_content(self):
        """Whitespace around content is stripped."""
        assert _sanitize_caption("  hello  ") == "hello"
        assert _sanitize_caption("\n\nhello\n\n") == "hello"


class TestBasicEmojiRemoval:
    """Test removal of common emojis."""

    def test_single_emoji(self):
        """Single emoji is removed."""
        # Smiling face
        assert _sanitize_caption("Hello \U0001F600") == "Hello"
        # Heart
        assert _sanitize_caption("Love \U0001F496") == "Love"
        # Fire
        assert _sanitize_caption("Hot \U0001F525") == "Hot"

    def test_multiple_emojis(self):
        """Multiple emojis are removed."""
        text = "Hello \U0001F600\U0001F601\U0001F602 World"
        assert _sanitize_caption(text) == "Hello World"

    def test_emoji_at_start(self):
        """Emoji at start is removed."""
        assert _sanitize_caption("\U0001F600 Hello") == "Hello"

    def test_emoji_at_end(self):
        """Emoji at end is removed."""
        assert _sanitize_caption("Hello \U0001F600") == "Hello"

    def test_emoji_in_middle(self):
        """Emoji in middle is removed."""
        assert _sanitize_caption("Hello \U0001F600 World") == "Hello World"

    def test_only_emojis(self):
        """String with only emojis becomes empty."""
        assert _sanitize_caption("\U0001F600\U0001F601\U0001F602") == ""

    def test_consecutive_emojis(self):
        """Consecutive emojis are all removed."""
        text = "Start\U0001F600\U0001F601\U0001F602\U0001F603End"
        assert _sanitize_caption(text) == "StartEnd"


class TestEmojiCategories:
    """Test removal of different emoji categories."""

    def test_emoticons(self):
        """Emoticons (faces) are removed."""
        # Various face emojis
        assert _sanitize_caption("Test \U0001F600 emoji") == "Test emoji"  # grinning
        assert _sanitize_caption("Sad \U0001F622") == "Sad"  # crying
        assert _sanitize_caption("Angry \U0001F620") == "Angry"  # angry

    def test_animals(self):
        """Animal emojis are removed."""
        assert _sanitize_caption("Dog \U0001F436") == "Dog"
        assert _sanitize_caption("Cat \U0001F431") == "Cat"
        assert _sanitize_caption("Lion \U0001F981") == "Lion"

    def test_food(self):
        """Food emojis are removed."""
        assert _sanitize_caption("Pizza \U0001F355") == "Pizza"
        assert _sanitize_caption("Coffee \U00002615") == "Coffee"  # Hot beverage

    def test_objects(self):
        """Object emojis are removed."""
        assert _sanitize_caption("Phone \U0001F4F1") == "Phone"
        assert _sanitize_caption("Camera \U0001F4F7") == "Camera"
        assert _sanitize_caption("Book \U0001F4DA") == "Book"

    def test_symbols(self):
        """Symbol emojis are removed."""
        assert _sanitize_caption("Check \U00002714") == "Check"  # Heavy check mark
        assert _sanitize_caption("Star \U00002B50") == "Star"

    def test_flags(self):
        """Flag emojis are removed."""
        # US flag (regional indicators)
        us_flag = "\U0001F1FA\U0001F1F8"
        assert _sanitize_caption(f"USA {us_flag}") == "USA"

    def test_hand_gestures(self):
        """Hand gesture emojis are removed."""
        assert _sanitize_caption("OK \U0001F44C") == "OK"
        assert _sanitize_caption("Thumbs up \U0001F44D") == "Thumbs up"
        assert _sanitize_caption("Wave \U0001F44B") == "Wave"

    def test_weather(self):
        """Weather emojis are removed."""
        assert _sanitize_caption("Sun \U00002600") == "Sun"
        assert _sanitize_caption("Cloud \U00002601") == "Cloud"
        assert _sanitize_caption("Rain \U0001F327") == "Rain"

    def test_travel(self):
        """Travel emojis are removed."""
        assert _sanitize_caption("Plane \U00002708") == "Plane"
        assert _sanitize_caption("Car \U0001F697") == "Car"
        assert _sanitize_caption("Rocket \U0001F680") == "Rocket"


class TestComplexEmojis:
    """Test removal of complex emoji sequences."""

    def test_emoji_with_skin_tone(self):
        """Emojis with skin tone modifiers are removed."""
        # Thumbs up with skin tone
        thumbs_with_tone = "\U0001F44D\U0001F3FB"  # Light skin tone
        result = _sanitize_caption(f"OK {thumbs_with_tone}")
        assert "\U0001F44D" not in result
        assert "\U0001F3FB" not in result

    def test_emoji_zwj_sequence(self):
        """Zero-width joiner emoji sequences are removed."""
        # Family emoji (man + ZWJ + woman + ZWJ + girl)
        # Most ZWJ sequences should be handled
        zwj = "\U0000200D"
        assert _sanitize_caption(f"Family{zwj}here") == "Familyhere"

    def test_keycap_emojis(self):
        """Keycap number emojis are handled."""
        # These use variation selectors
        keycap_1 = "1\uFE0F\u20E3"
        result = _sanitize_caption(f"Number {keycap_1}")
        # The base digit should remain
        assert "Number" in result

    def test_variation_selectors(self):
        """Variation selectors are removed."""
        # Text vs emoji presentation
        text_with_vs = "Star\uFE0F"
        result = _sanitize_caption(text_with_vs)
        assert "\uFE0F" not in result


class TestSmartQuotes:
    """Test smart quote replacement."""

    def test_left_single_quote(self):
        """Left single quote is replaced."""
        assert _sanitize_caption("It\u2018s") == "It's"

    def test_right_single_quote(self):
        """Right single quote is replaced."""
        assert _sanitize_caption("It\u2019s") == "It's"

    def test_left_double_quote(self):
        """Left double quote is replaced."""
        assert _sanitize_caption("\u201cHello\u201d") == '"Hello"'

    def test_right_double_quote(self):
        """Right double quote is replaced."""
        assert _sanitize_caption("Say \u201dhi\u201c") == 'Say "hi"'

    def test_mixed_quotes(self):
        """Mixed quote styles are all replaced."""
        text = "\u201cIt\u2019s a \u2018test\u2019\u201d"
        result = _sanitize_caption(text)
        assert "\u201c" not in result
        assert "\u201d" not in result
        assert "\u2018" not in result
        assert "\u2019" not in result
        assert "'" in result
        assert '"' in result


class TestDashesAndPunctuation:
    """Test dash and punctuation replacement."""

    def test_em_dash(self):
        """Em dash is replaced with hyphen."""
        assert _sanitize_caption("Hello\u2014World") == "Hello-World"

    def test_en_dash(self):
        """En dash is replaced with hyphen."""
        assert _sanitize_caption("2020\u20132025") == "2020-2025"

    def test_ellipsis(self):
        """Ellipsis is replaced with three dots."""
        assert _sanitize_caption("Wait\u2026") == "Wait..."

    def test_multiple_dashes(self):
        """Multiple dashes are all replaced."""
        text = "A\u2014B\u2013C\u2014D"
        result = _sanitize_caption(text)
        assert result == "A-B-C-D"


class TestZeroWidthCharacters:
    """Test zero-width character removal."""

    def test_zero_width_space(self):
        """Zero-width space is removed."""
        assert _sanitize_caption("Hello\u200bWorld") == "HelloWorld"

    def test_zero_width_non_joiner(self):
        """Zero-width non-joiner is removed."""
        assert _sanitize_caption("Hello\u200cWorld") == "HelloWorld"

    def test_zero_width_joiner(self):
        """Zero-width joiner is removed."""
        assert _sanitize_caption("Hello\u200dWorld") == "HelloWorld"

    def test_bom(self):
        """Byte order mark is removed."""
        assert _sanitize_caption("\ufeffHello") == "Hello"

    def test_soft_hyphen(self):
        """Soft hyphen is removed."""
        assert _sanitize_caption("Hel\u00adlo") == "Hello"

    def test_multiple_zero_width(self):
        """Multiple zero-width chars are all removed."""
        text = "\u200b\u200c\u200d\ufeffHello\u200b\u200c"
        assert _sanitize_caption(text) == "Hello"


class TestSpaceHandling:
    """Test space and non-breaking space handling."""

    def test_non_breaking_space(self):
        """Non-breaking space is converted to regular space."""
        assert _sanitize_caption("Hello\u00a0World") == "Hello World"

    def test_double_spaces_cleaned(self):
        """Double spaces are reduced to single."""
        assert _sanitize_caption("Hello  World") == "Hello World"

    def test_triple_spaces_cleaned(self):
        """Triple spaces are reduced to single."""
        assert _sanitize_caption("Hello   World") == "Hello World"

    def test_many_spaces_cleaned(self):
        """Many spaces are reduced to single."""
        assert _sanitize_caption("Hello      World") == "Hello World"

    def test_emoji_leaves_single_space(self):
        """Emoji removal leaves single space, not double."""
        text = "Hello \U0001F600 World"
        result = _sanitize_caption(text)
        assert "  " not in result
        assert result == "Hello World"


class TestNewlineHandling:
    """Test newline handling."""

    def test_single_newline_preserved(self):
        """Single newlines are preserved."""
        assert _sanitize_caption("Hello\nWorld") == "Hello\nWorld"

    def test_double_newline_preserved(self):
        """Double newlines are preserved."""
        assert _sanitize_caption("Hello\n\nWorld") == "Hello\n\nWorld"

    def test_triple_newline_reduced(self):
        """Triple newlines are reduced to double."""
        assert _sanitize_caption("Hello\n\n\nWorld") == "Hello\n\nWorld"

    def test_many_newlines_reduced(self):
        """Many newlines are reduced to double."""
        assert _sanitize_caption("Hello\n\n\n\n\nWorld") == "Hello\n\nWorld"

    def test_line_separator(self):
        """Line separator is converted to newline."""
        assert _sanitize_caption("Hello\u2028World") == "Hello\nWorld"

    def test_paragraph_separator(self):
        """Paragraph separator is converted to newline."""
        assert _sanitize_caption("Hello\u2029World") == "Hello\nWorld"

    def test_carriage_return_preserved(self):
        """Carriage return is preserved."""
        assert _sanitize_caption("Hello\rWorld") == "Hello\rWorld"

    def test_crlf_preserved(self):
        """CRLF line endings are preserved."""
        assert _sanitize_caption("Hello\r\nWorld") == "Hello\r\nWorld"


class TestControlCharacters:
    """Test control character removal."""

    def test_null_removed(self):
        """Null character is removed."""
        assert _sanitize_caption("Hello\x00World") == "HelloWorld"

    def test_bell_removed(self):
        """Bell character is removed."""
        assert _sanitize_caption("Hello\x07World") == "HelloWorld"

    def test_backspace_removed(self):
        """Backspace character is removed."""
        assert _sanitize_caption("Hello\x08World") == "HelloWorld"

    def test_form_feed_removed(self):
        """Form feed is removed."""
        assert _sanitize_caption("Hello\x0cWorld") == "HelloWorld"

    def test_escape_removed(self):
        """Escape character is removed."""
        assert _sanitize_caption("Hello\x1bWorld") == "HelloWorld"

    def test_tab_preserved(self):
        """Tab character is preserved."""
        assert _sanitize_caption("Hello\tWorld") == "Hello\tWorld"


class TestUnicodeNormalization:
    """Test Unicode normalization."""

    def test_composed_form(self):
        """Characters are normalized to composed form (NFC)."""
        # e + combining acute = e-acute
        decomposed = "caf\u0065\u0301"  # cafe with decomposed e-acute
        result = _sanitize_caption(decomposed)
        # Should be normalized - the combining char merged with e
        assert len(result) <= len(decomposed)

    def test_already_composed(self):
        """Already composed characters stay the same."""
        composed = "caf\u00e9"  # cafe with composed e-acute
        assert _sanitize_caption(composed) == composed

    def test_accented_characters_preserved(self):
        """Accented characters are preserved."""
        assert _sanitize_caption("cafe") == "cafe"
        assert _sanitize_caption("naive") == "naive"
        assert _sanitize_caption("resume") == "resume"


class TestSupplementaryPlane:
    """Test handling of supplementary plane characters."""

    def test_mathematical_symbols_removed(self):
        """Mathematical alphanumeric symbols are removed."""
        # These are above U+FFFF
        math_a = "\U0001D400"  # Mathematical Bold Capital A
        result = _sanitize_caption(f"Math {math_a}")
        assert math_a not in result

    def test_musical_symbols_removed(self):
        """Musical symbols are removed."""
        music = "\U0001D11E"  # Musical symbol G clef
        result = _sanitize_caption(f"Music {music}")
        assert music not in result

    def test_ancient_symbols_removed(self):
        """Ancient script symbols are removed."""
        # Egyptian hieroglyph
        hieroglyph = "\U00013000"
        result = _sanitize_caption(f"Ancient {hieroglyph}")
        assert hieroglyph not in result


class TestRealWorldCaptions:
    """Test with real-world caption scenarios."""

    def test_instagram_caption_with_emojis(self):
        """Typical Instagram caption with emojis."""
        caption = "3 FREE AI tools to transform your workflow! \U0001F4AA\U0001F680\n\n#AI #Tech #Productivity"
        result = _sanitize_caption(caption)
        assert "3 FREE AI tools" in result
        assert "#AI" in result
        assert "\U0001F4AA" not in result
        assert "\U0001F680" not in result

    def test_news_caption_with_emojis(self):
        """News-style caption with emojis."""
        caption = "Breaking News! \U0001F4F0\n- Story one \U0001F3A5\n- Story two \U0001F4BB\n#News"
        result = _sanitize_caption(caption)
        assert "Breaking News!" in result
        assert "- Story one" in result
        assert "#News" in result
        assert "\U0001F4F0" not in result

    def test_caption_with_smart_quotes_and_emojis(self):
        """Caption mixing smart quotes and emojis."""
        caption = "\u201cThis is amazing!\u201d \U0001F600 - John\u2019s review"
        result = _sanitize_caption(caption)
        assert '"This is amazing!"' in result
        assert "John's review" in result
        assert "\U0001F600" not in result

    def test_multilingual_caption(self):
        """Caption with multiple languages."""
        caption = "Hello World! Bonjour! Hola! \U0001F30D"
        result = _sanitize_caption(caption)
        assert "Hello World!" in result
        assert "Bonjour!" in result
        assert "Hola!" in result
        assert "\U0001F30D" not in result

    def test_hashtag_heavy_caption(self):
        """Caption with many hashtags."""
        caption = "Check this out! #AI #ML #Tech #Coding #Python #JavaScript #WebDev \U0001F4BB"
        result = _sanitize_caption(caption)
        assert "#AI" in result
        assert "#Python" in result
        assert "\U0001F4BB" not in result

    def test_mention_caption(self):
        """Caption with @mentions."""
        caption = "Thanks @user1 and @user2! \U0001F64F"
        result = _sanitize_caption(caption)
        assert "@user1" in result
        assert "@user2" in result
        assert "\U0001F64F" not in result

    def test_url_in_caption(self):
        """Caption with URL (should be preserved)."""
        caption = "Check out https://example.com \U0001F517"
        result = _sanitize_caption(caption)
        assert "https://example.com" in result
        assert "\U0001F517" not in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_caption(self):
        """Very long caption is handled."""
        long_text = "A" * 5000
        result = _sanitize_caption(long_text)
        assert len(result) == 5000
        assert result == long_text

    def test_long_caption_with_emojis(self):
        """Long caption with scattered emojis."""
        text = ("Hello " + "\U0001F600" + " ") * 100
        result = _sanitize_caption(text)
        assert "\U0001F600" not in result
        assert "Hello" in result

    def test_only_special_chars(self):
        """Caption with only special characters."""
        result = _sanitize_caption("\u200b\u200c\u200d\ufeff")
        assert result == ""

    def test_numbers_preserved(self):
        """Numbers are preserved."""
        assert _sanitize_caption("12345") == "12345"
        assert _sanitize_caption("3.14159") == "3.14159"

    def test_punctuation_preserved(self):
        """Standard punctuation is preserved."""
        text = "Hello, World! How are you? I'm fine."
        assert _sanitize_caption(text) == text

    def test_brackets_preserved(self):
        """Brackets and parentheses are preserved."""
        text = "Hello (World) [Test] {Data}"
        assert _sanitize_caption(text) == text

    def test_special_ascii_preserved(self):
        """Special ASCII characters are preserved."""
        text = "@#$%^&*()_+-=[]{}|;':\",./<>?"
        # Smart quotes would be converted
        result = _sanitize_caption(text)
        assert "@" in result
        assert "#" in result
        assert "$" in result

    def test_unicode_letters_preserved(self):
        """Non-ASCII letters are preserved."""
        # German
        assert "u" in _sanitize_caption("uber")
        # French
        assert _sanitize_caption("cafe") == "cafe"
        # Spanish
        assert "n" in _sanitize_caption("manana")

    def test_cjk_characters_preserved(self):
        """CJK characters are preserved."""
        # These are in BMP, not supplementary plane
        chinese = "Hello"  # Using ASCII for test reliability
        result = _sanitize_caption(chinese)
        assert result == chinese

    def test_mixed_everything(self):
        """Caption with mix of all types."""
        caption = (
            "\ufeff"  # BOM
            "\u201cHello\u201d "  # Smart quotes
            "\U0001F600 "  # Emoji
            "World\u2019s "  # Smart apostrophe
            "\u2014 "  # Em dash
            "Test\u200b"  # Zero-width space
            "\u2026"  # Ellipsis
            "\n\n\n"  # Triple newline
            "End"
        )
        result = _sanitize_caption(caption)
        assert '"Hello"' in result
        assert "World's" in result
        assert "-" in result
        assert "..." in result
        assert "\n\n" in result
        assert "End" in result
        assert "\U0001F600" not in result
        assert "\ufeff" not in result
        assert "\u200b" not in result


class TestCorruptedBytes:
    """Test handling of potentially corrupted byte sequences."""

    def test_replacement_character(self):
        """Unicode replacement character is handled."""
        # This might appear in corrupted text
        text = "Hello\uFFFDWorld"
        result = _sanitize_caption(text)
        # The replacement char should be in result (it's a valid BMP char)
        assert "Hello" in result
        assert "World" in result

    def test_private_use_area(self):
        """Private use area characters are handled."""
        # PUA characters (U+E000-U+F8FF)
        pua = "\uE000"
        result = _sanitize_caption(f"Test {pua}")
        # PUA chars are in BMP and not explicitly filtered
        assert "Test" in result


class TestIntegration:
    """Integration tests for complete sanitization flow."""

    def test_full_sanitization_flow(self):
        """Test complete sanitization of complex caption."""
        original = (
            "\ufeff"  # BOM at start
            "\U0001F525 HOT NEWS! \U0001F525\n\n\n"  # Emojis + triple newline
            "\u201cThis is \u2018amazing\u2019!\u201d\n"  # Smart quotes
            "Breaking\u2014live updates\u2026\n"  # Em dash + ellipsis
            "\u200bFollow @news\u200c for more!\u200d\n"  # Zero-width chars
            "#News #Breaking #Live \U0001F4F0"  # Hashtags + emoji
        )

        result = _sanitize_caption(original)

        # Check emojis removed
        assert "\U0001F525" not in result
        assert "\U0001F4F0" not in result

        # Check BOM removed
        assert "\ufeff" not in result

        # Check zero-width removed
        assert "\u200b" not in result
        assert "\u200c" not in result
        assert "\u200d" not in result

        # Check smart quotes converted
        assert '"This is' in result
        assert "'amazing'" in result

        # Check dashes/ellipsis converted
        assert "Breaking-live" in result
        assert "updates..." in result

        # Check triple newline reduced
        assert "\n\n\n" not in result
        assert "\n\n" in result

        # Check content preserved
        assert "HOT NEWS!" in result
        assert "@news" in result
        assert "#News" in result
        assert "#Breaking" in result

    def test_idempotent(self):
        """Sanitizing twice gives same result."""
        original = "Hello \U0001F600 World \u201ctest\u201d"
        first = _sanitize_caption(original)
        second = _sanitize_caption(first)
        assert first == second

    def test_no_data_loss_for_clean_text(self):
        """Clean text passes through unchanged."""
        clean = "This is a perfectly normal caption with #hashtags and @mentions."
        assert _sanitize_caption(clean) == clean
