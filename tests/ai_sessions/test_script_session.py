"""Tests for ScriptSession with hook tracking and constraints."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from socials_automator.ai_sessions.script_session import ScriptSession


class TestScriptSession:
    """Tests for ScriptSession class."""

    def test_init_creates_storage(self, tmp_path):
        """Test that session initializes storage correctly."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(
                profile_path=profile_path,
                target_duration=60.0,
                provider_override="zai",
            )

            assert session.profile_name == "test_profile"
            assert session.target_duration == 60.0
            assert session._storage is not None

    def test_detect_hook_type_question(self, tmp_path):
        """Test detecting question hook type."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(profile_path=profile_path)

            # Question hooks
            assert session._detect_hook_type("Do you know about AI?") == "question"
            assert session._detect_hook_type("Have you ever wondered...") == "question"
            assert session._detect_hook_type("What if I told you...") == "question"
            assert session._detect_hook_type("Why do people fail at this?") == "question"

    def test_detect_hook_type_number(self, tmp_path):
        """Test detecting number hook type."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(profile_path=profile_path)

            assert session._detect_hook_type("5 ways to boost productivity") == "number"
            assert session._detect_hook_type("3 tips for better AI prompts") == "number"
            assert session._detect_hook_type("10 secrets nobody tells you") == "number"

    def test_detect_hook_type_statement(self, tmp_path):
        """Test detecting statement hook type."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(profile_path=profile_path)

            assert session._detect_hook_type("This is the best AI tool ever") == "statement"
            assert session._detect_hook_type("Here's what nobody tells you") == "statement"
            assert session._detect_hook_type("The secret to productivity") == "statement"
            assert session._detect_hook_type("Stop doing this immediately") == "statement"

    def test_detect_hook_type_story(self, tmp_path):
        """Test detecting story hook type."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(profile_path=profile_path)

            assert session._detect_hook_type("I discovered something amazing") == "story"
            assert session._detect_hook_type("My journey with AI started...") == "story"
            assert session._detect_hook_type("When I first tried ChatGPT") == "story"
            assert session._detect_hook_type("Last week I found this tool") == "story"

    def test_get_constraints_empty_history(self, tmp_path):
        """Test constraints with no history."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(profile_path=profile_path)

            constraints = session.get_constraints()
            # With no history, constraints should be empty (no patterns to avoid)
            assert constraints == []

    def test_get_constraints_with_history(self, tmp_path):
        """Test constraints based on history."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            # Create first session and set metadata
            session1 = ScriptSession(profile_path=profile_path)
            session1.set_metadata(
                topic="AI automation tips",
                hook_text="Do you want to save time?",
                hook_type="question",
            )

            # Create second session - should see first session's data
            session2 = ScriptSession(profile_path=profile_path)

            constraints = session2.get_constraints()

            # Should warn about overused question hooks
            assert any("question" in c.lower() for c in constraints)
            # Should suggest other hook types
            assert any("Consider using" in c for c in constraints)

    def test_is_topic_recent(self, tmp_path):
        """Test topic deduplication check."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session1 = ScriptSession(profile_path=profile_path)
            session1.set_metadata(topic="AI automation for beginners")

            session2 = ScriptSession(profile_path=profile_path)

            # Exact topic should be detected
            assert session2.is_topic_recent("AI automation for beginners")
            # Similar topic should be detected (>50% overlap)
            assert session2.is_topic_recent("AI automation tips")
            # Different topic should not be detected
            assert not session2.is_topic_recent("Machine learning models")

    def test_get_recommended_hook_type(self, tmp_path):
        """Test hook type recommendation based on history."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            # Use question hooks twice
            for _ in range(2):
                session = ScriptSession(profile_path=profile_path)
                session.set_metadata(hook_type="question", topic="Some topic")

            # New session should recommend non-question hooks
            session = ScriptSession(profile_path=profile_path)
            recommended = session.get_recommended_hook_type()
            assert recommended != "question"
            assert recommended in ["number", "statement", "story"]

    def test_set_metadata_creates_session(self, tmp_path):
        """Test that set_metadata creates session if not exists."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            session = ScriptSession(profile_path=profile_path)

            # Session should not exist yet
            assert session._current_session is None

            # Set metadata should create session
            session.set_metadata(topic="Test topic", hook_type="question")

            # Session should now exist
            assert session._current_session is not None
            assert session.session_id is not None

    def test_session_persists_across_instances(self, tmp_path):
        """Test that session data persists across ScriptSession instances."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        with patch("socials_automator.ai_sessions.profile_session.TextProvider"):
            # Create and populate session
            session1 = ScriptSession(profile_path=profile_path)
            session1.set_metadata(
                topic="5 ChatGPT tips",
                hook_text="Did you know ChatGPT can...",
                hook_type="question",
                word_count=150,
            )

            # Create new instance and verify data
            session2 = ScriptSession(profile_path=profile_path)

            assert "5 chatgpt tips" in session2._recent_topics[0].lower()
            assert session2._hook_counts.get("question", 0) == 1
