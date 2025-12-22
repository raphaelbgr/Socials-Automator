"""Tests for SessionStorage persistence."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from socials_automator.ai_sessions.storage import SessionStorage


class TestSessionStorage:
    """Tests for SessionStorage class."""

    def test_create_session(self, tmp_path):
        """Test creating a new session."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)
        session = storage.create_session("script_generation")

        assert session["id"].startswith("session_")
        assert session["type"] == "script_generation"
        assert session["profile"] == "test_profile"
        assert session["history"] == []
        assert session["metadata"] == {}

    def test_session_persisted_to_file(self, tmp_path):
        """Test that session is saved to JSON file."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)
        session = storage.create_session("script_generation")

        # Verify file exists
        session_dir = profile_path / "ai_sessions" / "script_generation"
        assert session_dir.exists()

        session_file = session_dir / f"{session['id']}.json"
        assert session_file.exists()

        # Verify content
        with open(session_file) as f:
            saved = json.load(f)
        assert saved["id"] == session["id"]

    def test_add_to_history(self, tmp_path):
        """Test adding entries to session history."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)
        session = storage.create_session("script_generation")

        # Add user message
        storage.add_to_history(
            "script_generation",
            session["id"],
            role="user",
            content="Generate a topic about AI",
        )

        # Add assistant response
        storage.add_to_history(
            "script_generation",
            session["id"],
            role="assistant",
            content="5 ChatGPT tips for productivity",
            metadata={"provider": "zai", "model": "GLM-4.5-Air"},
        )

        # Reload and verify
        loaded = storage.load_session("script_generation", session["id"])
        assert len(loaded["history"]) == 2
        assert loaded["history"][0]["role"] == "user"
        assert loaded["history"][1]["role"] == "assistant"
        assert loaded["history"][1]["metadata"]["provider"] == "zai"

    def test_add_feedback(self, tmp_path):
        """Test adding feedback to session."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)
        session = storage.create_session("script_generation")

        storage.add_feedback(
            "script_generation",
            session["id"],
            quality="accepted",
            notes="Good topic generation",
            metrics={"word_count": 150, "duration": 35.5},
        )

        loaded = storage.load_session("script_generation", session["id"])
        assert loaded["feedback"]["quality"] == "accepted"
        assert loaded["feedback"]["metrics"]["word_count"] == 150

    def test_update_session_metadata(self, tmp_path):
        """Test updating session metadata."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)
        session = storage.create_session("script_generation")

        storage.update_session(
            "script_generation",
            session["id"],
            {"metadata": {"topic": "AI automation", "hook_type": "question"}},
        )

        loaded = storage.load_session("script_generation", session["id"])
        assert loaded["metadata"]["topic"] == "AI automation"
        assert loaded["metadata"]["hook_type"] == "question"

    def test_get_recent_sessions(self, tmp_path):
        """Test retrieving recent sessions."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)

        # Create multiple sessions
        for i in range(5):
            session = storage.create_session("script_generation")
            storage.update_session(
                "script_generation",
                session["id"],
                {"metadata": {"topic": f"Topic {i}"}},
            )

        recent = storage.get_recent_sessions("script_generation", days=7, limit=3)
        assert len(recent) == 3

    def test_get_history_summary(self, tmp_path):
        """Test getting history summary for constraints."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)

        # Create sessions with different hook types
        for hook_type in ["question", "question", "statement"]:
            session = storage.create_session("script_generation")
            storage.update_session(
                "script_generation",
                session["id"],
                {"metadata": {"hook_type": hook_type, "topic": f"Topic about {hook_type}"}},
            )

        summary = storage.get_history_summary("script_generation", days=14)
        assert summary["total_sessions"] == 3
        assert summary["patterns"]["hook_types"]["question"] == 2
        assert summary["patterns"]["hook_types"]["statement"] == 1

    def test_index_updated(self, tmp_path):
        """Test that index.json is updated on session creation."""
        profile_path = tmp_path / "test_profile"
        profile_path.mkdir()

        storage = SessionStorage(profile_path)
        session = storage.create_session("script_generation")

        index_file = profile_path / "ai_sessions" / "index.json"
        assert index_file.exists()

        with open(index_file) as f:
            index = json.load(f)

        assert "script_generation" in index["sessions"]
        assert len(index["sessions"]["script_generation"]) == 1
        assert index["sessions"]["script_generation"][0]["id"] == session["id"]
