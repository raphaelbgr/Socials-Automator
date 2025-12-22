"""Session storage for persisting AI conversation history.

Handles reading/writing session data to JSON files per profile.
Sessions are organized by type (script, content, caption) and date.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_logger = logging.getLogger("ai_sessions")


class SessionStorage:
    """Manages persistent storage for AI session data.

    Storage structure:
        profiles/<name>/ai_sessions/
            script_generation/
                session_20251222_001.json
                session_20251222_002.json
            content_planning/
                session_20251222_001.json
            index.json  # Quick lookup of recent sessions
    """

    def __init__(self, profile_path: Path):
        """Initialize session storage for a profile.

        Args:
            profile_path: Path to the profile directory.
        """
        self.profile_path = Path(profile_path)
        self.sessions_dir = self.profile_path / "ai_sessions"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure session directories exist."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_type: str) -> Path:
        """Get directory for a session type."""
        session_dir = self.sessions_dir / session_type
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _generate_session_id(self, session_type: str) -> str:
        """Generate unique session ID for today."""
        today = datetime.now().strftime("%Y%m%d")
        session_dir = self._get_session_dir(session_type)

        # Find next sequence number for today
        existing = list(session_dir.glob(f"session_{today}_*.json"))
        seq = len(existing) + 1

        return f"session_{today}_{seq:03d}"

    def create_session(self, session_type: str) -> dict[str, Any]:
        """Create a new session.

        Args:
            session_type: Type of session (script_generation, content_planning, etc.)

        Returns:
            Session data dict with id, type, created_at, and empty history.
        """
        session_id = self._generate_session_id(session_type)

        session = {
            "id": session_id,
            "type": session_type,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "profile": self.profile_path.name,
            "history": [],
            "constraints_used": [],
            "feedback": None,
            "metadata": {},
        }

        self._save_session(session_type, session_id, session)
        self._update_index(session_type, session_id)

        _logger.debug(f"Created session: {session_id} for {session_type}")
        return session

    def _save_session(
        self, session_type: str, session_id: str, session: dict[str, Any]
    ) -> None:
        """Save session to file."""
        session_dir = self._get_session_dir(session_type)
        session_file = session_dir / f"{session_id}.json"

        session["updated_at"] = datetime.now().isoformat()

        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2, ensure_ascii=False)

    def load_session(self, session_type: str, session_id: str) -> dict[str, Any] | None:
        """Load a specific session.

        Args:
            session_type: Type of session.
            session_id: Session ID.

        Returns:
            Session data or None if not found.
        """
        session_dir = self._get_session_dir(session_type)
        session_file = session_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_session(
        self, session_type: str, session_id: str, updates: dict[str, Any]
    ) -> None:
        """Update an existing session.

        Args:
            session_type: Type of session.
            session_id: Session ID.
            updates: Dict of fields to update.
        """
        session = self.load_session(session_type, session_id)
        if session is None:
            _logger.warning(f"Session not found: {session_id}")
            return

        session.update(updates)
        self._save_session(session_type, session_id, session)

    def add_to_history(
        self,
        session_type: str,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add an entry to session history.

        Args:
            session_type: Type of session.
            session_id: Session ID.
            role: Message role (user, assistant, system).
            content: Message content.
            metadata: Optional metadata (model, duration, etc.)
        """
        session = self.load_session(session_type, session_id)
        if session is None:
            return

        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            entry["metadata"] = metadata

        session["history"].append(entry)
        self._save_session(session_type, session_id, session)

    def add_feedback(
        self,
        session_type: str,
        session_id: str,
        quality: str,
        notes: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Add quality feedback to a session.

        Args:
            session_type: Type of session.
            session_id: Session ID.
            quality: Quality rating (accepted, rejected, retry).
            notes: Optional notes about the quality.
            metrics: Optional metrics (duration, word_count, etc.)
        """
        session = self.load_session(session_type, session_id)
        if session is None:
            return

        session["feedback"] = {
            "quality": quality,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
        }
        self._save_session(session_type, session_id, session)

    def _update_index(self, session_type: str, session_id: str) -> None:
        """Update the sessions index for quick lookup."""
        index_file = self.sessions_dir / "index.json"

        if index_file.exists():
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
        else:
            index = {"sessions": {}}

        if session_type not in index["sessions"]:
            index["sessions"][session_type] = []

        index["sessions"][session_type].append({
            "id": session_id,
            "created_at": datetime.now().isoformat(),
        })

        # Keep only last 100 per type
        index["sessions"][session_type] = index["sessions"][session_type][-100:]

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def get_recent_sessions(
        self, session_type: str, days: int = 7, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get recent sessions of a type.

        Args:
            session_type: Type of session.
            days: Number of days to look back.
            limit: Maximum sessions to return.

        Returns:
            List of session data dicts, most recent first.
        """
        session_dir = self._get_session_dir(session_type)
        cutoff = datetime.now() - timedelta(days=days)

        sessions = []
        for session_file in sorted(session_dir.glob("session_*.json"), reverse=True):
            if len(sessions) >= limit:
                break

            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session = json.load(f)

                created = datetime.fromisoformat(session["created_at"])
                if created >= cutoff:
                    sessions.append(session)
            except Exception as e:
                _logger.warning(f"Failed to load session {session_file}: {e}")

        return sessions

    def get_history_summary(
        self, session_type: str, days: int = 14
    ) -> dict[str, Any]:
        """Get summary of recent session history for constraint generation.

        Args:
            session_type: Type of session.
            days: Number of days to analyze.

        Returns:
            Summary dict with patterns, counts, and recommendations.
        """
        sessions = self.get_recent_sessions(session_type, days=days, limit=50)

        summary = {
            "total_sessions": len(sessions),
            "accepted_count": 0,
            "rejected_count": 0,
            "patterns": {},
            "providers_used": {},
            "recent_topics": [],
            "recent_hooks": [],
        }

        for session in sessions:
            # Track feedback
            feedback = session.get("feedback")
            if feedback:
                if feedback.get("quality") == "accepted":
                    summary["accepted_count"] += 1
                elif feedback.get("quality") == "rejected":
                    summary["rejected_count"] += 1

            # Track metadata patterns
            metadata = session.get("metadata", {})
            if "hook_type" in metadata:
                hook_type = metadata["hook_type"]
                summary["patterns"].setdefault("hook_types", {})
                summary["patterns"]["hook_types"][hook_type] = (
                    summary["patterns"]["hook_types"].get(hook_type, 0) + 1
                )

            if "topic" in metadata:
                summary["recent_topics"].append(metadata["topic"])

            if "hook_text" in metadata:
                summary["recent_hooks"].append(metadata["hook_text"][:50])

            # Track provider usage
            for entry in session.get("history", []):
                entry_meta = entry.get("metadata", {})
                if "provider" in entry_meta:
                    provider = entry_meta["provider"]
                    summary["providers_used"][provider] = (
                        summary["providers_used"].get(provider, 0) + 1
                    )

        return summary

    def cleanup_old_sessions(self, days: int = 30) -> int:
        """Remove sessions older than specified days.

        Args:
            days: Sessions older than this will be deleted.

        Returns:
            Number of sessions deleted.
        """
        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0

        for session_type_dir in self.sessions_dir.iterdir():
            if not session_type_dir.is_dir() or session_type_dir.name == "index.json":
                continue

            for session_file in session_type_dir.glob("session_*.json"):
                try:
                    with open(session_file, "r", encoding="utf-8") as f:
                        session = json.load(f)

                    created = datetime.fromisoformat(session["created_at"])
                    if created < cutoff:
                        session_file.unlink()
                        deleted += 1
                except Exception as e:
                    _logger.warning(f"Failed to check/delete {session_file}: {e}")

        _logger.info(f"Cleaned up {deleted} old sessions")
        return deleted
