"""Base ProfileAISession class for profile-scoped AI conversations.

Provides persistent conversation context across AI calls for a profile,
enabling multi-turn memory and learning from feedback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Awaitable, TypeVar

from pydantic import BaseModel

from .storage import SessionStorage
from ..providers.text import TextProvider
from ..providers.config import load_provider_config

_logger = logging.getLogger("ai_sessions")

T = TypeVar("T", bound=BaseModel)

# Type for AI event callback
AIEventCallback = Callable[[dict[str, Any]], Awaitable[None]] | None


class ProfileAISession(ABC):
    """Base class for profile-scoped AI sessions.

    Maintains conversation context across AI calls for a specific profile,
    enabling:
    - Multi-turn conversation memory
    - Constraint generation from history
    - Quality feedback tracking
    - Provider performance learning

    Subclasses implement specific session types (script, content, etc.)
    with their own constraint logic and prompt templates.
    """

    # Session type identifier (override in subclasses)
    SESSION_TYPE: str = "generic"

    def __init__(
        self,
        profile_path: Path,
        provider_override: str | None = None,
        event_callback: AIEventCallback = None,
        history_days: int = 7,
    ):
        """Initialize a profile AI session.

        Args:
            profile_path: Path to the profile directory.
            provider_override: Override LLM provider (e.g., 'lmstudio').
            event_callback: Optional callback for AI events.
            history_days: Days of history to consider for context.
        """
        self.profile_path = Path(profile_path)
        self.profile_name = self.profile_path.name
        self.history_days = history_days

        # Storage for persistence
        self._storage = SessionStorage(profile_path)

        # Current session (created on first generate)
        self._current_session: dict[str, Any] | None = None

        # Text provider for AI calls
        self._text_provider = TextProvider(
            config=load_provider_config(),
            event_callback=event_callback,
            provider_override=provider_override,
        )

        # Load history summary for constraints
        self._history_summary = self._storage.get_history_summary(
            self.SESSION_TYPE, days=history_days
        )

        _logger.info(
            f"Initialized {self.SESSION_TYPE} session for {self.profile_name} "
            f"({self._history_summary['total_sessions']} recent sessions)"
        )

    def _ensure_session(self) -> None:
        """Ensure a current session exists."""
        if self._current_session is None:
            self._current_session = self._storage.create_session(self.SESSION_TYPE)
            _logger.debug(f"Created session: {self._current_session['id']}")

    @property
    def session_id(self) -> str | None:
        """Get current session ID."""
        return self._current_session["id"] if self._current_session else None

    @property
    def current_provider(self) -> str | None:
        """Get the current provider being used."""
        return self._text_provider.current_provider

    @property
    def current_model(self) -> str | None:
        """Get the current model being used."""
        return self._text_provider.current_model

    @abstractmethod
    def get_constraints(self) -> list[str]:
        """Generate constraints from session history.

        Override in subclasses to provide session-type-specific constraints.

        Returns:
            List of constraint strings to include in prompts.
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this session type.

        Override in subclasses to provide session-type-specific prompts.

        Returns:
            System prompt string.
        """
        pass

    def _build_context_prompt(self) -> str:
        """Build context section from history and constraints."""
        constraints = self.get_constraints()

        if not constraints:
            return ""

        context = "\n\nCONTEXT FROM RECENT SESSIONS:\n"
        context += "\n".join(f"- {c}" for c in constraints)
        return context

    def _format_history_for_prompt(self) -> str:
        """Format recent conversation history for prompt inclusion."""
        if not self._current_session:
            return ""

        history = self._current_session.get("history", [])
        if not history:
            return ""

        # Include last 5 turns
        recent = history[-10:]  # 5 exchanges = 10 entries

        formatted = "\n\nPREVIOUS CONVERSATION:\n"
        for entry in recent:
            role = entry["role"].upper()
            content = entry["content"]
            # Truncate long content
            if len(content) > 500:
                content = content[:500] + "..."
            formatted += f"[{role}] {content}\n"

        return formatted

    async def generate(
        self,
        prompt: str,
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        include_history: bool = True,
    ) -> str:
        """Generate text with session context.

        Args:
            prompt: User prompt.
            task: Task name for logging.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            include_history: Whether to include conversation history.

        Returns:
            Generated text response.
        """
        self._ensure_session()

        # Build full system prompt with context
        system = self.get_system_prompt()
        system += self._build_context_prompt()

        if include_history:
            system += self._format_history_for_prompt()

        # Record user message in history
        self._storage.add_to_history(
            self.SESSION_TYPE,
            self._current_session["id"],
            "user",
            prompt,
        )

        # Generate
        result = await self._text_provider.generate(
            prompt=prompt,
            system=system,
            task=task or self.SESSION_TYPE,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Record assistant response
        self._storage.add_to_history(
            self.SESSION_TYPE,
            self._current_session["id"],
            "assistant",
            result,
            metadata={
                "provider": self._text_provider.current_provider,
                "model": self._text_provider.current_model,
            },
        )

        return result

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        task: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        include_history: bool = True,
    ) -> T:
        """Generate structured output with session context.

        Args:
            prompt: User prompt.
            response_model: Pydantic model for response.
            task: Task name for logging.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            include_history: Whether to include conversation history.

        Returns:
            Instance of response_model.
        """
        self._ensure_session()

        # Build full system prompt with context
        system = self.get_system_prompt()
        system += self._build_context_prompt()

        if include_history:
            system += self._format_history_for_prompt()

        # Record user message in history
        self._storage.add_to_history(
            self.SESSION_TYPE,
            self._current_session["id"],
            "user",
            prompt,
        )

        # Generate structured
        result = await self._text_provider.generate_structured(
            prompt=prompt,
            response_model=response_model,
            system=system,
            task=task or self.SESSION_TYPE,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Record assistant response
        result_json = result.model_dump_json(indent=2)
        self._storage.add_to_history(
            self.SESSION_TYPE,
            self._current_session["id"],
            "assistant",
            result_json,
            metadata={
                "provider": self._text_provider.current_provider,
                "model": self._text_provider.current_model,
                "response_model": response_model.__name__,
            },
        )

        return result

    def add_feedback(
        self,
        quality: str,
        notes: str = "",
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """Add quality feedback to current session.

        Args:
            quality: Quality rating (accepted, rejected, retry).
            notes: Optional notes.
            metrics: Optional metrics (duration, word_count, etc.)
        """
        if not self._current_session:
            _logger.warning("No active session to add feedback to")
            return

        self._storage.add_feedback(
            self.SESSION_TYPE,
            self._current_session["id"],
            quality=quality,
            notes=notes,
            metrics=metrics,
        )

        _logger.info(
            f"Added feedback to session {self._current_session['id']}: "
            f"quality={quality}"
        )

    def set_metadata(self, **kwargs: Any) -> None:
        """Set metadata on current session.

        Args:
            **kwargs: Metadata key-value pairs.
        """
        # Ensure session exists (create if needed)
        self._ensure_session()

        self._current_session.setdefault("metadata", {}).update(kwargs)
        self._storage.update_session(
            self.SESSION_TYPE,
            self._current_session["id"],
            {"metadata": self._current_session["metadata"]},
        )

    def close(self) -> None:
        """Close the current session."""
        if self._current_session:
            _logger.debug(f"Closed session: {self._current_session['id']}")
            self._current_session = None

    def get_recent_outputs(self, limit: int = 10) -> list[str]:
        """Get recent assistant outputs from history.

        Args:
            limit: Maximum outputs to return.

        Returns:
            List of recent output strings.
        """
        sessions = self._storage.get_recent_sessions(
            self.SESSION_TYPE, days=self.history_days, limit=50
        )

        outputs = []
        for session in sessions:
            for entry in session.get("history", []):
                if entry["role"] == "assistant":
                    outputs.append(entry["content"])
                    if len(outputs) >= limit:
                        return outputs

        return outputs
