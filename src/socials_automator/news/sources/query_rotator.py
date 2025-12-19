"""Query rotation with round-robin batches and state persistence.

This module manages query rotation to ensure diverse search coverage
while avoiding API rate limits. Queries are organized into batches
and rotated based on cooldown periods.

Usage:
    from socials_automator.news.sources import QueryRotator

    rotator = QueryRotator()

    # Get current batch queries
    queries = rotator.get_current_queries()

    # Advance to next batch (called after successful fetch)
    rotator.advance_batch()

    # Get rotation state
    state = rotator.get_state()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from socials_automator.news.sources.models import QueryConfig
from socials_automator.news.sources.registry import SourceRegistry, get_source_registry

logger = logging.getLogger("ai_calls")


@dataclass
class RotationState:
    """State for query rotation persistence."""

    current_batch: int = 1
    last_rotation_at: Optional[str] = None
    batch_history: list[dict] = field(default_factory=list)
    total_rotations: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RotationState":
        """Create from dictionary."""
        return cls(
            current_batch=data.get("current_batch", 1),
            last_rotation_at=data.get("last_rotation_at"),
            batch_history=data.get("batch_history", []),
            total_rotations=data.get("total_rotations", 0),
        )

    @property
    def last_rotation_datetime(self) -> Optional[datetime]:
        """Get last rotation time as datetime."""
        if not self.last_rotation_at:
            return None
        try:
            return datetime.fromisoformat(self.last_rotation_at)
        except ValueError:
            return None


class QueryRotator:
    """Manages round-robin query batch rotation.

    Queries are organized into batches (typically 3) and rotated
    based on a cooldown period. This ensures:
    - Even distribution of query usage over time
    - Avoidance of API rate limits
    - Diversity in search results

    Example:
        rotator = QueryRotator()

        # Normal usage - get queries for current batch
        queries = rotator.get_current_queries()

        # After fetching, mark batch as used
        rotator.mark_batch_used()

        # Force rotation (e.g., for testing)
        rotator.advance_batch()
    """

    # Default state file location
    DEFAULT_STATE_PATH = Path("data/query_rotation_state.json")

    def __init__(
        self,
        registry: Optional[SourceRegistry] = None,
        state_path: Optional[Path] = None,
    ):
        """Initialize the rotator.

        Args:
            registry: SourceRegistry instance. Uses default if None.
            state_path: Path for state persistence. Uses default if None.
        """
        self._registry = registry or get_source_registry()
        self._state_path = state_path or self.DEFAULT_STATE_PATH
        self._state = self._load_state()

    def _load_state(self) -> RotationState:
        """Load rotation state from file."""
        if not self._state_path.exists():
            logger.debug(f"QUERY_ROTATOR | no state file, starting fresh")
            return RotationState()

        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            state = RotationState.from_dict(data)
            logger.debug(
                f"QUERY_ROTATOR | loaded state | batch={state.current_batch} | "
                f"rotations={state.total_rotations}"
            )
            return state
        except Exception as e:
            logger.warning(f"QUERY_ROTATOR | failed to load state: {e}")
            return RotationState()

    def _save_state(self) -> None:
        """Save rotation state to file."""
        try:
            # Ensure directory exists
            self._state_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, indent=2)

            logger.debug(
                f"QUERY_ROTATOR | saved state | batch={self._state.current_batch}"
            )
        except Exception as e:
            logger.warning(f"QUERY_ROTATOR | failed to save state: {e}")

    @property
    def current_batch(self) -> int:
        """Get current batch number (1-indexed)."""
        return self._state.current_batch

    @property
    def total_batches(self) -> int:
        """Get total number of batches."""
        return self._registry.get_batch_count()

    @property
    def cooldown_hours(self) -> int:
        """Get cooldown period between rotations."""
        return self._registry.get_cooldown_hours()

    def is_cooldown_expired(self) -> bool:
        """Check if cooldown period has expired.

        Returns:
            True if rotation is allowed (cooldown expired or never rotated).
        """
        last_rotation = self._state.last_rotation_datetime
        if last_rotation is None:
            return True

        cooldown_delta = timedelta(hours=self.cooldown_hours)
        return datetime.utcnow() >= last_rotation + cooldown_delta

    def get_time_until_next_rotation(self) -> Optional[timedelta]:
        """Get time remaining until next allowed rotation.

        Returns:
            Timedelta until rotation allowed, or None if already allowed.
        """
        if self.is_cooldown_expired():
            return None

        last_rotation = self._state.last_rotation_datetime
        cooldown_delta = timedelta(hours=self.cooldown_hours)
        next_rotation = last_rotation + cooldown_delta
        return next_rotation - datetime.utcnow()

    def get_current_queries(self) -> list[QueryConfig]:
        """Get queries for the current batch.

        Returns:
            List of query configurations for current batch.
        """
        return self._registry.get_queries_by_batch(self.current_batch)

    def get_all_current_batch_query_strings(self) -> list[str]:
        """Get just the query strings for current batch.

        Returns:
            List of search query strings.
        """
        return [q.query for q in self.get_current_queries()]

    def advance_batch(self) -> int:
        """Advance to the next batch (round-robin).

        This respects cooldown period - if cooldown hasn't expired,
        it will not advance and return current batch.

        Returns:
            New current batch number.
        """
        if not self.is_cooldown_expired():
            remaining = self.get_time_until_next_rotation()
            logger.info(
                f"QUERY_ROTATOR | cooldown active | remaining={remaining} | "
                f"batch={self.current_batch}"
            )
            return self.current_batch

        # Advance to next batch (wrap around)
        old_batch = self.current_batch
        new_batch = (self.current_batch % self.total_batches) + 1
        self._state.current_batch = new_batch
        self._state.last_rotation_at = datetime.utcnow().isoformat()
        self._state.total_rotations += 1

        # Add to history (keep last 10)
        self._state.batch_history.append({
            "batch": old_batch,
            "rotated_at": self._state.last_rotation_at,
        })
        if len(self._state.batch_history) > 10:
            self._state.batch_history = self._state.batch_history[-10:]

        self._save_state()

        logger.info(
            f"QUERY_ROTATOR | rotated | {old_batch} -> {new_batch} | "
            f"total_rotations={self._state.total_rotations}"
        )

        return new_batch

    def force_batch(self, batch: int) -> None:
        """Force rotation to a specific batch (for testing).

        Args:
            batch: Batch number to set (1-indexed).
        """
        if batch < 1 or batch > self.total_batches:
            raise ValueError(
                f"Invalid batch {batch}, must be 1-{self.total_batches}"
            )

        old_batch = self.current_batch
        self._state.current_batch = batch
        self._state.last_rotation_at = datetime.utcnow().isoformat()
        self._save_state()

        logger.info(
            f"QUERY_ROTATOR | forced batch | {old_batch} -> {batch}"
        )

    def mark_batch_used(self) -> None:
        """Mark the current batch as used (update last rotation time).

        Call this after successfully fetching with the current batch.
        """
        self._state.last_rotation_at = datetime.utcnow().isoformat()
        self._save_state()

    def reset(self) -> None:
        """Reset rotation state to defaults."""
        self._state = RotationState()
        self._save_state()
        logger.info("QUERY_ROTATOR | reset to default state")

    def get_state(self) -> dict:
        """Get current rotation state.

        Returns:
            Dictionary with rotation state info.
        """
        return {
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "queries_in_batch": len(self.get_current_queries()),
            "cooldown_hours": self.cooldown_hours,
            "cooldown_expired": self.is_cooldown_expired(),
            "time_until_next": str(self.get_time_until_next_rotation()),
            "total_rotations": self._state.total_rotations,
            "last_rotation_at": self._state.last_rotation_at,
        }

    def get_batch_summary(self) -> dict:
        """Get summary of all batches.

        Returns:
            Dictionary with batch information.
        """
        summary = {}
        for batch in range(1, self.total_batches + 1):
            queries = self._registry.get_queries_by_batch(batch)
            summary[f"batch_{batch}"] = {
                "count": len(queries),
                "languages": list(set(q.language for q in queries)),
                "categories": list(set(q.category.value for q in queries)),
                "is_current": batch == self.current_batch,
            }
        return summary


# Module-level cached instance
_rotator_instance: Optional[QueryRotator] = None


def get_query_rotator(reload: bool = False) -> QueryRotator:
    """Get or create the default QueryRotator instance.

    Args:
        reload: If True, reload from state file even if already loaded.

    Returns:
        QueryRotator instance.
    """
    global _rotator_instance

    if _rotator_instance is None or reload:
        _rotator_instance = QueryRotator()

    return _rotator_instance
