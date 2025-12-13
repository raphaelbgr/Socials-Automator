"""Progress management service.

Handles tracking and reporting of generation progress, extracted from
the monolithic generator.py to follow Single Responsibility Principle.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..content.models import GenerationProgress


# Logger for progress events
_logger = logging.getLogger("ai_calls")

# Type for progress callback
ProgressCallback = Callable[["GenerationProgress"], Awaitable[None]]


class ProgressManager:
    """Manages generation progress tracking and reporting.

    Centralizes all progress-related logic:
    - Progress state management
    - Callback invocation
    - Stats accumulation
    - Event logging

    Usage:
        manager = ProgressManager(post_id="20241212-001", callback=display_progress)
        await manager.start_phase("Research")
        await manager.update(current_action="Searching...")
        await manager.complete_phase()
    """

    def __init__(
        self,
        post_id: str,
        callback: ProgressCallback | None = None,
        total_steps: int = 5,
        topic: str = "",
    ):
        """Initialize the progress manager.

        Args:
            post_id: Post ID for tracking.
            callback: Optional callback for progress updates.
            total_steps: Initial estimate of total steps.
            topic: The topic being generated.
        """
        self.post_id = post_id
        self.callback = callback

        # Create initial progress state
        from ..content.models import GenerationProgress

        self._progress = GenerationProgress(
            post_id=post_id,
            topic=topic,
            status="starting",
            total_steps=total_steps,
        )

        # Stats tracking
        self._total_text_calls = 0
        self._total_image_calls = 0
        self._total_cost = 0.0
        self._total_tool_calls = 0

    @property
    def progress(self) -> "GenerationProgress":
        """Get the current progress state."""
        return self._progress

    @property
    def total_text_calls(self) -> int:
        """Get total text API calls made."""
        return self._total_text_calls

    @property
    def total_image_calls(self) -> int:
        """Get total image API calls made."""
        return self._total_image_calls

    @property
    def total_cost(self) -> float:
        """Get total cost accumulated."""
        return self._total_cost

    async def emit(self, progress: "GenerationProgress | None" = None) -> None:
        """Emit a progress update.

        Args:
            progress: Optional progress to emit. Uses internal state if not provided.
        """
        if progress:
            self._progress = progress

        if self.callback:
            await self.callback(self._progress)

    async def update(self, **kwargs: Any) -> None:
        """Update progress state and emit.

        Args:
            **kwargs: Fields to update on the progress model.
        """
        self._progress = self._progress.model_copy(update=kwargs)
        await self.emit()

    async def start_phase(
        self,
        phase_name: str,
        phase_num: int | None = None,
        current_step: str | None = None,
    ) -> None:
        """Start a new generation phase.

        Args:
            phase_name: Name of the phase (Research, Planning, etc.)
            phase_num: Phase number (0-indexed).
            current_step: Description of current step.
        """
        updates: dict[str, Any] = {
            "phase_name": phase_name,
            "status": phase_name.lower(),
        }

        if phase_num is not None:
            updates["current_phase"] = phase_num

        if current_step:
            updates["current_step"] = current_step

        _logger.info(
            f"POST:{self.post_id} | PHASE_START | phase:{phase_num or 0} | "
            f"name:{phase_name}"
        )

        await self.update(**updates)

    async def complete_phase(self, phase_name: str | None = None) -> None:
        """Mark current phase as complete.

        Args:
            phase_name: Optional phase name for logging.
        """
        name = phase_name or self._progress.phase_name
        await self.update(completed_steps=self._progress.completed_steps + 1)

        _logger.info(
            f"POST:{self.post_id} | PHASE_END | phase:{self._progress.current_phase} | "
            f"name:{name}"
        )

    async def set_total_slides(self, count: int) -> None:
        """Set the total number of slides.

        Args:
            count: Number of slides.
        """
        # Recalculate total steps: research + planning + slides + caption + save
        total_steps = count + 3
        await self.update(
            total_slides=count,
            total_steps=total_steps,
        )

    async def start_slide(self, slide_number: int) -> None:
        """Start generating a slide.

        Args:
            slide_number: Slide number (1-indexed).
        """
        await self.update(
            current_slide=slide_number,
            current_step=f"Generating slide {slide_number}/{self._progress.total_slides}",
        )

    async def complete_slide(self, slide_number: int) -> None:
        """Mark a slide as complete.

        Args:
            slide_number: Slide number that was completed.
        """
        await self.update(
            completed_steps=self._progress.completed_steps + 1,
        )

    async def handle_ai_event(self, event: dict[str, Any], source: str) -> None:
        """Handle an AI event from providers.

        Args:
            event: Event dictionary from provider.
            source: Source of event ("text" or "image").
        """
        event_type = event.get("type", "")

        # Update stats
        if event_type == "text_response":
            self._total_text_calls += 1
        elif event_type == "image_response":
            self._total_image_calls += 1

        if event.get("cost_usd"):
            self._total_cost += event["cost_usd"]

        # Build update dict
        update: dict[str, Any] = {
            "event_type": event_type,
            "provider": event.get("provider"),
            "model": event.get("model"),
            "prompt_preview": event.get("prompt_preview"),
            "response_preview": event.get("response_preview"),
            "duration_seconds": event.get("duration_seconds"),
            "cost_usd": event.get("cost_usd"),
            "total_text_calls": self._total_text_calls,
            "total_image_calls": self._total_image_calls,
            "total_cost_usd": self._total_cost,
        }

        # Handle text AI events
        if event_type.startswith("text_"):
            update["text_provider"] = event.get("provider")
            update["text_model"] = event.get("model")
            update["text_prompt_preview"] = event.get("prompt_preview")
            update["text_failed_providers"] = event.get("failed_providers", [])

        # Handle image AI events
        if event_type.startswith("image_"):
            update["image_provider"] = event.get("provider")
            update["image_model"] = event.get("model")
            update["image_prompt_preview"] = event.get("prompt_preview")
            update["image_failed_providers"] = event.get("failed_providers", [])

        await self.update(**update)

        # Log the event
        self._log_ai_event(event, source)

    def _log_ai_event(self, event: dict[str, Any], source: str) -> None:
        """Log an AI event to the file logger.

        Args:
            event: Event dictionary.
            source: Event source.
        """
        event_type = event.get("type", "")
        provider = event.get("provider", "unknown")
        model = event.get("model", "unknown")
        duration = event.get("duration_seconds", 0)
        cost = event.get("cost_usd", 0)
        task = event.get("task", "unknown")
        failed_providers = event.get("failed_providers", [])

        if event_type == "text_call":
            prompt_preview = (event.get("prompt_preview", "") or "")[:150]
            has_tools = event.get("has_tools", False)
            tool_info = f" | tools:{event.get('tool_count', 0)}" if has_tools else ""
            failed_info = f" | failed_first:{','.join(failed_providers)}" if failed_providers else ""
            _logger.info(
                f"POST:{self.post_id} | TEXT_CALL | provider:{provider} | "
                f"model:{model} | task:{task}{tool_info}{failed_info} | prompt:{prompt_preview}..."
            )
        elif event_type == "text_response":
            response_preview = (event.get("response_preview", "") or "")[:100]
            tool_calls_count = event.get("tool_calls_count", 0)
            tool_info = f" | tool_calls:{tool_calls_count}" if tool_calls_count else ""
            _logger.info(
                f"POST:{self.post_id} | TEXT_RESPONSE | provider:{provider} | model:{model} | task:{task} | "
                f"duration:{duration:.2f}s | cost:${cost:.4f}{tool_info} | response:{response_preview}..."
            )
        elif event_type == "image_response":
            prompt_preview = (event.get("prompt_preview", "") or "")[:100]
            _logger.info(
                f"POST:{self.post_id} | IMAGE_RESPONSE | provider:{provider} | model:{model} | "
                f"task:{task} | duration:{duration:.2f}s | cost:${cost:.4f} | prompt:{prompt_preview}..."
            )
        elif event_type in ["text_error", "image_error"]:
            error = event.get("error", "unknown")
            failed_info = f" | fallback_from:{','.join(failed_providers)}" if failed_providers else ""
            _logger.error(f"POST:{self.post_id} | {event_type.upper()} | provider:{provider} | error:{error}{failed_info}")

    async def complete(self) -> None:
        """Mark generation as complete."""
        await self.update(
            status="completed",
            current_step="Done",
            completed_steps=self._progress.total_steps,
        )

        _logger.info(f"POST:{self.post_id} | GENERATION_COMPLETE | steps:{self._progress.total_steps}")

    async def fail(self, error: str) -> None:
        """Mark generation as failed.

        Args:
            error: Error message.
        """
        errors = list(self._progress.errors) + [error]
        await self.update(
            status="failed",
            errors=errors,
        )

        _logger.error(f"POST:{self.post_id} | GENERATION_FAILED | error:{error}")
