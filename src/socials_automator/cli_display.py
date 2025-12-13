"""CLI display components for progress visualization - Simple print version."""

from __future__ import annotations

from rich.panel import Panel
from rich.console import Console

from .content.models import GenerationProgress

console = Console()


class ContentGenerationDisplay:
    """Simple print-based display. Logs every step as it happens."""

    def __init__(self):
        self.stats = {
            "text_calls": 0,
            "total_cost": 0.0,
        }
        self._printed: set[str] = set()  # Track what we've printed
        self._web_search_count = 0

    def _print_once(self, key: str, msg: str) -> bool:
        """Print message only once per key. Returns True if printed."""
        if key not in self._printed:
            print(msg)
            self._printed.add(key)
            return True
        return False

    def add_event(self, progress: GenerationProgress | dict):
        """Process event and print status."""
        if isinstance(progress, dict):
            self._handle_dict_event(progress)
        else:
            self._handle_progress(progress)

    def _handle_dict_event(self, event: dict):
        """Handle tool/AI events."""
        event_type = event.get("type", "")

        if event_type == "tool_call_start":
            tool = event.get("tool_name", "")
            args = event.get("arguments", {})
            call_id = event.get("call_id", "")

            if tool in ("web_search", "news_search"):
                queries = args.get("queries", [])
                self._web_search_count += 1
                if queries:
                    query = queries[0][:55]
                    self._print_once(f"search_{call_id}", f"  [Search #{self._web_search_count}] {query}...")

        elif event_type == "tool_call_complete":
            tool = event.get("tool_name", "")
            success = event.get("success", False)
            meta = event.get("metadata", {})
            call_id = event.get("call_id", "")

            if tool in ("web_search", "news_search"):
                results = meta.get("total_results", 0)
                status = "[OK]" if success else "[FAIL]"
                self._print_once(f"search_done_{call_id}", f"    {status} Found {results} results")

        elif event_type == "text_response":
            self.stats["text_calls"] += 1
            self.stats["total_cost"] += event.get("cost_usd", 0)

    def _handle_progress(self, p: GenerationProgress):
        """Handle GenerationProgress events."""
        phase = p.current_phase
        step = p.current_step or ""
        action = p.current_action or ""
        status = p.status or ""
        phase_name = p.phase_name or ""
        phase_output = p.phase_output or ""

        # Update stats
        if p.total_text_calls:
            self.stats["text_calls"] = p.total_text_calls
        if p.total_cost_usd:
            self.stats["total_cost"] = p.total_cost_usd

        # === PHASE 0: Preparation ===
        if phase == 0:
            self._print_once("phase_0", f"\n[Phase 0] {phase_name or 'Preparation'}")

            # Print step updates
            if step:
                key = f"p0_{step[:25]}"
                if "history" in step.lower() or "loading" in step.lower():
                    self._print_once(key, "  ... Loading post history")
                elif "duplicate" in step.lower() or "avoid" in step.lower() or "checking" in step.lower():
                    self._print_once(key, "  ... Checking recent posts")
                elif "trending" in step.lower() or "research" in step.lower():
                    self._print_once(key, "  ... Researching trending topics")
                elif "ai" in step.lower() and "generat" in step.lower():
                    self._print_once(key, "  ... AI generating topics")
                elif "found" in step.lower() or "selected" in step.lower():
                    self._print_once(key, f"  [OK] {step}")
                else:
                    self._print_once(key, f"  ... {step}")

        # === PHASE 1: Planning ===
        elif phase == 1:
            self._print_once("phase_1", f"\n[Phase 1] Planning")

            if "analyzing" in action.lower():
                self._print_once("p1_analyze", "  ... Analyzing topic")
            elif "complete" in action.lower():
                # Show result
                if p.content_count and p.content_type:
                    self._print_once("p1_done", f"  [OK] {p.content_count} {p.content_type}s identified")
                elif phase_output:
                    self._print_once("p1_done", f"  [OK] {phase_output}")

        # === PHASE 2: Structure ===
        elif phase == 2:
            self._print_once("phase_2", f"\n[Phase 2] Structure")

            if "creating" in action.lower() or "hook" in action.lower():
                self._print_once("p2_create", "  ... Creating hook and structure")
            elif "complete" in action.lower():
                if phase_output:
                    self._print_once("p2_done", f"  [OK] {phase_output}")
                # Show ALL slide titles
                if p.slide_titles:
                    for i, title in enumerate(p.slide_titles, 1):
                        self._print_once(f"p2_title_{i}", f"       Slide {i}: {title[:45]}")

        # === PHASE 3: Content ===
        elif phase == 3:
            self._print_once("phase_3", f"\n[Phase 3] Content")

            slide = p.current_slide
            if slide:
                # Generating
                if "generating" in action.lower():
                    total = p.total_slides or "?"
                    self._print_once(f"p3_gen_{slide}", f"  ... Slide {slide}/{total}")
                # Validation retry
                elif "regenerat" in action.lower():
                    self._print_once(f"p3_retry_{slide}_{p.validation_attempt}",
                                    f"      [!] Regenerating (quality check)")
                # Complete
                elif "complete" in action.lower():
                    heading = phase_output[:40] if phase_output else f"Slide {slide}"
                    self._print_once(f"p3_done_{slide}", f"  [OK] {heading}")
            elif p.validation_error:
                self._print_once(f"p3_err_{p.validation_attempt}",
                                f"      [!] {p.validation_error[:50]}")

        # === PHASE 4: CTA ===
        elif phase == 4:
            self._print_once("phase_4", f"\n[Phase 4] CTA")

            if "creating" in action.lower() or "call-to-action" in action.lower():
                self._print_once("p4_create", "  ... Creating call-to-action")
            elif "complete" in action.lower():
                if phase_output:
                    self._print_once("p4_done", f"  [OK] {phase_output}")

        # === Caption ===
        if "caption" in step.lower():
            self._print_once("caption_start", "\n[Caption]")
            self._print_once("caption_gen", "  ... Generating caption")

        # === Completion ===
        if status == "completed":
            cost = self.stats.get("total_cost", 0)
            calls = self.stats.get("text_calls", 0)
            self._print_once("done", f"\n[Done] {calls} AI calls, ${cost:.4f} total")
        elif status == "failed" and p.errors:
            self._print_once("failed", f"\n[FAILED] {p.errors[-1]}")

    def render(self, progress: GenerationProgress) -> Panel:
        """Return empty panel (Live display not used)."""
        return Panel("", title="", border_style="dim")


class InstagramPostingDisplay:
    """Simple display for Instagram posting progress."""

    def __init__(self):
        self.current_status = "Initializing..."
        self.images_uploaded = 0
        self.total_images = 0
        self.containers_created = 0
        self._printed: set[str] = set()

    def _print_once(self, key: str, msg: str):
        """Print message only once."""
        if key not in self._printed:
            print(msg)
            self._printed.add(key)

    def update(self, progress=None, **kwargs):
        """Update posting status and print progress."""
        if progress is not None:
            # Extract from progress object
            if hasattr(progress, 'current_step'):
                step = progress.current_step or ""
            elif isinstance(progress, dict):
                step = progress.get('current_step', '')
            else:
                step = ""

            if hasattr(progress, 'images_uploaded'):
                self.images_uploaded = progress.images_uploaded or 0
            if hasattr(progress, 'total_images'):
                self.total_images = progress.total_images or 0
            if hasattr(progress, 'containers_created'):
                self.containers_created = progress.containers_created or 0

            # Print step
            if step:
                self._print_step(step)
            return

        # Handle keyword arguments
        step = kwargs.get('step', '')
        if kwargs.get('total_images'):
            self.total_images = kwargs['total_images']
        if kwargs.get('images_uploaded'):
            self.images_uploaded = kwargs['images_uploaded']
        if kwargs.get('containers_created'):
            self.containers_created = kwargs['containers_created']

        if step:
            self._print_step(step)

    def _print_step(self, step: str):
        """Print a step (deduplicated)."""
        key = step[:30]
        if key in self._printed:
            return
        self._printed.add(key)

        s = step.lower()
        if "upload" in s:
            print(f"  [Upload] {step}")
        elif "container" in s:
            print(f"  [Container] {step}")
        elif "carousel" in s:
            print(f"  [Carousel] {step}")
        elif "publish" in s:
            print(f"  [Publish] {step}")
        elif "rate limit" in s:
            print(f"  [!] Rate limit - checking status...")
        elif "success" in s or "published" in s:
            print(f"  [OK] {step}")
        elif "error" in s or "fail" in s:
            print(f"  [X] {step}")
        else:
            print(f"  ... {step}")

    def render(self) -> Panel:
        """Render summary panel."""
        return Panel(
            f"Status: {self.current_status}\n"
            f"Images: {self.images_uploaded}/{self.total_images}\n"
            f"Containers: {self.containers_created}",
            title="Instagram Publishing",
            border_style="cyan",
        )
