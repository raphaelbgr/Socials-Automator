"""CLI display components for progress visualization."""

from __future__ import annotations

from rich.panel import Panel
from rich import box

from .content.models import GenerationProgress


class ContentGenerationDisplay:
    """Rich display for content generation progress with phase visualization.

    Displays:
    - Phase 0: Web Research (search queries, results, domains)
    - Phase 1: Planning (topic analysis)
    - Phase 2: Structure (hook and titles)
    - Phase 3: Content (individual slides)
    - Phase 4: CTA (call-to-action)
    """

    def __init__(self):
        self.events: list[dict] = []
        self.current_status = "Initializing..."
        self.stats = {
            "text_calls": 0,
            "image_calls": 0,
            "total_cost": 0.0,
            "providers_used": set(),
            "text_provider": None,
            "text_model": None,
            "text_prompt_preview": None,
            "image_provider": None,
            "image_model": None,
            "image_prompt_preview": None,
        }
        # Phase tracking - store results for each completed phase
        self.phase_results: dict[int, dict] = {}
        self.generated_slides: list[dict] = []
        self.slide_providers: dict[int, dict] = {}
        self._last_phase = 0

        # Web search tracking
        self.web_search_complete = False
        self.web_search_results: dict = {}

    def add_event(self, progress: GenerationProgress):
        """Add a progress event."""
        event = {
            "step": progress.current_step,
            "event_type": progress.event_type,
            "provider": progress.provider,
            "model": progress.model,
            "prompt": progress.prompt_preview,
            "response": progress.response_preview,
            "duration": progress.duration_seconds,
            "cost": progress.cost_usd,
        }
        self.events.append(event)
        self.current_status = progress.current_step

        # Update stats from progress totals
        self.stats["text_calls"] = progress.total_text_calls
        self.stats["image_calls"] = progress.total_image_calls
        self.stats["total_cost"] = progress.total_cost_usd

        # Track providers (persist values)
        if progress.text_provider:
            self.stats["text_provider"] = progress.text_provider
        if progress.text_model:
            self.stats["text_model"] = progress.text_model
        if progress.text_prompt_preview:
            self.stats["text_prompt_preview"] = progress.text_prompt_preview
        if progress.text_provider:
            self.stats["providers_used"].add(progress.text_provider)

        if progress.image_provider:
            self.stats["image_provider"] = progress.image_provider
            self.stats["providers_used"].add(progress.image_provider)
        if progress.image_model:
            self.stats["image_model"] = progress.image_model
        if progress.image_prompt_preview:
            self.stats["image_prompt_preview"] = progress.image_prompt_preview

        # Track generated slides
        if progress.generated_slides:
            self.generated_slides = progress.generated_slides

        # Track web search results
        if progress.web_search_status:
            self.web_search_results = {
                "status": progress.web_search_status,
                "queries": progress.web_search_queries,
                "results": progress.web_search_results,
                "sources": progress.web_search_sources,
                "domains": progress.web_search_domains,
                "duration_ms": progress.web_search_duration_ms,
            }
            if progress.web_search_status == "complete":
                self.web_search_complete = True

        # Track phase completion
        current_phase = progress.current_phase
        if current_phase > 0:
            if current_phase not in self.phase_results or progress.text_provider:
                self.phase_results[current_phase] = {
                    "provider": progress.text_provider or self.stats.get("text_provider") or "",
                    "model": progress.text_model or self.stats.get("text_model") or "",
                    "prompt_preview": progress.text_prompt_preview or progress.phase_input or "",
                    "output": progress.phase_output or "",
                }

            # Track slide-level providers for Phase 3
            if current_phase == 3 and progress.current_slide > 0:
                slide_num = progress.current_slide
                if progress.text_provider:
                    self.slide_providers[slide_num] = {
                        "provider": progress.text_provider,
                        "model": progress.text_model or "",
                    }

        self._last_phase = current_phase

    def _render_web_search_box(self) -> str:
        """Render the web search phase box."""
        ws = self.web_search_results

        if not ws:
            return ""

        status = ws.get("status", "")
        is_complete = status == "complete"
        is_current = status == "searching"
        is_failed = status == "failed"
        is_skipped = status == "skipped"

        if is_current:
            border = "[bold yellow]"
            status_icon = "[yellow]...[/]"
        elif is_complete:
            border = "[green]"
            status_icon = "[green][OK][/]"
        elif is_failed:
            border = "[red]"
            status_icon = "[red][X][/]"
        elif is_skipped:
            border = "[dim]"
            status_icon = "[dim][-][/]"
        else:
            border = "[dim]"
            status_icon = "[dim][ ][/]"

        lines = []
        lines.append(f"{border}+{'-' * 58}+[/]")
        lines.append(f"{border}|[/] {status_icon} [bold]Phase 0: Web Research[/]{' ' * 33} {border}|[/]")

        if is_current:
            queries = ws.get("queries", [])
            if queries:
                lines.append(f"{border}|[/]   [dim]Searching {len(queries)} queries...[/]{' ' * 30} {border}|[/]")
        elif is_complete:
            results = ws.get("results", 0)
            sources = ws.get("sources", 0)
            duration = ws.get("duration_ms", 0)
            domains = ws.get("domains", [])[:4]

            lines.append(f"{border}|[/]   [cyan]Results:[/] {results} found, {sources} unique sources{' ' * (26 - len(str(results)) - len(str(sources)))} {border}|[/]")

            if domains:
                domain_str = ", ".join(domains)
                if len(domain_str) > 45:
                    domain_str = domain_str[:42] + "..."
                padding = 45 - len(domain_str)
                lines.append(f"{border}|[/]   [dim]Domains:[/] {domain_str}{' ' * padding} {border}|[/]")

            duration_str = f"{duration}ms"
            lines.append(f"{border}|[/]   [dim]Duration:[/] {duration_str}{' ' * (44 - len(duration_str))} {border}|[/]")
        elif is_failed:
            lines.append(f"{border}|[/]   [red]Search failed - continuing without research[/]{' ' * 8} {border}|[/]")
        elif is_skipped:
            lines.append(f"{border}|[/]   [dim]Skipped (context provided)[/]{' ' * 26} {border}|[/]")

        lines.append(f"{border}+{'-' * 58}+[/]")

        return "\n".join(lines)

    def _render_phase_box(
        self,
        phase_num: int,
        name: str,
        input_text: str,
        output_text: str,
        is_current: bool,
        is_complete: bool,
        provider: str = "",
        model: str = "",
        prompt_preview: str = "",
    ) -> str:
        """Render a single phase box with provider/model info."""
        if is_current:
            border = "[bold yellow]"
            status_icon = "[yellow]...[/]"
        elif is_complete:
            border = "[green]"
            status_icon = "[green][OK][/]"
        else:
            border = "[dim]"
            status_icon = "[dim][ ][/]"

        lines = []
        lines.append(f"{border}+{'-' * 58}+[/]")

        # Show provider/model on header line
        provider_model = ""
        if provider:
            if model:
                provider_model = f" [cyan]({provider}/{model})[/]"
            else:
                provider_model = f" [cyan]({provider})[/]"

        if provider_model:
            padding = max(0, 42 - len(name) - len(provider) - (len(model) + 1 if model else 0))
            lines.append(f"{border}|[/] {status_icon} [bold]Phase {phase_num}: {name}[/]{provider_model}{' ' * padding} {border}|[/]")
        else:
            lines.append(f"{border}|[/] {status_icon} [bold]Phase {phase_num}: {name}[/]{' ' * (45 - len(name))} {border}|[/]")

        # Show prompt preview
        if prompt_preview:
            prompt_display = prompt_preview[:45] + "..." if len(prompt_preview) > 45 else prompt_preview
            prompt_display = prompt_display.replace("\n", " ").strip()
            lines.append(f"{border}|[/]   [dim]Prompt:[/] {prompt_display[:45]:<45} {border}|[/]")

        if output_text:
            output_display = output_text[:45] + "..." if len(output_text) > 45 else output_text
            lines.append(f"{border}|[/]   [cyan]Output:[/] {output_display:<45} {border}|[/]")

        lines.append(f"{border}+{'-' * 58}+[/]")

        return "\n".join(lines)

    def render(self, progress: GenerationProgress) -> Panel:
        """Render the phase-based progress display."""
        lines = []

        # Header with topic
        lines.append(f"[bold]Topic:[/] {progress.post_id}")
        lines.append("")

        # Phase visualization
        current_phase = progress.current_phase
        content_count = progress.content_count or 5
        content_type = progress.content_type or "item"

        # Get current provider info
        current_provider = progress.text_provider or ""
        current_model = progress.text_model or ""
        current_prompt = progress.text_prompt_preview or ""

        # Phase 0: Web Research
        web_search_box = self._render_web_search_box()
        if web_search_box:
            lines.append(web_search_box)
            if self.web_search_complete or progress.web_search_status in ("complete", "failed", "skipped"):
                lines.append("                              [dim]v[/]")

        # Phase 1: Planning
        phase1_complete = current_phase > 1
        phase1_current = current_phase == 1
        phase1_output = f"{content_count} {content_type}s" if phase1_complete else ""

        p1_result = self.phase_results.get(1, {})
        p1_provider = p1_result.get("provider", "") if phase1_complete else (current_provider if phase1_current else "")
        p1_model = p1_result.get("model", "") if phase1_complete else (current_model if phase1_current else "")
        p1_prompt = p1_result.get("prompt_preview", "") if phase1_complete else (current_prompt if phase1_current else "")

        lines.append(self._render_phase_box(
            1, "Planning",
            "",
            phase1_output if phase1_complete else (progress.phase_output if phase1_current else ""),
            phase1_current, phase1_complete,
            provider=p1_provider,
            model=p1_model,
            prompt_preview=p1_prompt if (phase1_current or phase1_complete) else "",
        ))

        if current_phase >= 1:
            lines.append("                              [dim]v[/]")

        # Phase 2: Structure
        phase2_complete = current_phase > 2
        phase2_current = current_phase == 2
        if current_phase >= 2:
            p2_result = self.phase_results.get(2, {})
            p2_provider = p2_result.get("provider", "") if phase2_complete else (current_provider if phase2_current else "")
            p2_model = p2_result.get("model", "") if phase2_complete else (current_model if phase2_current else "")
            p2_prompt = p2_result.get("prompt_preview", "") if phase2_complete else (current_prompt if phase2_current else "")

            lines.append(self._render_phase_box(
                2, "Structure",
                "",
                progress.phase_output if (phase2_current or phase2_complete) else "",
                phase2_current, phase2_complete,
                provider=p2_provider,
                model=p2_model,
                prompt_preview=p2_prompt if (phase2_current or phase2_complete) else "",
            ))
            lines.append("                              [dim]v[/]")

        # Phase 3: Content (show individual slides)
        if current_phase >= 3:
            p3_result = self.phase_results.get(3, {})
            p3_provider = p3_result.get("provider", "") or current_provider
            p3_model = p3_result.get("model", "") or current_model

            phase3_header = f"Phase 3: Content ({content_count} {content_type}s)"
            provider_model_str = ""
            if p3_provider:
                provider_model_str = f" [cyan]({p3_provider}" + (f"/{p3_model})" if p3_model else ")") + "[/]"
            lines.append(f"[{'bold yellow' if current_phase == 3 else 'green' if current_phase > 3 else 'dim'}]+{'-' * 58}+[/]")
            lines.append(f"[{'bold yellow' if current_phase == 3 else 'green' if current_phase > 3 else 'dim'}]|[/] [bold]{phase3_header}[/]{provider_model_str}")

            # Show slide titles and their status
            slide_titles = progress.slide_titles or []
            generated = self.generated_slides

            for i, title in enumerate(slide_titles):
                slide_num = i + 1
                if slide_num <= len(generated):
                    heading = generated[i].get("heading", title)[:30]
                    slide_info = self.slide_providers.get(slide_num, {})
                    slide_prov = slide_info.get("provider", p3_provider)
                    prov_str = f" [dim]({slide_prov})[/]" if slide_prov else ""
                    lines.append(f"[green]|[/]   [green][OK][/] Slide {slide_num}: {heading}...{prov_str}")
                elif current_phase == 3 and progress.phase_input and f"Slide {slide_num}" in progress.phase_input:
                    provider_suffix = f" [cyan]({current_provider}/{current_model})[/]" if current_provider else "[yellow](generating)[/]"
                    lines.append(f"[yellow]|[/]   [yellow]...[/] Slide {slide_num}: {title[:25]}... {provider_suffix}")
                else:
                    lines.append(f"[dim]|[/]   [ ] Slide {slide_num}: {title[:35]}...")

            lines.append(f"[{'bold yellow' if current_phase == 3 else 'green' if current_phase > 3 else 'dim'}]+{'-' * 58}+[/]")
            lines.append("                              [dim]v[/]")

        # Phase 4: CTA
        if current_phase >= 4:
            phase4_complete = progress.phase_name == "Complete"
            phase4_current = current_phase == 4 and not phase4_complete

            p4_result = self.phase_results.get(4, {})
            p4_provider = p4_result.get("provider", "") if phase4_complete else (current_provider if phase4_current else "")
            p4_model = p4_result.get("model", "") if phase4_complete else (current_model if phase4_current else "")
            p4_prompt = p4_result.get("prompt_preview", "") if phase4_complete else (current_prompt if phase4_current else "")

            lines.append(self._render_phase_box(
                4, "CTA",
                "",
                progress.phase_output if phase4_complete else "",
                phase4_current, phase4_complete,
                provider=p4_provider,
                model=p4_model,
                prompt_preview=p4_prompt if (phase4_current or phase4_complete) else "",
            ))

        lines.append("")

        # Current action
        if progress.current_action:
            action_color = "yellow" if "..." in progress.current_action else "green"
            lines.append(f"[bold {action_color}]Current:[/] {progress.current_action}")

        # Stats footer
        lines.append("")
        lines.append(f"[dim]API Calls: {progress.total_text_calls} text | Cost: ${progress.total_cost_usd:.4f}[/]")

        # Provider info
        text_provider = progress.text_provider or self.stats.get("text_provider")
        if text_provider:
            text_model = progress.text_model or self.stats.get("text_model") or ""
            lines.append(f"[dim]Provider: {text_provider} / {text_model}[/]")

        return Panel(
            "\n".join(lines),
            title="[bold blue]Content Generation[/]",
            border_style="blue",
            box=box.ROUNDED,
        )


class InstagramPostingDisplay:
    """Rich display for Instagram publishing progress."""

    def __init__(self):
        from .instagram.models import InstagramProgress
        self.progress = InstagramProgress()

    def update(self, progress):
        """Update the progress state."""
        self.progress = progress

    def render(self) -> Panel:
        """Render the progress display."""
        from .instagram.models import InstagramPostStatus

        progress = self.progress
        lines = []

        # Status with color
        status_color = {
            InstagramPostStatus.PENDING: "white",
            InstagramPostStatus.UPLOADING: "yellow",
            InstagramPostStatus.CREATING_CONTAINERS: "cyan",
            InstagramPostStatus.PUBLISHING: "blue",
            InstagramPostStatus.PUBLISHED: "green",
            InstagramPostStatus.FAILED: "red",
        }.get(progress.status, "white")

        lines.append(f"[bold {status_color}]Status:[/] {progress.status.value.upper()}")
        lines.append(f"[bold]Step:[/] {progress.current_step}")
        lines.append(f"[bold]Progress:[/] {progress.progress_percent:.0f}%")
        lines.append("")

        # Image upload progress
        if progress.total_images > 0:
            lines.append(f"[bold cyan]Images:[/] {progress.images_uploaded}/{progress.total_images} uploaded")
            lines.append(f"[bold cyan]Containers:[/] {progress.containers_created}/{progress.total_images} created")
            lines.append("")

        # Error display
        if progress.error:
            lines.append(f"[bold red]Error:[/] {progress.error}")

        return Panel(
            "\n".join(lines),
            title="[bold]Instagram Publishing[/]",
            border_style="blue",
            box=box.ROUNDED,
        )
