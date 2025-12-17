"""Rich console singleton for CLI output."""

from rich.console import Console

# Global console instance - used across all CLI modules
console = Console()


def print_error(message: str, details: dict | None = None) -> None:
    """Print an error message.

    Args:
        message: Error message
        details: Optional details dict
    """
    console.print(f"[red]Error: {message}[/red]")
    if details:
        for key, value in details.items():
            console.print(f"  [dim]{key}:[/dim] [yellow]{value}[/yellow]")


def print_warning(message: str) -> None:
    """Print a warning message.

    Args:
        message: Warning message
    """
    console.print(f"[yellow]Warning: {message}[/yellow]")


def print_success(message: str) -> None:
    """Print a success message.

    Args:
        message: Success message
    """
    console.print(f"[green]{message}[/green]")


def print_info(message: str) -> None:
    """Print an info message.

    Args:
        message: Info message
    """
    console.print(f"[cyan]{message}[/cyan]")
