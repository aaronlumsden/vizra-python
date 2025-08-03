"""
Rich display utilities for Vizra CLI.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.syntax import Syntax
from rich.tree import Tree
from rich.text import Text
from rich.layout import Layout
from rich.align import Align
from rich import box
from typing import List, Dict, Any, Optional


# Global console instance
console = Console()

# Color scheme
COLORS = {
    'primary': 'cyan',
    'success': 'green',
    'error': 'red',
    'warning': 'yellow',
    'info': 'blue',
    'accent': 'magenta',
    'dim': 'bright_black'
}

# Emoji mappings
EMOJIS = {
    'rocket': '🚀',
    'checkmark': '✅',
    'cross': '❌',
    'warning': '⚠️',
    'info': '📋',
    'robot': '🤖',
    'package': '📦',
    'chart': '📊',
    'trophy': '🏆',
    'fire': '🔥',
    'sparkles': '✨',
    'target': '🎯',
    'gear': '⚙️',
    'save': '💾',
    'document': '📄',
    'folder': '📁',
    'clock': '⏱️',
    'magnify': '🔍',
    'pencil': '✏️',
    'book': '📚'
}


def get_banner():
    """Get the Vizra ASCII banner with gradient colors."""
    banner_text = """██╗   ██╗██╗███████╗██████╗  █████╗ 
██║   ██║██║╚══███╔╝██╔══██╗██╔══██╗
██║   ██║██║  ███╔╝ ██████╔╝███████║
╚██╗ ██╔╝██║ ███╔╝  ██╔══██╗██╔══██║
 ╚████╔╝ ██║███████╗██║  ██║██║  ██║
  ╚═══╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝"""
    
    # Create gradient effect
    lines = banner_text.split('\n')
    gradient_lines = []
    colors = ['bright_cyan', 'cyan', 'blue', 'bright_blue', 'magenta', 'bright_magenta']
    
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        gradient_lines.append(f"[{color}]{line}[/{color}]")
    
    return '\n'.join(gradient_lines)


def show_welcome():
    """Display the welcome banner."""
    banner = get_banner()
    tagline = "[bright_white]AI Agent Framework with Evaluation and Training[/bright_white]"
    
    # Create a renderable group with centered content
    from rich.console import Group
    content = Group(
        Align.center(Text.from_markup(banner)),
        Text(""),  # Empty line
        Align.center(Text.from_markup(tagline))
    )
    
    # Create a panel with the banner
    panel = Panel(
        content,
        border_style="cyan",
        box=box.DOUBLE_EDGE,
        padding=(1, 2)
    )
    
    console.print(panel)


def create_table(title: str, columns: List[str], rows: List[List[str]], 
                 show_header: bool = True, show_edge: bool = True) -> Table:
    """Create a styled table."""
    table = Table(
        title=title,
        show_header=show_header,
        header_style="bold cyan",
        show_edge=show_edge,
        box=box.ROUNDED,
        title_style="bold bright_cyan",
        row_styles=["", "bright_black"]  # Alternating row colors
    )
    
    # Add columns
    for col in columns:
        table.add_column(col, style="white")
    
    # Add rows
    for row in rows:
        table.add_row(*row)
    
    return table


def create_panel(content: str, title: str = "", style: str = "cyan", 
                 box_style: Any = box.ROUNDED) -> Panel:
    """Create a styled panel."""
    return Panel(
        content,
        title=title,
        border_style=style,
        box=box_style,
        padding=(1, 2)
    )


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a styled progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    )


def print_error(message: str):
    """Print an error message."""
    console.print(f"{EMOJIS['cross']} [bold red]Error:[/bold red] {message}")


def print_success(message: str):
    """Print a success message."""
    console.print(f"{EMOJIS['checkmark']} [bold green]{message}[/bold green]")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"{EMOJIS['warning']} [bold yellow]Warning:[/bold yellow] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"{EMOJIS['info']} [cyan]{message}[/cyan]")


def print_json(data: Dict[str, Any], title: str = "JSON Output"):
    """Print JSON data with syntax highlighting."""
    import json
    json_str = json.dumps(data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    
    panel = Panel(syntax, title=title, border_style="blue", box=box.ROUNDED)
    console.print(panel)


def create_status_table(status_data: Dict[str, Any]) -> Table:
    """Create a status table for showing key-value pairs."""
    table = Table(show_header=False, box=box.SIMPLE, show_edge=False)
    table.add_column("Key", style="cyan", width=20)
    table.add_column("Value", style="white")
    
    for key, value in status_data.items():
        table.add_row(key, str(value))
    
    return table


def create_tree(title: str, items: Dict[str, List[str]]) -> Tree:
    """Create a tree structure for hierarchical display."""
    tree = Tree(f"[bold cyan]{title}[/bold cyan]")
    
    for category, subitems in items.items():
        branch = tree.add(f"[yellow]{category}[/yellow]")
        for item in subitems:
            branch.add(f"[white]{item}[/white]")
    
    return tree


def format_metric_value(value: Any, metric_type: str = "default") -> str:
    """Format metric values with appropriate styling."""
    if metric_type == "percentage":
        color = "green" if value >= 80 else "yellow" if value >= 50 else "red"
        return f"[{color}]{value:.1f}%[/{color}]"
    elif metric_type == "boolean":
        return f"[green]✓[/green]" if value else f"[red]✗[/red]"
    elif metric_type == "score":
        return f"[cyan]{value:.3f}[/cyan]"
    else:
        return str(value)


def create_live_display() -> Layout:
    """Create a layout for live updates during long operations."""
    layout = Layout()
    
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    
    layout["body"].split_row(
        Layout(name="main", ratio=2),
        Layout(name="sidebar", ratio=1)
    )
    
    return layout


def print_separator(style: str = "cyan"):
    """Print a styled separator line."""
    console.print("─" * console.width, style=style)