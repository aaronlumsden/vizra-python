"""
Vizra CLI - Command Line Interface for the Vizra AI Agent Framework.
"""

import click
from .. import __version__
from .display import (
    console, show_welcome, print_success, print_info,
    create_table, create_panel, EMOJIS, COLORS
)


@click.group()
@click.version_option(version=__version__, prog_name="vizra")
@click.pass_context
def cli(ctx):
    """
    Vizra - AI Agent Framework with Evaluation and Training.
    
    Build, evaluate, and train AI agents with ease.
    """
    # Ensure that ctx.obj exists and is a dict (in case we need to pass data between commands)
    ctx.ensure_object(dict)
    
    # Show welcome banner on main command (not on subcommands)
    if ctx.invoked_subcommand is None:
        show_welcome()


# Import and register subcommands
from .eval import eval_group
from .train import train_group
from .make import make_group

cli.add_command(eval_group)
cli.add_command(train_group)
cli.add_command(make_group)


# Add a simple status command at the root level
@cli.command()
def status():
    """Show Vizra installation status."""
    # Show welcome banner
    show_welcome()
    
    console.print()
    
    # Create status panel
    status_content = f"""
{EMOJIS['checkmark']} [bold green]Vizra v{__version__} is installed and ready![/bold green]

[bold cyan]Available Commands:[/bold cyan]
  {EMOJIS['chart']} [cyan]vizra eval[/cyan]  - Run and manage evaluations
  {EMOJIS['rocket']} [cyan]vizra train[/cyan] - Run and manage training
  {EMOJIS['info']} [cyan]vizra status[/cyan] - Show this status

[dim]Run 'vizra --help' for more information.[/dim]
    """
    
    panel = create_panel(
        status_content.strip(),
        title="System Status",
        style="green"
    )
    
    console.print(panel)