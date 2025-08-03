"""
Training commands for Vizra CLI.
"""

import json
import sys
from pathlib import Path
import click
from ..training import TrainingRunner
from .display import (
    console, print_error, print_success, print_warning, print_info,
    create_table, create_panel, create_progress_bar, print_json,
    EMOJIS, COLORS, format_metric_value, print_separator
)
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout


@click.group(name='train')
def train_group():
    """Run and manage training."""
    pass


@train_group.command(name='list')
def list_trainings():
    """List all available training routines."""
    try:
        runner = TrainingRunner()
        trainings = runner.list_trainings()
        
        if not trainings:
            print_warning("No training routines found.")
            console.print("\n[dim]Make sure you have training classes defined and accessible.[/dim]")
            return
        
        # Create a beautiful table
        table = create_table(
            f"Available Training Routines ({len(trainings)})",
            ["Name", "Description", "Agent", "Algorithm", "Class"],
            []
        )
        
        for train_info in trainings:
            table.add_row(
                f"[bold green]{train_info['name']}[/bold green]",
                train_info['description'],
                f"{EMOJIS['robot']} {train_info['agent_name']}",
                f"{EMOJIS['target']} {train_info['algorithm']}",
                f"[dim]{train_info['class']}[/dim]"
            )
        
        console.print()
        console.print(table)
        console.print()
        console.print(f"Run training with: [cyan]vizra train run <name>[/cyan]")
        
    except Exception as e:
        print_error(f"Error listing training routines: {e}")
        sys.exit(1)


@train_group.command(name='run')
@click.argument('training_name')
@click.option('--iterations', '-i', type=int, help='Override number of iterations')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--output', '-o', help='Save results to file (JSON format)')
def run_training(training_name, iterations, verbose, output):
    """Run a specific training routine."""
    try:
        runner = TrainingRunner()
        
        # Check if training exists
        available = [t['name'] for t in runner.list_trainings()]
        if training_name not in available:
            print_error(f"Training '{training_name}' not found.")
            console.print(f"\nAvailable training routines: [cyan]{', '.join(available)}[/cyan]")
            console.print(f"Run '[cyan]vizra train list[/cyan]' to see details.")
            sys.exit(1)
        
        # Modify iterations if specified
        if iterations:
            # Get the training class and modify its iterations
            train_class = runner.trainings[training_name]
            original_iterations = train_class.n_iterations
            train_class.n_iterations = iterations
            print_info(f"Overriding iterations: {original_iterations} → {iterations}")
        
        # Run training
        console.print()
        console.print(f"{EMOJIS['rocket']} [bold green]Starting training: {training_name}[/bold green]")
        
        if not verbose:
            # Create a rich progress display
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task(f"[cyan]Training {training_name}...", total=iterations or 100)
                
                # Create a live display for metrics
                layout = Layout()
                layout.split_column(
                    Layout(name="progress", size=3),
                    Layout(name="metrics", size=6)
                )
                
                # Run training
                results = runner.run_training(training_name)
                progress.update(task, completed=iterations or 100)
        else:
            results = runner.run_training(training_name)
        
        # Display summary
        console.print()
        print_separator()
        
        if 'error' in results:
            print_error(f"Training failed: {results['error']}")
            sys.exit(1)
        
        # Extract key metrics
        total_iterations = results.get('total_iterations', 0)
        best_reward = results.get('best_reward', 0)
        final_metrics = results.get('final_metrics', {})
        hyperparams = results.get('hyperparameters', {})
        
        # Create summary panel
        summary_content = f"""
{EMOJIS['chart']} [bold cyan]Training Summary[/bold cyan]

[bold]Configuration:[/bold]
  Algorithm: [white]{hyperparams.get('algorithm', 'Unknown')}[/white]
  Learning rate: [white]{hyperparams.get('learning_rate', 'N/A')}[/white]
  Batch size: [white]{hyperparams.get('batch_size', 'N/A')}[/white]
  Iterations: [white]{total_iterations}[/white]

[bold]Results:[/bold]
  {EMOJIS['trophy']} Best reward: [bold green]{best_reward:.3f}[/bold green]
        """
        
        summary_panel = create_panel(
            summary_content.strip(),
            title="Training Results",
            style="cyan"
        )
        console.print(summary_panel)
        
        if final_metrics:
            # Create metrics table
            metrics_table = create_table(
                "Final Metrics",
                ["Metric", "Value"],
                [
                    ["Average reward", f"{final_metrics.get('avg_reward', 0):.3f}"],
                    ["Success rate", format_metric_value(final_metrics.get('success_rate', 0) * 100, 'percentage')],
                    ["Min reward", f"{final_metrics.get('min_reward', 0):.3f}"],
                    ["Max reward", f"{final_metrics.get('max_reward', 0):.3f}"]
                ],
                show_edge=False
            )
            console.print()
            console.print(metrics_table)
        
        # Show training progress if available
        if 'training_history' in results and results['training_history']:
            history = results['training_history']
            start_reward = history[0]['avg_reward']
            end_reward = history[-1]['avg_reward']
            improvement = end_reward - start_reward
            
            # Create progress panel
            progress_content = f"""
{EMOJIS['chart']} [bold]Training Progress[/bold]
  Starting reward: [white]{start_reward:.3f}[/white]
  Final reward: [white]{end_reward:.3f}[/white]
  Improvement: {f'[green]+{improvement:.3f}[/green]' if improvement > 0 else f'[red]{improvement:.3f}[/red]'}
            """
            
            progress_panel = create_panel(
                progress_content.strip(),
                title="Performance",
                style="magenta"
            )
            console.print()
            console.print(progress_panel)
        
        # Save results if requested
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n{EMOJIS['save']} Results saved to: [cyan]{output_path}[/cyan]")
        
        console.print()
        print_success("Training completed successfully!")
            
    except Exception as e:
        print_error(f"Error running training: {e}")
        if verbose:
            console.print_exception(show_locals=True)
        sys.exit(1)