"""
Evaluation commands for Vizra CLI.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import click
from ..evaluation import EvaluationRunner
from .display import (
    console, print_error, print_success, print_warning, print_info,
    create_table, create_panel, create_progress_bar, print_json,
    EMOJIS, COLORS, format_metric_value, print_separator
)


@click.group(name='eval')
def eval_group():
    """Run and manage evaluations."""
    pass


@eval_group.command(name='list')
def list_evaluations():
    """List all available evaluations."""
    try:
        runner = EvaluationRunner()
        evaluations = runner.list_evaluations()
        
        if not evaluations:
            print_warning("No evaluations found.")
            console.print("\n[dim]Make sure you have evaluation classes defined and accessible.[/dim]")
            return
        
        # Create a beautiful table
        table = create_table(
            f"Available Evaluations ({len(evaluations)})",
            ["Name", "Description", "Agent", "Class"],
            []
        )
        
        for eval_info in evaluations:
            table.add_row(
                f"[bold green]{eval_info['name']}[/bold green]",
                eval_info['description'],
                f"{EMOJIS['robot']} {eval_info['agent_name']}",
                f"[dim]{eval_info['class']}[/dim]"
            )
        
        console.print()
        console.print(table)
        console.print()
        console.print(f"Run an evaluation with: [cyan]vizra eval run <name>[/cyan]")
        
    except Exception as e:
        print_error(f"Error listing evaluations: {e}")
        sys.exit(1)


@eval_group.command(name='run')
@click.argument('evaluation_name')
@click.option('--output', '-o', help='Save results to custom file path (JSON format)')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.option('--limit', '-l', type=int, help='Limit number of test cases to run')
@click.option('--detailed', '-d', is_flag=True, help='Generate detailed CSV with all metric columns')
@click.option('--json', '-j', is_flag=True, help='Generate JSON results file')
def run_evaluation(evaluation_name, output, verbose, limit, detailed, json):
    """Run a specific evaluation."""
    try:
        runner = EvaluationRunner()
        
        # Check if evaluation exists
        available = [e['name'] for e in runner.list_evaluations()]
        if evaluation_name not in available:
            print_error(f"Evaluation '{evaluation_name}' not found.")
            console.print(f"\nAvailable evaluations: [cyan]{', '.join(available)}[/cyan]")
            console.print(f"Run '[cyan]vizra eval list[/cyan]' to see details.")
            sys.exit(1)
        
        # Run evaluation
        console.print()
        console.print(f"{EMOJIS['rocket']} [bold green]Running evaluation: {evaluation_name}[/bold green]")
        
        if limit:
            print_info(f"Limited to {limit} test case(s)")
        
        if not verbose:
            with create_progress_bar(f"Running {evaluation_name}") as progress:
                task = progress.add_task(f"[cyan]Evaluating...", total=limit or 100)
                results = runner.run_evaluation(evaluation_name, limit=limit)
                progress.update(task, completed=limit or 100)
        else:
            results = runner.run_evaluation(evaluation_name, limit=limit)
        
        # Display summary
        console.print()
        print_separator()
        
        total = results.get('total_cases', 0)
        passed = results.get('passed', 0)
        failed = results.get('failed', 0)
        success_rate = results.get('success_rate', 0)
        
        # Create summary panel
        summary_content = f"""
{EMOJIS['chart']} [bold cyan]Summary[/bold cyan]
  Total cases: [white]{total}[/white]
  {EMOJIS['checkmark']} Passed: [bold green]{passed}[/bold green]
  {EMOJIS['cross']} Failed: [bold red]{failed}[/bold red]
  {EMOJIS['trophy']} Success rate: {format_metric_value(success_rate, 'percentage')}
        """
        
        summary_panel = create_panel(
            summary_content.strip(),
            title="Evaluation Results",
            style="cyan"
        )
        console.print(summary_panel)
        
        # Show failed cases if any
        if failed > 0 and 'results' in results:
            console.print()
            console.print(f"[bold red]Failed Cases:[/bold red]\n")
            failed_cases = [r for r in results['results'] if not r.get('passed', False)]
            
            # Create failed cases table
            failed_table = create_table(
                "",
                ["#", "Prompt", "Failed Metrics", "Error"],
                [],
                show_edge=False
            )
            
            for i, case in enumerate(failed_cases[:5]):  # Show first 5
                prompt = case.get('prompt', 'N/A')[:60] + '...' if len(case.get('prompt', '')) > 60 else case.get('prompt', 'N/A')
                
                failed_metrics = []
                if 'metrics' in case:
                    for metric_name, metric_result in case['metrics'].items():
                        if not metric_result['passed']:
                            failed_metrics.append(f"{EMOJIS['cross']} {metric_name}")
                
                error = case.get('error', '-')[:40] + '...' if len(case.get('error', '')) > 40 else case.get('error', '-')
                
                failed_table.add_row(
                    str(i+1),
                    f"[yellow]{prompt}[/yellow]",
                    "\n".join(failed_metrics) if failed_metrics else "-",
                    f"[red]{error}[/red]" if error != '-' else "-"
                )
            
            console.print(failed_table)
            
            if len(failed_cases) > 5:
                console.print(f"\n[dim]... and {len(failed_cases) - 5} more failed cases[/dim]")
        
        # Always save results to evaluations/results folder
        # Create results directory if it doesn't exist
        results_dir = Path('evaluations/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model name from results
        model_name = results.get('model', 'unknown')
        # Clean model name for filename (remove special chars)
        model_name = model_name.replace('/', '-').replace(':', '-')
        
        # Generate timestamp and base filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{timestamp}_{evaluation_name}_{model_name}"
        
        # Save JSON only if requested
        if json:
            json_filename = f"{base_filename}.json"
            json_path = results_dir / json_filename
            with open(json_path, 'w') as f:
                import json as json_module
                json_module.dump(results, f, indent=2)
            console.print(f"\n{EMOJIS['save']} JSON results saved to: [cyan]{json_path}[/cyan]")
        
        # Create CSV files
        
        # 1. Summary CSV
        summary_data = {
            'timestamp': [results.get('timestamp', timestamp)],
            'evaluation_name': [results.get('evaluation_name', evaluation_name)],
            'agent_name': [results.get('agent_name', '')],
            'model': [results.get('model', 'unknown')],
            'total_cases': [results.get('total_cases', 0)],
            'passed': [results.get('passed', 0)],
            'failed': [results.get('failed', 0)],
            'success_rate': [results.get('success_rate', 0)]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_csv_path = results_dir / f"{base_filename}_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        console.print(f"{EMOJIS['document']} Summary CSV saved to: [cyan]{summary_csv_path}[/cyan]")
        
        # 2. Simplified results CSV (new format)
        if 'results' in results and results['results']:
            # First create the simplified format
            simple_rows = []
            for idx, row_result in enumerate(results['results']):
                # Truncate prompt for readability
                prompt = row_result.get('prompt', '')
                prompt_short = prompt[:80] + '...' if len(prompt) > 80 else prompt
                
                # Generate failure summary
                failure_summary = ""
                failed_metrics = []
                
                if not row_result.get('passed', False):
                    if row_result.get('error'):
                        failure_summary = f"Error: {row_result['error']}"
                    elif 'metrics' in row_result:
                        summaries = []
                        for metric_name, metric_result in row_result['metrics'].items():
                            if not metric_result.get('passed', False):
                                failed_metrics.append(metric_name)
                                # Generate human-readable summary based on metric type
                                details = metric_result.get('details', {})
                                
                                if 'exact_match' in metric_name:
                                    expected = details.get('expected', 'N/A')
                                    actual = details.get('actual', 'N/A')
                                    summaries.append(f"Expected '{expected}' but got '{actual}'")
                                elif 'tool_usage' in metric_name:
                                    expected_tools = details.get('expected_tools', [])
                                    tools_used = details.get('tools_used', [])
                                    if not tools_used:
                                        summaries.append(f"Tool '{expected_tools[0] if expected_tools else 'N/A'}' was not used")
                                    else:
                                        summaries.append(f"Wrong tool used: expected {expected_tools} but used {tools_used}")
                                elif 'format' in metric_name:
                                    summaries.append("Response format incorrect")
                                else:
                                    # Generic failure message
                                    summaries.append(f"{metric_name} check failed")
                        
                        failure_summary = "; ".join(summaries)
                
                simple_row = {
                    'index': idx + 1,
                    'prompt': prompt_short,
                    'agent_response': row_result.get('response', row_result.get('agent_response', '')),
                    'passed': row_result.get('passed', False),
                    'failure_summary': failure_summary,
                    'failed_metrics': ', '.join(failed_metrics) if failed_metrics else ''
                }
                simple_rows.append(simple_row)
            
            # Save simplified CSV
            simple_df = pd.DataFrame(simple_rows)
            simple_csv_path = results_dir / f"{base_filename}_simple.csv"
            simple_df.to_csv(simple_csv_path, index=False)
            console.print(f"{EMOJIS['document']} Simplified results CSV saved to: [cyan]{simple_csv_path}[/cyan]")
            
            # Create detailed format only if requested
            if detailed:
                rows_data = []
                for idx, row_result in enumerate(results['results']):
                    row_entry = {
                        'index': idx + 1,
                        'prompt': row_result.get('prompt', ''),
                        'agent_response': row_result.get('response', row_result.get('agent_response', '')),
                        'passed': row_result.get('passed', False),
                        'error': row_result.get('error', '')
                    }
                    
                    # Add metric columns
                    if 'metrics' in row_result:
                        for metric_name, metric_result in row_result['metrics'].items():
                            # Add pass/fail status
                            row_entry[f'metric_{metric_name}_passed'] = metric_result.get('passed', False)
                            # Add score if available
                            if 'score' in metric_result:
                                row_entry[f'metric_{metric_name}_score'] = metric_result['score']
                            # Add key details if needed
                            if 'details' in metric_result and isinstance(metric_result['details'], dict):
                                # Only add simple values, not complex nested structures
                                for detail_key, detail_value in metric_result['details'].items():
                                    if isinstance(detail_value, (str, int, float, bool)):
                                        row_entry[f'metric_{metric_name}_{detail_key}'] = detail_value
                    
                    # Add original CSV data columns
                    if 'row_data' in row_result:
                        for key, value in row_result['row_data'].items():
                            if key not in ['prompt']:  # Don't duplicate prompt
                                row_entry[f'csv_{key}'] = value
                    
                    # Add conversation history as JSON string
                    if 'conversation_history' in row_result:
                        row_entry['conversation_history'] = json.dumps(row_result['conversation_history'])
                    
                    rows_data.append(row_entry)
                
                detailed_df = pd.DataFrame(rows_data)
                detailed_csv_path = results_dir / f"{base_filename}_detailed.csv"
                detailed_df.to_csv(detailed_csv_path, index=False)
                console.print(f"{EMOJIS['document']} Detailed results CSV saved to: [cyan]{detailed_csv_path}[/cyan]")
        
        # Save to custom output path if specified (always JSON format)
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                import json as json_module
                json_module.dump(results, f, indent=2)
            console.print(f"{EMOJIS['save']} Custom JSON output saved to: [cyan]{output_path}[/cyan]")
        
        # Exit with appropriate code
        console.print()
        if failed > 0:
            sys.exit(1)
        else:
            print_success("All tests passed!")
            
    except Exception as e:
        print_error(f"Error running evaluation: {e}")
        if verbose:
            import traceback
            console.print_exception(show_locals=True)
        sys.exit(1)