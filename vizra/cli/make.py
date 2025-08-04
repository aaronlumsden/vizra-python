"""
Make commands for generating Vizra components.
"""

import os
import click
from pathlib import Path
from .templates import (
    get_agent_template,
    get_tool_template,
    get_evaluation_template,
    get_training_template,
    get_metric_template
)
from .display import console, print_success, print_error


def to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    return ''.join(word.capitalize() for word in snake_str.split('_'))


def ensure_suffix(name: str, suffixes: list, default_suffix: str) -> tuple[str, str]:
    """
    Ensure the name has an appropriate suffix for class naming.
    Returns (file_name, class_name).
    """
    name_lower = name.lower()
    
    # Check if name already ends with any of the suffixes
    has_suffix = any(name_lower.endswith(suffix.lower()) for suffix in suffixes)
    
    # File name is always the input name
    file_name = name
    
    # Class name needs proper casing and suffix
    if has_suffix:
        class_name = to_pascal_case(name)
    else:
        class_name = to_pascal_case(name) + default_suffix
    
    return file_name, class_name


def create_file(directory: str, file_name: str, content: str) -> Path:
    """Create a file with the given content in the specified directory."""
    # Create directory if it doesn't exist
    dir_path = Path(directory)
    dir_path.mkdir(exist_ok=True)
    
    # Create __init__.py if it doesn't exist
    init_file = dir_path / '__init__.py'
    if not init_file.exists():
        init_file.touch()
    
    # Create the file
    file_path = dir_path / f"{file_name}.py"
    if file_path.exists():
        raise click.ClickException(f"File already exists: {file_path}")
    
    file_path.write_text(content)
    return file_path


@click.group(name='make')
def make_group():
    """Generate boilerplate code for Vizra components."""
    pass


@make_group.command()
@click.argument('name')
def agent(name):
    """Create a new agent class."""
    file_name, class_name = ensure_suffix(
        name, 
        ['_agent', 'agent'], 
        'Agent'
    )
    
    content = get_agent_template(class_name)
    
    try:
        file_path = create_file('agents', file_name, content)
        print_success(f"Created agent: {file_path}")
        console.print(f"[dim]Class name: {class_name}[/dim]")
    except click.ClickException as e:
        print_error(str(e))
        raise


@make_group.command()
@click.argument('name')
def tool(name):
    """Create a new tool class."""
    file_name, class_name = ensure_suffix(
        name,
        ['_tool', 'tool'],
        'Tool'
    )
    
    content = get_tool_template(class_name)
    
    try:
        file_path = create_file('tools', file_name, content)
        print_success(f"Created tool: {file_path}")
        console.print(f"[dim]Class name: {class_name}[/dim]")
    except click.ClickException as e:
        print_error(str(e))
        raise


@make_group.command()
@click.argument('name')
def evaluation(name):
    """Create a new evaluation class."""
    file_name, class_name = ensure_suffix(
        name,
        ['_eval', '_evaluation', 'eval', 'evaluation'],
        'Evaluation'
    )
    
    content = get_evaluation_template(class_name)
    
    try:
        file_path = create_file('evaluations', file_name, content)
        print_success(f"Created evaluation: {file_path}")
        console.print(f"[dim]Class name: {class_name}[/dim]")
    except click.ClickException as e:
        print_error(str(e))
        raise


@make_group.command()
@click.argument('name')
def training(name):
    """Create a new training class."""
    file_name, class_name = ensure_suffix(
        name,
        ['_training', 'training'],
        'Training'
    )
    
    content = get_training_template(class_name)
    
    try:
        file_path = create_file('training', file_name, content)
        print_success(f"Created training routine: {file_path}")
        console.print(f"[dim]Class name: {class_name}[/dim]")
    except click.ClickException as e:
        print_error(str(e))
        raise


@make_group.command()
@click.argument('name')
def metric(name):
    """Create a new metric class."""
    file_name, class_name = ensure_suffix(
        name,
        ['_metric', 'metric'],
        'Metric'
    )
    
    content = get_metric_template(class_name)
    
    try:
        file_path = create_file('metrics', file_name, content)
        print_success(f"Created metric: {file_path}")
        console.print(f"[dim]Class name: {class_name}[/dim]")
    except click.ClickException as e:
        print_error(str(e))
        raise