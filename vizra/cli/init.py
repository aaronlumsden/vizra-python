"""
Init command for scaffolding new Vizra projects.
"""

import os
import click
from pathlib import Path
from .display import console, print_success, print_info, print_error, create_panel, EMOJIS


def get_vizra_config_template() -> str:
    """Generate vizra_config.py template."""
    return '''"""
Vizra project configuration file.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
AGENTS_DIR = PROJECT_ROOT / "agents"
TOOLS_DIR = PROJECT_ROOT / "tools"
EVALUATIONS_DIR = PROJECT_ROOT / "evaluations"
TRAINING_DIR = PROJECT_ROOT / "training"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
METRICS_DIR = PROJECT_ROOT / "metrics"

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))

EVALUATION_CONFIG = {
    "batch_size": 10,
    "max_workers": 5,
    "timeout": 60,
    "retry_attempts": 3,
}

TRAINING_CONFIG = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10,
    "validation_split": 0.2,
}

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
'''


def get_gitignore_template() -> str:
    """Generate .gitignore template."""
    return '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Environment variables
.env
.env.local
.env.*.local

# Vizra specific
*.log
*.cache
results/
outputs/
checkpoints/
*.pkl
*.joblib

# Data files (uncomment if you want to exclude data)
# data/*.csv
# data/*.json

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.coverage
.pytest_cache/
htmlcov/
.tox/
'''


def get_env_example_template() -> str:
    """Generate .env.example template."""
    return '''# OpenAI API Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic API Configuration (optional)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Google API Configuration (optional)
GOOGLE_API_KEY=your-google-api-key-here

# Azure OpenAI Configuration (optional)
AZURE_OPENAI_API_KEY=your-azure-api-key-here
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-here
AZURE_OPENAI_DEPLOYMENT=your-deployment-name-here

# Default Model Settings
DEFAULT_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=2048

# Logging
LOG_LEVEL=INFO

# Other API Keys (add as needed)
# CUSTOM_API_KEY=your-custom-api-key-here
'''


def get_sample_agent_template() -> str:
    """Generate a sample agent template."""
    return '''from vizra import BaseAgent


class ExampleAgent(BaseAgent):
    name = 'example_agent'
    description = 'An example agent to demonstrate Vizra capabilities'
    instructions = \'\'\'You are a helpful AI assistant.
    Provide clear, accurate, and concise responses.
    Be friendly and professional in your interactions.\'\'\'
    model = 'gpt-4o-mini'
    tools = []
'''


def get_sample_tool_template() -> str:
    """Generate a sample tool template."""
    return '''import json
from vizra import ToolInterface, AgentContext


class ExampleTool(ToolInterface):
    def definition(self) -> dict:
        return {
            'name': 'example_tool',
            'description': 'An example tool that processes input',
            'parameters': {
                'type': 'object',
                'properties': {
                    'message': {
                        'type': 'string',
                        'description': 'The message to process',
                    },
                },
                'required': ['message'],
            },
        }

    def execute(self, arguments: dict, context: AgentContext) -> str:
        message = arguments.get('message', '')
        
        result = {
            'success': True,
            'processed_message': f'Processed: {message}',
            'timestamp': context.metadata.get('timestamp', 'N/A')
        }
        
        return json.dumps(result)
'''


def get_readme_template(project_name: str) -> str:
    """Generate README.md template."""
    return f'''# {project_name}

A Vizra AI Agent Framework project.

## Setup

1. Install dependencies:
```bash
pip install vizra
```

2. Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

3. Run an agent:
```python
from agents.example_agent import ExampleAgent

result = ExampleAgent.run("Hello, how can you help me?")
print(result)
```

## Project Structure

```
{project_name}/
├── agents/          # Agent class definitions
├── data/            # CSV files for evaluation and training
├── evaluations/     # Evaluation class definitions
├── metrics/         # Custom metric implementations
├── prompts/         # Markdown files for agent instructions
├── tools/           # Tool implementations
├── training/        # Training routine definitions
├── vizra_config.py  # Configuration file
├── .env.example     # Environment variables template
└── .gitignore       # Git ignore file
```

## Commands

### Create Components
```bash
vizra make agent my_agent
vizra make tool my_tool
vizra make evaluation my_eval
vizra make training my_training
vizra make metric my_metric
```

### Run Evaluations
```bash
vizra eval run MyEvaluation
vizra eval list
```

### Run Training
```bash
vizra train run MyTraining
vizra train list
```

## Documentation

For more information, visit the [Vizra documentation](https://github.com/aaronlumsden/vizra-python).

---

Built with ❤️ using [Vizra](https://github.com/aaronlumsden/vizra-python)
'''


@click.command(name='init')
@click.option('--name', prompt='Project name', help='Name of the project to create')
def init_command(name):
    """Initialize a new Vizra project with standard structure."""
    project_path = Path.cwd() / name
    
    if project_path.exists():
        print_error(f"Directory '{name}' already exists!")
        raise click.Abort()
    
    try:
        console.print(f"\n{EMOJIS['rocket']} Creating Vizra project: [bold cyan]{name}[/bold cyan]\n")
        
        project_path.mkdir()
        
        directories = [
            'agents',
            'data',
            'evaluations',
            'metrics',
            'prompts',
            'tools',
            'training',
        ]
        
        for dir_name in directories:
            dir_path = project_path / dir_name
            dir_path.mkdir()
            (dir_path / '__init__.py').touch()
            console.print(f"  {EMOJIS['folder']} Created [dim]{dir_name}/[/dim]")
        
        files_to_create = {
            'vizra_config.py': get_vizra_config_template(),
            '.gitignore': get_gitignore_template(),
            '.env.example': get_env_example_template(),
            'README.md': get_readme_template(name),
            'agents/example_agent.py': get_sample_agent_template(),
            'tools/example_tool.py': get_sample_tool_template(),
        }
        
        for file_name, content in files_to_create.items():
            file_path = project_path / file_name
            file_path.write_text(content)
            console.print(f"  {EMOJIS['document']} Created [dim]{file_name}[/dim]")
        
        console.print()
        
        success_message = f"""
{EMOJIS['checkmark']} [bold green]Project "{name}" initialized successfully![/bold green]

[bold cyan]What's been created:[/bold cyan]
  • Project structure with all necessary directories
  • Configuration file (vizra_config.py) with sensible defaults
  • Environment template (.env.example) for API keys
  • Git ignore file (.gitignore) configured for Python projects
  • Example agent and tool to get you started
  • README.md with project documentation

[bold cyan]Next steps:[/bold cyan]
  1. Navigate to your project: [yellow]cd {name}[/yellow]
  2. Copy .env.example to .env: [yellow]cp .env.example .env[/yellow]
  3. Add your API keys to the .env file
  4. Start building your agents!

[bold cyan]Useful commands:[/bold cyan]
  • Create an agent: [yellow]vizra make agent my_agent[/yellow]
  • Create a tool: [yellow]vizra make tool my_tool[/yellow]
  • Run evaluations: [yellow]vizra eval run MyEvaluation[/yellow]

[dim]For more information, check out the documentation at:
https://github.com/aaronlumsden/vizra-python[/dim]
        """
        
        panel = create_panel(
            success_message.strip(),
            title="✨ Project Initialized",
            style="green"
        )
        console.print(panel)
        
        console.print(f"\n{EMOJIS['star']} [bold yellow]If you find Vizra helpful, please support us with a star![/bold yellow]")
        console.print(f"   [link=https://github.com/aaronlumsden/vizra-python]https://github.com/aaronlumsden/vizra-python[/link]\n")
        
    except Exception as e:
        print_error(f"Failed to create project: {str(e)}")
        if project_path.exists():
            import shutil
            shutil.rmtree(project_path)
        raise click.Abort()