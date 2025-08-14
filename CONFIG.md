# Vizra Configuration System

Vizra provides a simple, pythonic configuration system using Python dictionaries.

## Quick Start

1. Copy `vizra_config.py.example` to `vizra_config.py` in your project root
2. Customize the settings dictionary
3. Access configuration values in your code:

```python
from vizra import config

# Get configuration values with defaults
model = config('llm.model', 'gpt-4o')
api_key = config('api.openai.key')
batch_size = config('training.batch_size', 32)
```

## Configuration File

Create a `vizra_config.py` file in your project root:

```python
import os

settings = {
    'llm': {
        'model': 'gpt-4o',
        'temperature': 0.7,
    },
    
    'api': {
        'openai': {
            'key': os.getenv('OPENAI_API_KEY'),
        },
    },
    
    # Add any custom configuration
    'my_app': {
        'feature_flag': True,
        'custom_value': 42,
    },
}
```

## Usage Examples

### In Agents

```python
from vizra import BaseAgent, config

class MyAgent(BaseAgent):
    name = 'my_agent'
    model = config('llm.model', 'gpt-4o')
    instructions = 'You are a helpful assistant.'
```

### In Training

```python
from vizra import BaseRLTraining, config

class MyTraining(BaseRLTraining):
    learning_rate = config('training.learning_rate', 1e-4)
    batch_size = config('training.batch_size', 32)
```

### Custom Values

```python
from vizra import config

# Access nested values with dot notation
debug_mode = config('features.debug_mode', False)
api_endpoint = config('my_api.endpoint', 'https://api.example.com')

# Values not in config return default (or None)
missing_value = config('non.existent.path', 'default_value')
```

## Advanced Usage

```python
from vizra import get_config

# Get the Config object for advanced operations
cfg = get_config()

# Set values dynamically
cfg.set('runtime.value', 'dynamic')

# Update multiple values
cfg.update({
    'new_section': {
        'key': 'value'
    }
})
```

## Environment Variables

You can use environment variables in your configuration:

```python
settings = {
    'api': {
        'key': os.getenv('MY_API_KEY'),
        'endpoint': os.getenv('API_ENDPOINT', 'https://default.com'),
    },
}
```

Or point to a different config file:
```bash
export VIZRA_CONFIG=/path/to/custom_config.py
```

## Benefits

- **Pythonic**: Just a Python dictionary in a .py file
- **Flexible**: Add any configuration values you need
- **Simple**: No complex parsing or validation
- **IDE-friendly**: Full autocomplete and type hints
- **No dependencies**: Uses only Python standard library