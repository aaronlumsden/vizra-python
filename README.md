<div align="center">
  <p align="center">
  <img src="https://vizra.ai/img/vizra-logo.svg" alt="Vizra Logo" width="200">
</p>
  <p><strong>The lightweight AI agent framework that gets out of your way</strong></p>
  
  [![PyPI Version](https://img.shields.io/pypi/v/vizra?color=blue)](https://pypi.org/project/vizra/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/vizra)](https://pypi.org/project/vizra/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

**Vizra** is a lightweight, class-based AI agent framework for Python. No heavy abstractions, no complex setups – just inherit from `BaseAgent` and you're building AI agents. Built on litellm for broad LLM support, it stays simple while giving you the power you need.

## 📚 Table of Contents

[Why Vizra?](#-why-vizra) • [Quick Start](#-quick-start) • [Core Concepts](#-core-concepts) • [Real Examples](#-real-examples) • [Configuration](#️-configuration) • [Testing & Training](#-testing--training) • [Project Structure](#-project-structure) • [Advanced Features](#️-advanced-features) • [CLI Reference](#-cli-reference) • [Support](#-support)

## 🎯 Why Vizra?

I built Vizra because I wanted an AI agent framework that was simple to understand and use. No complex abstractions, minimal dependencies, just straightforward and clean.

> 🪶 **Lightweight** – Minimal dependencies, just litellm at its core  
> 🎯 **Simple** – One base class, straightforward patterns  
> 🐍 **Pythonic** – Follows Python conventions and patterns  
> ⚡ **Fast Setup** – From install to working agent in under 30 seconds  
> 🔧 **Flexible** – Add tools, hooks, context, or keep it minimal  
> 📊 **Production Ready** – Built-in evaluation and training capabilities  

## 🚀 Quick Start

```bash
pip install vizra
```

**Your first agent in 8 lines:**

```python
from vizra import BaseAgent

class MyAgent(BaseAgent):
    name: str = 'my_agent'
    instructions: str = 'You are a helpful assistant.'
    model: str = 'gpt-4o'

response: str = MyAgent.run("Hello!")
print(response)  # "Hi there! How can I help you today?"
```

That's it. You've got a working AI agent.

## 📖 Core Concepts

### Agents - Just Python Classes

Every agent is just a Python class. Define what it does, give it instructions, and run it:

```python
from vizra import BaseAgent, AgentContext

class CodeReviewerAgent(BaseAgent):
    name: str = 'code_reviewer'
    instructions: str = 'You review code for clarity and best practices.'
    model: str = 'gpt-4o'
    
# Or use a markdown file for longer instructions:
class DetailedReviewerAgent(BaseAgent):
    name: str = 'detailed_reviewer'
    instructions_file: str = 'reviewer_prompt.md'  # Loads from prompts/ directory
    model: str = 'gpt-4o'
    
# Single response
feedback: str = CodeReviewerAgent.run("def add(x,y): return x+y")

# Or with conversation memory
context = AgentContext()
reviewer = CodeReviewerAgent.with_context(context)
response1: str = reviewer.run("Review this: def calculate(): return 42")
response2: str = reviewer.run("What about error handling?")  # Remembers previous code
```

### Tools - Give Your Agents Abilities

Tools let your agents actually do things. They're just classes with two methods:

```python
from vizra import ToolInterface, AgentContext, BaseAgent
import json

class SearchTool(ToolInterface):
    def definition(self):
        return {
            'name': 'search',
            'description': 'Search for information',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query'}
                },
                'required': ['query']
            }
        }
    
    def execute(self, arguments, context):
        query = arguments['query']
        # Your search logic here
        results = perform_search(query)
        return json.dumps(results)

# Give your agent the tool
class ResearchAgent(BaseAgent):
    name: str = 'researcher'
    instructions: str = 'You help users find information.'
    model: str = 'gpt-4o'
    tools = [SearchTool]
```

### Hooks - Customize Agent Behavior

Want to log, monitor, or modify agent behavior? Use hooks:

```python
from vizra import BaseAgent, AgentContext

class MonitoredAgent(BaseAgent):
    name: str = 'monitored'
    instructions: str = 'You are a helpful assistant.'
    model: str = 'gpt-4o'
    
    def before_llm_call(self, messages, tools):
        print(f"📤 Sending {len(messages)} messages to LLM")
    
    def after_llm_response(self, response, messages):
        print(f"📥 Got response: {response.choices[0].message.content[:50]}...")
    
    def before_tool_call(self, tool_name, arguments, context):
        print(f"🔧 Calling tool: {tool_name}")
    
    def after_tool_result(self, tool_name, result, context):
        print(f"✅ Tool {tool_name} completed")
```

## 🔥 Real Examples

### Customer Support Agent

```python
from vizra import BaseAgent

class SupportAgent(BaseAgent):
    name: str = 'support'
    instructions: str = '''You are a friendly customer support agent.
    Be helpful, concise, and always offer to escalate if needed.'''
    model: str = 'gpt-4o'
    tools = [SearchDocsTool, CreateTicketTool]

# Usage
response = SupportAgent.run("I can't login to my account")
# "I'm sorry to hear you're having trouble logging in. Let me help you..."
```

### Code Generator

```python
from vizra import BaseAgent

class CodeGeneratorAgent(BaseAgent):
    name: str = 'codegen'
    instructions_file: str = 'codegen_prompt.md'  # Load from prompts/
    model: str = 'gpt-4o'
    tools = [WriteFileTool, RunTestsTool]

# Usage
response = CodeGeneratorAgent.run("Create a Python function to calculate fibonacci")
# Generates complete, tested code
```

### Data Analyst

```python
from vizra import BaseAgent

class DataAnalystAgent(BaseAgent):
    name: str = 'analyst'
    instructions: str = 'You analyze data and provide insights.'
    model: str = 'gpt-4o'
    tools = [QueryDatabaseTool, CreateChartTool, ExportReportTool]

# Usage
response = DataAnalystAgent.run("Show me sales trends for Q4")
# Queries data, creates visualizations, and provides insights
```

## ⚙️ Configuration

Vizra provides a simple configuration system using Python dictionaries:

```python
from vizra import config

# Access configuration with dot notation
model = config('llm.model', 'gpt-4o')
api_key = config('api.openai.key')
batch_size = config('training.batch_size', 32)
```

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
        'custom_value': 42,
    },
}
```

See `vizra_config.py.example` for a complete template.

## 🧪 Testing & Training

### Evaluation

Test your agents with real data:

```python
from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ExactMatchMetric, ContainsMetric

class MyEvaluation(BaseEvaluation):
    name: str = 'test_my_agent'
    agent_name: str = 'my_agent'
    csv_path: str = 'data/test_cases.csv'
    
    metrics = [
        ExactMatchMetric('expected_output'),
        ContainsMetric('must_include')
    ]
```

Run evaluations from the command line:

```bash
vizra eval run test_my_agent -v
```

### Training

Train your agents with reinforcement learning:

```python
from vizra.training import BaseRLTraining
from vizra.providers import VerifiersProvider

class MyTraining(BaseRLTraining):
    name: str = 'train_agent'
    agent_name: str = 'my_agent'
    csv_path: str = 'data/training.csv'
    algorithm: str = 'grpo'  # Options: ppo, dpo, grpo, reinforce, etc.
    
    # Optional: Use providers for real model weight updates
    provider = VerifiersProvider(
        model_name='my_finetuned_model',
        base_model='Qwen/Qwen2.5-0.5B-Instruct'
    )
    
    def calculate_reward(self, csv_row_data, agent_response):
        # Give partial rewards to guide learning
        expected = csv_row_data.get('expected_output', '')
        if expected in agent_response:
            return 1.0  # Perfect match
        elif expected.lower() in agent_response.lower():
            return 0.7  # Close match
        elif len(agent_response) > 20:
            return 0.3  # At least tried
        return 0.0  # Too short or wrong
```

Run training with `vizra train run train_agent`. Without a provider, it runs in placeholder mode. With VerifiersProvider, it performs real model weight updates.

## 📁 Project Structure

Keep your project organized:

```
your-project/
├── agents/          # Agent class definitions
├── data/            # CSV files for evaluation and training
├── evaluations/     # Evaluation class definitions
├── metrics/         # Custom metric implementations (optional)
├── prompts/         # Markdown files for agent instructions (use with instructions_file)
├── tools/           # Tool implementations
├── training/        # Training routine definitions
├── vizra_config.py  # Configuration file (optional)
├── .env.example     # Environment variables template
└── .gitignore       # Git ignore file
```

**Important**: Create `__init__.py` files in all Python package directories (agents, tools, evaluations, training, metrics).

## 🛠️ Advanced Features

### XML-Style Tools

For models that prefer XML:

```python
from vizra import ToolInterface, AgentContext

class XMLTool(ToolInterface):
    xml_tag: str = 'search'  # Enables <search>query</search>
    
    def parse_xml_content(self, content):
        return {"query": content}
    
    def execute(self, arguments, context):
        return f"Searched for: {arguments['query']}"
```

### Custom Metrics

Create your own evaluation metrics:

```python
from vizra.evaluation.metrics import BaseMetric

class QualityMetric(BaseMetric):
    name: str = "quality_check"
    
    def evaluate(self, row_data, response):
        score = calculate_quality_score(response)
        return {
            'passed': score > 0.8,
            'score': score,
            'details': {'quality_score': score}
        }
```

### Context Management

Full control over conversation state:

```python
from vizra import AgentContext

context = AgentContext()
# Context automatically manages:
# - Message history
# - Tool call tracking  
# - Metadata storage
# - Token counting
```

## 💻 CLI Reference

Vizra comes with a beautiful CLI powered by Rich:

### General Commands

| Command | Description |
|---------|-------------|
| `vizra status` | Show installation status |
| `vizra --version` | Show version info |

### Evaluation Commands

| Command | Description | Options |
|---------|-------------|---------|
| `vizra eval list` | List available evaluations | |
| `vizra eval run <name>` | Run an evaluation | `-v` (verbose)<br>`-l N` (limit test cases)<br>`-d` (detailed CSV)<br>`-j` (JSON output)<br>`-o FILE` (custom output) |

### Training Commands

| Command | Description | Options |
|---------|-------------|---------|
| `vizra train list` | List training routines | |
| `vizra train run <name>` | Run training | `-v` (verbose)<br>`-i N` (iterations)<br>`-t` (test mode)<br>`-o FILE` (save results) |

### Code Generation Commands

| Command | Description |
|---------|-------------|
| `vizra make agent <name>` | Create new agent class |
| `vizra make tool <name>` | Create new tool class |
| `vizra make evaluation <name>` | Create new evaluation |
| `vizra make training <name>` | Create new training routine |
| `vizra make metric <name>` | Create custom metric |

## 🤝 Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/aaronlumsden/vizra-python/issues)
- 🐦 **X**: [@aaronlumsden](https://x.com/aaronlumsden)

## 📝 License

MIT License - Use it however you want.

---

<div align="center">
  <p>Built with ❤️ for developers who want a simple AI agent framework</p>
  <p><a href="https://github.com/aaronlumsden/vizra-python">⭐ Star on GitHub</a> if you find it useful!</p>
</div>