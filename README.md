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

class CodeReviewer(BaseAgent):
    name: str = 'code_reviewer'
    instructions: str = 'You review code for clarity and best practices.'
    model: str = 'gpt-4o'
    
# Single response
feedback: str = CodeReviewer.run("def add(x,y): return x+y")

# Or with conversation memory
context = AgentContext()
reviewer = CodeReviewer.with_context(context)
response1: str = reviewer.run("Review this: def calculate(): return 42")
response2: str = reviewer.run("What about error handling?")  # Remembers previous code
```

### Tools - Give Your Agents Abilities

Tools let your agents actually do things. They're just classes with two methods:

```python
from typing import Dict, Any, List, Type
from vizra import ToolInterface, AgentContext, BaseAgent
import json

class SearchTool(ToolInterface):
    def definition(self) -> Dict[str, Any]:
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
    
    def execute(self, arguments: Dict[str, Any], context: AgentContext) -> str:
        query = arguments['query']
        # Your search logic here
        results = perform_search(query)
        return json.dumps(results)

# Give your agent the tool
class ResearchAgent(BaseAgent):
    name: str = 'researcher'
    instructions: str = 'You help users find information.'
    model: str = 'gpt-4o'
    tools: List[Type[ToolInterface]] = [SearchTool]
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

## 🧪 Evaluation & Testing

Test your agents with real data:

```python
from typing import List
from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ExactMatchMetric, ContainsMetric, BaseMetric

class MyEvaluation(BaseEvaluation):
    name: str = 'test_my_agent'
    agent_name: str = 'my_agent'
    csv_path: str = 'data/test_cases.csv'
    
    metrics: List[BaseMetric] = [
        ExactMatchMetric('expected_output'),
        ContainsMetric('must_include')
    ]
```

Run evaluations from the command line:

```bash
vizra eval run test_my_agent -v
```

## 🎓 Training

Train your agents with reinforcement learning:

```python
from typing import Dict, Any
from vizra.training import BaseRLTraining

class MyTraining(BaseRLTraining):
    name: str = 'train_agent'
    agent_name: str = 'my_agent'
    csv_path: str = 'data/training.csv'
    
    def calculate_reward(self, csv_row_data: Dict[str, Any], agent_response: str) -> float:
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

Run training locally or integrate with external RL providers like OpenPipe ART for production-grade training.

## 💻 CLI Commands

Vizra comes with a beautiful CLI powered by Rich:

| Command | Description | Options |
|---------|-------------|---------|
| **General** | | |
| `vizra status` | Show installation status | |
| `vizra --version` | Show version info | |
| **Evaluation** | | |
| `vizra eval list` | List available evaluations | |
| `vizra eval run <name>` | Run an evaluation | `-v` (verbose)<br>`-l N` (limit test cases)<br>`-d` (detailed CSV)<br>`-j` (JSON output)<br>`-o FILE` (custom output) |
| **Training** | | |
| `vizra train list` | List training routines | |
| `vizra train run <name>` | Run training | `-v` (verbose)<br>`-i N` (iterations)<br>`-t` (test mode)<br>`-o FILE` (save results) |
| **Code Generation** | | |
| `vizra make agent <name>` | Create new agent class | |
| `vizra make tool <name>` | Create new tool class | |
| `vizra make evaluation <name>` | Create new evaluation | |
| `vizra make training <name>` | Create new training routine | |
| `vizra make metric <name>` | Create custom metric | |

## 📁 Project Structure

Keep your project organized:

```
your-project/
├── agents/          # Agent class definitions
├── data/            # CSV files for evaluation and training
├── evaluations/     # Evaluation class definitions
├── metrics/         # Custom metric implementations (optional)
├── prompts/         # Markdown files with agent instructions
├── tools/           # Tool implementations
├── training/        # Training routine definitions
├── .env.example     # Environment variables template
└── .gitignore       # Git ignore file
```

**Important**: Create `__init__.py` files in all Python package directories (agents, tools, evaluations, training, metrics).

## 🔥 Real Examples

### Customer Support Agent

```python
from typing import List, Type
from vizra import BaseAgent, ToolInterface

class SupportAgent(BaseAgent):
    name: str = 'support'
    instructions: str = '''You are a friendly customer support agent.
    Be helpful, concise, and always offer to escalate if needed.'''
    model: str = 'gpt-4o'
    tools: List[Type[ToolInterface]] = [SearchDocsToool, CreateTicketTool]
```

### Code Generator

```python
from typing import List, Type
from vizra import BaseAgent, ToolInterface

class CodeGenerator(BaseAgent):
    name: str = 'codegen'
    instructions_file: str = 'codegen_prompt.md'  # Load from prompts/
    model: str = 'gpt-4o'
    tools: List[Type[ToolInterface]] = [WriteFileTool, RunTestsTool]
```

### Data Analyst

```python
from typing import List, Type
from vizra import BaseAgent, ToolInterface

class DataAnalyst(BaseAgent):
    name: str = 'analyst'
    instructions: str = 'You analyze data and provide insights.'
    model: str = 'gpt-4o'
    tools: List[Type[ToolInterface]] = [QueryDatabaseTool, CreateChartTool, ExportReportTool]
```

## 🛠️ Advanced Features

### XML-Style Tools

For models that prefer XML:

```python
from typing import Dict, Any
from vizra import ToolInterface, AgentContext

class XMLTool(ToolInterface):
    xml_tag: str = 'search'  # Enables <search>query</search>
    
    def parse_xml_content(self, content: str) -> Dict[str, Any]:
        return {"query": content}
    
    def execute(self, arguments: Dict[str, Any], context: AgentContext) -> str:
        return f"Searched for: {arguments['query']}"
```

### Custom Metrics

Create your own evaluation metrics:

```python
from typing import Dict, Any
from vizra.evaluation.metrics import BaseMetric

class QualityMetric(BaseMetric):
    name: str = "quality_check"
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
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