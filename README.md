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

I built Vizra because I wanted an AI agent framework that felt like writing regular Python code. No massive learning curve, no heavy dependencies, just clean and simple.

> 🪶 **Lightweight** – Minimal dependencies, just litellm at its core  
> 🎯 **Simple** – One base class, straightforward patterns  
> 🐍 **Pythonic** – It feels like Python, not a framework  
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
    name = 'my_agent'
    instructions = 'You are a helpful assistant.'
    model = 'gpt-4o'

response = MyAgent.run("Hello!")
print(response)  # "Hi there! How can I help you today?"
```

That's it. You've got a working AI agent.

## 📖 Core Concepts

### Agents - Just Python Classes

Every agent is just a Python class. Define what it does, give it instructions, and run it:

```python
from vizra import BaseAgent

class CodeReviewer(BaseAgent):
    name = 'code_reviewer'
    instructions = 'You review code for clarity and best practices.'
    model = 'gpt-4o'
    
# Single response
feedback = CodeReviewer.run("def add(x,y): return x+y")

# Or with conversation memory
from vizra import AgentContext

context = AgentContext()
reviewer = CodeReviewer.with_context(context)
response1 = reviewer.run("Review this: def calculate(): return 42")
response2 = reviewer.run("What about error handling?")  # Remembers previous code
```

### Tools - Give Your Agents Abilities

Tools let your agents actually do things. They're just classes with two methods:

```python
from vizra import ToolInterface, AgentContext
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
    
    def execute(self, arguments: dict, context: AgentContext):
        query = arguments['query']
        # Your search logic here
        results = perform_search(query)
        return json.dumps(results)

# Give your agent the tool
class ResearchAgent(BaseAgent):
    name = 'researcher'
    instructions = 'You help users find information.'
    model = 'gpt-4o'
    tools = [SearchTool]
```

### Hooks - Customize Agent Behavior

Want to log, monitor, or modify agent behavior? Use hooks:

```python
class MonitoredAgent(BaseAgent):
    name = 'monitored'
    instructions = 'You are a helpful assistant.'
    model = 'gpt-4o'
    
    def before_llm_call(self, messages, tools):
        print(f"📤 Sending {len(messages)} messages to LLM")
    
    def after_llm_response(self, response, messages):
        print(f"📥 Got response: {response.choices[0].message.content[:50]}...")
```

## 🧪 Evaluation & Testing

Test your agents with real data:

```python
from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ExactMatchMetric, ContainsMetric

class MyEvaluation(BaseEvaluation):
    name = 'test_my_agent'
    agent_name = 'my_agent'
    csv_path = 'data/test_cases.csv'
    
    metrics = [
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
from vizra.training import BaseRLTraining

class MyTraining(BaseRLTraining):
    name = 'train_agent'
    agent_name = 'my_agent'
    csv_path = 'data/training_data.csv'
    
    def calculate_reward(self, csv_row_data: dict, agent_response: str):
        expected = csv_row_data.get('expected', '')
        return 1.0 if expected in agent_response else 0.0
```

## 💻 CLI Commands

Vizra comes with a beautiful CLI powered by Rich:

```bash
# Check your setup
vizra status

# List what you can evaluate
vizra eval list

# Run an evaluation
vizra eval run my_eval -v

# Train an agent
vizra train run my_training -i 100
```

## 📁 Project Structure

Keep your project organized:

```
your-project/
├── agents/          # Your agent classes
├── tools/           # Custom tools
├── evaluations/     # Evaluation definitions
├── training/        # Training routines
├── data/           # CSV files for testing/training
└── prompts/        # Markdown instruction files
```

## 🔥 Real Examples

### Customer Support Agent

```python
class SupportAgent(BaseAgent):
    name = 'support'
    instructions = '''You are a friendly customer support agent.
    Be helpful, concise, and always offer to escalate if needed.'''
    model = 'gpt-4o'
    tools = [SearchDocsToool, CreateTicketTool]
```

### Code Generator

```python
class CodeGenerator(BaseAgent):
    name = 'codegen'
    instructions_file = 'codegen_prompt.md'  # Load from prompts/
    model = 'gpt-4o'
    tools = [WriteFileTool, RunTestsTool]
```

### Data Analyst

```python
class DataAnalyst(BaseAgent):
    name = 'analyst'
    instructions = 'You analyze data and provide insights.'
    model = 'gpt-4o'
    tools = [QueryDatabaseTool, CreateChartTool, ExportReportTool]
```

## 🛠️ Advanced Features

### XML-Style Tools

For models that prefer XML:

```python
class XMLTool(ToolInterface):
    xml_tag = 'search'  # Enables <search>query</search>
    
    def parse_xml_content(self, content: str):
        return {"query": content}
    
    def execute(self, arguments: dict, context: AgentContext):
        return f"Searched for: {arguments['query']}"
```

### Custom Metrics

Create your own evaluation metrics:

```python
from vizra.evaluation.metrics import BaseMetric

class QualityMetric(BaseMetric):
    name = "quality_check"
    
    def evaluate(self, row_data: dict, response: str):
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

## 🤝 Community & Support

Built something cool with Vizra? I'd love to hear about it!

- 🐛 **Issues**: [GitHub Issues](https://github.com/aaronlumsden/vizra-python/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/aaronlumsden/vizra-python/discussions)
- 🐦 **X**: [@aaronlumsden](https://x.com/aaronlumsden)

## 📝 License

MIT License - Use it however you want.

---

<div align="center">
  <p>Built with ❤️ by developers who just wanted a simple AI agent framework</p>
  <p><a href="https://github.com/aaronlumsden/vizra-python">⭐ Star on GitHub</a> if you find it useful!</p>
</div>