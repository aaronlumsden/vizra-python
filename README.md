# Vizra - Simple AI Agent Framework for Python

A lightweight, class-based AI agent framework for Python that uses litellm for LLM integration.

## Installation

```bash
pip install vizra
```

## Quick Start

### Define an Agent

```python
from vizra import BaseAgent

class CustomerSupportAgent(BaseAgent):
    name = 'customer_support'
    description = 'Helps customers with their inquiries'
    instructions = 'You are a friendly customer support assistant.'
    model = 'gpt-4o'
    tools = [OrderLookupTool, RefundProcessorTool]  # Explicitly specify tools

# Or auto-discover all tools from the tools/ directory
class SmartAgent(BaseAgent):
    name = 'smart_agent'
    description = 'Agent that uses all available tools'
    instructions = 'You are a helpful assistant with access to many tools.'
    model = 'gpt-4o'
    # No tools specified - will auto-discover from tools/ directory

# Or load instructions from a file
class AdvancedSupportAgent(BaseAgent):
    name = 'advanced_support'
    description = 'Advanced support agent with complex instructions'
    instructions_file = 'advanced_support.md'  # Looks in prompts/ folder by default
    model = 'gpt-4o'
    tools = [OrderLookupTool, RefundProcessorTool]
```

### Run the Agent

```python
from my_agents import CustomerSupportAgent  # Import your agent

# Simple usage
response = CustomerSupportAgent.run('How do I reset my password?')

# With context for conversation continuity
from vizra import AgentContext

context = AgentContext()
agent_runner = CustomerSupportAgent.with_context(context)

response1 = agent_runner.run("Hi, I need help")
response2 = agent_runner.run("Can you check my order?")
```

### Define Tools

```python
from vizra import ToolInterface, AgentContext
import json

class OrderLookupTool(ToolInterface):
    def definition(self) -> dict:
        return {
            'name': 'order_lookup',
            'description': 'Look up order information by order ID',
            'parameters': {
                'type': 'object',
                'properties': {
                    'order_id': {
                        'type': 'string',
                        'description': 'The order ID to look up',
                    },
                },
                'required': ['order_id'],
            },
        }

    def execute(self, arguments: dict, context: AgentContext) -> str:
        order_id = arguments['order_id']
        # Your implementation here
        return json.dumps({"order_id": order_id, "status": "shipped"})
```

## Features

- Class-based agent definition
- Tool integration with automatic execution loop (max 3 iterations)
- Context management for conversation history
- Support for multiple LLM providers via litellm
- Hook methods for monitoring and customization
- File-based instruction loading from `prompts/` directory
- Simple and intuitive API
- **CLI for evaluations and training**
- **CSV-based evaluation framework**
- **Reinforcement learning training support**

## CLI Usage

Vizra includes a powerful CLI for running evaluations and training agents.

### Installation

After installing vizra, the CLI is available:

```bash
vizra --help
```

### Commands

#### Status Check
```bash
vizra status
```

#### Evaluations

List available evaluations:
```bash
vizra eval list
```

Run an evaluation:
```bash
vizra eval run <evaluation_name>

# Results are automatically saved to evaluations/results/ with:
# - JSON file: YYYYMMDD_HHMMSS_<eval_name>_<model_name>.json
# - Summary CSV: YYYYMMDD_HHMMSS_<eval_name>_<model_name>_summary.csv
# - Detailed CSV: YYYYMMDD_HHMMSS_<eval_name>_<model_name>_detailed.csv

# With options
vizra eval run chord_identifier_eval -v  # Verbose output
vizra eval run chord_identifier_eval -o custom_results.json  # Additional save location
vizra eval run chord_identifier_eval -l 10  # Limit to first 10 test cases
vizra eval run chord_identifier_eval -l 5 -v  # Limit to 5 cases with verbose output
```

#### Training

List available training routines:
```bash
vizra train list
```

Run training:
```bash
vizra train run <training_name>

# With options
vizra train run chord_training -i 50  # Override iterations
vizra train run chord_training -v -o training_results.json
```

### Project Structure

Vizra expects the following structure in your project:

```
your-project/
├── agents/          # Your agent definitions
├── evaluations/     # Evaluation classes
├── training/        # Training routines
├── data/           # CSV files for evaluation/training
├── prompts/        # Markdown files for agent instructions
└── tools/          # Tool implementations
```

The framework automatically discovers:
- **Agents** from the `agents/` folder (referenced by `agent_name` in evaluations/training)
- **Tools** from the `tools/` folder (when no tools are specified in the agent class)
- **Evaluations** from the `evaluations/` folder (for the CLI)
- **Training routines** from the `training/` folder (for the CLI)

**Important**: Make sure to create `__init__.py` files in all these folders to make them Python packages.

### Creating Evaluations

Create an evaluation by subclassing `BaseEvaluation` in your `evaluations/` folder:

```python
# evaluations/my_agent_eval.py
from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ContainsMetric, ExactMatchMetric

class MyAgentEvaluation(BaseEvaluation):
    name = 'my_agent_eval'
    description = 'Evaluate my agent accuracy'
    agent_name = 'my_agent'  # Must match the 'name' attribute of your agent class in agents/
    csv_path = 'data/test_cases.csv'
    
    # Define metrics to use
    metrics = [
        ContainsMetric('expected_response'),  # Check if response contains expected text
        ExactMatchMetric('expected_response', case_sensitive=False)  # Check exact match
    ]
    
    # The base class automatically runs all metrics for you
```

#### Using Custom Metrics

You can create custom metrics by subclassing `BaseMetric`:

```python
from vizra.evaluation.metrics import BaseMetric

class CustomMetric(BaseMetric):
    name = "custom_check"  # This becomes the CSV column name
    
    def evaluate(self, row_data, response):
        # Your custom evaluation logic
        passed = "specific_keyword" in response
        score = 1.0 if passed else 0.0
        
        return {
            'passed': passed,
            'score': score,
            'details': {
                'custom_field': 'custom_value'
            }
        }

# Then use it in your evaluation
class MyAgentEvaluation(BaseEvaluation):
    metrics = [
        ContainsMetric('expected_response'),
        CustomMetric()
    ]
```

#### Built-in Metrics

Vizra provides several built-in metrics:

- **ExactMatchMetric**: Checks if response exactly matches expected value
- **ContainsMetric**: Checks if response contains expected substring
- **NotContainsMetric**: Ensures response doesn't contain specific text
- **RegexMetric**: Evaluates response against a regex pattern
- **SentimentMetric**: Analyzes sentiment (positive/negative/neutral)
- **LengthMetric**: Validates response length constraints

CSV format for evaluations:
```csv
prompt,expected_response
"What is 2+2?","4"
"What is the capital of France?","Paris"
```

Evaluation results are automatically saved to `evaluations/results/` with:
- Summary CSV with overall metrics
- Detailed CSV with individual metric columns for each test case
- Full conversation history for each evaluation (including tool calls) in JSON format

The conversation history column in the CSV includes:
- All messages between user and assistant
- Tool calls and their results
- Complete context for debugging and analysis

### Creating Training Routines

Create a training routine by subclassing `BaseRLTraining` in your `training/` folder:

```python
# training/my_agent_training.py
from vizra.training import BaseRLTraining

class MyAgentTraining(BaseRLTraining):
    name = 'my_agent_training'
    description = 'Train my agent with RL'
    agent_name = 'my_agent'
    csv_path = 'data/training_data.csv'
    n_iterations = 100
    batch_size = 32
    
    def calculate_reward(self, csv_row_data, agent_response):
        # Custom reward logic
        expected = csv_row_data.get('expected_response', '')
        if expected.lower() in agent_response.lower():
            return 1.0
        return 0.0
```

## Hooks

Agents can override hook methods to add custom behavior:

```python
class MonitoredAgent(BaseAgent):
    def before_llm_call(self, messages, tools):
        print(f"Making LLM call with {len(messages)} messages")
    
    def after_llm_response(self, response, messages):
        print(f"Response received with {response.usage.total_tokens} tokens")
    
    def before_tool_call(self, tool_name, arguments, context):
        print(f"Calling tool: {tool_name}")
    
    def after_tool_result(self, tool_name, result, context):
        print(f"Tool {tool_name} returned: {result}")
```

## License

MIT