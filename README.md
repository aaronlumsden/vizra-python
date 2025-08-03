# Vizra - AI Agent Framework for Python

A lightweight, class-based AI agent framework for Python that uses litellm for LLM integration. Build agents with tools, evaluations, and training capabilities.

## Installation

```bash
pip install vizra
```

## Quick Start

### 1. Create an Agent

```python
from vizra import BaseAgent

class MyAgent(BaseAgent):
    name = 'my_agent'
    description = 'A helpful AI assistant'
    instructions = 'You are a helpful AI assistant.'
    model = 'gpt-4o'
    tools = []  # Add your tools here
```

### 2. Run the Agent

```python
# Simple usage
response = MyAgent.run("Hello, how are you?")
print(response)

# With conversation context
from vizra import AgentContext

context = AgentContext()
agent_runner = MyAgent.with_context(context)

response1 = agent_runner.run("Hi there!")
response2 = agent_runner.run("What did I just say?")  # Maintains context
```

## Project Structure

A typical Vizra project should be organized as follows:

```
your-project/
├── agents/          # Agent class definitions
├── data/           # CSV files for evaluation and training data
├── evaluations/    # Evaluation class definitions
├── metrics/        # Custom metric implementations (optional)
├── prompts/        # Markdown files with agent instructions
├── tools/          # Tool implementations
├── training/       # Training routine definitions
├── .env.example    # Environment variables template
└── .gitignore      # Git ignore file
```

**Important**: Create `__init__.py` files in all Python package directories (agents, tools, evaluations, training, metrics).

## Agents

### BaseAgent Class

All agents inherit from `BaseAgent` and must define these class attributes:

```python
from vizra import BaseAgent

class MyAgent(BaseAgent):
    name = 'my_agent'                    # Unique identifier
    description = 'What this agent does' # Human-readable description
    instructions = 'Your behavior...'    # System prompt
    model = 'gpt-4o'                     # LLM model to use
    tools = [MyTool]                     # List of tool classes
```

### Instructions

You can provide instructions in two ways:

**Inline instructions:**
```python
class MyAgent(BaseAgent):
    instructions = 'You are a helpful assistant that...'
```

**File-based instructions:**
```python
class MyAgent(BaseAgent):
    instructions_file = 'my_agent.md'  # Looks in prompts/ folder
```

### Running Agents

**Single interaction:**
```python
response = MyAgent.run("Your message here")
```

**Conversation with context:**
```python
from vizra import AgentContext

context = AgentContext()
agent_runner = MyAgent.with_context(context)

response1 = agent_runner.run("First message")
response2 = agent_runner.run("Follow-up message")  # Remembers conversation
```

### Agent Hooks

Override these methods to add custom behavior:

```python
class MonitoredAgent(BaseAgent):
    def before_llm_call(self, messages, tools):
        print(f"Making LLM call with {len(messages)} messages")
    
    def after_llm_response(self, response, messages):
        print(f"Response: {response.choices[0].message.content[:50]}...")
    
    def before_tool_call(self, tool_name, arguments, context):
        print(f"Calling tool: {tool_name}")
    
    def after_tool_result(self, tool_name, result, context):
        print(f"Tool {tool_name} completed")
```

### Tool Execution Loop

Agents automatically handle tool calls with a maximum of 3 iterations to prevent infinite loops.

## Tools

### ToolInterface

All tools inherit from `ToolInterface`:

```python
from vizra import ToolInterface, AgentContext
import json

class MyTool(ToolInterface):
    def definition(self) -> dict:
        """OpenAI function calling format"""
        return {
            'name': 'my_tool',
            'description': 'What this tool does',
            'parameters': {
                'type': 'object',
                'properties': {
                    'param1': {
                        'type': 'string',
                        'description': 'Parameter description'
                    }
                },
                'required': ['param1']
            }
        }
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        """Execute the tool logic"""
        param1 = arguments['param1']
        # Your tool logic here
        return json.dumps({"result": f"Processed {param1}"})
```

### OpenAI vs XML Tools

**OpenAI-style tools** use function calling with JSON schemas:
- Define `definition()` method returning OpenAI function schema
- Agent automatically calls tools when LLM requests them

**XML-style tools** use XML tags in responses:
```python
class XMLTool(ToolInterface):
    xml_tag = 'my_tool'  # Enables <my_tool>content</my_tool> usage
    
    def parse_xml_content(self, content: str) -> dict:
        """Parse XML content into arguments"""
        return {"content": content}
    
    def execute(self, arguments: dict, context: AgentContext) -> str:
        # Your tool logic here
        return "Tool result"
```

## Evaluations & Metrics

### BaseEvaluation

Create evaluations by subclassing `BaseEvaluation`:

```python
from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ContainsMetric, ExactMatchMetric

class MyEvaluation(BaseEvaluation):
    name = 'my_evaluation'
    description = 'Evaluate my agent'
    agent_name = 'my_agent'  # Must match agent's name attribute
    csv_path = 'data/test_cases.csv'
    
    metrics = [
        ExactMatchMetric('expected_output'),
        ContainsMetric('expected_keywords')
    ]
```

### CSV Format

Your CSV files should contain test cases:

```csv
prompt,expected_output,expected_keywords
"What is 2+2?","4","four,math"
"Hello","Hi there","greeting,hello"
```

### Built-in Metrics

- **ExactMatchMetric** - Checks exact string match
- **ContainsMetric** - Checks if response contains expected text
- **NotContainsMetric** - Ensures response doesn't contain specific text
- **RegexMetric** - Evaluates against regex patterns
- **SentimentMetric** - Analyzes response sentiment
- **LengthMetric** - Validates response length constraints
- **ToolUsageMetric** - Checks if specific tools were used

### Custom Metrics

```python
from vizra.evaluation.metrics import BaseMetric

class CustomMetric(BaseMetric):
    name = "custom_check"
    
    def evaluate(self, row_data: dict, response: str) -> dict:
        # Your evaluation logic
        passed = "keyword" in response.lower()
        
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'details': {'found_keyword': passed}
        }
```

### Evaluation Results

Results are automatically saved to `evaluations/results/` with timestamps:
- `YYYYMMDD_HHMMSS_evalname_model_summary.csv` - Overall metrics
- `YYYYMMDD_HHMMSS_evalname_model_simple.csv` - Test case results
- `YYYYMMDD_HHMMSS_evalname_model_detailed.csv` - Full metric details (with `-d` flag)

## Training

### BaseRLTraining

Create reinforcement learning training routines:

```python
from vizra.training import BaseRLTraining

class MyTraining(BaseRLTraining):
    name = 'my_training'
    description = 'Train my agent with RL'
    agent_name = 'my_agent'
    csv_path = 'data/training_data.csv'
    
    # Training configuration
    algorithm = 'ppo'
    learning_rate = 1e-4
    batch_size = 32
    n_iterations = 100
    
    def calculate_reward(self, csv_row_data: dict, agent_response: str) -> float:
        """Custom reward logic"""
        expected = csv_row_data.get('expected_response', '')
        if expected.lower() in agent_response.lower():
            return 1.0
        return 0.0
```

### Training Configuration

Override these attributes:
- `algorithm` - Training algorithm (ppo, dpo, reinforce)
- `learning_rate` - Learning rate for training
- `batch_size` - Batch size for training
- `n_iterations` - Number of training iterations

### Custom Rewards

Override `calculate_reward()` to implement custom reward functions that return values between 0.0 and 1.0.

### Using Metrics in Training

You can reuse evaluation metrics in training by using trainable versions:

```python
from vizra.evaluation.metrics import TrainableExactMatchMetric, TrainableContainsMetric

class MyTraining(BaseRLTraining):
    # ... other configuration ...
    
    def calculate_reward(self, csv_row_data: dict, agent_response: str) -> float:
        # Use trainable metrics for consistent evaluation
        exact_match = TrainableExactMatchMetric('expected_response')
        contains_metric = TrainableContainsMetric('keywords')
        
        # Evaluate using the same logic as evaluations
        exact_result = exact_match.evaluate(csv_row_data, agent_response)
        contains_result = contains_metric.evaluate(csv_row_data, agent_response)
        
        # Compute rewards from metric results
        exact_reward = exact_match.compute_reward(exact_result)
        contains_reward = contains_metric.compute_reward(contains_result)
        
        # Combine rewards as needed
        return (exact_reward + contains_reward) / 2
```

**Sharing Metrics**: Metrics can be shared between evaluations and training by using the trainable versions (`TrainableExactMatchMetric`, `TrainableContainsMetric`, `TrainableSentimentMetric`) which extend regular metrics with reward computation capabilities.

### Custom Training Metrics

Create custom trainable metrics by inheriting from both your custom metric and `TrainableMetric`:

```python
from vizra.evaluation.metrics import BaseMetric, TrainableMetric

class CustomMetric(BaseMetric):
    name = "custom_check"
    
    def evaluate(self, row_data: dict, response: str) -> dict:
        passed = "keyword" in response.lower()
        return {
            'passed': passed,
            'score': 1.0 if passed else 0.0,
            'details': {'found_keyword': passed}
        }

class TrainableCustomMetric(CustomMetric, TrainableMetric):
    def compute_reward(self, result: dict) -> float:
        # Custom reward logic
        if result['passed']:
            return 1.0  # High reward for success
        else:
            return -0.5  # Penalty for failure
```

## CLI Commands

Vizra includes a beautiful CLI with Rich styling:

| Command | Description | Options | Examples |
|---------|-------------|---------|----------|
| `vizra status` | Show installation status | None | `vizra status` |
| `vizra eval list` | List available evaluations | None | `vizra eval list` |
| `vizra eval run <name>` | Run an evaluation | `-v` (verbose)<br>`-o <file>` (custom output)<br>`-l <num>` (limit test cases)<br>`-d` (detailed CSV)<br>`-j` (JSON output) | `vizra eval run my_eval -v -l 10`<br>`vizra eval run my_eval -d -j` |
| `vizra train list` | List training routines | None | `vizra train list` |
| `vizra train run <name>` | Run training | `-v` (verbose)<br>`-o <file>` (output file)<br>`-i <num>` (override iterations) | `vizra train run my_training -i 50 -v` |

## Advanced Features

### AgentContext

The context object maintains conversation state:
- Message history
- Tool call tracking
- Metadata storage

```python
from vizra import AgentContext

context = AgentContext()
# Context automatically tracks conversation as agent runs
```

### Error Handling

- Tools return errors as JSON instead of throwing exceptions
- Agents continue execution even if tools fail
- Hook methods don't interrupt agent execution if they fail

### LLM Integration

Uses litellm for broad LLM provider support:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Azure OpenAI
- And many more via litellm

## Examples

Check the `examples/` directory for complete working examples of agents, tools, evaluations, and training routines.

## License

MIT