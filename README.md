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
    tools = [OrderLookupTool, RefundProcessorTool]

# Or load instructions from a file
class AdvancedSupportAgent(BaseAgent):
    name = 'advanced_support'
    description = 'Advanced support agent with complex instructions'
    instructions_file = 'prompts/advanced_support.md'  # Path relative to your project
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
- File-based instruction loading
- Simple and intuitive API

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