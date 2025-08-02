# Claude Code Assistant Context

This document provides context about the Vizra Python AI Agent Framework to help Claude Code understand the project structure and assist effectively.

## Project Overview

Vizra is a lightweight, class-based AI agent framework for Python that uses litellm for LLM integration. It enables developers to create AI agents with tools, conversation management, and customizable behavior through hooks.

## Key Components

### BaseAgent (`vizra/agent.py`)
- Main agent class that users inherit from
- Supports inline instructions or file-based instructions
- Implements tool execution loop with max 3 iterations
- Provides 4 hook methods for customization
- Uses class method `run()` for execution

### ToolInterface (`vizra/tool.py`)
- Abstract base class for creating tools
- Requires `definition()` and `execute()` methods
- Tools receive arguments and AgentContext

### AgentContext (`vizra/context.py`)
- Manages conversation history
- Tracks tool calls and enforces iteration limits
- Provides metadata storage

### Hooks
1. `before_llm_call(messages, tools)` - Called before LLM requests
2. `after_llm_response(response, messages)` - Called after LLM responses
3. `before_tool_call(tool_name, arguments, context)` - Called before tool execution
4. `after_tool_result(tool_name, result, context)` - Called after tool results

## Code Style Guidelines

- No comments unless explicitly requested
- Use type hints consistently
- Follow existing patterns in the codebase
- Keep methods concise and focused
- Handle errors gracefully (tools return errors as JSON, not exceptions)

## Testing

- Run tests: `pytest`
- Run with coverage: `pytest --cov=vizra --cov-report=term-missing`
- Tests use mocks to avoid actual LLM API calls
- All tests should pass before committing changes

## Common Tasks

### Adding a New Feature
1. Check existing patterns in the codebase
2. Write tests first (TDD approach)
3. Implement the feature
4. Ensure all tests pass
5. Update README.md if needed

### Creating a New Tool
```python
class MyTool(ToolInterface):
    def definition(self):
        return {
            'name': 'my_tool',
            'description': 'Tool description',
            'parameters': {...}
        }
    
    def execute(self, arguments, context):
        # Implementation
        return json.dumps(result)
```

### Creating a New Agent
```python
class MyAgent(BaseAgent):
    name = 'my_agent'
    instructions = 'Agent instructions'
    model = 'gpt-4o'
    tools = [MyTool]
```

## Dependencies

- litellm: LLM integration
- openai: Required by litellm
- pytest, pytest-mock, pytest-cov: Testing

## File Structure
```
vizra-python/
├── vizra/              # Main package
├── examples/           # Example implementations
├── tests/              # Test suite
├── requirements.txt    # Dependencies
├── setup.py           # Package setup
└── README.md          # Documentation
```

## Important Notes

- The framework is designed to be simple and extensible
- Tool failures are handled gracefully (return errors as JSON)
- Hooks should not interrupt agent execution
- The `run()` method is a class method that creates an instance internally
- Context management allows for conversation continuity