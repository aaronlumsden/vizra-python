"""
Pytest configuration and shared fixtures for vizra tests.
"""

import pytest
from unittest.mock import MagicMock, patch
import json


@pytest.fixture
def mock_completion():
    """Mock litellm completion function."""
    with patch('vizra.agent.completion') as mock:
        # Default response
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "Test response"
        response.choices[0].message.tool_calls = None
        response.usage.total_tokens = 100
        mock.return_value = response
        yield mock


@pytest.fixture
def mock_completion_with_tool_call():
    """Mock litellm completion that returns a tool call."""
    with patch('vizra.agent.completion') as mock:
        # First call - tool request
        tool_response = MagicMock()
        tool_response.choices = [MagicMock()]
        tool_response.choices[0].message.content = ""
        
        # Mock tool call
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "test_tool"
        tool_call.function.arguments = json.dumps({"arg": "value"})
        tool_response.choices[0].message.tool_calls = [tool_call]
        
        # Second call - final response
        final_response = MagicMock()
        final_response.choices = [MagicMock()]
        final_response.choices[0].message.content = "Final response"
        final_response.choices[0].message.tool_calls = None
        
        mock.side_effect = [tool_response, final_response]
        yield mock


@pytest.fixture
def sample_tool_class():
    """Create a sample tool class for testing."""
    from vizra import ToolInterface
    
    class TestTool(ToolInterface):
        def definition(self):
            return {
                'name': 'test_tool',
                'description': 'A test tool',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'arg': {'type': 'string', 'description': 'Test argument'}
                    },
                    'required': ['arg']
                }
            }
        
        def execute(self, arguments, context):
            return json.dumps({"result": f"Processed {arguments.get('arg')}"})
    
    return TestTool


@pytest.fixture
def sample_agent_class(sample_tool_class):
    """Create a sample agent class for testing."""
    from vizra import BaseAgent
    
    class TestAgent(BaseAgent):
        name = 'test_agent'
        description = 'A test agent'
        instructions = 'You are a test agent.'
        model = 'gpt-4o'
        tools = [sample_tool_class]
    
    return TestAgent