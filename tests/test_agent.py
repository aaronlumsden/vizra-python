"""
Tests for BaseAgent class.
"""

import pytest
from unittest.mock import patch, MagicMock
import os
from vizra import BaseAgent, AgentContext
from vizra.exceptions import AgentExecutionError


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    def test_agent_attributes(self):
        """Test agent class attributes are set correctly."""
        class CustomAgent(BaseAgent):
            name = 'custom'
            description = 'Custom agent'
            instructions = 'Custom instructions'
            model = 'gpt-3.5-turbo'
            tools = []
        
        assert CustomAgent.name == 'custom'
        assert CustomAgent.description == 'Custom agent'
        assert CustomAgent.instructions == 'Custom instructions'
        assert CustomAgent.model == 'gpt-3.5-turbo'
        assert CustomAgent.tools == []
    
    def test_simple_run(self, mock_completion):
        """Test simple agent run without tools."""
        # Create a custom side effect to capture the messages at call time
        captured_messages = None
        
        def capture_and_respond(**kwargs):
            nonlocal captured_messages
            # Make a copy of messages at the time of the call
            captured_messages = kwargs['messages'].copy()
            # Return the mock response
            response = MagicMock()
            response.choices = [MagicMock()]
            response.choices[0].message.content = "Test response"
            response.choices[0].message.tool_calls = None
            response.usage.total_tokens = 100
            return response
        
        mock_completion.side_effect = capture_and_respond
        
        response = BaseAgent.run("Hello")
        
        assert response == "Test response"
        mock_completion.assert_called_once()
        
        # Check the captured messages
        assert len(captured_messages) == 2
        assert captured_messages[0]['role'] == 'system'
        assert captured_messages[0]['content'] == BaseAgent.instructions
        assert captured_messages[1]['role'] == 'user'
        assert captured_messages[1]['content'] == 'Hello'
    
    def test_run_with_context(self, mock_completion):
        """Test agent run with existing context."""
        context = AgentContext()
        context.add_message("system", "Previous instructions")
        context.add_message("user", "Previous message")
        context.add_message("assistant", "Previous response")
        
        response = BaseAgent.run("New message", context)
        
        assert response == "Test response"
        # Should not add system message again
        assert len(context.messages) == 5  # Previous 3 + new user + new assistant
    
    def test_instructions_from_file(self, mock_completion):
        """Test loading instructions from file."""
        class FileBasedAgent(BaseAgent):
            instructions_file = 'tests/fixtures/test_prompt.md'
            model = 'gpt-4o'
        
        response = FileBasedAgent.run("Hello")
        
        assert response == "Test response"
        
        # Check that file content was used
        call_args = mock_completion.call_args
        messages = call_args.kwargs['messages']
        assert "Test Instructions" in messages[0]['content']
        assert "test assistant created for unit testing" in messages[0]['content']
    
    def test_instructions_file_not_found(self):
        """Test error when instructions file not found."""
        class BadFileAgent(BaseAgent):
            instructions_file = 'nonexistent.md'
        
        with pytest.raises(AgentExecutionError, match="Instructions file not found"):
            BadFileAgent.run("Hello")
    
    def test_with_context_method(self):
        """Test the with_context class method."""
        context = AgentContext()
        runner = BaseAgent.with_context(context)
        
        assert runner.context is context
        assert runner.agent_class is BaseAgent
    
    def test_agent_runner(self, mock_completion):
        """Test AgentRunner functionality."""
        context = AgentContext()
        runner = BaseAgent.with_context(context)
        
        response1 = runner.run("First message")
        assert response1 == "Test response"
        
        response2 = runner.run("Second message")
        assert response2 == "Test response"
        
        # Context should accumulate messages
        assert len(context.messages) == 5  # system + 2 user + 2 assistant
    
    def test_model_configuration(self, mock_completion):
        """Test that model configuration is passed correctly."""
        class CustomModelAgent(BaseAgent):
            model = 'claude-3-sonnet-20240229'
        
        CustomModelAgent.run("Hello")
        
        call_args = mock_completion.call_args
        assert call_args.kwargs['model'] == 'claude-3-sonnet-20240229'
    
    def test_empty_tools_list(self, mock_completion):
        """Test agent with empty tools list."""
        class NoToolsAgent(BaseAgent):
            tools = []
        
        response = NoToolsAgent.run("Hello")
        
        assert response == "Test response"
        call_args = mock_completion.call_args
        assert 'tools' in call_args.kwargs
        assert call_args.kwargs['tools'] is None