"""
Tests for AgentContext class.
"""

import pytest
from vizra import AgentContext


class TestAgentContext:
    """Test AgentContext functionality."""
    
    def test_context_initialization(self):
        """Test context is initialized correctly."""
        context = AgentContext()
        
        assert context.messages == []
        assert context.tool_call_count == 0
        assert context.max_tool_iterations == 3
        assert context.metadata == {}
    
    def test_add_message(self):
        """Test adding messages to context."""
        context = AgentContext()
        
        context.add_message("user", "Hello")
        assert len(context.messages) == 1
        assert context.messages[0] == {"role": "user", "content": "Hello"}
        
        context.add_message("assistant", "Hi there", extra_field="value")
        assert len(context.messages) == 2
        assert context.messages[1] == {
            "role": "assistant",
            "content": "Hi there",
            "extra_field": "value"
        }
    
    def test_add_tool_call(self):
        """Test adding tool calls to assistant message."""
        context = AgentContext()
        
        # Add assistant message first
        context.add_message("assistant", "Let me help")
        
        # Add tool call
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "test_tool",
                "arguments": '{"arg": "value"}'
            }
        }
        context.add_tool_call(tool_call)
        
        assert "tool_calls" in context.messages[-1]
        assert len(context.messages[-1]["tool_calls"]) == 1
        assert context.messages[-1]["tool_calls"][0] == tool_call
    
    def test_add_tool_call_without_assistant_message(self):
        """Test adding tool call without assistant message does nothing."""
        context = AgentContext()
        context.add_message("user", "Hello")
        
        tool_call = {"id": "call_123"}
        context.add_tool_call(tool_call)
        
        # Should not add tool_calls to user message
        assert "tool_calls" not in context.messages[-1]
    
    def test_add_tool_result(self):
        """Test adding tool results."""
        context = AgentContext()
        
        context.add_tool_result("call_123", "Tool result")
        
        assert len(context.messages) == 1
        assert context.messages[0] == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Tool result"
        }
        assert context.tool_call_count == 1
    
    def test_can_call_tools(self):
        """Test tool call limit checking."""
        context = AgentContext()
        
        assert context.can_call_tools() is True
        
        # Add tool results up to the limit
        context.add_tool_result("call_1", "Result 1")
        assert context.can_call_tools() is True
        
        context.add_tool_result("call_2", "Result 2")
        assert context.can_call_tools() is True
        
        context.add_tool_result("call_3", "Result 3")
        assert context.can_call_tools() is False  # Reached limit
    
    def test_reset_tool_count(self):
        """Test resetting tool call counter."""
        context = AgentContext()
        
        # Add some tool calls
        context.add_tool_result("call_1", "Result 1")
        context.add_tool_result("call_2", "Result 2")
        assert context.tool_call_count == 2
        
        # Reset
        context.reset_tool_count()
        assert context.tool_call_count == 0
        assert context.can_call_tools() is True
    
    def test_custom_max_iterations(self):
        """Test custom max tool iterations."""
        context = AgentContext(max_tool_iterations=5)
        
        assert context.max_tool_iterations == 5
        
        # Add 4 tool calls
        for i in range(4):
            context.add_tool_result(f"call_{i}", f"Result {i}")
        
        assert context.can_call_tools() is True
        
        # Add 5th call
        context.add_tool_result("call_5", "Result 5")
        assert context.can_call_tools() is False
    
    def test_metadata(self):
        """Test metadata storage."""
        context = AgentContext()
        
        context.metadata["user_id"] = "123"
        context.metadata["session"] = "abc"
        
        assert context.metadata["user_id"] == "123"
        assert context.metadata["session"] == "abc"
    
    def test_message_history_preservation(self):
        """Test that message history is preserved correctly."""
        context = AgentContext()
        
        # Build a conversation
        context.add_message("system", "You are helpful")
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi!")
        context.add_tool_call({
            "id": "call_1",
            "type": "function",
            "function": {"name": "tool", "arguments": "{}"}
        })
        context.add_tool_result("call_1", "Result")
        context.add_message("assistant", "Here's your answer")
        
        assert len(context.messages) == 5  # system, user, assistant, tool, assistant
        assert context.messages[0]["role"] == "system"
        assert context.messages[1]["role"] == "user"
        assert context.messages[2]["role"] == "assistant"
        assert context.messages[3]["role"] == "tool"
        assert context.messages[4]["role"] == "assistant"