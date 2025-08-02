"""
Tests for agent hook functionality.
"""

import pytest
from unittest.mock import MagicMock, patch, call
import json
from vizra import BaseAgent, AgentContext, ToolInterface


class TestHooks:
    """Test agent hook methods."""
    
    def test_before_llm_call_hook(self, mock_completion):
        """Test before_llm_call hook is called correctly."""
        hook_calls = []
        
        class HookedAgent(BaseAgent):
            def before_llm_call(self, messages, tools):
                hook_calls.append({
                    'messages': len(messages),
                    'tools': len(tools) if tools else 0
                })
        
        HookedAgent.run("Hello")
        
        assert len(hook_calls) == 1
        assert hook_calls[0]['messages'] == 2  # system + user
        assert hook_calls[0]['tools'] == 0
    
    def test_after_llm_response_hook(self, mock_completion):
        """Test after_llm_response hook is called correctly."""
        hook_calls = []
        
        class HookedAgent(BaseAgent):
            def after_llm_response(self, response, messages):
                hook_calls.append({
                    'response_content': response.choices[0].message.content,
                    'message_count': len(messages)
                })
        
        HookedAgent.run("Hello")
        
        assert len(hook_calls) == 1
        assert hook_calls[0]['response_content'] == "Test response"
        assert hook_calls[0]['message_count'] == 2
    
    def test_tool_hooks(self, mock_completion_with_tool_call, sample_tool_class):
        """Test before_tool_call and after_tool_result hooks."""
        before_calls = []
        after_calls = []
        
        class HookedAgent(BaseAgent):
            tools = [sample_tool_class]
            
            def before_tool_call(self, tool_name, arguments, context):
                before_calls.append({
                    'tool_name': tool_name,
                    'arguments': arguments
                })
            
            def after_tool_result(self, tool_name, result, context):
                after_calls.append({
                    'tool_name': tool_name,
                    'result': result
                })
        
        HookedAgent.run("Use the tool")
        
        # Check before_tool_call was called
        assert len(before_calls) == 1
        assert before_calls[0]['tool_name'] == 'test_tool'
        assert before_calls[0]['arguments'] == {'arg': 'value'}
        
        # Check after_tool_result was called
        assert len(after_calls) == 1
        assert after_calls[0]['tool_name'] == 'test_tool'
        assert 'Processed value' in after_calls[0]['result']
    
    def test_hook_error_handling(self, mock_completion):
        """Test that hook errors don't interrupt agent execution."""
        class BrokenHookAgent(BaseAgent):
            def before_llm_call(self, messages, tools):
                raise ValueError("Hook error")
            
            def after_llm_response(self, response, messages):
                raise RuntimeError("Another hook error")
        
        # Should still work despite hook errors
        response = BrokenHookAgent.run("Hello")
        assert response == "Test response"
    
    def test_multiple_llm_calls_with_hooks(self, mock_completion_with_tool_call, sample_tool_class):
        """Test hooks are called for each LLM interaction."""
        llm_call_count = 0
        
        class CountingAgent(BaseAgent):
            tools = [sample_tool_class]
            
            def before_llm_call(self, messages, tools):
                nonlocal llm_call_count
                llm_call_count += 1
        
        CountingAgent.run("Use the tool")
        
        # Should be called twice: once for tool request, once for final response
        assert llm_call_count == 2
    
    def test_hooks_with_max_iterations(self, sample_tool_class):
        """Test hooks when max tool iterations are reached."""
        before_calls = []
        after_calls = []
        
        class MultiToolAgent(BaseAgent):
            tools = [sample_tool_class]
            
            def before_llm_call(self, messages, tools):
                before_calls.append({
                    'has_tools': tools is not None
                })
            
            def after_llm_response(self, response, messages):
                after_calls.append({
                    'message_count': len(messages)
                })
        
        # Mock completion to always return tool calls until limit
        with patch('vizra.agent.completion') as mock:
            tool_response = MagicMock()
            tool_response.choices = [MagicMock()]
            tool_response.choices[0].message.content = ""
            
            tool_call = MagicMock()
            tool_call.id = "call_123"
            tool_call.function.name = "test_tool"
            tool_call.function.arguments = json.dumps({"arg": "value"})
            tool_response.choices[0].message.tool_calls = [tool_call]
            
            final_response = MagicMock()
            final_response.choices = [MagicMock()]
            final_response.choices[0].message.content = "Final"
            final_response.choices[0].message.tool_calls = None
            
            # Return tool calls 3 times, then final response
            mock.side_effect = [tool_response, tool_response, tool_response, final_response]
            
            MultiToolAgent.run("Use tools")
        
        # Should have 4 before_llm_call hooks (3 with tools, 1 without)
        assert len(before_calls) == 4
        assert before_calls[0]['has_tools'] is True
        assert before_calls[1]['has_tools'] is True
        assert before_calls[2]['has_tools'] is True
        assert before_calls[3]['has_tools'] is False  # Final call without tools
    
    def test_hook_access_to_self(self, mock_completion):
        """Test that hooks have access to self."""
        class StatefulAgent(BaseAgent):
            def __init__(self):
                super().__init__()
                self.call_count = 0
            
            def before_llm_call(self, messages, tools):
                self.call_count += 1
            
            def after_llm_response(self, response, messages):
                # Can access instance state
                assert self.call_count > 0
        
        response = StatefulAgent.run("Hello")
        assert response == "Test response"