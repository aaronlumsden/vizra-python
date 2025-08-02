"""
Tests for ToolInterface class.
"""

import pytest
import json
from vizra import ToolInterface, AgentContext


class TestToolInterface:
    """Test ToolInterface functionality."""
    
    def test_tool_interface_abstract(self):
        """Test that ToolInterface is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ToolInterface()
    
    def test_tool_implementation(self):
        """Test implementing a tool."""
        class MyTool(ToolInterface):
            def definition(self):
                return {
                    'name': 'my_tool',
                    'description': 'My test tool',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'message': {
                                'type': 'string',
                                'description': 'A message'
                            }
                        },
                        'required': ['message']
                    }
                }
            
            def execute(self, arguments, context):
                return json.dumps({
                    'response': f"Received: {arguments['message']}"
                })
        
        tool = MyTool()
        definition = tool.definition()
        
        assert definition['name'] == 'my_tool'
        assert definition['description'] == 'My test tool'
        assert 'parameters' in definition
        assert definition['parameters']['required'] == ['message']
        
        # Test execution
        context = AgentContext()
        result = tool.execute({'message': 'Hello'}, context)
        result_data = json.loads(result)
        
        assert result_data['response'] == 'Received: Hello'
    
    def test_tool_error_handling(self):
        """Test tool error handling."""
        class ErrorTool(ToolInterface):
            def definition(self):
                return {
                    'name': 'error_tool',
                    'description': 'Tool that raises errors',
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': []
                    }
                }
            
            def execute(self, arguments, context):
                raise ValueError("Tool execution failed")
        
        tool = ErrorTool()
        context = AgentContext()
        
        # Tool should raise the error (error handling is done in the agent)
        with pytest.raises(ValueError, match="Tool execution failed"):
            tool.execute({}, context)
    
    def test_tool_with_optional_parameters(self):
        """Test tool with optional parameters."""
        class OptionalParamTool(ToolInterface):
            def definition(self):
                return {
                    'name': 'optional_tool',
                    'description': 'Tool with optional parameters',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'required_param': {
                                'type': 'string',
                                'description': 'Required parameter'
                            },
                            'optional_param': {
                                'type': 'number',
                                'description': 'Optional parameter'
                            }
                        },
                        'required': ['required_param']
                    }
                }
            
            def execute(self, arguments, context):
                result = {
                    'required': arguments['required_param'],
                    'optional': arguments.get('optional_param', 'not provided')
                }
                return json.dumps(result)
        
        tool = OptionalParamTool()
        context = AgentContext()
        
        # Test with only required parameter
        result = tool.execute({'required_param': 'test'}, context)
        data = json.loads(result)
        assert data['required'] == 'test'
        assert data['optional'] == 'not provided'
        
        # Test with both parameters
        result = tool.execute({
            'required_param': 'test',
            'optional_param': 42
        }, context)
        data = json.loads(result)
        assert data['required'] == 'test'
        assert data['optional'] == 42
    
    def test_tool_context_access(self):
        """Test that tools can access agent context."""
        class ContextAwareTool(ToolInterface):
            def definition(self):
                return {
                    'name': 'context_tool',
                    'description': 'Tool that uses context',
                    'parameters': {
                        'type': 'object',
                        'properties': {},
                        'required': []
                    }
                }
            
            def execute(self, arguments, context):
                # Access context information
                message_count = len(context.messages)
                tool_count = context.tool_call_count
                
                return json.dumps({
                    'message_count': message_count,
                    'tool_call_count': tool_count
                })
        
        tool = ContextAwareTool()
        context = AgentContext()
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi")
        context.tool_call_count = 2
        
        result = tool.execute({}, context)
        data = json.loads(result)
        
        assert data['message_count'] == 2
        assert data['tool_call_count'] == 2