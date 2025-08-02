"""
Example showing how to use hooks in agents for monitoring and logging.
"""

import json
from vizra import BaseAgent, AgentContext
from examples.tools.order_lookup import OrderLookupTool
from examples.tools.refund_processor import RefundProcessorTool


class MonitoredSupportAgent(BaseAgent):
    """
    Example agent that uses hooks for monitoring and logging.
    """
    name = 'monitored_support'
    description = 'Customer support agent with monitoring hooks'
    instructions = '''You are a friendly customer support assistant. 
    Always be helpful and provide accurate information.
    When customers ask about orders, use the order lookup tool.
    When customers request refunds, gather necessary information and process the refund.'''
    model = 'gpt-4o'
    tools = [OrderLookupTool, RefundProcessorTool]
    
    def before_llm_call(self, messages, tools):
        """Log before making an LLM call."""
        print(f"\n🔵 Making LLM call:")
        print(f"  - Messages: {len(messages)}")
        print(f"  - Tools available: {len(tools) if tools else 0}")
        print(f"  - Last message: {messages[-1]['content'][:50]}...")
    
    def after_llm_response(self, response, messages):
        """Log after receiving LLM response."""
        print(f"\n🟢 LLM Response received:")
        # Track token usage if available
        if hasattr(response, 'usage'):
            print(f"  - Tokens used: {response.usage.total_tokens}")
        # Check if tools were called
        message = response.choices[0].message
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"  - Tool calls requested: {len(message.tool_calls)}")
        else:
            print(f"  - Response preview: {message.content[:100]}...")
    
    def before_tool_call(self, tool_name, arguments, context):
        """Log before executing a tool."""
        print(f"\n🔧 Executing tool: {tool_name}")
        print(f"  - Arguments: {json.dumps(arguments, indent=2)}")
    
    def after_tool_result(self, tool_name, result, context):
        """Log after tool execution."""
        print(f"\n✅ Tool result from {tool_name}:")
        try:
            # Try to pretty-print JSON results
            result_data = json.loads(result)
            print(f"  - Result: {json.dumps(result_data, indent=2)}")
        except:
            print(f"  - Result: {result[:100]}...")


class SecurityAuditAgent(BaseAgent):
    """
    Example agent that uses hooks for security auditing.
    """
    name = 'security_audit'
    description = 'Agent with security audit hooks'
    instructions = 'You are a helpful assistant with security auditing enabled.'
    model = 'gpt-4o'
    tools = [OrderLookupTool, RefundProcessorTool]
    
    def before_tool_call(self, tool_name, arguments, context):
        """Audit tool calls for security purposes."""
        # Check for sensitive operations
        if tool_name == 'process_refund':
            amount = arguments.get('amount', 'full')
            print(f"\n⚠️  SECURITY AUDIT: Refund requested")
            print(f"  - Order ID: {arguments.get('order_id')}")
            print(f"  - Amount: {amount}")
            print(f"  - Reason: {arguments.get('reason')}")
            
            # In a real system, you might:
            # - Log to security audit trail
            # - Check against fraud patterns
            # - Require additional authentication for large amounts
            # - Send alerts for suspicious activity
    
    def after_llm_response(self, response, messages):
        """Check for potential data leaks in responses."""
        content = response.choices[0].message.content or ""
        
        # Check for patterns that might indicate sensitive data
        sensitive_patterns = ['password', 'credit card', 'ssn', 'api key']
        for pattern in sensitive_patterns:
            if pattern.lower() in content.lower():
                print(f"\n⚠️  SECURITY WARNING: Response may contain sensitive data: {pattern}")


if __name__ == "__main__":
    print("=== Monitored Support Agent Example ===")
    
    # Create an agent with monitoring hooks
    response = MonitoredSupportAgent.run("What's the status of order ORDER123?")
    print(f"\nFinal response: {response}")
    
    print("\n" + "="*50 + "\n")
    print("=== Security Audit Agent Example ===")
    
    # Create an agent with security audit hooks
    context = AgentContext()
    agent_runner = SecurityAuditAgent.with_context(context)
    
    # First message
    response1 = agent_runner.run("I need to process a refund for order ORDER456")
    print(f"\nResponse: {response1}")
    
    # Follow-up that triggers the refund
    response2 = agent_runner.run("Yes, please process a full refund. The product was defective.")
    print(f"\nResponse: {response2}")