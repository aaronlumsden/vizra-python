from typing import List, Type
from vizra import BaseAgent, ToolInterface
from examples.tools.order_lookup import OrderLookupTool
from examples.tools.refund_processor import RefundProcessorTool


class CustomerSupportAgent(BaseAgent):
    name: str = 'customer_support'
    description: str = 'Helps customers with their inquiries'
    instructions: str = '''You are a friendly customer support assistant. 
    Always be helpful and provide accurate information.
    When customers ask about orders, use the order lookup tool.
    When customers request refunds, gather necessary information and process the refund.
    Be empathetic and professional in all interactions.'''
    model: str = 'gpt-4o'
    tools: List[Type[ToolInterface]] = [
        OrderLookupTool,
        RefundProcessorTool,
    ]


# Example usage
if __name__ == "__main__":
    # Simple usage
    response = CustomerSupportAgent.run("What's the status of order ORDER123?")
    print(response)
    
    # Usage with context for conversation continuity
    from vizra import AgentContext
    
    context = AgentContext()
    agent_runner = CustomerSupportAgent.with_context(context)
    
    # First message
    response1 = agent_runner.run("Hi, I need help with my order")
    print("Response 1:", response1)
    
    # Follow-up message (maintains context)
    response2 = agent_runner.run("Can you check order ORDER456?")
    print("Response 2:", response2)
    
    # Another follow-up
    response3 = agent_runner.run("I'd like to request a refund for it")
    print("Response 3:", response3)