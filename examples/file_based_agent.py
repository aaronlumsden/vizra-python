"""
Example showing how to use file-based instructions with agents.

In your project, you might have a structure like:
    your_project/
    ├── agents/
    │   └── support_agent.py
    ├── prompts/
    │   └── customer_support.md
    └── main.py
"""

from vizra import BaseAgent
from examples.tools.order_lookup import OrderLookupTool
from examples.tools.refund_processor import RefundProcessorTool


class FileBasedSupportAgent(BaseAgent):
    """
    Example agent that loads instructions from a markdown file.
    """
    name = 'file_based_support'
    description = 'Customer support agent with file-based instructions'
    
    # Path relative to where the script is run
    # Users would typically use paths like:
    # - 'prompts/customer_support.md'
    # - './instructions/agent_prompt.md'
    # - '/absolute/path/to/prompt.md'
    instructions_file = 'example_prompt.md'  # This would be in user's project
    
    model = 'gpt-4o'
    tools = [OrderLookupTool, RefundProcessorTool]


# Example with inline instructions for comparison
class InlineSupportAgent(BaseAgent):
    """
    Traditional agent with inline instructions.
    """
    name = 'inline_support'
    description = 'Customer support agent with inline instructions'
    instructions = """You are a friendly customer support assistant.
Always be helpful and provide accurate information.
When customers ask about orders, use the order lookup tool.
When customers request refunds, gather necessary information and process the refund.
Be empathetic and professional in all interactions."""
    model = 'gpt-4o'
    tools = [OrderLookupTool, RefundProcessorTool]


if __name__ == "__main__":
    # Note: For the file-based agent to work, you would need to create
    # an 'example_prompt.md' file in your project directory
    
    # Using inline agent (always works)
    print("Using inline instructions:")
    response = InlineSupportAgent.run("What's the status of order ORDER123?")
    print(response)
    
    # Using file-based agent (requires prompt file)
    # Uncomment the following to test with a file:
    # print("\nUsing file-based instructions:")
    # response = FileBasedSupportAgent.run("What's the status of order ORDER123?")
    # print(response)