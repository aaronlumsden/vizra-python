import json
from typing import Dict, Any
from vizra import ToolInterface, AgentContext


class OrderLookupTool(ToolInterface):
    """
    A tool to look up order information by order ID.
    """
    def definition(self) -> Dict[str, Any]:
        """
        Defines the tool's name, description, and required parameters.
        """
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

    def execute(self, arguments: Dict[str, Any], context: AgentContext) -> str:
        """
        Executes the tool logic to find and return order details.

        Args:
            arguments (dict): The arguments provided by the agent, e.g., {'order_id': 'ORDER123'}.
            context (AgentContext): The current agent context.

        Returns:
            str: A JSON-formatted string containing the order details or an error message.
        """
        # This is where you would implement actual order lookup logic
        # For demonstration, we'll return mock data
        order_id = arguments.get('order_id')
        
        # Mock order data
        mock_orders = {
            'ORDER123': {
                'order_id': 'ORDER123',
                'status': 'shipped',
                'total': 99.99,
                'items': ['Widget A', 'Widget B']
            },
            'ORDER456': {
                'order_id': 'ORDER456',
                'status': 'processing',
                'total': 149.99,
                'items': ['Gadget X']
            }
        }
        
        if order_id in mock_orders:
            return json.dumps(mock_orders[order_id])
        else:
            return json.dumps({"error": f"Order {order_id} not found"})