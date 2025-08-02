import json
from vizra import ToolInterface, AgentContext


class RefundProcessorTool(ToolInterface):
    """
    A tool to process refunds for orders.
    """
    def definition(self) -> dict:
        """
        Defines the tool's name, description, and required parameters.
        """
        return {
            'name': 'process_refund',
            'description': 'Process a refund for an order',
            'parameters': {
                'type': 'object',
                'properties': {
                    'order_id': {
                        'type': 'string',
                        'description': 'The order ID to refund',
                    },
                    'reason': {
                        'type': 'string',
                        'description': 'The reason for the refund',
                    },
                    'amount': {
                        'type': 'number',
                        'description': 'The amount to refund (optional, defaults to full refund)',
                    },
                },
                'required': ['order_id', 'reason'],
            },
        }

    def execute(self, arguments: dict, context: AgentContext) -> str:
        """
        Executes the refund processing logic.

        Args:
            arguments (dict): The arguments provided by the agent.
            context (AgentContext): The current agent context.

        Returns:
            str: A JSON-formatted string with the refund result.
        """
        order_id = arguments.get('order_id')
        reason = arguments.get('reason')
        amount = arguments.get('amount')
        
        # Mock refund processing
        # In a real implementation, this would integrate with payment systems
        
        # Simulate checking if order exists
        valid_orders = ['ORDER123', 'ORDER456', 'ORDER789']
        
        if order_id not in valid_orders:
            return json.dumps({
                "error": f"Order {order_id} not found",
                "success": False
            })
        
        # Simulate processing the refund
        refund_data = {
            "success": True,
            "order_id": order_id,
            "refund_id": f"REFUND-{order_id}-001",
            "reason": reason,
            "amount": amount if amount else "Full refund",
            "status": "processed",
            "message": "Refund has been successfully processed. The customer will receive their refund within 3-5 business days."
        }
        
        return json.dumps(refund_data)