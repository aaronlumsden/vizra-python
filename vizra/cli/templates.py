"""
Template generators for Vizra CLI make commands.
"""

def get_agent_template(class_name: str) -> str:
    """Generate agent class template."""
    return f'''from vizra import BaseAgent


class {class_name}(BaseAgent):
    name = '{class_name.lower()}'
    description = 'Description of what this agent does'
    instructions = \'\'\'You are a helpful AI assistant.
    Provide clear and accurate responses.\'\'\'
    model = 'gpt-4o'
    tools = []
'''


def get_tool_template(class_name: str) -> str:
    """Generate tool class template."""
    return f'''import json
from vizra import ToolInterface, AgentContext


class {class_name}(ToolInterface):
    def definition(self) -> dict:
        return {{
            'name': '{class_name.replace("Tool", "").lower()}_tool',
            'description': 'Description of what this tool does',
            'parameters': {{
                'type': 'object',
                'properties': {{
                    'input': {{
                        'type': 'string',
                        'description': 'The input parameter',
                    }},
                }},
                'required': ['input'],
            }},
        }}

    def execute(self, arguments: dict, context: AgentContext) -> str:
        input_value = arguments.get('input', '')
        
        # Implement your tool logic here
        result = {{
            'success': True,
            'output': f'Processed: {{input_value}}'
        }}
        
        return json.dumps(result)
'''


def get_evaluation_template(class_name: str) -> str:
    """Generate evaluation class template."""
    return f'''from vizra.evaluation import BaseEvaluation
from vizra.evaluation.metrics import ContainsMetric


class {class_name}(BaseEvaluation):
    name = '{class_name.lower()}'
    description = 'Description of what this evaluation tests'
    agent_name = 'agent_to_evaluate'  # Update this
    csv_path = 'data/test_cases.csv'  # Update this
    
    metrics = [
        ContainsMetric('expected_response'),
    ]
'''


def get_training_template(class_name: str) -> str:
    """Generate training class template."""
    return f'''from vizra.training import BaseRLTraining


class {class_name}(BaseRLTraining):
    name = '{class_name.lower()}'
    description = 'Description of this training routine'
    agent_name = 'agent_to_train'  # Update this
    csv_path = 'data/training_data.csv'  # Update this
    
    # Training hyperparameters
    algorithm = 'ppo'
    learning_rate = 5e-5
    batch_size = 16
    n_iterations = 50
    
    def calculate_reward(self, csv_row_data: dict, agent_response: str) -> float:
        """
        Calculate reward based on response quality.
        Returns a value between 0.0 and 1.0.
        """
        expected = csv_row_data.get('expected_response', '').lower()
        response_lower = agent_response.lower()
        
        # Simple reward: 1.0 if expected text is in response, 0.0 otherwise
        if expected in response_lower:
            return 1.0
        else:
            return 0.0
'''


def get_metric_template(class_name: str) -> str:
    """Generate metric class template."""
    return f'''from typing import Dict, Any
from vizra.evaluation.metrics import BaseMetric


class {class_name}(BaseMetric):
    name = '{class_name.lower()}'
    
    def __init__(self):
        """Initialize the metric."""
        pass
    
    def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Evaluate the response against the metric criteria.
        
        Args:
            row_data: Dictionary containing CSV row data
            response: The agent's response
            
        Returns:
            dict: Result dictionary with:
                - passed: bool indicating if metric passed
                - score: numeric score (0-1)
                - details: dict with additional information
        """
        # Implement your metric logic here
        passed = True  # Update based on your criteria
        score = 1.0 if passed else 0.0
        
        return {{
            'passed': passed,
            'score': score,
            'details': {{
                'response_preview': response[:100] + '...' if len(response) > 100 else response
            }}
        }}
'''