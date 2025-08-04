"""
Tests for Vizra CLI template generators.
"""

import pytest
from vizra.cli.templates import (
    get_agent_template,
    get_tool_template,
    get_evaluation_template,
    get_training_template,
    get_metric_template
)


class TestTemplates:
    """Test template generation functions."""
    
    def test_agent_template(self):
        """Test agent template generation."""
        template = get_agent_template('CustomerSupportAgent')
        
        # Check imports
        assert 'from vizra import BaseAgent' in template
        
        # Check class definition
        assert 'class CustomerSupportAgent(BaseAgent):' in template
        
        # Check required attributes
        assert "name = 'customersupportagent'" in template
        assert 'description = ' in template
        assert 'instructions = ' in template
        assert "model = 'gpt-4o'" in template
        assert 'tools = []' in template
        
        # Check structure
        assert 'You are a helpful AI assistant' in template
    
    def test_tool_template(self):
        """Test tool template generation."""
        template = get_tool_template('OrderLookupTool')
        
        # Check imports
        assert 'import json' in template
        assert 'from vizra import ToolInterface, AgentContext' in template
        
        # Check class definition
        assert 'class OrderLookupTool(ToolInterface):' in template
        
        # Check methods
        assert 'def definition(self) -> dict:' in template
        assert 'def execute(self, arguments: dict, context: AgentContext) -> str:' in template
        
        # Check tool name extraction
        assert "'name': 'orderlookup_tool'" in template
        
        # Check structure
        assert "'type': 'object'" in template
        assert "'properties':" in template
        assert "'required': ['input']" in template
        assert 'return json.dumps(result)' in template
    
    def test_evaluation_template(self):
        """Test evaluation template generation."""
        template = get_evaluation_template('ChordIdentifierEvaluation')
        
        # Check imports
        assert 'from vizra.evaluation import BaseEvaluation' in template
        assert 'from vizra.evaluation.metrics import ContainsMetric' in template
        
        # Check class definition
        assert 'class ChordIdentifierEvaluation(BaseEvaluation):' in template
        
        # Check required attributes
        assert "name = 'chordidentifierevaluation'" in template
        assert 'description = ' in template
        assert 'agent_name = ' in template
        assert 'csv_path = ' in template
        
        # Check metrics
        assert 'metrics = [' in template
        assert "ContainsMetric('expected_response')" in template
    
    def test_training_template(self):
        """Test training template generation."""
        template = get_training_template('AgentRLTraining')
        
        # Check imports
        assert 'from vizra.training import BaseRLTraining' in template
        
        # Check class definition
        assert 'class AgentRLTraining(BaseRLTraining):' in template
        
        # Check required attributes
        assert "name = 'agentrltraining'" in template
        assert 'description = ' in template
        assert 'agent_name = ' in template
        assert 'csv_path = ' in template
        
        # Check hyperparameters
        assert "algorithm = 'ppo'" in template
        assert 'learning_rate = 5e-5' in template
        assert 'batch_size = 16' in template
        assert 'n_iterations = 50' in template
        
        # Check method
        assert 'def calculate_reward(self, csv_row_data: dict, agent_response: str) -> float:' in template
        assert 'Returns a value between 0.0 and 1.0' in template
    
    def test_metric_template(self):
        """Test metric template generation."""
        template = get_metric_template('ResponseQualityMetric')
        
        # Check imports
        assert 'from typing import Dict, Any' in template
        assert 'from vizra.evaluation.metrics import BaseMetric' in template
        
        # Check class definition
        assert 'class ResponseQualityMetric(BaseMetric):' in template
        
        # Check required attributes
        assert "name = 'responsequalitymetric'" in template
        
        # Check methods
        assert 'def __init__(self):' in template
        assert 'def evaluate(self, row_data: Dict[str, Any], response: str) -> Dict[str, Any]:' in template
        
        # Check docstring
        assert 'Evaluate the response against the metric criteria' in template
        assert 'Args:' in template
        assert 'Returns:' in template
        
        # Check return structure
        assert "'passed': passed" in template
        assert "'score': score" in template
        assert "'details':" in template
    
    def test_template_name_variations(self):
        """Test templates with various class name formats."""
        # Test with different casing
        agent_template = get_agent_template('GPT4Agent')
        assert "class GPT4Agent(BaseAgent):" in agent_template
        assert "name = 'gpt4agent'" in agent_template
        
        # Test tool name extraction
        tool_template = get_tool_template('ComplexAnalysisTool')
        assert "'name': 'complexanalysis_tool'" in tool_template
        
        # Test long names
        eval_template = get_evaluation_template('VeryLongAndComplexEvaluationNameEvaluation')
        assert "name = 'verylongandcomplexevaluationnameevaluation'" in eval_template