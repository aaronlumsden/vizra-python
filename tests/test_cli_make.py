"""
Tests for Vizra CLI make commands.
"""

import pytest
from click.testing import CliRunner
from pathlib import Path
from vizra.cli import cli
from vizra.cli.make import to_pascal_case, ensure_suffix


class TestMakeHelpers:
    """Test helper functions for make commands."""
    
    def test_to_pascal_case(self):
        """Test snake_case to PascalCase conversion."""
        assert to_pascal_case('customer_support') == 'CustomerSupport'
        assert to_pascal_case('customer_support_agent') == 'CustomerSupportAgent'
        assert to_pascal_case('simple') == 'Simple'
        assert to_pascal_case('order_lookup_tool') == 'OrderLookupTool'
    
    def test_ensure_suffix_agent(self):
        """Test suffix handling for agents."""
        # Already has suffix
        file_name, class_name = ensure_suffix('customer_support_agent', ['_agent', 'agent'], 'Agent')
        assert file_name == 'customer_support_agent'
        assert class_name == 'CustomerSupportAgent'
        
        # No suffix - should add
        file_name, class_name = ensure_suffix('customer_support', ['_agent', 'agent'], 'Agent')
        assert file_name == 'customer_support'
        assert class_name == 'CustomerSupportAgent'
    
    def test_ensure_suffix_tool(self):
        """Test suffix handling for tools."""
        # Already has suffix
        file_name, class_name = ensure_suffix('order_lookup_tool', ['_tool', 'tool'], 'Tool')
        assert file_name == 'order_lookup_tool'
        assert class_name == 'OrderLookupTool'
        
        # No suffix - should add
        file_name, class_name = ensure_suffix('order_lookup', ['_tool', 'tool'], 'Tool')
        assert file_name == 'order_lookup'
        assert class_name == 'OrderLookupTool'
    
    def test_ensure_suffix_evaluation(self):
        """Test suffix handling for evaluations."""
        # Short suffix
        file_name, class_name = ensure_suffix('chord_eval', ['_eval', '_evaluation', 'eval', 'evaluation'], 'Evaluation')
        assert file_name == 'chord_eval'
        assert class_name == 'ChordEval'
        
        # Full suffix
        file_name, class_name = ensure_suffix('chord_evaluation', ['_eval', '_evaluation', 'eval', 'evaluation'], 'Evaluation')
        assert file_name == 'chord_evaluation'
        assert class_name == 'ChordEvaluation'
        
        # No suffix - should add
        file_name, class_name = ensure_suffix('chord_accuracy', ['_eval', '_evaluation', 'eval', 'evaluation'], 'Evaluation')
        assert file_name == 'chord_accuracy'
        assert class_name == 'ChordAccuracyEvaluation'


class TestMakeCommands:
    """Test make commands."""
    
    def test_make_help(self):
        """Test make command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['make', '--help'])
        
        assert result.exit_code == 0
        assert 'Generate boilerplate code' in result.output
        assert 'agent' in result.output
        assert 'tool' in result.output
        assert 'evaluation' in result.output
        assert 'training' in result.output
        assert 'metric' in result.output
    
    def test_make_agent(self):
        """Test creating an agent."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test with suffix
            result = runner.invoke(cli, ['make', 'agent', 'customer_support_agent'])
            assert result.exit_code == 0
            assert 'Created agent: agents/customer_support_agent.py' in result.output
            assert 'Class name: CustomerSupportAgent' in result.output
            
            # Check file exists and content
            agent_file = Path('agents/customer_support_agent.py')
            assert agent_file.exists()
            content = agent_file.read_text()
            assert 'class CustomerSupportAgent(BaseAgent):' in content
            assert 'from vizra import BaseAgent' in content
            assert "name = 'customersupportagent'" in content
            
            # Check __init__.py was created
            assert Path('agents/__init__.py').exists()
            
            # Test without suffix
            result = runner.invoke(cli, ['make', 'agent', 'order_processor'])
            assert result.exit_code == 0
            assert 'Created agent: agents/order_processor.py' in result.output
            assert 'Class name: OrderProcessorAgent' in result.output
            
            # Test duplicate file
            result = runner.invoke(cli, ['make', 'agent', 'customer_support_agent'])
            assert result.exit_code == 1
            assert 'File already exists' in result.output
    
    def test_make_tool(self):
        """Test creating a tool."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test with suffix
            result = runner.invoke(cli, ['make', 'tool', 'order_lookup_tool'])
            assert result.exit_code == 0
            assert 'Created tool: tools/order_lookup_tool.py' in result.output
            assert 'Class name: OrderLookupTool' in result.output
            
            # Check file content
            tool_file = Path('tools/order_lookup_tool.py')
            assert tool_file.exists()
            content = tool_file.read_text()
            assert 'class OrderLookupTool(ToolInterface):' in content
            assert 'from vizra import ToolInterface, AgentContext' in content
            assert 'def definition(self)' in content
            assert 'def execute(self, arguments: dict, context: AgentContext)' in content
            
            # Test without suffix
            result = runner.invoke(cli, ['make', 'tool', 'weather_check'])
            assert result.exit_code == 0
            assert 'Class name: WeatherCheckTool' in result.output
    
    def test_make_evaluation(self):
        """Test creating an evaluation."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test with short suffix
            result = runner.invoke(cli, ['make', 'evaluation', 'chord_eval'])
            assert result.exit_code == 0
            assert 'Created evaluation: evaluations/chord_eval.py' in result.output
            assert 'Class name: ChordEval' in result.output
            
            # Check file content
            eval_file = Path('evaluations/chord_eval.py')
            assert eval_file.exists()
            content = eval_file.read_text()
            assert 'class ChordEval(BaseEvaluation):' in content
            assert 'from vizra.evaluation import BaseEvaluation' in content
            assert 'from vizra.evaluation.metrics import ContainsMetric' in content
            
            # Test without suffix
            result = runner.invoke(cli, ['make', 'evaluation', 'response_quality'])
            assert result.exit_code == 0
            assert 'Class name: ResponseQualityEvaluation' in result.output
    
    def test_make_training(self):
        """Test creating a training routine."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test with suffix
            result = runner.invoke(cli, ['make', 'training', 'agent_rl_training'])
            assert result.exit_code == 0
            assert 'Created training routine: training/agent_rl_training.py' in result.output
            assert 'Class name: AgentRlTraining' in result.output
            
            # Check file content
            training_file = Path('training/agent_rl_training.py')
            assert training_file.exists()
            content = training_file.read_text()
            assert 'class AgentRlTraining(BaseRLTraining):' in content
            assert 'from vizra.training import BaseRLTraining' in content
            assert 'def calculate_reward(self, csv_row_data: dict, agent_response: str)' in content
            
            # Test without suffix
            result = runner.invoke(cli, ['make', 'training', 'chord_identifier'])
            assert result.exit_code == 0
            assert 'Class name: ChordIdentifierTraining' in result.output
    
    def test_make_metric(self):
        """Test creating a metric."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Test with suffix
            result = runner.invoke(cli, ['make', 'metric', 'response_quality_metric'])
            assert result.exit_code == 0
            assert 'Created metric: metrics/response_quality_metric.py' in result.output
            assert 'Class name: ResponseQualityMetric' in result.output
            
            # Check file content
            metric_file = Path('metrics/response_quality_metric.py')
            assert metric_file.exists()
            content = metric_file.read_text()
            assert 'class ResponseQualityMetric(BaseMetric):' in content
            assert 'from vizra.evaluation.metrics import BaseMetric' in content
            assert 'def evaluate(self, row_data: Dict[str, Any], response: str)' in content
            
            # Test without suffix
            result = runner.invoke(cli, ['make', 'metric', 'sentiment_accuracy'])
            assert result.exit_code == 0
            assert 'Class name: SentimentAccuracyMetric' in result.output
    
    def test_make_invalid_command(self):
        """Test invalid make subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, ['make', 'invalid'])
        
        assert result.exit_code == 2
        assert 'No such command' in result.output
    
    def test_make_agent_complex_names(self):
        """Test agent creation with various naming patterns."""
        runner = CliRunner()
        
        with runner.isolated_filesystem():
            # Multi-word names
            result = runner.invoke(cli, ['make', 'agent', 'multi_word_customer_support'])
            assert result.exit_code == 0
            assert 'Class name: MultiWordCustomerSupportAgent' in result.output
            
            # Name ending with 'agent' (not '_agent')
            result = runner.invoke(cli, ['make', 'agent', 'travelagent'])
            assert result.exit_code == 0
            assert 'Class name: Travelagent' in result.output
            
            # Name with numbers
            result = runner.invoke(cli, ['make', 'agent', 'gpt4_agent'])
            assert result.exit_code == 0
            assert 'Class name: Gpt4Agent' in result.output