from abc import ABC, abstractmethod
from typing import Dict, Any


class ToolInterface(ABC):
    """
    Abstract base class for tools that can be used by agents.
    """
    
    @abstractmethod
    def definition(self) -> Dict[str, Any]:
        """
        Return the tool definition in OpenAI function calling format.
        
        Returns:
            dict: Tool definition with name, description, and parameters
        """
        pass
    
    @abstractmethod
    def execute(self, arguments: Dict[str, Any], context: 'AgentContext') -> str:
        """
        Execute the tool with given arguments and context.
        
        Args:
            arguments: The arguments passed to the tool
            context: The agent context containing conversation history
            
        Returns:
            str: Result of the tool execution
        """
        pass