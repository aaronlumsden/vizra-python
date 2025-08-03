from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ToolInterface(ABC):
    """
    Abstract base class for tools that can be used by agents.
    """
    
    # Optional XML tag for tools that support XML format
    xml_tag: Optional[str] = None
    
    def definition(self) -> Dict[str, Any]:
        """
        Return the tool definition in OpenAI function calling format.
        
        Only required for tools that support OpenAI-style function calling.
        
        Returns:
            dict: Tool definition with name, description, and parameters
        """
        # Default implementation for XML-only tools
        return {}
    
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
    
    def parse_xml_content(self, content: str) -> Dict[str, Any]:
        """
        Parse XML content into arguments for execute().
        
        Override this method to provide custom parsing logic.
        
        Args:
            content: The content between XML tags
            
        Returns:
            dict: Arguments to pass to execute()
        """
        # Default implementation - just pass content as is
        return {"content": content}