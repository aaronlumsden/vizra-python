from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentContext:
    """
    Context for agent execution containing conversation history and metadata.
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    max_tool_iterations: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to the conversation history."""
        message = {"role": role, "content": content}
        message.update(kwargs)
        self.messages.append(message)
    
    def add_tool_call(self, tool_call: Dict[str, Any]) -> None:
        """Add a tool call to the last assistant message."""
        if self.messages and self.messages[-1]["role"] == "assistant":
            if "tool_calls" not in self.messages[-1]:
                self.messages[-1]["tool_calls"] = []
            self.messages[-1]["tool_calls"].append(tool_call)
    
    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        """Add a tool result message."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result
        })
        self.tool_call_count += 1
    
    def can_call_tools(self) -> bool:
        """Check if more tool calls are allowed."""
        return self.tool_call_count < self.max_tool_iterations
    
    def reset_tool_count(self) -> None:
        """Reset the tool call counter."""
        self.tool_call_count = 0