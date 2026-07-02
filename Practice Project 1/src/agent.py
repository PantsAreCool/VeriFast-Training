import json
import re
from typing import List, Dict, Any, Optional
from src.llm_interface import LLMInterface
from src.tools import ToolRegistry

class LogAgent:
    def __init__(self, llm_client: LLMInterface, tool_registry: ToolRegistry):
        self.llm = llm_client
        self.registry = tool_registry
        
        tool_desc_list = []
        for tool in self.registry.tools_schema:
            func = tool["function"]
            name = func["name"]
            desc = func["description"]
            params = json.dumps(func.get("parameters", {}).get("properties", {}))
            tool_desc_list.append(f"- {name}: {desc}\n  Parameter Schema Properties: {params}")
        tool_descriptions = "\n".join(tool_desc_list)
        
        self.system_prompt = f"""You are a Log Analysis Agent. You help users understand errors,
patterns, and issues in their system logs.

You have access to these tools:
{tool_descriptions}

INSTRUCTIONS:
1. When you receive a user question, THINK about which tool(s) you need.
2. Respond with a tool call in this exact format when you need data:
   ACTION: tool_name
   PARAMETERS: {{"param1": "value1"}}
3. After receiving the tool result, analyze it and either:
   - Call another tool if you need more data
   - Provide your final answer
4. When you have enough information, respond with your analysis directly.

Format your final answers clearly with:
- Bullet points for lists
- **Bold** for emphasis
- Code blocks for technical details

Always base your answers on the data from your tools, not assumptions."""

        self.history: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

    def query(self, user_question: str, max_iterations: int = 5) -> str:
        self.history.append({"role": "user", "content": user_question})
        
        for iteration in range(max_iterations):
            response_text = self.llm.generate_response(messages=self.history)
            
            self.history.append({"role": "assistant", "content": response_text})
            
            action_match = re.search(r'ACTION:\s*(\w+)', response_text)
            params_match = re.search(r'PARAMETERS:\s*(\{.*?\})', response_text, re.DOTALL)
            
            if action_match:
                tool_name = action_match.group(1).strip()
                tool_params_raw = params_match.group(1).strip() if params_match else "{}"
                
                try:
                    tool_arguments = json.loads(tool_params_raw)
                except Exception:
                    tool_arguments = {}
                
                print(f"   [Tool Triggered]: Invoking '{tool_name}' with args {tool_arguments}...")
                observation = self.registry.execute(name=tool_name, arguments=tool_arguments)
                
                self.history.append({
                    "role": "user",
                    "content": f"OBSERVATION:\n{observation}"
                })
            else:
                return response_text
                
        return "Error: Agent maximum internal reasoning limit loops exhausted without finding terminal resolution."

    def reset_session(self):
        """Clears existing tracking instances back to original states."""
        self.history = [{"role": "system", "content": self.system_prompt}]