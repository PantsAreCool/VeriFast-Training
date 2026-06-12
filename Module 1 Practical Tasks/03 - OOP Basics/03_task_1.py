class Message:
    """A chat message with role and content."""

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def __repr__(self):
        return f"Message({self.role}: {self.content[:40]}...)"

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """Manages conversation history with a sliding window."""

    def __init__(self, max_messages=20):
        self._messages = []
        self.max_messages = max_messages

    def add(self, role, content):
        msg = Message(role, content)
        self._messages.append(msg)
        if len(self._messages) > self.max_messages:
            self._messages = self._messages[-self.max_messages:]

    def get_messages(self):
        return [m.to_dict() for m in self._messages]

    def __len__(self):
        return len(self._messages)

    def clear(self):
        self._messages.clear()


class Agent:
    """An AI agent with memory and tool access."""

    def __init__(self, name, system_prompt="", memory=None):
        self.name = name
        self.system_prompt = system_prompt
        self.memory = memory or ConversationMemory()
        self.tools = {}

    def add_tool(self, name, func, description=""):
        self.tools[name] = {"func": func, "description": description}

    def respond(self, user_input):
        """Process user input and generate response."""
        self.memory.add("user", user_input)
        # In a real app, this calls an LLM
        response = f"[{self.name}] Acknowledged: {user_input}"
        self.memory.add("assistant", response)
        return response

    def __repr__(self):
        tool_names = list(self.tools.keys())
        return f"Agent(name='{self.name}', tools={tool_names}, memory={len(self.memory)} msgs)"


class ToolCallingAgent(Agent):
    def respond(self, user_input):
        self.memory.add("user", user_input)
        cleaned_input = user_input.lower()
        tool_used = None
        tool_output = ""

        for tool_name in self.tools:
            if tool_name in cleaned_input:
                tool_used = tool_name
                tool_output = self.tools[tool_name]["func"](user_input)
        
        if tool_used:
            response = f"[{self.name}] Used tool '{tool_used}'. Result: {tool_output}"
        else:
            response = f"[{self.name}] No specific tool matched. Generic response to: {user_input}"
            
        self.memory.add("assistant", response)
        return response


if __name__ == "__main__":
    agent = ToolCallingAgent(
        name="ResearchBot",
        system_prompt="You are a helpful research assistant.",
    )

    agent.add_tool("calculate", lambda text: "4", "Performs arithmetic tasks")
    agent.add_tool("search", lambda text: "Found 3 results for AI", "Queries the web")
    
    print(agent.respond("calculate 2+2"))
    print(agent.respond("search for machine learning tutorials"))
    print(agent.respond("Nothing should happen here."))
