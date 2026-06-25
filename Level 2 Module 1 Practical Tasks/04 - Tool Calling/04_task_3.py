# Task 3: Create a Conversational Tool-Using Agent
# Build an interactive agent that maintains conversation history and can use tools across multiple turns. 
# The agent should handle: (1) referring to previous tool results ("check that city's weather again"), 
# (2) combining information from current and previous queries, and (3) asking clarifying questions when tool parameters are ambiguous. 
# Implement a simple REPL interface and log the complete conversation including tool calls.

import json

def get_weather(city: str) -> str:
    weather_data = {"tokyo": "22°C, Partly Cloudy", "london": "14°C, Rainy", "paris": "16°C, Clear"}
    return json.dumps({"city": city, "result": weather_data.get(city.lower(), "18°C, Clear")})

class ConversationalAgent:
    def __init__(self):
        self.history = []
        self.last_city = None

    def handle_message(self, user_input: str) -> str:
        text = user_input.lower().strip()
        self.history.append(f"User: {user_input}")

        if "weather" in text or "check again" in text:
            extracted_city = None
            for city in ["tokyo", "london", "paris"]:
                if city in text:
                    extracted_city = city
            
            if not extracted_city and ("that city" in text or "it" in text or "again" in text):
                extracted_city = self.last_city

            if not extracted_city:
                assistant_reply = "Which city would you like to check the weather for? (Tokyo, London, or Paris)"
                self.history.append(f"Agent: {assistant_reply}")
                return assistant_reply

            self.last_city = extracted_city
            tool_output = get_weather(extracted_city)
            self.history.append(f"Tool Call (get_weather): {tool_output}")
            
            parsed = json.loads(tool_output)
            assistant_reply = f"The weather in {parsed['city'].title()} is currently {parsed['result']}."
            self.history.append(f"Agent: {assistant_reply}")
            return assistant_reply

        assistant_reply = "I can help you check the weather. Try asking 'What's the weather in London?'"
        self.history.append(f"Agent: {assistant_reply}")
        return assistant_reply

if __name__ == "__main__":
    agent = ConversationalAgent()
    print("Type 'exit' to quit, or 'history' to view the complete log.\n")

    while True:
        try:
            user_msg = input("You: ")
            if user_msg.lower() == "exit":
                break
            if user_msg.lower() == "history":
                print("\nComplete Conversation & Tool Logs:")
                print("\n".join(agent.history))
                print("-" * 40 + "\n")
                continue

            response = agent.handle_message(user_msg)
            print(f"Agent: {response}\n")
        except (KeyboardInterrupt, EOFError):
            break