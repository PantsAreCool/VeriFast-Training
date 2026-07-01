# Task 2: Implement a Function Calling Router
# Build a router that analyzes incoming user queries and determines which function(s) to call without sending the query to the LLM first. 
# Train or hard-code a classification system using keyword patterns and intent matching. 
# Compare the router's function selection accuracy against the LLM's function calling decisions for 20 test queries. 
# Create a confusion matrix showing agreement and disagreement between the two approaches.



import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()

def keyword_router(query: str) -> str:
    query = query.lower()
    if any(w in query for w in ["weather", "forecast", "temperature", "rain"]):
        return "get_weather"
    elif any(w in query for w in ["database", "search", "find", "lookup", "records"]):
        return "search_database"
    elif any(w in query for w in ["calculate", "math", "compute", "plus", "%", "percent"]):
        return "calculate"
    return "text_response"


def get_llm_decision(query: str, tools: list) -> str:
    # Expects OPENROUTER_API_KEY environment variable to be set
    api_key= os.environ.get("OPENROUTER_API_KEY")
    client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(
        model="mistralai/mistral-large",
        messages=[{"role": "user", "content": query}],
        tools=tools,
        tool_choice="auto"
    )
    msg = response.choices[0].message
    if msg.tool_calls:
        return msg.tool_calls[0].function.name
    return "text_response"



test_queries = [
    "What's the weather in Tokyo?", "Is it raining in London?", "Current temperature in Paris", "Do I need an umbrella today?", "Forecast for New York",

    "Search database for customer 45", "Find all laptop records", "Lookup order status in the system", "Find files updated today", 
    "Search for product ID 101",

    "Calculate 15% of 250", "What is 45 times 12?", "Compute the square root of 144", "What is 100 plus 55?", "How much is 20 percent off 90?",

    "Tell me a joke.", "Who wrote Romeo and Juliet?", "Explain quantum physics simply.", "Hi, how are you?", "What is the capital of France?"
]

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specified location. Returns temperature, conditions, humidity, and wind speed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo', 'London', 'New York'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit. Defaults to celsius."
                    }
                },
                "required": ["location"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database for records matching the given criteria.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "Table to search: 'customers', 'products', or 'orders'"
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query or filter criteria"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results. Default 10."
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date", "name"],
                        "description": "Sort order. Default relevance."
                    }
                },
                "required": ["table", "query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation. Supports arithmetic and percentages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression, e.g. '15% of 250', '100 * 1.08'"
                    },
                    "precision": {
                        "type": "integer",
                        "description": "Decimal places. Default 2."
                    }
                },
                "required": ["expression"],
                "additionalProperties": False
            }
        }
    }
]

results = []
categories = ["get_weather", "search_database", "calculate", "text_response"]
matrix = {actual: {pred: 0 for pred in categories} for actual in categories}

for query in test_queries:
    router_choice = keyword_router(query)
    llm_choice = get_llm_decision(query, TOOL_DEFINITIONS)

    matrix[llm_choice][router_choice] += 1
    results.append({"query": query, "Router": router_choice, "LLM": llm_choice})

# Confusion Matrix
print(f"{'LLM v / Router >':<18} | {'weather':<12} | {'database':<12} | {'calculate':<12} | {'text_only':<12}")
print("-" * 80)
for llm_cat in categories:
    print(f"{llm_cat:<18} | " + " | ".join(f"{matrix[llm_cat][r_cat]:<12}" for r_cat in categories))

