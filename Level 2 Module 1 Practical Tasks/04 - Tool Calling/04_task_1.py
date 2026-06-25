# Task 1: Build a Multi-Provider Tool Router
# Create a system that routes user queries to either OpenAI or Anthropic based on query characteristics. 
# Define at least 4 tools that work identically across both providers. 
# The router should log which provider was selected, why, and compare the quality of tool selection between providers. 
# Run 10 test queries through both providers and create a comparison report showing differences in tool selection, latency, and response quality.


# Note: Unsure if this actually works; don't have a Claude API Key, just one Openrouter one.

import os
import json
import time
import anthropic
import openai


#from typing import Dict, List, Any

def get_stock(ticker: str) -> str:
    stocks = {"AAPL": 175.50, "GOOG": 150.25, "MSFT": 420.10}
    return json.dumps({"ticker": ticker, "price": stocks.get(ticker.upper(), "Unknown stock")})

def convert_currency(amount: float, to_currency: str) -> str:
    rates = {"EUR": 0.92, "GBP": 0.79, "JPY": 155.0}
    rate = rates.get(to_currency.upper(), 1.0)
    return json.dumps({"original_amount": amount, "converted": round(amount * rate, 2), "currency": to_currency})

def search_faqs(topic: str) -> str:
    faqs = {"password": "Reset via the login screen link.", "refund": "Allowed within 14 days."}
    return json.dumps({"topic": topic, "answer": faqs.get(topic.lower(), "Contact human support.")})

def get_sys_status() -> str:
    return json.dumps({"api_gateway": "operational", "database": "nominal", "latency": "14ms"})

TOOL_MAP = {
    "get_stock": get_stock,
    "convert_currency": convert_currency,
    "search_faqs": search_faqs,
    "get_sys_status": get_sys_status
}

OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "get_stock", "description": "Get current stock price", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Convert USD to another currency", "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "to_currency": {"type": "string"}}, "required": ["amount", "to_currency"]}}},
    {"type": "function", "function": {"name": "search_faqs", "description": "Search customer FAQs", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}},
    {"type": "function", "function": {"name": "get_sys_status", "description": "Get current system infrastructure status", "parameters": {"type": "object", "properties": {}}}}
]

ANTHROPIC_TOOLS = [
    {"name": "get_stock", "description": "Get current stock price", "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
    {"name": "convert_currency", "description": "Convert USD to another currency", "input_schema": {"type": "object", "properties": {"amount": {"type": "number"}, "to_currency": {"type": "string"}}, "required": ["amount", "to_currency"]}},
    {"name": "search_faqs", "description": "Search customer FAQs", "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
    {"name": "get_sys_status", "description": "Get current system infrastructure status", "input_schema": {"type": "object", "properties": {}}}
]


class ToolRouter:
    def route_query(self, query: str) -> str:
        """Determines provider based on keyword complexity or specific domains."""
        q = query.lower()
        if "status" in q or "system" in q:
            return "anthropic"
        return "openai"

    def run_openai(self, query: str) -> tuple:
        client = openai.OpenAI()
        start = time.time()
        res = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": query}], tools=OPENAI_TOOLS, tool_choice="auto"
        )
        latency = (time.time() - start) * 1000
        msg = res.choices.message
        tool_called = msg.tool_calls.function.name if msg.tool_calls else None
        return tool_called, latency

    def run_anthropic(self, query: str) -> tuple:
        client = anthropic.Anthropic()
        start = time.time()
        res = client.messages.create(
            model="claude-3-haiku-20240307", max_tokens=1000, tools=ANTHROPIC_TOOLS, messages=[{"role": "user", "content": query}]
        )
        latency = (time.time() - start) * 1000
        tool_blocks = [b for b in res.content if b.type == "tool_use"]
        tool_called = tool_blocks.name if tool_blocks else None
        return tool_called, latency


if __name__ == "__main__":
    has_keys = os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_API_KEY")
    
    test_queries = [
        "How is Apple's stock doing? Check AAPL.",
        "What is the price of Google right now?",
        "Can you convert $150 to EUR for me?",
        "How many Euros do I get for 50 dollars?",
        "I forgot my password, how do I reset it?",
        "How is Apple's stock doing? Check AAPL.",
        "What is the price of Google right now?",
        "Can you convert $150 to EUR for me?",
        "How many Euros do I get for 50 dollars?",
        "I forgot my password, how do I reset it?",
    ]

    router = ToolRouter()
    print(f"| Query | Chosen Provider | Reason |")
    print(f"-" * 40)
    
    for idx, q in enumerate(test_queries):
        chosen = router.route_query(q)
        reason = "System/Infra heavy query" if chosen == "anthropic" else "General business query"
            
        print(f"| {idx+1}. {q[:30]} | {chosen} | {reason} |")