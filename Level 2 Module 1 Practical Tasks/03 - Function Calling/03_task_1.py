# Task 1: Build a Customer Service Function Suite
# Create a function calling system for a customer service chatbot with at least 5 functions: 
# lookup_customer, check_order_status, process_refund, update_shipping_address, and escalate_to_human. 
# Define complete schemas for each, implement the functions with simulated data, and 
# build an interactive loop where a user can type questions and the system automatically calls the right functions. 
# Include conversation history so the model can handle follow-up questions.

import json
import openai
import os
from dotenv import load_dotenv

load_dotenv()


tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Find customer ID by email.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check status of an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "process_refund",
            "description": "Process a refund for an order.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["order_id", "reason"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_shipping_address",
            "description": "Update customer shipping address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "new_address": {"type": "string"}
                },
                "required": ["customer_id", "new_address"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate the issue to a human agent.",
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
                "additionalProperties": False
            }
        }
    }
]


def lookup_customer(email):
    return json.dumps({"customer_id": "CUST-99", "name": "Panth", "email": email})

def check_order_status(order_id):
    return json.dumps({"order_id": order_id, "status": "Shipped", "delivery_date": "Tomorrow"})

def process_refund(order_id, reason):
    return json.dumps({"order_id": order_id, "status": "Refund Approved", "reason": reason})

def update_shipping_address(customer_id, new_address):
    return json.dumps({"customer_id": customer_id, "status": "Address Updated Successfully"})

def escalate_to_human(reason):
    return json.dumps({"status": "Escalated", "ticket_id": "TK-099", "reason": reason})

available_functions = {
    "lookup_customer": lookup_customer,
    "check_order_status": check_order_status,
    "process_refund": process_refund,
    "update_shipping_address": update_shipping_address,
    "escalate_to_human": escalate_to_human,
}

# Expects OPENROUTER_API_KEY environment variable to be set
api_key= os.environ.get("OPENROUTER_API_KEY")
client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

def run_customer_agent(user_prompt, conversation_history=[]):
    conversation_history.append({"role": "user", "content": user_prompt})
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        tools=tools,
        tool_choice="auto"
    )
    
    assistant_message = response.choices[0].message
    conversation_history.append(assistant_message)
    
    if assistant_message.tool_calls:
        for tool_call in assistant_message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            
            function_to_call = available_functions[func_name]
            tool_output = function_to_call(**func_args)
            
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_output
            })
        
        final_response = client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=conversation_history
        )
        return final_response.choices[0].message.content, conversation_history
        
    return assistant_message.content, conversation_history

history = []

reply, history = run_customer_agent("Hi, where is my order #11111?", history)
print(f"Bot: {reply}\n")

reply, history = run_customer_agent("Actually, I want a refund because it took too long.", history)
print(f"Bot: {reply}")