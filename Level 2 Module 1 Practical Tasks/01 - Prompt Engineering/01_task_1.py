# Task 1: Build a Multi-Strategy Prompt Comparator
# Create a Python script that takes a user question and runs it through at least 4 different prompting strategies 
# (zero-shot, few-shot with 3 examples, chain-of-thought, and persona-based with a custom persona). 
# Display all results side-by-side in a formatted table showing the strategy name, response length, estimated cost, and a quality score (1-5) 
# that you manually assign. Save the results to a JSON file for later analysis.

import os
import json
import time
from typing import List, Dict, Any
import openai
import tiktoken
from tabulate import tabulate
from dotenv import load_dotenv


load_dotenv()

COST_PER_1M_INPUT = 2.50
COST_PER_1M_OUTPUT = 10.00

class PromptComparator:
    """
    Executes a user query against 4 distinct prompt engineering strategies,
    calculates exact costs, captures user quality feedback, and saves metrics to JSON.
    """
    def __init__(self, model: str = "mistralai/mistral-large"):
        self.model = model
        api_key= os.environ.get("MISTRAL_API_KEY")

        self.client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

        try:
            self.encoder = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculates the total cost of an API call."""
        input_cost = (prompt_tokens / 1_000_000) * COST_PER_1M_INPUT
        output_cost = (completion_tokens / 1_000_000) * COST_PER_1M_OUTPUT
        return input_cost + output_cost

    def _execute_strategy(self, strategy_name: str, messages: List[Dict[str, str]], temperature: float) -> Dict[str, Any]:
        print(f"Running strategy: {strategy_name}...")
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        
        latency = (time.time() - start_time) * 1000
        usage = response.usage
        
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost = self._calculate_cost(prompt_tokens, completion_tokens)
        content = response.choices[0].message.content

        return {
            "strategy": strategy_name,
            "response": content,
            "response_length_chars": len(content),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost,
            "latency_ms": latency
        }

    def run_comparison(self, question: str) -> List[Dict[str, Any]]:
        results = []

        # 1. Zero-Shot Strategy
        zero_shot_msgs = [
            {"role": "system", "content": "You are a helpful assistant. Give clear, direct answers."},
            {"role": "user", "content": question}
        ]
        results.append(self._execute_strategy("Zero-Shot", zero_shot_msgs, temperature=0.0))

        # 2. Few-Shot Strategy
        few_shot_msgs = [
            {"role": "system", "content": "You provide structured explanations mapping definitions, core mechanics, and key use cases."},
            # Example 1
            {"role": "user", "content": "Explain what a Database Index is."},
            {"role": "assistant", "content": "Definition: A data structure that improves data retrieval speeds.\nMechanic: It builds a B-Tree lookup pointing to rows.\nUse Case: High-read query optimization."},
            # Example 2
            {"role": "user", "content": "Explain what CI/CD is."},
            {"role": "assistant", "content": "Definition: Automated software delivery pipelines.\nMechanic: Builds code, runs test suites, and pushes binaries to production on every commit.\nUse Case: Rapid feature deployment pipelines."},
            # Example 3
            {"role": "user", "content": "Explain what a REST API is."},
            {"role": "assistant", "content": "Definition: An architectural pattern using standard stateless HTTP verbs.\nMechanic: Exposes endpoints manipulating resource JSON strings.\nUse Case: Decoupled web client-to-server communications."},
            # Target
            {"role": "user", "content": question}
        ]
        results.append(self._execute_strategy("Few-Shot (3 Examples)", few_shot_msgs, temperature=0.0))

        # 3. Chain-of-Thought Strategy
        cot_msgs = [
            {
                "role": "system", 
                "content": "You are an analytical reasoning agent. Think through problems step-by-step. Break down your reasoning explicitly before declaring a conclusion."
            },
            {"role": "user", "content": f"{question}\n\nDeconstruct this topic logically step-by-step."}
        ]
        results.append(self._execute_strategy("Chain-of-Thought", cot_msgs, temperature=0.0))

        # 4. Persona-Based Strategy
        persona_msgs = [
            {
                "role": "system", 
                "content": "You are a distinguished Principal Systems Engineer and industry visionary. Explain topics using rigorous, technically precise architecture jargon, highlighting underlying edge cases and long-term implications."
            },
            {"role": "user", "content": question}
        ]
        results.append(self._execute_strategy("Persona-Based", persona_msgs, temperature=0.7))

        return results

def main():
    user_query = input("Enter your evaluation prompt/question: ") or "Explain how horizontal scaling differs from vertical scaling."
    
    comparator = PromptComparator()
    raw_results = comparator.run_comparison(user_query)
    
    final_results = []
    for res in raw_results:
        print(f"\n[Strategy: {res['strategy']}]")
        print(res['response'])
        print("-" * 40)
        
        while True:
            try:
                score = int(input(f"Assign Quality Score (1-5) for '{res['strategy']}': "))
                if 1 <= score <= 5:
                    res["quality_score"] = score
                    break
                print("Invalid range. Enter an integer between 1 and 5.")
            except ValueError:
                print("Please enter a valid integer.")
        
        final_results.append(res)

    # Compile Summary Data Table View
    table_headers = ["Strategy", "Length (chars)", "Tokens (In/Out)", "Cost (USD)", "Latency (ms)", "Quality Score"]
    table_rows = [
        [
            r["strategy"],
            r["response_length_chars"],
            f"{r['prompt_tokens']} / {r['completion_tokens']}",
            f"${r['cost_usd']:.6f}",
            f"{r['latency_ms']:.1f}",
            f"{r['quality_score']}/5"
        ]
        for r in final_results
    ]
    
    print("\n\n" + "="*80)
    print("Performance Matrix Summary:")
    print("="*80)
    print(tabulate(table_rows, headers=table_headers, tablefmt="grid"))

    output_filename = f"prompt_analysis_{int(time.time())}.json"
    log_data = {
        "metadata": {
            "evaluated_query": user_query,
            "model": comparator.model
        },
        "results": final_results
    }
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=4)
        print(f"\n[Metrics Saved Successfully] Log details written to: {output_filename}\n")

if __name__ == "__main__":
    main()