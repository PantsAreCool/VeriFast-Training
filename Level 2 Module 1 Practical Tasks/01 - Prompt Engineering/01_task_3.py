# Task 3: Implement a Prompt Chaining Pipeline
# Create a prompt chaining system that takes a research topic and produces a final report through a 4-step chain: 
# (1) generate 10 research questions, (2) answer each question, (3) synthesize answers into sections, and (4) produce a final polished report. 
# Each step should use a different temperature setting appropriate for the task. 
# Log the intermediate outputs at each stage and measure the latency of the full pipeline. Include error handling for API failures at each chain step.

import os
import time
import logging
import openai
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

class ResearchPipeline:
    def __init__(self, model: str = "mistralai/mistral-large"):
        self.model = model
        api_key= os.environ.get("MISTRAL_API_KEY")

        self.client = openai.OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    def _run_step(self, system_prompt: str, user_prompt: str, temp: float) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temp,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip()

    def execute(self, topic: str) -> str:
        start_time = time.time()
        logger.info(f"Starting pipeline for topic: '{topic}'")

        # Step 1
        logger.info("Step 1/4: Generating research questions...")
        step1_system = "You are a research director. Generate exactly 10 diverse, critical research questions regarding the provided topic."
        questions = self._run_step(step1_system, f"Topic: {topic}", temp=0.8)
        logger.debug(f"Intermediate Output (Questions):\n{questions}\n")

        # Step 2
        logger.info("Step 2/4: Answering research questions...")
        step2_system = "You are an expert researcher. Provide accurate, data-backed, and concise answers to each of the research questions provided."
        answers = self._run_step(step2_system, f"Questions:\n{questions}", temp=0.2)
        logger.debug(f"Intermediate Output (Answers):\n{answers}\n")

        # Step 3
        logger.info("Step 3/4: Synthesizing raw answers into cohesive sections...")
        step3_system = "You are a technical editor. Organize raw questions and answers into logical, themed report sections with clear markdown headings."
        sections = self._run_step(step3_system, f"Raw Research Material:\n{answers}", temp=0.4)
        logger.debug(f"Intermediate Output (Sections):\n{sections}\n")

        # Step 4
        logger.info("Step 4/4: Polishing final report...")
        step4_system = "You are a senior copyeditor. Elevate the tone, ensure fluid transitions, and format the sections into a highly professional executive report."
        final_report = self._run_step(step4_system, f"Draft Layout:\n{sections}", temp=0.5)

        total_latency = (time.time() - start_time) * 1000
        logger.info(f"Pipeline complete! Total Latency: {total_latency:.2f}ms")
        
        return final_report

# --- Execution Demo ---
if __name__ == "__main__":
    # Ensure API key is configured before testing: export MISTRAL_API_KEY="your-key"
   # if not os.environ.get("MISTRAL_API_KEY"):
   #     print("Please set your MISTRAL_API_KEY environment variable to test the script.")
   # else:
    pipeline = ResearchPipeline()
    report = pipeline.execute("The impact of quantum computing on modern RSA cryptography paradigms")
    print("\n" + "="*40 + " FINAL REPORT " + "="*40)
    print(report)