import asyncio
import random
import time

async def simulate_api_call(prompt):
    delay = random.uniform(0.2, 0.8)
    await asyncio.sleep(delay)
    
    if random.random() < 0.15:
        raise RuntimeError("API Timeout")
        
    return f"Response to: {prompt}"

async def process_batch(prompts, batch_size):
    all_results = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        print(f"\nStarting Batch (Items {i} to {i + len(batch)})")
        
        start_time = time.time()

        tasks = []
        for prompt in batch:
            tasks.append(simulate_api_call(prompt))
        batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.time() - start_time

        successes = 0
        failures = 0
        for res in batch_responses:
            if isinstance(res, Exception):
                failures += 1
                all_results.append(None)
            else:
                successes += 1
                all_results.append(res)
                
        print(f"Finished in {elapsed:.2f}s | Success: {successes} | Failures: {failures}")
        
    return all_results

if __name__ == "__main__":
    sample_prompts = [f"Prompt #{x}" for x in range(1, 11)]
    
    results = asyncio.run(process_batch(sample_prompts, batch_size=3))
    
    for prompt, res in zip(sample_prompts, results):
        print(f" - {prompt} -> {res}")