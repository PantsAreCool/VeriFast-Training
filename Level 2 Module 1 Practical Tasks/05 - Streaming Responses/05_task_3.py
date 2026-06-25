# Task 3: Create a Streaming Comparison Tool
# Build a tool that sends the same prompt to both OpenAI and Anthropic simultaneously and streams both responses side-by-side in the terminal 
# (using two columns). Measure and display: time to first token for each provider, total generation time, tokens per second, and total token count. 
# Run 5 different prompts and generate a comparison report. Include error handling for cases where one provider fails or is much slower.


import asyncio
import time

async def mock_openai_stream(prompt):
    tokens = f"OpenAI response for '{prompt}': The quick brown fox jumps over the lazy dog.".split()
    await asyncio.sleep(0.2)
    for t in tokens:
        yield t + " "
        await asyncio.sleep(0.04)

async def mock_anthropic_stream(prompt):
    tokens = f"Claude response for '{prompt}': The quick brown fox jumps over the lazy dog.".split()
    await asyncio.sleep(0.4)
    for t in tokens:
        yield t + " "
        await asyncio.sleep(0.06)

async def run_benchmark(prompt_idx, prompt):
    print(f"\n[Prompt {prompt_idx}]: '{prompt}'")

    metrics = {
        "openai": {"ttft": None, "start": time.time(), "tokens": 0, "text": ""},
        "anthropic": {"ttft": None, "start": time.time(), "tokens": 0, "text": ""}
    }


    oi_iter = aiter(mock_openai_stream(prompt))
    an_iter = aiter(mock_anthropic_stream(prompt))
    
    oi_done, an_done = False, False
    col_width = 40

    while not (oi_done and an_done):
        oi_chunk, an_chunk = "", ""
        
        if not oi_done:
            try:
                oi_chunk = await anext(oi_iter)
                metrics["openai"]["tokens"] += 1
                if metrics["openai"]["ttft"] is None:
                    metrics["openai"]["ttft"] = (time.time() - metrics["openai"]["start"]) * 1000
                metrics["openai"]["text"] += oi_chunk
            except StopAsyncIteration:
                oi_done = True

        if not an_done:
            try:
                an_chunk = await anext(an_iter)
                metrics["anthropic"]["tokens"] += 1
                if metrics["anthropic"]["ttft"] is None:
                    metrics["anthropic"]["ttft"] = (time.time() - metrics["anthropic"]["start"]) * 1000
                metrics["anthropic"]["text"] += an_chunk
            except StopAsyncIteration:
                an_done = True


       # print(f"{oi_chunk[:col_width]:<{col_width}} | {an_chunk[:col_width]:<{col_width}}")
       # await asyncio.sleep(0.01)


    for p in ["openai", "anthropic"]:
        metrics[p]["total_time"] = time.time() - metrics[p]["start"]
        metrics[p]["tps"] = metrics[p]["tokens"] / metrics[p]["total_time"]

    print("-" * 80)
    print(f"TTFT: {metrics['openai']['ttft']:.0f}ms{r'':<32} | TTFT: {metrics['anthropic']['ttft']:.0f}ms")
    print(f"TPS:  {metrics['openai']['tps']:.1f} tok/s{r'':<29} | TPS:  {metrics['anthropic']['tps']:.1f} tok/s")
    
    return metrics


async def main():
    prompts = [
        "How is Apple's stock doing? Check AAPL.",
        "What is the price of Google right now?",
        "Can you convert $150 to EUR for me?",
        "How many Euros do I get for 50 dollars?",
        "I forgot my password, how do I reset it?",
    ]

    reports = []
    for idx, prompt in enumerate(prompts, 1):
        result = await run_benchmark(idx, prompt)
        reports.append((prompt, result))


    print("\n" + "="*40 + " FINAL COMPARISON REPORT " + "="*40)
    print(f"{'Prompt Preview':<30} | {'OpenAI (TTFT / TPS)':<22} | {'Anthropic (TTFT / TPS)':<22}")
    print("-" * 80)
    for p, data in reports:
        oi_str = f"{data['openai']['ttft']:.0f}ms / {data['openai']['tps']:.1f} tps"
        an_str = f"{data['anthropic']['ttft']:.0f}ms / {data['anthropic']['tps']:.1f} tps"
        print(f"{p[:28]+'...':<30} | {oi_str:<22} | {an_str:<22}")

if __name__ == "__main__":
    asyncio.run(main())