import asyncio
import random

async def response_token_streamer(prompt):
    response_text = f"This is a simulated streaming response answer for: '{prompt}'."
    tokens = response_text.split()
    
    for token in tokens:
        await asyncio.sleep(random.uniform(0.05, 0.2))
        yield token

async def consume_stream(prompt):
    print(f"Sending prompt: '{prompt}'")
    print("Streaming response output: ", end="", flush=True)

    async for token in response_token_streamer(prompt):
        print(token + " ", end="", flush=True)
        

if __name__ == "__main__":
    asyncio.run(consume_stream("Explain machine learning simply"))