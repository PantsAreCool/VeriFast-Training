# Task 1: Build a Streaming Chat Interface
# Create a command-line chat application that streams responses token-by-token. 
# The interface should: (1) display a prompt character while waiting for user input, (2) stream the response with a typewriter effect, 
# (3) show a spinning indicator while waiting for the first token, (4) display streaming statistics after each response (TTFT, tokens/sec, total time). 
# Support conversation history so the model has context from previous turns.


import asyncio
import os
import sys
import time
from openai import AsyncOpenAI, APIError
from dotenv import load_dotenv

load_dotenv()
api_key=os.environ.get("OPENROUTER_API_KEY")
client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

async def show_spinner(stop_event: asyncio.Event):
    """Displays the spinning indicator while waiting for the first token."""
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    idx = 0
    sys.stdout.write(f"AI is thinking... ")
    while not stop_event.is_set():
        sys.stdout.write(f"\rAI is thinking... {spinner_frames[idx % len(spinner_frames)]}")
        sys.stdout.flush()
        idx += 1
        await asyncio.sleep(0.08)

    sys.stdout.flush()

async def stream_chat_turn(conversation_history: list) -> list:
    start_time = time.time()
    first_token_time = None
    token_count = 0
    accumulated_content = ""

    spinner_stop_event = asyncio.Event()
    spinner_task = asyncio.create_task(show_spinner(spinner_stop_event))

    try:
        stream = await client.chat.completions.create(
            model="mistralai/mistral-large",
            messages=conversation_history,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                
                if first_token_time is None:
                    first_token_time = time.time()
                    spinner_stop_event.set()
                    await spinner_task
                    sys.stdout.write(f"AI: ")

                sys.stdout.write(token)
                sys.stdout.flush()
                
                accumulated_content += token
                token_count += 1


        end_time = time.time()

        if first_token_time and token_count > 0:
            ttft_ms = (first_token_time - start_time) * 1000
            total_time_s = end_time - start_time
            tokens_per_sec = token_count / (end_time - first_token_time) if (end_time - first_token_time) > 0 else token_count

            print(f"\nStreaming Stats:")
            print(f"Time to First Token (TTFT): {ttft_ms:.1f}ms")
            print(f"Total Generation Time:      {total_time_s:.2f}s")
            print(f"Throughput:                 {tokens_per_sec:.1f} tokens/sec")
            print(f"Tokens Generated:           {token_count}\n" + "-"*50)

        conversation_history.append({"role": "assistant", "content": accumulated_content})

    except APIError as e:
        spinner_stop_event.set()
        await spinner_task
        print(f"\nAPI Error occurred: {e}")

    return conversation_history

async def main():
    history = [{"role": "system", "content": "You are a helpful, concise AI assistant."}]
    print("Type 'exit' or 'quit' to end the session.\n" + "="*50)

    while True:
        try:
            user_input = input(f"\nUser › ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                break

            history.append({"role": "user", "content": user_input})
            history = await stream_chat_turn(history)

        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    asyncio.run(main())