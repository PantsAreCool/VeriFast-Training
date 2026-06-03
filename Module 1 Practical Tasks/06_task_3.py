import asyncio
import time

class RateLimitedClient:
    def __init__(self, max_per_second=5):
        self.delay_between_calls = 1.0 / max_per_second
        self.last_call_time = 0.0
        self.lock = asyncio.Lock()

    async def secure_call(self, request_id):
        async with self.lock:
            current_time = time.time()
            time_passed = current_time - self.last_call_time

            if time_passed < self.delay_between_calls:
                sleep_needed = self.delay_between_calls - time_passed
                await asyncio.sleep(sleep_needed)

            self.last_call_time = time.time()

        await asyncio.sleep(0.1)
        return f"Result {request_id} processed at timestamp {time.strftime('%H:%M:%S')}"

async def main():
    client = RateLimitedClient(max_per_second=5)
    
    print("Scheduling 20 API requests simultaneously.")
    start_time = time.time()
    
    tasks = []
    for i in range(1, 21):
        tasks.append(client.secure_call(i))

    results = await asyncio.gather(*tasks)
    
    total_duration = time.time() - start_time
    print("\nAPI CALL TIMESTAMPS OVERVIEW:")
    for row in results:
        print(f" -> {row}")
        
    print(f"\nFinished processing all 20 calls in: {total_duration:.2f} seconds.")

if __name__ == "__main__":
    asyncio.run(main())