import time
import random
from functools import wraps

def retry_backoff(max_attempts=3, base_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    
                    if attempt == max_attempts:
                        print("Max attempts reached. Raising exception.")
                        raise

                    sleep_time = delay + random.random()
                    print(f"Waiting {sleep_time:.2f} seconds before trying again...")
                    time.sleep(sleep_time)
                    delay = delay * 2
        return wrapper
    return decorator

@retry_backoff(max_attempts=3, base_delay=1)
def fetch_ai_data():
    if random.random() < 0.6:
        raise ConnectionError("API connection timeout")
    return "Successfully retrieved model response!"



if __name__ == "__main__":
    print("STARTING API CALL WITH RETRY:")
    try:
        result = fetch_ai_data()
        print(f"Result: {result}")
    except Exception:
        print("The API call permanently failed.")