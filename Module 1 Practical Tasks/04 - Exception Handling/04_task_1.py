import time
import random

class CircuitBreaker:
    def __init__(self, max_failures=5, cooldown=10):
        self.max_failures = max_failures
        self.cooldown = cooldown
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0

    def check_status(self):
        if self.state == "OPEN":
            current_time = time.time()
            if current_time - self.last_failure_time > self.cooldown:
                self.state = "HALF-OPEN"
                print("Circuit is HALF-OPEN. Testing next request...")
                return True
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.max_failures:
            self.state = "OPEN"
            print(f"Circuit tripped to OPEN! Blocking requests for {self.cooldown}s.")

class CircuitOpenError(Exception):
    pass

class SafeLLMClient:
    def __init__(self):
        self.breaker = CircuitBreaker(max_failures=3, cooldown=4)

    def call_api(self, prompt):
        if not self.breaker.check_status():
            raise CircuitOpenError("Call rejected: Circuit is currently OPEN.")

        try:
            if random.random() < 0.7:
                raise RuntimeError("API Timeout Error")

            self.breaker.record_success()
            return f"Response to: {prompt}"
            
        except Exception as e:
            print(f"Request failed: {e}")
            self.breaker.record_failure()
            raise e

if __name__ == "__main__":
    client = SafeLLMClient()
    
    for i in range(1, 8):
        print(f"\nRequest #{i}:")
        try:
            res = client.call_api("Hello AI")
            print(f"Success: {res}")
        except Exception as err:
            print(f"Caught Error in App: {err}")
        time.sleep(1)