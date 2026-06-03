import time
import traceback

def log_tracer(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        arg_str = f"args={args} kwargs={kwargs}"
        log_msg = f"[{timestamp}] start: Running {func_name} with {arg_str}\n"

        with open("app_history.log", "a") as f:
            f.write(log_msg)

        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            success_msg = f"[{timestamp}] success: {func_name} finished in {elapsed:.3f}s. Result: {result}\n"
            with open("app_history.log", "a") as f:
                f.write(success_msg)
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            tb_details = traceback.format_exc()
            
            error_msg = f"[{timestamp}] exception in {func_name} after {elapsed:.3f}s: {e}\nTraceback:\n{tb_details}\n"
            with open("app_history.log", "a") as f:
                f.write(error_msg)
            raise e
            
    return wrapper

@log_tracer
def clean_text(text):
    return text.strip().lower()

@log_tracer
def calculate_cost(tokens, price_per_token):
    return tokens * price_per_token

@log_tracer
def fetch_database_record(record_id):
    if record_id < 0:
        raise ValueError("Invalid Database Record ID requested.")
    time.sleep(0.1)
    return {"id": record_id, "status": "active"}

if __name__ == "__main__":
    
    clean_text("AI Systems")
    calculate_cost(500, 0.0002)
    fetch_database_record(42)

    try:
        fetch_database_record(-5)
    except ValueError:
        print("Caught intended test error")

    print("\nLog file created at: app_history.log.")