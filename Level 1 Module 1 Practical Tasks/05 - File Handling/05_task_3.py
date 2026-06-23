import json
from pathlib import Path

def analyze_api_logs(log_filepath, report_filepath):
    log_path = Path(log_filepath)
    report_path = Path(report_filepath)

    if not log_path.exists():
        print(f"Log source file '{log_path}' does not exist.")
        return

    total_calls = 0
    total_latency = 0.0
    error_count = 0
    model_counts = {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            log_entry = json.loads(line)
            total_calls += 1

            total_latency += log_entry.get("latency", 0.0)
            
            if log_entry.get("status") == "error":
                error_count += 1
                
            model = log_entry.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

    avg_latency = total_latency / total_calls if total_calls > 0 else 0
    error_rate = error_count / total_calls if total_calls > 0 else 0

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# API Performance Metrics Summary\n\n")
        f.write(f"- **Total API Calls Processed**: {total_calls}\n")
        f.write(f"- **Average Processing Latency**: {avg_latency:.2f} seconds\n")
        f.write(f"- **System Failures / Error Rate**: {error_rate * 100:.1f}%\n\n")
        
        f.write("## Usage Distribution per Model\n")
        for model_name, count in model_counts.items():
            f.write(f"- `{model_name}`: {count} calls\n")

    print(f"Analysis complete. Report written to: '{report_path}'")

if __name__ == "__main__":
    mock_logs = [
        '{"model": "gpt-4", "latency": 1.2, "status": "success"}',
        '{"model": "gpt-3.5", "latency": 0.4, "status": "success"}',
        '{"model": "gpt-4", "latency": 0.0, "status": "error"}',
        '{"model": "claude-3", "latency": 2.1, "status": "success"}'
    ]
    
    log_file = Path("api_calls.jsonl")
    log_file.write_text("\n".join(mock_logs))

    analyze_api_logs("api_calls.jsonl", "api_report.md")

    print(Path("api_report.md").read_text())

    log_file.unlink()
    Path("api_report.md").unlink()