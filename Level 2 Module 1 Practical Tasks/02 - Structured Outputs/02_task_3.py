# Task 3: Create a Batch Processing Pipeline with Error Recovery
# Build a batch processing pipeline that reads a CSV file of 50+ text entries, processes each through structured sentiment analysis, and saves results. 
# Implement: (1) progress tracking with a progress bar, (2) per-item error handling that logs failures without stopping, 
# (3) rate limiting to stay within API limits, (4) automatic retry with exponential backoff for transient failures, 
# and (5) a final summary report with success/failure counts and average confidence scores.

import csv
import json
import os
import time
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from tqdm import tqdm
import openai
import random


class MockClient:
    def analyze_sentiment(self, text: str) -> Any:
        class MockResult:
            sentiment = "positive" if len(text) % 2 == 0 else "negative"
            confidence = round(random.uniform(0.75, 0.99), 2)
        return MockResult()


def run_sentiment_batch(input_csv: str, output_csv: str, rpm_limit: int = 60):
    client = MockClient()
    delay_between_requests = 60.0 / rpm_limit
    
    with open(input_csv, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    total_items = len(rows)
    results = []
    errors_log = []
    success_count = 0
    total_confidence = 0.0

    print(f"processing batch of {total_items} items\n")

    for idx, row in enumerate(rows, 1):
        text_to_analyze = row.get("text", "")
        item_id = row.get("id", idx)
        
        max_retries = 3
        backoff_delay = 1.0
        success = False
        res = None

        for i in range(max_retries):
            try:
                res = client.analyze_sentiment(text_to_analyze)
                success = True
                break
            except Exception as e:
                if i == max_retries - 1:
                    errors_log.append({"id": item_id, "error": str(e)})
                else:
                    time.sleep(backoff_delay)
                    backoff_delay *= 2

        if success and res:
            success_count += 1
            total_confidence += res.confidence
            results.append({
                "id": item_id,
                "text": text_to_analyze,
                "sentiment": res.sentiment,
                "confidence": res.confidence,
                "status": "SUCCESS"
            })
        else:
            results.append({
                "id": item_id,
                "text": text_to_analyze,
                "sentiment": "N/A",
                "confidence": 0.0,
                "status": "FAILED"
            })


    if results:
        with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    failure_count = total_items - success_count
    avg_conf = (total_confidence / success_count) if success_count > 0 else 0.0
    

    print("Report Summary")
    print("="*30)
    print(f"Rows Processed: {total_items}")
    print(f"Success Count: {success_count}")
    print(f"Failure Count: {failure_count}")
    print(f"Avg Confidence Score: {avg_conf:.2f}")
    print(f"Error logs recorded: {len(errors_log)}")


if __name__ == "__main__":
    input_file = "batch_input.csv"
    output_file = "batch_results.csv"
    
    with open(input_file, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text"])
        for i in range(1, 53):
            writer.writerow([f"ID_{i}", f"Sample phrase number {i}."])


    run_sentiment_batch(input_file, output_file, rpm_limit=120)