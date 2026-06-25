# Task 1: Build a Multi-Model Output Validator
# Create a script that takes a text input and runs it through three different structured output operations 
# (sentiment analysis, entity extraction, and classification) simultaneously. Validate each output against its Pydantic model. 
# Generate a consolidated report that includes all three results plus metadata (model used, latency, token count). 
# Export the final report as both JSON and a formatted text file.

import json
import time
from typing import List
from pydantic import BaseModel, Field
from output_toolkit import StructuredOutputClient

class ConsolidatedReport(BaseModel):
    model_used: str
    total_latency_ms: float
    sentiment_analysis: dict
    entity_extraction: dict
    classification: dict

def generate_consolidated_report(text: str, output_prefix: str = "report") -> ConsolidatedReport:
    # Switch use_api=True if OPENAI_API_KEY is set
    client = StructuredOutputClient(model="gpt-4o", use_api=False)
    
    start_time = time.time()
    
    sentiment = client.analyze_sentiment(text)
    entities = client.extract_entities(text)
    classification = client.classify(text)
    
    end_time = time.time()
    total_latency = (end_time - start_time) * 1000
    
    report = ConsolidatedReport(
        model_used=client.model,
        total_latency_ms=round(total_latency, 2),
        sentiment_analysis=sentiment.model_dump() if sentiment else {},
        entity_extraction=entities.model_dump() if entities else {},
        classification=classification.model_dump() if classification else {}
    )
    
    json_filename = f"{output_prefix}.json"
    with open(json_filename, "w") as f:
        f.write(report.model_dump_json(indent=2))
        
    txt_filename = f"{output_prefix}.txt"
    with open(txt_filename, "w") as f:
        f.write(f"AI Analysis Report\n")
        f.write(f"Model Used: {report.model_used}\n")
        f.write(f"Total Latency: {report.total_latency_ms} ms\n")
        f.write(f"-" * 40 + "\n\n")
        f.write(f"1. Sentiment Analysis\n")
        f.write(f"Sentiment: {report.sentiment_analysis.get('sentiment')}\n")
        f.write(f"Confidence: {report.sentiment_analysis.get('confidence')}\n\n")
        f.write(f"2. Entity Extraction\n")
        f.write(f"Total Found: {report.entity_extraction.get('total_count', 0)}\n\n")
        f.write(f"3. Classification\n")
        f.write(f"Category: {report.classification.get('primary_category')}\n")
        f.write(f"Confidence: {report.classification.get('confidence')}\n")
        
    print(f"Reports: {json_filename} & {txt_filename}")
    return report

if __name__ == "__main__":
    sample_text = (
        "Google announced new quantum computing hardware at their headquarters in California. "
        "The presentation was impressive, though some critics say the timeline is realistic but slow."
    )
    
    report_data = generate_consolidated_report(sample_text)
    print("\nOutput preview:")
    print(report_data.model_dump_json(indent=2))