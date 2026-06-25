"""
Structured Output Toolkit: Pydantic models for common AI outputs
with guaranteed valid JSON using OpenAI's structured output features.

Works with real API calls if OPENAI_API_KEY is set, otherwise uses simulated responses.
"""

import os
import json
import re
import time
from typing import List, Optional, Type, TypeVar
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from dataclasses import dataclass

T = TypeVar("T", bound=BaseModel)


# ---------- Pydantic Models for Common AI Tasks ----------

class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SentimentAnalysisResult(BaseModel):
    """Structured result for sentiment analysis."""
    text_snippet: str = Field(description="First 100 chars of analyzed text")
    sentiment: SentimentType = Field(description="Overall sentiment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    positive_aspects: List[str] = Field(default_factory=list, description="Positive things mentioned")
    negative_aspects: List[str] = Field(default_factory=list, description="Negative things mentioned")
    reasoning: str = Field(description="Why this sentiment was chosen")


class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    PRODUCT = "product"
    EVENT = "event"
    TECHNOLOGY = "technology"


class ExtractedEntity(BaseModel):
    """A single extracted entity."""
    text: str = Field(description="Entity text as it appears")
    type: EntityType = Field(description="Entity type")
    start_position: Optional[int] = Field(default=None, description="Character offset in original text")
    confidence: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = Field(default=None, description="Brief description of the entity")


class EntityExtractionResult(BaseModel):
    """Structured result for entity extraction."""
    entities: List[ExtractedEntity] = Field(description="All extracted entities")
    total_count: int = Field(ge=0, description="Total entities found")
    entity_types_found: List[str] = Field(description="Unique entity types present")


class StructuredSummary(BaseModel):
    """Structured result for text summarization."""
    title: str = Field(description="Concise title for the content")
    one_line_summary: str = Field(description="One sentence capturing the essence")
    key_points: List[str] = Field(min_length=2, max_length=7, description="Key points as bullet items")
    topics: List[str] = Field(description="Main topics or themes")
    audience: str = Field(default="general", description="Target audience level")
    sentiment: SentimentType = Field(default=SentimentType.NEUTRAL, description="Overall tone")


class ClassificationResult(BaseModel):
    """Structured result for text classification."""
    primary_category: str = Field(description="Main category")
    subcategories: List[str] = Field(default_factory=list, description="Sub-categories")
    confidence: float = Field(ge=0.0, le=1.0)
    alternative_categories: List[str] = Field(default_factory=list, description="Other possible categories")
    reasoning: str = Field(description="Why this classification was chosen")


# ---------- Simulated LLM for Offline Use ----------

class SimulatedStructuredLLM:
    """Simulates structured LLM outputs for offline demonstrations."""

    @staticmethod
    def get_sentiment(text: str) -> dict:
        text_lower = text.lower()
        pos_words = ["great", "amazing", "love", "excellent", "wonderful", "best"]
        neg_words = ["terrible", "hate", "worst", "awful", "bad", "horrible"]

        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)

        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        elif pos_count > 0 and neg_count > 0:
            sentiment = "mixed"
        else:
            sentiment = "neutral"

        return {
            "text_snippet": text[:100],
            "sentiment": sentiment,
            "confidence": 0.85,
            "positive_aspects": ["quality"] if pos_count > 0 else [],
            "negative_aspects": ["price"] if neg_count > 0 else [],
            "reasoning": f"Based on keyword analysis, sentiment appears {sentiment}."
        }

    @staticmethod
    def get_entities(text: str) -> dict:
        entities = [
            {"text": "Example Corp", "type": "organization", "confidence": 0.9, "description": "A company"},
            {"text": "New York", "type": "location", "confidence": 0.95, "description": "A city"},
        ]
        return {
            "entities": entities,
            "total_count": len(entities),
            "entity_types_found": list(set(e["type"] for e in entities))
        }

    @staticmethod
    def get_summary(text: str) -> dict:
        return {
            "title": "Simulated Summary Title",
            "one_line_summary": "This text discusses important topics.",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "topics": ["Technology", "Innovation"],
            "audience": "general",
            "sentiment": "neutral"
        }


# ---------- Main Client Class ----------

class StructuredOutputClient:
    """
    A client for producing typed, validated structured outputs from LLM calls.
    Supports both real API calls and simulated responses.
    """

    def __init__(self, model: str = "gpt-4o", use_api: bool = True):
        self.model = model
        self.use_api = use_api and bool(os.environ.get("OPENAI_API_KEY"))
        self.simulator = SimulatedStructuredLLM()
        self.call_log: List[dict] = []

    def _log_call(self, task: str, model_name: str, success: bool, latency: float):
        self.call_log.append({
            "task": task,
            "model": model_name,
            "success": success,
            "latency_ms": round(latency, 1),
            "timestamp": time.time()
        })

    def _call_openai_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system_message: str = "",
        temperature: float = 0.0,
        max_retries: int = 3,
    ) -> Optional[T]:
        """Call OpenAI with structured output enforcement."""
        import openai
        client = openai.OpenAI()

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": response_model.__name__,
                            "strict": True,
                            "schema": response_model.model_json_schema()
                        }
                    },
                    temperature=temperature,
                )
                return response_model.model_validate_json(response.choices[0].message.content)
            except (ValidationError, json.JSONDecodeError) as e:
                print(f"  Retry {attempt + 1}/{max_retries}: {type(e).__name__}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(1)

    def analyze_sentiment(self, text: str) -> Optional[SentimentAnalysisResult]:
        """Perform structured sentiment analysis."""
        start = time.time()

        if self.use_api:
            result = self._call_openai_structured(
                prompt=f"Analyze the sentiment of this text:\n\n{text}",
                response_model=SentimentAnalysisResult,
                system_message="You are a sentiment analysis expert. Provide detailed, accurate analysis."
            )
        else:
            data = self.simulator.get_sentiment(text)
            result = SentimentAnalysisResult.model_validate(data)

        self._log_call("sentiment", self.model, result is not None, (time.time() - start) * 1000)
        return result

    def extract_entities(self, text: str) -> Optional[EntityExtractionResult]:
        """Perform structured entity extraction."""
        start = time.time()

        if self.use_api:
            result = self._call_openai_structured(
                prompt=f"Extract all named entities from this text:\n\n{text}",
                response_model=EntityExtractionResult,
                system_message="You are an NLP entity extraction system. Extract all named entities."
            )
        else:
            data = self.simulator.get_entities(text)
            result = EntityExtractionResult.model_validate(data)

        self._log_call("entities", self.model, result is not None, (time.time() - start) * 1000)
        return result

    def summarize(self, text: str) -> Optional[StructuredSummary]:
        """Produce a structured summary."""
        start = time.time()

        if self.use_api:
            result = self._call_openai_structured(
                prompt=f"Summarize the following text:\n\n{text}",
                response_model=StructuredSummary,
                system_message="You are a summarization assistant. Produce clear, structured summaries."
            )
        else:
            data = self.simulator.get_summary(text)
            result = StructuredSummary.model_validate(data)

        self._log_call("summary", self.model, result is not None, (time.time() - start) * 1000)
        return result

    def classify(self, text: str, categories: List[str] = None) -> Optional[ClassificationResult]:
        """Classify text into categories with structured output."""
        start = time.time()
        categories = categories or ["technology", "sports", "politics", "entertainment", "science", "business"]

        if self.use_api:
            result = self._call_openai_structured(
                prompt=f"Classify this text into one of these categories: {', '.join(categories)}\n\nText: {text}",
                response_model=ClassificationResult,
                system_message="You are a text classification system."
            )
        else:
            result = ClassificationResult(
                primary_category="technology",
                subcategories=["software"],
                confidence=0.78,
                alternative_categories=["science"],
                reasoning="Text discusses technology topics."
            )

        self._log_call("classification", self.model, result is not None, (time.time() - start) * 1000)
        return result

    def print_log(self):
        """Print the call log."""
        print("\n--- API Call Log ---")
        for entry in self.call_log:
            status = "OK" if entry["success"] else "FAIL"
            print(f"  [{status}] {entry['task']:20s} | {entry['model']:15s} | {entry['latency_ms']:8.1f}ms")


# ---------- Demo ----------
if __name__ == "__main__":
    client = StructuredOutputClient(use_api=True)

    print("=" * 70)
    print("STRUCTURED OUTPUT TOOLKIT DEMO")
    print("=" * 70)

    # Sentiment Analysis
    print("\n--- SENTIMENT ANALYSIS ---")
    sentiment_result = client.analyze_sentiment(
        "The new laptop has an incredible display and fast processor, "
        "but the keyboard feels cheap and the fan is too loud under load."
    )
    if sentiment_result:
        print(f"Sentiment: {sentiment_result.sentiment.value}")
        print(f"Confidence: {sentiment_result.confidence}")
        print(f"Positive: {sentiment_result.positive_aspects}")
        print(f"Negative: {sentiment_result.negative_aspects}")

    # Entity Extraction
    print("\n--- ENTITY EXTRACTION ---")
    entity_result = client.extract_entities(
        "Apple Inc. announced Tim Cook will present the new iPhone 16 "
        "at their headquarters in Cupertino, California on September 9th, 2024."
    )
    if entity_result:
        print(f"Total entities: {entity_result.total_count}")
        for entity in entity_result.entities:
            print(f"  [{entity.type.value}] {entity.text} (confidence: {entity.confidence})")

    # Structured Summary
    print("\n--- STRUCTURED SUMMARY ---")
    summary_result = client.summarize(
        "Artificial intelligence has transformed numerous industries in recent years. "
        "Healthcare uses AI for drug discovery and diagnostics. Finance employs AI for "
        "fraud detection and algorithmic trading. The technology continues to advance "
        "rapidly, with large language models demonstrating remarkable capabilities in "
        "understanding and generating human language."
    )
    if summary_result:
        print(f"Title: {summary_result.title}")
        print(f"Summary: {summary_result.one_line_summary}")
        print(f"Key Points: {summary_result.key_points}")
        print(f"Topics: {summary_result.topics}")

    # Classification
    print("\n--- CLASSIFICATION ---")
    classify_result = client.classify(
        "SpaceX successfully launched their Starship rocket on a test flight, "
        "achieving a full orbital trajectory for the first time."
    )
    if classify_result:
        print(f"Category: {classify_result.primary_category}")
        print(f"Confidence: {classify_result.confidence}")
        print(f"Alternatives: {classify_result.alternative_categories}")

    # Print call log
    client.print_log()