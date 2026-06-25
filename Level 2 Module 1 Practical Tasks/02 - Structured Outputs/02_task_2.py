# Task 2: Implement a Schema Evolution Handler
# Build a system that handles versioned Pydantic schemas. Define V1, V2, and V3 versions of a ProductAnalysis model where each version adds new fields. 
# Write migration functions that can convert V1 outputs to V2 and V2 to V3 (filling defaults for new fields). 
# Create a test suite that validates migration correctness and demonstrates backward compatibility by processing outputs generated with older schemas.

from pydantic import BaseModel, Field
from typing import List, Optional

# Schemas

class ProductAnalysisV1(BaseModel):
    product_name: str
    rating: float

class ProductAnalysisV2(BaseModel):
    product_name: str
    rating: float
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)

class ProductAnalysisV3(BaseModel):
    product_name: str
    rating: float
    pros: List[str]
    cons: List[str]
    sentiment: str = "neutral"
    recommended: Optional[bool] = None



# Migration Functions

def migrate_v1_to_v2(v1_data: dict) -> dict:
    """Migrates a V1 dictionary to a V2 dictionary via defaults."""
    v2_data = v1_data.copy()
    v2_data.setdefault("pros", [])
    v2_data.setdefault("cons", [])
    return v2_data

def migrate_v2_to_v3(v2_data: dict) -> dict:
    """Migrates a V2 dictionary to a V3 dictionary via defaults."""
    v3_data = v2_data.copy()
    v3_data.setdefault("sentiment", "neutral")
    v3_data.setdefault("recommended", None)
    return v3_data

def process_to_latest(raw_json_str: str, version: str) -> ProductAnalysisV3:
    """
    Accepts data from any version, runs it through the migration pipeline, and returns a V3 model.
    """
    import json
    data = json.loads(raw_json_str)
    
    if version == "V1":
        ProductAnalysisV1.model_validate(data)
        data = migrate_v1_to_v2(data)
        data = migrate_v2_to_v3(data)
    elif version == "V2":
        ProductAnalysisV2.model_validate(data)
        data = migrate_v2_to_v3(data)
    else:
        pass
        
    return ProductAnalysisV3.model_validate(data)



if __name__ == "__main__":
    v1_payload = '{"product_name": "Wireless Mouse", "rating": 4.5}'
    v2_payload = '{"product_name": "Mechanical Keyboard", "rating": 4.8, "pros": ["sleek"], "cons": ["loud"]}'
    
    print("V1 -> V3 Migration:")
    v1_migrated = process_to_latest(v1_payload, version="V1")
    print(f"Result: {v1_migrated.model_dump_json(indent=2)}\n")
    assert v1_migrated.sentiment == "neutral"
    assert v1_migrated.pros == []

    print("V2 -> V3 Migration:")
    v2_migrated = process_to_latest(v2_payload, version="V2")
    print(f"Result: {v2_migrated.model_dump_json(indent=2)}\n")
    assert v2_migrated.sentiment == "neutral"
    assert v2_migrated.pros == ["sleek"]