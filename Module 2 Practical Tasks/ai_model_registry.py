"""
AI Model Registry API
=====================
A complete CRUD API for managing AI/ML models.
Run: uvicorn ai_model_registry:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid

app = FastAPI(
    title="AI Model Registry",
    description="Register, list, get, and delete AI/ML models.",
    version="1.0.0",
)

# ---------- In-memory storage ----------
db: dict[str, dict] = {}

# ---------- Pydantic schemas ----------

class ModelCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    description: str = Field(..., min_length=10, description="What this model does")
    task_type: str = Field(..., description="e.g. text-generation, image-classification")
    framework: str = Field(default="pytorch", description="ML framework")
    version: str = Field(default="1.0.0", description="Model version string")
    parameters_count: Optional[int] = Field(default=None, gt=0, description="Trainable params")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "gpt2-medium",
                    "description": "OpenAI GPT-2 medium for text generation tasks",
                    "task_type": "text-generation",
                    "framework": "pytorch",
                    "version": "1.0.0",
                    "parameters_count": 354823168,
                    "tags": ["nlp", "generative", "decoder-only"]
                }
            ]
        }
    }

class ModelUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, min_length=10)
    task_type: Optional[str] = None
    framework: Optional[str] = None
    version: Optional[str] = None
    parameters_count: Optional[int] = Field(default=None, gt=0)
    tags: Optional[List[str]] = None

class ModelResponse(BaseModel):
    id: str
    name: str
    description: str
    task_type: str
    framework: str
    version: str
    parameters_count: Optional[int]
    tags: List[str]
    created_at: str
    updated_at: str

class MessageResponse(BaseModel):
    message: str

# ---------- Endpoints ----------

@app.get("/", response_model=MessageResponse)
def root():
    return {"message": "Welcome to the AI Model Registry API. Visit /docs for interactive docs."}

@app.post("/models", response_model=ModelResponse, status_code=201)
def register_model(model: ModelCreateRequest):
    """Register a new AI model in the registry."""
    model_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    record = {
        "id": model_id,
        "created_at": now,
        "updated_at": now,
        **model.model_dump(),
    }
    db[model_id] = record
    return record

@app.get("/models", response_model=List[ModelResponse])
def list_models(
    task_type: Optional[str] = Query(default=None, description="Filter by task type"),
    framework: Optional[str] = Query(default=None, description="Filter by framework"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List all registered models with optional filtering and pagination."""
    results = list(db.values())
    if task_type:
        results = [m for m in results if m["task_type"] == task_type]
    if framework:
        results = [m for m in results if m["framework"] == framework]
    if tag:
        results = [m for m in results if tag in m.get("tags", [])]
    return results[offset : offset + limit]

@app.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: str):
    """Retrieve a single model by its ID."""
    if model_id not in db:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return db[model_id]

@app.put("/models/{model_id}", response_model=ModelResponse)
def update_model(model_id: str, update: ModelUpdateRequest):
    """Update fields of an existing model."""
    if model_id not in db:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    record = db[model_id]
    update_data = update.model_dump(exclude_unset=True)
    record.update(update_data)
    record["updated_at"] = datetime.utcnow().isoformat()
    db[model_id] = record
    return record

@app.delete("/models/{model_id}", response_model=MessageResponse)
def delete_model(model_id: str):
    """Remove a model from the registry."""
    if model_id not in db:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    del db[model_id]
    return {"message": f"Model '{model_id}' deleted successfully"}

# ---------- Run directly ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_model_registry:app", host="0.0.0.0", port=8000, reload=True)


# 01 Task 1

@app.get("/models/search", response_model=List[ModelResponse])
def search_models(q: str = Query(..., description="Case-insensitive text search term")):
    """Search models by name and description fields."""
    results = []
    search_term = q.lower()
    
    for model in db.values():
        name_match = search_term in model["name"].lower()
        desc_match = search_term in model["description"].lower()
        
        if name_match or desc_match:
            results.append(model)
            
    return results



# 01 Task 2

class StatsResponse(BaseModel):
    total_models: int
    by_task_type: dict[str, int]
    by_framework: dict[str, int]

@app.get("/stats", response_model=StatsResponse)
def get_registry_stats():
    """Response model for registry statistics."""
    total = len(db)
    task_counts = {}
    framework_counts = {}
    
    for model in db.values():
        task = model["task_type"]
        framework = model["framework"]

        task_counts[task] = task_counts.get(task, 0) + 1
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
        
    return {
        "total_models": total,
        "by_task_type": task_counts,
        "by_framework": framework_counts
    }


# 01 Task 3

@app.post("/models/bulk", response_model=List[ModelResponse], status_code=201)
def bulk_register_models(models_list: List[ModelCreateRequest]):
    """Register multiple models in a single request."""
    created_records = []
    
    for model in models_list:
        model_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        
        record = {
            "id": model_id,
            "created_at": now,
            "updated_at": now,
            **model.model_dump(),
        }
        
        db[model_id] = record
        created_records.append(record)
        
    return created_records