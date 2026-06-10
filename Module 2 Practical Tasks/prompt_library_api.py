"""
LLM Prompt Library API
======================
A fully featured REST API for managing LLM prompts with CRUD,
pagination, search, filtering, and proper error handling.

Run:  uvicorn prompt_library_api:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum
from math import ceil
import uuid

# ---------- App setup ----------
app = FastAPI(
    title="LLM Prompt Library",
    description="Manage, search, and organize LLM prompts with a clean REST API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Enums ----------
class CategoryEnum(str, Enum):
    coding = "coding"
    writing = "writing"
    analysis = "analysis"
    creative = "creative"
    general = "general"

# ---------- Schemas ----------

class PromptCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    content: str = Field(..., min_length=10, description="The prompt template text")
    category: CategoryEnum
    tags: List[str] = Field(default_factory=list)
    is_public: bool = Field(default=True)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Code Review Assistant",
                    "content": "Review the following code for bugs, performance issues, and best practices: {{code}}",
                    "category": "coding",
                    "tags": ["code-review", "quality"],
                    "is_public": True,
                }
            ]
        }
    }

class PromptUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=3, max_length=200)
    content: Optional[str] = Field(default=None, min_length=10)
    category: Optional[CategoryEnum] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None

class PromptResponse(BaseModel):
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    is_public: bool
    author: str
    created_at: str
    updated_at: str

class PaginatedPromptResponse(BaseModel):
    data: List[PromptResponse]
    meta: dict
    links: dict

class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    status: int
    error: str
    message: str
    details: Optional[dict] = None

# ---------- In-memory database ----------
db: dict[str, dict] = {}

# ---------- Seed data ----------
def seed_data():
    samples = [
        {
            "title": "Code Review Assistant",
            "content": "Review the following code for bugs, performance issues, and best practices:\n\n{{code}}",
            "category": "coding",
            "tags": ["code-review", "quality"],
        },
        {
            "title": "Blog Post Writer",
            "content": "Write a blog post about {{topic}} in a {{tone}} tone. Target audience: {{audience}}.",
            "category": "writing",
            "tags": ["blog", "content-creation"],
        },
        {
            "title": "Data Analysis Helper",
            "content": "Analyze the following dataset and provide key insights, trends, and anomalies:\n\n{{data}}",
            "category": "analysis",
            "tags": ["data", "insights"],
        },
        {
            "title": "Creative Story Starter",
            "content": "Write the opening paragraph of a {{genre}} story set in {{setting}} with a protagonist who is {{character}}.",
            "category": "creative",
            "tags": ["fiction", "storytelling"],
        },
    ]
    for s in samples:
        model_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow().isoformat()
        db[model_id] = {
            "id": model_id,
            "author": "system",
            "created_at": now,
            "updated_at": now,
            "is_public": True,
            **s,
        }

seed_data()

# ---------- Helper: pagination ----------
def paginate_results(items: list, limit: int, offset: int, base_path: str) -> dict:
    total = len(items)
    page = items[offset : offset + limit]
    total_pages = ceil(total / limit) if limit else 0
    return {
        "data": page,
        "meta": {
            "total": total,
            "total_pages": total_pages,
            "current_page": (offset // limit) + 1 if limit else 1,
            "limit": limit,
            "offset": offset,
        },
        "links": {
            "self": f"{base_path}?limit={limit}&offset={offset}",
            "next": f"{base_path}?limit={limit}&offset={offset + limit}"
                if offset + limit < total else None,
            "prev": f"{base_path}?limit={limit}&offset={max(0, offset - limit)}"
                if offset > 0 else None,
        },
    }

# ---------- Error helper ----------
def not_found_error(resource_id: str):
    raise HTTPException(
        status_code=404,
        detail={
            "status": 404,
            "error": "not_found",
            "message": f"Prompt '{resource_id}' does not exist",
            "details": {"prompt_id": resource_id},
        }
    )

# ---------- Endpoints ----------

@app.get("/", response_model=MessageResponse)
def root():
    return {"message": "LLM Prompt Library API v1.0. See /docs for interactive documentation."}

@app.post(
    "/api/v1/prompts",
    response_model=PromptResponse,
    status_code=201,
    summary="Create a new prompt",
)
def create_prompt(prompt: PromptCreate):
    """Create and store a new prompt in the library."""
    prompt_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    record = {
        "id": prompt_id,
        "author": "anonymous",
        "created_at": now,
        "updated_at": now,
        **prompt.model_dump(),
    }
    db[prompt_id] = record
    return record
'''
@app.get(
    "/api/v1/prompts",
    response_model=PaginatedPromptResponse,
    summary="List prompts with filtering and pagination",
)
def list_prompts(
    category: Optional[CategoryEnum] = Query(default=None, description="Filter by category"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    q: Optional[str] = Query(default=None, description="Search title and content"),
    is_public: Optional[bool] = Query(default=None, description="Filter by visibility"),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List prompts with optional filters and pagination."""
    results = list(db.values())

    if category:
        results = [p for p in results if p["category"] == category]
    if tag:
        results = [p for p in results if tag in p.get("tags", [])]
    if q:
        q_lower = q.lower()
        results = [
            p for p in results
            if q_lower in p["title"].lower() or q_lower in p["content"].lower()
        ]
    if is_public is not None:
        results = [p for p in results if p["is_public"] == is_public]

    return paginate_results(results, limit, offset, "/api/v1/prompts")
'''
@app.get("/api/v1/prompts/{prompt_id}", response_model=PromptResponse)
def get_prompt(prompt_id: str):
    """Retrieve a single prompt by ID."""
    if prompt_id not in db:
        not_found_error(prompt_id)
    return db[prompt_id]

@app.put("/api/v1/prompts/{prompt_id}", response_model=PromptResponse)
def update_prompt(prompt_id: str, update: PromptUpdate):
    """Update an existing prompt. Only provided fields are changed."""
    if prompt_id not in db:
        not_found_error(prompt_id)
    record = db[prompt_id]
    changes = update.model_dump(exclude_unset=True)
    record.update(changes)
    record["updated_at"] = datetime.utcnow().isoformat()
    db[prompt_id] = record
    return record

@app.delete("/api/v1/prompts/{prompt_id}", response_model=MessageResponse)
def delete_prompt(prompt_id: str):
    """Delete a prompt from the library."""
    if prompt_id not in db:
        not_found_error(prompt_id)
    del db[prompt_id]
    return {"message": f"Prompt '{prompt_id}' deleted successfully"}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prompt_library_api:app", host="0.0.0.0", port=8000, reload=True)



# 02 Task 1: Add a Categories Endpoint

class CategoryItem(BaseModel):
    name: str
    count: int


class CategoriesResponse(BaseModel):
    categories: List[CategoryItem]


@app.get("/api/v1/categories", response_model=CategoriesResponse, summary="List available categories with counts")
def list_categories():
    """Return all available categories with a count of prompts in each."""
    counts = {c.value: 0 for c in CategoryEnum}
    for p in db.values():
        cat = p.get("category")
        if isinstance(cat, CategoryEnum):
            cat = cat.value
        if cat in counts:
            counts[cat] += 1
        else:
            counts[cat] = counts.get(cat, 0) + 1

    categories = [{"name": name, "count": count} for name, count in counts.items()]
    return {"categories": categories}


# 02 Task 2: Add Sorting to the List Endpoint

@app.get("/api/v1/prompts", response_model=PaginatedPromptResponse, summary="List prompts with filtering, sorting, and pagination")
def list_prompts(
    category: Optional[CategoryEnum] = Query(default=None, description="Filter by category"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    q: Optional[str] = Query(default=None, description="Search title and content"),
    is_public: Optional[bool] = Query(default=None, description="Filter by visibility"),
    sort_by: Optional[str] = Query(default=None, description="Sort by field (title, created_at, updated_at)"),
    sort_order: Optional[str] = Query(default="asc", description="Sort order (asc or desc)"),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List prompts with optional filters, sorting, and pagination."""
    results = list(db.values())

    if category:
        results = [p for p in results if p["category"] == category]
    if tag:
        results = [p for p in results if tag in p.get("tags", [])]
    if q:
        q_lower = q.lower()
        results = [
            p for p in results
            if q_lower in p["title"].lower() or q_lower in p["content"].lower()
        ]
    if is_public is not None:
        results = [p for p in results if p["is_public"] == is_public]

    valid_sort_fields = {"title", "created_at", "updated_at"}
    if sort_by:
        if sort_by not in valid_sort_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "status": 400,
                    "error": "invalid_sort_field",
                    "message": f"Invalid sort field '{sort_by}'. Valid fields are: {', '.join(valid_sort_fields)}",
                    "details": {"sort_by": sort_by},
                }
            )
        reverse = sort_order == "desc"
        results.sort(key=lambda x: x.get(sort_by), reverse=reverse)

    return paginate_results(results, limit, offset, "/api/v1/prompts")


# 02 Task 3: Implement Soft Delete

@app.delete("/api/v1/prompts/{prompt_id}", response_model=MessageResponse)
def delete_prompt(prompt_id: str):
    """Soft delete a prompt from the library."""
    if prompt_id not in db:
        not_found_error(prompt_id)
    record = db[prompt_id]
    record["deleted_at"] = datetime.utcnow().isoformat()
    record["is_public"] = False
    db[prompt_id] = record
    return {"message": f"Prompt '{prompt_id}' soft-deleted successfully"}

@app.get("/api/v1/prompts/trash", response_model=PaginatedPromptResponse, summary="List soft-deleted prompts")
def list_deleted_prompts(limit: int = Query(default=10, ge=1, le=100),
                         offset: int = Query(default=0, ge=0),
                         ):
    """List all soft-deleted prompts."""
    results = [p for p in db.values() if p.get("deleted_at") is not None]
    return paginate_results(results, limit, offset, "/api/v1/prompts/trash")