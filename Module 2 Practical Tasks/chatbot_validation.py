"""
AI Chatbot API -- Full Pydantic Validation
==========================================
Request/response models for an AI chatbot API with comprehensive
validation, custom validators, nested models, and serialization.

Run:  uvicorn chatbot_validation:app --reload
Docs: http://127.0.0.1:8000/docs
"""

from urllib import response

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import time

# ---------- App ----------
app = FastAPI(
    title="AI Chatbot API",
    description="Comprehensive request/response validation for chatbot interactions.",
    version="1.0.0",
)

# ---------- Custom validation error handler ----------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field_path = ".".join(str(loc) for loc in err["loc"] if loc != "body")
        errors.append({
            "field": field_path,
            "message": err["msg"],
            "error_type": err["type"],
        })
    return JSONResponse(
        status_code=422,
        content={
            "status": 422,
            "error": "validation_error",
            "message": "Request validation failed. Check 'details' for specific field errors.",
            "details": errors,
        }
    )

# ---------- Enums ----------
class ModelEnum(str, Enum):
    gpt4 = "gpt-4"
    gpt35 = "gpt-3.5-turbo"
    claude3 = "claude-3-opus"
    llama2 = "llama-2-70b"
    mistral = "mistral-7b"

class RoleEnum(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"

class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

# ---------- Nested Models ----------

class ChatMessage(BaseModel):
    role: RoleEnum = Field(..., description="Who sent this message")
    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message text"
    )

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, v: str) -> str:
        if v.strip() != v:
            v = v.strip()
        if not v:
            raise ValueError("Message content cannot be blank or whitespace only")
        return v

class SafetySettings(BaseModel):
    toxicity_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Block responses above this toxicity score"
    )
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Rate limit"
    )
    block_pii: bool = Field(
        default=True,
        description="Detect and block personally identifiable information"
    )

class ModelParameters(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop: List[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def check_penalties(self):
        if self.frequency_penalty != 0.0 and self.temperature == 0.0:
            raise ValueError(
                "frequency_penalty has no effect when temperature is 0 (greedy decoding)"
            )
        return self

# ---------- Request Schema ----------

class ChatRequest(BaseModel):
    model: ModelEnum = Field(..., description="Which LLM to use")
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Conversation messages"
    )
    parameters: ModelParameters = Field(default_factory=ModelParameters)
    safety: SafetySettings = Field(default_factory=SafetySettings)
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Custom metadata to pass through"
    )

    @field_validator("messages")
    @classmethod
    def messages_must_have_user(cls, v: List[ChatMessage]) -> List[ChatMessage]:
        roles = [msg.role for msg in v]
        if "user" not in roles:
            raise ValueError("At least one message must have role 'user'")
        return v

    @model_validator(mode="after")
    def check_conversation_length(self):
        total_length = sum(len(msg.content) for msg in self.messages)
        if total_length > 20000:
            raise ValueError(
                f"Total conversation length exceeds 20,000 characters (got {total_length})"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-4",
                    "messages": [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": "Write a Python function to reverse a string."},
                    ],
                    "parameters": {"temperature": 0.2, "max_tokens": 512},
                }
            ]
        }
    }

# ---------- Response Schemas ----------

class TokenUsage(BaseModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)

class ResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None

class SafetyResult(BaseModel):
    flagged: bool
    categories: Dict[str, bool] = Field(default_factory=dict)
    severity: Optional[SeverityEnum] = None

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: List[ResponseChoice]
    usage: TokenUsage
    safety: SafetyResult
    created: str

# ---------- Simulated LLM call ----------
def simulate_llm_response(request: ChatRequest) -> ChatResponse:
    """Simulate an LLM response for demonstration purposes."""
    last_user_msg = [m for m in request.messages if m.role == "user"][-1]
    simulated_reply = (
        f"[Simulated response from {request.model.value}] "
        f"You asked: '{last_user_msg.content[:50]}...'. "
        f"This is a placeholder response for testing validation."
    )
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=request.model.value,
        choices=[
            ResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=simulated_reply),
                finish_reason="stop",
            )
        ],
        usage=TokenUsage(
            prompt_tokens=sum(len(m.content.split()) for m in request.messages),
            completion_tokens=len(simulated_reply.split()),
            total_tokens=sum(len(m.content.split()) for m in request.messages)
                         + len(simulated_reply.split()),
        ),
        safety=SafetyResult(flagged=False, categories={}, severity=None),
        created=datetime.utcnow().isoformat(),
    )

# ---------- Endpoints ----------

@app.get("/")
def root():
    return {
        "service": "AI Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.post("/v1/chat/completions", response_model=ChatResponse)
def chat_completions(request: ChatRequest):
    """
    Send messages to the chatbot and receive a validated response.

    The request is fully validated by Pydantic before this function runs.
    Invalid requests get a 422 with detailed field-level error messages.
    """
    return simulate_llm_response(request)

@app.post("/v1/chat/validate", response_model=dict)
def validate_only(request: ChatRequest):
    """
    Validate a request without generating a response.
    Useful for client-side form validation.
    """
    return {
        "valid": True,
        "model": request.model.value,
        "message_count": len(request.messages),
        "parameters": request.parameters.model_dump(),
        "safety_settings": request.safety.model_dump(),
    }

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot_validation:app", host="0.0.0.0", port=8000, reload=True)

# 03 Task 1: Add a Conversation Length Validator
## Inserted in ChatRequest model after messages_must_have_user validator

'''
@model_validator(mode="after")
def check_conversation_length(self):
    total_length = sum(len(msg.content) for msg in self.messages)
    if total_length > 20000:
        raise ValueError(
            f"Total conversation length exceeds 20,000 characters (got {total_length})"
        )
    return self
'''

# 03 Task 2: Add a Streaming Response Model

class ChatStreamChunk(BaseModel):
    id: str
    delta: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Partial message content and/or role updates"
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason for completion (e.g., 'stop', 'length')"
    )

@app.post("/v1/chat/stream", response_model=List[ChatStreamChunk])
def chat_stream(request: ChatRequest):
    """Simulate a streaming response by returning a list of ChatStreamChunk objects."""
    full_response = simulate_llm_response(request)
    chunks = []
    for choice in full_response.choices:
        content = choice.message.content
        for i in range(0, len(content), 10):
            chunk_content = content[i:i+10]
            chunk = ChatStreamChunk(
                id=f"chunk-{uuid.uuid4().hex[:8]}",
                delta={"content": chunk_content},
                finish_reason=None
            )
            chunks.append(chunk)
        chunks.append(ChatStreamChunk(
            id=f"chunk-{uuid.uuid4().hex[:8]}",
            delta={},
            finish_reason=choice.finish_reason
        ))
    return chunks


# 03 Task 3: Build a Configuration Profile Model

class ProfileModel(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    default_model: ModelEnum
    default_parameters: ModelParameters = Field(default_factory=ModelParameters)
    safety_settings: SafetySettings = Field(default_factory=SafetySettings)
    description: Optional[str] = None

profiles_db: Dict[str, ProfileModel] = {}

@app.post("/v1/profiles")
def create_profile(profile: ProfileModel):
    profile_id = f"profile-{uuid.uuid4().hex[:8]}"
    profiles_db[profile_id] = profile
    return {"id": profile_id, **profile.model_dump(exclude_none=True)}

@app.get("/v1/profiles")
def list_profiles():
    return [profile.model_dump(exclude_none=True) for profile in profiles_db.values()]

