"""
AI API with Complete Authentication
====================================
User registration, login, JWT token management, and protected
LLM endpoints. Demonstrates the full auth lifecycle.

Install:  pip install fastapi uvicorn pyjwt passlib[bcrypt] python-multipart
Run:      uvicorn ai_api_auth:app --reload
Docs:     http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field, field_validator
from passlib.context import CryptContext
from typing import Optional, List
from datetime import datetime, timedelta
from enum import Enum
import jwt
import uuid

# ---------- Configuration ----------
SECRET_KEY = "dev-secret-change-in-production-use-env-vars"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ---------- App ----------
app = FastAPI(
    title="AI API with Authentication",
    description="Complete auth system protecting AI/LLM endpoints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Password hashing ----------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------- In-memory storage ----------
users_db: dict[str, dict] = {}
prompts_db: dict[str, dict] = {}

# ---------- Enums ----------
class UserRole(str, Enum):
    user = "user"
    premium = "premium"
    admin = "admin"

class ModelEnum(str, Enum):
    gpt4 = "gpt-4"
    gpt35 = "gpt-3.5-turbo"
    claude3 = "claude-3-opus"
    llama2 = "llama-2-70b"

# ---------- Schemas ----------

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=30, pattern=r"^[a-zA-Z0-9_]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    role: UserRole = Field(default=UserRole.user)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    role: str
    created_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class PromptRequest(BaseModel):
    model: ModelEnum = Field(..., description="Which LLM to use")
    prompt: str = Field(..., min_length=1, max_length=5000)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-4",
                    "prompt": "Explain quantum computing in simple terms",
                    "max_tokens": 256,
                    "temperature": 0.7,
                }
            ]
        }
    }

class LLMResponse(BaseModel):
    id: str
    model: str
    prompt: str
    response: str
    tokens_used: int
    created_at: str

class MessageResponse(BaseModel):
    message: str

# ---------- JWT utilities ----------

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token. Please log in again.")

# ---------- Auth dependencies ----------

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    payload = decode_token(token)
    username = payload.get("sub")
    if not username or username not in users_db:
        raise HTTPException(status_code=401, detail="User not found")
    user = users_db[username]
    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
    }

class RoleChecker:
    """Dependency that enforces the user has one of the allowed roles."""
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in self.allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Your role: '{current_user['role']}'. "
                       f"Required: {self.allowed_roles}",
            )
        return current_user

# ---------- Public endpoints ----------

@app.get("/")
def root():
    return {
        "service": "AI API with Authentication",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "register": "POST /auth/register",
            "login": "POST /auth/login",
            "me": "GET /auth/me (authenticated)",
            "chat": "POST /v1/chat (authenticated)",
            "history": "GET /v1/history (authenticated)",
        }
    }

# ---------- Auth endpoints ----------

@app.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=201,
    summary="Register a new user",
)
def register(user_data: UserRegister):
    """Create a new user account. Password is hashed before storage."""
    if user_data.username in users_db:
        raise HTTPException(
            status_code=409,
            detail=f"Username '{user_data.username}' is already taken"
        )

    # Check for duplicate email
    for existing in users_db.values():
        if existing["email"] == user_data.email:
            raise HTTPException(
                status_code=409,
                detail=f"Email '{user_data.email}' is already registered"
            )

    user_id = str(uuid.uuid4())[:8]
    hashed_password = pwd_context.hash(user_data.password)

    users_db[user_data.username] = {
        "id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hashed_password,
        "role": user_data.role.value,
        "created_at": datetime.utcnow().isoformat(),
    }

    return UserResponse(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        role=user_data.role.value,
        created_at=users_db[user_data.username]["created_at"],
    )

@app.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="Log in and get a JWT token",
)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with username and password (sent as form data).
    Returns a JWT access token on success.
    """
    user = users_db.get(form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    if not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )

    return TokenResponse(
        access_token=token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

@app.get(
    "/auth/me",
    response_model=UserResponse,
    summary="Get current user profile",
)
def get_me(current_user: dict = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
        created_at=users_db[current_user["username"]]["created_at"],
    )

# ---------- Protected AI endpoints ----------
'''
@app.post(
    "/v1/chat",
    response_model=LLMResponse,
    summary="Send a prompt to an LLM (requires authentication)",
)
'''
def chat_completion(
    request: PromptRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Send a prompt to an LLM model. Requires a valid JWT token.
    """
    # Check if user role has access to the requested model
    premium_models = ["gpt-4", "claude-3-opus"]
    if request.model.value in premium_models and current_user["role"] == "user":
        raise HTTPException(
            status_code=403,
            detail=f"Model '{request.model.value}' requires premium or admin role. "
                   f"Your role: '{current_user['role']}'",
        )

    # Simulate LLM response
    response_id = str(uuid.uuid4())[:8]
    simulated = (
        f"[Simulated {request.model.value}] Response to: "
        f"'{request.prompt[:60]}...'"
    )
    tokens = min(request.max_tokens, len(simulated.split()))

    # Store in history
    record = {
        "id": response_id,
        "user": current_user["username"],
        "model": request.model.value,
        "prompt": request.prompt,
        "response": simulated,
        "tokens_used": tokens,
        "created_at": datetime.utcnow().isoformat(),
    }
    prompts_db[response_id] = record

    return LLMResponse(**record)

@app.get(
    "/v1/history",
    response_model=List[LLMResponse],
    summary="Get your prompt history",
)
def get_history(
    current_user: dict = Depends(get_current_user),
    limit: int = 20,
):
    """Retrieve the authenticated user's prompt history."""
    user_prompts = [
        record for record in prompts_db.values()
        if record["user"] == current_user["username"]
    ]
    return user_prompts[:limit]

@app.delete(
    "/v1/history/{prompt_id}",
    response_model=MessageResponse,
    summary="Delete a prompt from history",
)
def delete_history_item(
    prompt_id: str,
    current_user: dict = Depends(get_current_user),
):
    if prompt_id not in prompts_db:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if prompts_db[prompt_id]["user"] != current_user["username"]:
        raise HTTPException(status_code=403, detail="Not authorized to delete this prompt")
    del prompts_db[prompt_id]
    return {"message": f"Prompt '{prompt_id}' deleted"}

# ---------- Admin-only endpoints ----------

admin_only = RoleChecker(["admin"])

@app.get(
    "/admin/users",
    response_model=List[UserResponse],
    summary="List all users (admin only)",
)
def list_users(current_user: dict = Depends(admin_only)):
    """Admin endpoint: list all registered users."""
    return [
        UserResponse(
            id=u["id"],
            username=u["username"],
            email=u["email"],
            role=u["role"],
            created_at=u["created_at"],
        )
        for u in users_db.values()
    ]

@app.delete(
    "/admin/users/{username}",
    response_model=MessageResponse,
    summary="Delete a user (admin only)",
)
def delete_user(
    username: str,
    current_user: dict = Depends(admin_only),
):
    if username not in users_db:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")
    if username == current_user["username"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    del users_db[username]
    return {"message": f"User '{username}' deleted"}

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_api_auth:app", host="0.0.0.0", port=8000, reload=True)


# Task 1: Add a Token Refresh Endpoint

def refresh_token(current_user: dict = Depends(get_current_user)) -> TokenResponse:
    """Refresh the JWT token for the currently authenticated user."""
    new_token = create_access_token(
        data={"sub": current_user["username"], "role": current_user["role"]}
    )
    return TokenResponse(
        access_token=new_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

app.post(
    "/auth/refresh",
    response_model=TokenResponse,
    summary="Refresh your JWT token",
)(refresh_token)

# Task 2: Add Rate Limiting Per User
## Implement a simple in-memory rate limiter that tracks the number of requests per user within a time window. Create a dependency `RateLimiter(max_requests=10, window_seconds=60)` that checks if the current user has exceeded the limit. Apply it to the `POST /v1/chat` endpoint so each user can only make 10 chat requests per minute. Return a 429 Too Many Requests response with a `Retry-After` header when the limit is exceeded.

from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def __call__(self, current_user: dict = Depends(get_current_user)):
        now = time()
        user_requests = self.requests[current_user["username"]]

        while user_requests and user_requests[0] < now - self.window_seconds:
            user_requests.pop(0)

        if len(user_requests) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - user_requests[0]))
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )

        user_requests.append(now)

rate_limiter = RateLimiter(max_requests=10, window_seconds=60)

app.post(
    "/v1/chat",
    response_model=LLMResponse,
    summary="Send a prompt to an LLM (requires authentication and rate limiting)",
    dependencies=[Depends(rate_limiter)]
)(chat_completion)

# Task 3: Add API Key Management for Admins
# Create admin-only endpoints for managing API keys: `POST /admin/api-keys` to generate a new key for a given username (returns the key once), `GET /admin/api-keys` to list all keys (masked), and `DELETE /admin/api-keys/{key_id}` to revoke a key. Then add an alternative authentication method where requests can use either a JWT Bearer token OR an `X-API-Key` header to access `POST /v1/chat`. This demonstrates supporting multiple auth methods on the same endpoint.

api_keys_db: dict[str, dict] = {}

class APIKeyRequest(BaseModel):
    username: str = Field(..., description="Username to associate with the API key")

class APIKeyResponse(BaseModel):
    key_id: str
    api_key: Optional[str] = None
    username: str
    created_at: str

@app.post(
    "/admin/api-keys",
    response_model=APIKeyResponse,
    summary="Generate a new API key for a user (admin only)",
)
def create_api_key(
    request: APIKeyRequest,
    current_user: dict = Depends(admin_only),
):
    if request.username not in users_db:
        raise HTTPException(status_code=404, detail=f"User '{request.username}' not found")

    key_id = str(uuid.uuid4())[:8]
    api_key = str(uuid.uuid4())
    api_keys_db[key_id] = {
        "key_id": key_id,
        "api_key": api_key,
        "username": request.username,
        "created_at": datetime.utcnow().isoformat(),
    }
    return APIKeyResponse(**api_keys_db[key_id])

@app.get(
    "/admin/api-keys",
    response_model=List[APIKeyResponse],
    summary="List all API keys (admin only)",
)(lambda current_user: [
    APIKeyResponse(
        key_id=k["key_id"],
        username=k["username"],
        created_at=k["created_at"],
    )
    for k in api_keys_db.values()
])

@app.delete(
    "/admin/api-keys/{key_id}",
    response_model=MessageResponse,
    summary="Revoke an API key (admin only)",
)
def delete_api_key(
    key_id: str,
    current_user: dict = Depends(admin_only),
):
    if key_id not in api_keys_db:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found")
    del api_keys_db[key_id]
    return {"message": f"API key '{key_id}' revoked"}

def authenticate_api_key(api_key: str) -> Optional[dict]:
    for key in api_keys_db.values():
        if key["api_key"] == api_key:
            user = users_db.get(key["username"])
            if user:
                return {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user["role"],
                }
    return None
