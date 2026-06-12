"""
crud.py - Create, Read, Update, Delete operations using SQLAlchemy ORM.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from sqlalchemy import select, func, update, delete

from models import Model, Conversation, Prompt


# ==================================================================
# Model CRUD
# ==================================================================

def create_model(db: Session, name: str, provider: str, model_type: str,
                 max_tokens: int = 4096) -> Model:
    """Create a new AI model record."""
    model = Model(
        name=name,
        provider=provider,
        model_type=model_type,
        max_tokens=max_tokens,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return model


def get_model_by_name(db: Session, name: str) -> Optional[Model]:
    """Get a model by its name."""
    stmt = select(Model).where(Model.name == name)
    result = db.execute(stmt)
    return result.scalar_one_or_none()


def get_active_models(db: Session, model_type: Optional[str] = None) -> list[Model]:
    """Get all active models, optionally filtered by type."""
    stmt = select(Model).where(Model.is_active == True)
    if model_type:
        stmt = stmt.where(Model.model_type == model_type)
    stmt = stmt.order_by(Model.name)
    result = db.execute(stmt)
    return list(result.scalars().all())


def update_model(db: Session, model_id: int, **kwargs) -> Optional[Model]:
    """Update a model's fields."""
    model = db.get(Model, model_id)
    if not model:
        return None
    for key, value in kwargs.items():
        if hasattr(model, key):
            setattr(model, key, value)
    model.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(model)
    return model


def deactivate_model(db: Session, model_id: int) -> bool:
    """Soft-delete a model by setting is_active to False."""
    stmt = (
        update(Model)
        .where(Model.id == model_id)
        .values(is_active=False, updated_at=datetime.utcnow())
    )
    result = db.execute(stmt)
    db.commit()
    return result.rowcount > 0


# ==================================================================
# Conversation CRUD
# ==================================================================

def create_conversation(db: Session, user_id: str, model_id: Optional[int],
                        title: Optional[str] = None,
                        system_prompt: Optional[str] = None,
                        temperature: float = 0.7) -> Conversation:
    """Create a new conversation."""
    conversation = Conversation(
        user_id=user_id,
        model_id=model_id,
        title=title,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def get_user_conversations(db: Session, user_id: str,
                           limit: int = 20, offset: int = 0) -> list[Conversation]:
    """Get paginated conversations for a user, newest first."""
    stmt = (
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = db.execute(stmt)
    return list(result.scalars().all())


def get_conversation_with_messages(db: Session, conversation_id: str) -> Optional[Conversation]:
    """Get a conversation with all its prompts eagerly loaded."""
    stmt = (
        select(Conversation)
        .where(Conversation.id == conversation_id)
    )
    result = db.execute(stmt)
    conversation = result.scalar_one_or_none()
    if conversation:
        # Access prompts to trigger loading (or use joinedload)
        _ = conversation.prompts
    return conversation


def update_conversation_tokens(db: Session, conversation_id: str,
                                tokens_used: int) -> bool:
    """Add tokens to a conversation's total."""
    stmt = (
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(
            total_tokens=Conversation.total_tokens + tokens_used,
            updated_at=datetime.utcnow(),
        )
    )
    result = db.execute(stmt)
    db.commit()
    return result.rowcount > 0


def delete_conversation(db: Session, conversation_id: str) -> bool:
    """Delete a conversation and all its prompts (cascade)."""
    stmt = delete(Conversation).where(Conversation.id == conversation_id)
    result = db.execute(stmt)
    db.commit()
    return result.rowcount > 0


# ==================================================================
# Prompt CRUD
# ==================================================================

def add_prompt(db: Session, conversation_id: str, role: str, content: str,
               token_count: int = 0, latency_ms: Optional[float] = None) -> Prompt:
    """Add a message to a conversation."""
    prompt = Prompt(
        conversation_id=conversation_id,
        role=role,
        content=content,
        token_count=token_count,
        latency_ms=latency_ms,
    )
    db.add(prompt)
    db.commit()
    db.refresh(prompt)
    return prompt


def get_conversation_messages(db: Session, conversation_id: str) -> list[Prompt]:
    """Get all messages in a conversation, ordered by time."""
    stmt = (
        select(Prompt)
        .where(Prompt.conversation_id == conversation_id)
        .order_by(Prompt.created_at)
    )
    result = db.execute(stmt)
    return list(result.scalars().all())


# ==================================================================
# Analytics Queries
# ==================================================================

def get_user_stats(db: Session, user_id: str) -> dict:
    """Get usage statistics for a user."""
    conv_count = db.execute(
        select(func.count()).select_from(Conversation).where(Conversation.user_id == user_id)
    ).scalar() or 0

    total_tokens = db.execute(
        select(func.coalesce(func.sum(Conversation.total_tokens), 0))
        .where(Conversation.user_id == user_id)
    ).scalar() or 0

    avg_latency = db.execute(
        select(func.coalesce(func.avg(Prompt.latency_ms), 0.0))
        .join(Conversation, Prompt.conversation_id == Conversation.id)
        .where(Conversation.user_id == user_id)
        .where(Prompt.role == "assistant")
    ).scalar() or 0.0

    return {
        "user_id": user_id,
        "total_conversations": conv_count,
        "total_tokens": int(total_tokens),
        "avg_latency_ms": round(float(avg_latency), 2),
    }
