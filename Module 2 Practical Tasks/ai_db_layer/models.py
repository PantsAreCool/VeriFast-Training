"""
models.py - SQLAlchemy ORM models for the AI application.
"""

import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy import String, Integer, Float, Boolean, Text, ForeignKey, DateTime, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Model(Base):
    """AI Model configuration table."""
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    max_tokens: Mapped[int] = mapped_column(Integer, default=4096)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    conversations: Mapped[List["Conversation"]] = relationship(back_populates="model")

    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.name}', provider='{self.provider}')>"


class Conversation(Base):
    """Chat conversation session table."""
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("models.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    model: Mapped[Optional["Model"]] = relationship(back_populates="conversations")
    prompts: Mapped[List["Prompt"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Prompt.created_at",
    )

    __table_args__ = (
        Index("idx_conversations_user_created", "user_id", "created_at"),
    )

    def __repr__(self):
        return f"<Conversation(id={self.id[:8]}, title='{self.title}')>"


class Prompt(Base):
    """Individual message within a conversation."""
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    conversation: Mapped["Conversation"] = relationship(back_populates="prompts")

    __table_args__ = (
        Index("idx_prompts_conversation", "conversation_id"),
        Index("idx_prompts_conv_role", "conversation_id", "role"),
    )

    def __repr__(self):
        return f"<Prompt(id={self.id}, role='{self.role}', conv={self.conversation_id[:8]})>"