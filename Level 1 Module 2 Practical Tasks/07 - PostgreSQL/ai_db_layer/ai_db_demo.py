"""
ai_db_demo.py - Complete demonstration of the database layer.
Run this script to create tables, insert sample data, and run queries.

Requirements: pip install sqlalchemy psycopg2-binary
Ensure PostgreSQL is running and the database exists.
"""

import uuid
from datetime import datetime

from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker

from database import Base, DATABASE_URL
from models import Model, Conversation, Prompt
import crud


def setup_database():
    """Create all tables in the database."""
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.drop_all(bind=engine)   # reset (for demo only!)
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully.")
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


def seed_data(db):
    """Insert sample data for demonstration."""
    print("\n--- Seeding Sample Data ---")

    # Create models
    models_data = [
        {"name": "gpt-4o", "provider": "openai", "model_type": "chat", "max_tokens": 128000},
        {"name": "gpt-4o-mini", "provider": "openai", "model_type": "chat", "max_tokens": 128000},
        {"name": "claude-sonnet-4-20250514", "provider": "anthropic", "model_type": "chat", "max_tokens": 200000},
        {"name": "text-embedding-3-large", "provider": "openai", "model_type": "embedding", "max_tokens": 8191},
        {"name": "llama-3.1-70b", "provider": "meta", "model_type": "chat", "max_tokens": 8192},
    ]
    created_models = []
    for m in models_data:
        model = crud.create_model(db, **m)
        created_models.append(model)
        print(f"  Created model: {model.name} ({model.provider})")

    # Create conversations
    user_id = "user_001"
    gpt4o = created_models[0]
    claude = created_models[2]

    conv1 = crud.create_conversation(
        db, user_id=user_id, model_id=gpt4o.id,
        title="Python Decorators Explained",
        system_prompt="You are a helpful Python tutor.",
        temperature=0.7,
    )
    print(f"  Created conversation: {conv1.id[:8]} - {conv1.title}")

    conv2 = crud.create_conversation(
        db, user_id=user_id, model_id=claude.id,
        title="Building RAG Systems",
        system_prompt="You are an AI engineering expert.",
        temperature=0.5,
    )
    print(f"  Created conversation: {conv2.id[:8]} - {conv2.title}")

    # Add messages to conversation 1
    messages = [
        ("user", "What is a decorator in Python?", 10),
        ("assistant", "A decorator is a function that wraps another function to extend its behavior...", 85),
        ("user", "Can you show an example with arguments?", 12),
        ("assistant", "Sure! Here is a decorator that accepts arguments...", 120),
    ]
    for role, content, tokens in messages:
        crud.add_prompt(db, conv1.id, role, content, token_count=tokens)
        crud.update_conversation_tokens(db, conv1.id, tokens)
    print(f"  Added {len(messages)} messages to conversation 1")

    # Add messages to conversation 2
    messages2 = [
        ("user", "How do I build a RAG system with LangChain?", 15),
        ("assistant", "To build a RAG system, you need: 1) A document loader, 2) An embedding model...", 200),
    ]
    for role, content, tokens in messages2:
        crud.add_prompt(db, conv2.id, role, content, token_count=tokens)
        crud.update_conversation_tokens(db, conv2.id, tokens)
    print(f"  Added {len(messages2)} messages to conversation 2")

    return user_id


def demonstrate_queries(db, user_id: str):
    """Run various queries to demonstrate the database layer."""
    print("\n--- Query Demonstrations ---")

    # 1. List all active models
    print("\n1. All active chat models:")
    models = crud.get_active_models(db, model_type="chat")
    for m in models:
        print(f"   {m.name:<30} | {m.provider:<12} | {m.max_tokens:>8} tokens")

    # 2. List user conversations
    print(f"\n2. Conversations for {user_id}:")
    convs = crud.get_user_conversations(db, user_id)
    for c in convs:
        print(f"   [{c.id[:8]}] {c.title:<35} | tokens: {c.total_tokens}")

    # 3. Get conversation with messages
    if convs:
        conv = crud.get_conversation_with_messages(db, convs[0].id)
        print(f"\n3. Messages in '{conv.title}':")
        for p in conv.prompts:
            preview = p.content[:60] + "..." if len(p.content) > 60 else p.content
            print(f"   [{p.role:<9}] {preview}")

    # 4. User statistics
    stats = crud.get_user_stats(db, user_id)
    print(f"\n4. User statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 5. Update a model
    print("\n5. Deactivating a model...")
    crud.deactivate_model(db, 2)  # deactivate gpt-4o-mini
    active = crud.get_active_models(db)
    print(f"   Active models after deactivation: {len(active)}")

    # 6. Search conversations by content
    print("\n6. Searching prompts containing 'decorator':")
    stmt = (
        select(Prompt)
        .where(Prompt.content.ilike("%decorator%"))
    )
    result = db.execute(stmt)
    found = list(result.scalars().all())
    for p in found:
        print(f"   [{p.role}] in conv {p.conversation_id[:8]}: {p.content[:50]}...")

    # 7. Token usage by model
    print("\n7. Token usage by model:")
    stmt = (
        select(
            Model.name,
            func.coalesce(func.sum(Conversation.total_tokens), 0).label("total_tokens"),
            func.count(Conversation.id).label("conv_count"),
        )
        .outerjoin(Conversation, Model.id == Conversation.model_id)
        .group_by(Model.name)
        .order_by(func.sum(Conversation.total_tokens).desc())
    )
    result = db.execute(stmt)
    for row in result:
        print(f"   {row.name:<30} | tokens: {row.total_tokens:>8} | convs: {row.conv_count}")


def main():
    print("=" * 60)
    print("AI Database Layer Demonstration")
    print("=" * 60)

    Session = setup_database()
    db = Session()

    try:
        user_id = seed_data(db)
        demonstrate_queries(db, user_id)
    finally:
        db.close()

    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()