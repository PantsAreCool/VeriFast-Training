"""
database.py - Database connection configuration using SQLAlchemy 2.0.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Mapped, mapped_column
from sqlalchemy.pool import QueuePool
import os

# Synchronous connection URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://aiuser:aipass@localhost:5432/aidb"
)

# Async connection URL (replace postgresql:// with postgresql+asyncpg://)
ASYNC_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://", "postgresql+asyncpg://"
)


# ---- Synchronous Engine (for scripts and background tasks) ----
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,          # number of permanent connections
    max_overflow=20,       # additional connections during spikes
    pool_timeout=30,       # seconds to wait for a connection
    pool_recycle=3600,     # recycle connections after 1 hour
    echo=False,            # Set True for SQL logging (debugging)
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ---- Base Class for ORM Models ----
class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


# ---- Dependency for FastAPI ----
def get_db():
    """FastAPI dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()