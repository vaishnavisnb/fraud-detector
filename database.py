"""
Database setup for the fraud API.

By default it uses a local SQLite file (fraud.db), which needs no installation.
To use PostgreSQL later, just set the DATABASE_URL environment variable, for example:
    postgresql://user:password@localhost:5432/fraud
"""

import os
from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fraud.db")

# Newer SQLAlchemy versions pick a different PostgreSQL driver by default, so name the one we install
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# SQLite needs this extra setting because FastAPI uses several threads
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class PredictionRecord(Base):
    """One row for every transaction the API has scored."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    amount = Column(Float)
    fraud_probability = Column(Float)
    is_fraud = Column(Boolean)
    threshold = Column(Float)
    payload = Column(Text)  # the full transaction, saved as JSON text


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    """Gives each request its own database session and closes it afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
