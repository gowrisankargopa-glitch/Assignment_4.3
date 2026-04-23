"""
Database configuration and models using SQLAlchemy
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_URL = "sqlite:///./housing_predictions.db"

# Create database engine
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=False
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class InferenceRecord(Base):
    """
    SQLAlchemy model for storing inference records in the database
    """
    __tablename__ = "inference_records"

    id = Column(Integer, primary_key=True, index=True)
    
    # Input features
    longitude = Column(Float, nullable=False)
    latitude = Column(Float, nullable=False)
    housing_median_age = Column(Float, nullable=False)
    total_rooms = Column(Float, nullable=False)
    total_bedrooms = Column(Float, nullable=False)
    population = Column(Float, nullable=False)
    households = Column(Float, nullable=False)
    median_income = Column(Float, nullable=False)
    ocean_proximity = Column(String, nullable=True)
    
    # Model and prediction information
    model_name = Column(String, nullable=False, index=True)
    prediction = Column(Float, nullable=False)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """
    Initialize the database by creating all tables
    """
    Base.metadata.create_all(bind=engine)
    print("✓ Database initialized successfully")


def get_db():
    """
    Dependency injection function for database sessions
    Usage: In FastAPI route, add parameter: db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
