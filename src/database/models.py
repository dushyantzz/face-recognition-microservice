"""SQLAlchemy database models"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Identity(Base):
    """Identity/Person model"""
    __tablename__ = "identities"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    embeddings = relationship("Embedding", back_populates="identity", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Identity(id={self.id}, name='{self.name}')>"


class Embedding(Base):
    """Face embedding model"""
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    identity_id = Column(Integer, ForeignKey("identities.id"), nullable=False)
    embedding = Column(JSON, nullable=False)
    image_path = Column(String(255), nullable=True)
    quality_score = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    identity = relationship("Identity", back_populates="embeddings")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, identity_id={self.identity_id})>"


class RecognitionLog(Base):
    """Log of recognition attempts for analytics"""
    __tablename__ = "recognition_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    identity_id = Column(Integer, ForeignKey("identities.id"), nullable=True)
    confidence = Column(Float, nullable=True)
    image_path = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    recognized = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<RecognitionLog(id={self.id}, identity_id={self.identity_id})>"