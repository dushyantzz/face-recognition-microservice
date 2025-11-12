"""Database connection and session management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
from typing import Generator
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database.models import Base


class Database:
    """Database connection manager"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or settings.database_url
        
        if self.database_url.startswith("sqlite"):
            self.engine = create_engine(
                self.database_url,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        else:
            self.engine = create_engine(self.database_url, pool_pre_ping=True)
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all tables"""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


db = Database()


def get_db() -> Generator[Session, None, None]:
    """Dependency for FastAPI routes"""
    session = db.SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db():
    """Initialize database"""
    db.create_tables()
    print("✅ Database initialized successfully!")


if __name__ == "__main__":
    init_db()