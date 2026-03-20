"""База данных: SQLAlchemy engine, сессии, модели."""

import enum
from typing import Generator

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from sqlalchemy.sql import func

from backend.core.config import settings


class UserRole(str, enum.Enum):
    """Доступные роли пользователей."""

    RESEARCHER = "researcher"
    GOVERNMENT = "government"
    STUDENT = "student"


engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.DATABASE_URL
    else {},
)

session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):  # type: ignore[valid-type, misc]
    """Модель пользователя."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role: Column[str] = Column(
        Enum(UserRole), nullable=False, default=UserRole.STUDENT
    )
    is_active = Column(Boolean, default=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),  # pylint: disable=not-callable
    )
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),  # pylint: disable=not-callable
    )


def get_db() -> Generator[Session, None, None]:
    """Yield a database session."""
    db = session_local()
    try:
        yield db
    finally:
        db.close()
