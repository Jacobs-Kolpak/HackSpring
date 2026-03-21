"""Утилиты для сохранения действий пользователя в историю сессии."""

import json
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from backend.core.database import ChatSession, SessionFile, SessionMessage


def save_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
    message_type: str = "text",
    meta: Optional[Dict[str, Any]] = None,
) -> SessionMessage:
    msg = SessionMessage(
        session_id=session_id,
        role=role,
        content=content,
        message_type=message_type,
        meta=json.dumps(meta, ensure_ascii=False) if meta else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def save_file(
    db: Session,
    session_id: int,
    filename: str,
    file_size: Optional[int] = None,
    collection: Optional[str] = None,
) -> SessionFile:
    f = SessionFile(
        session_id=session_id,
        filename=filename,
        file_size=file_size,
        collection=collection,
    )
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


def get_or_create_session(
    db: Session,
    user_id: int,
    session_id: Optional[int] = None,
    title: str = "Новая сессия",
) -> ChatSession:
    if session_id:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )
        if session:
            return session

    session = ChatSession(user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session
