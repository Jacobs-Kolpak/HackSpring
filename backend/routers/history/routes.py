import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.core.database import ChatSession, SessionFile, SessionMessage, User, get_db
from backend.core.dependencies import get_current_user

router = APIRouter(prefix="/api/jacobs/history", tags=["History"])


# ── Schemas ────────────────────────────────────────────────

class SessionCreate(BaseModel):
    title: Optional[str] = "Новая сессия"


class SessionUpdate(BaseModel):
    title: str


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    message_type: str
    meta: Optional[Dict[str, Any]] = None
    created_at: str

    class Config:
        from_attributes = True


class FileOut(BaseModel):
    id: int
    filename: str
    file_size: Optional[int] = None
    collection: Optional[str] = None
    uploaded_at: str

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: int
    title: str
    created_at: str
    updated_at: str
    files_count: int = 0
    messages_count: int = 0

    class Config:
        from_attributes = True


class SessionDetailOut(SessionOut):
    messages: List[MessageOut] = []
    files: List[FileOut] = []


# ── Endpoints ──────────────────────────────────────────────

@router.get("/sessions", response_model=List[SessionOut])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [
        SessionOut(
            id=s.id,
            title=s.title,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
            files_count=len(s.files),
            messages_count=len(s.messages),
        )
        for s in sessions
    ]


@router.post("/sessions", response_model=SessionOut, status_code=201)
async def create_session(
    body: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = ChatSession(user_id=current_user.id, title=body.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return SessionOut(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
    )


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _get_user_session(db, session_id, current_user.id)
    messages = [
        MessageOut(
            id=m.id,
            role=m.role,
            content=m.content,
            message_type=m.message_type,
            meta=json.loads(m.meta) if m.meta else None,
            created_at=m.created_at.isoformat(),
        )
        for m in session.messages
    ]
    files = [
        FileOut(
            id=f.id,
            filename=f.filename,
            file_size=f.file_size,
            collection=f.collection,
            uploaded_at=f.uploaded_at.isoformat(),
        )
        for f in session.files
    ]
    return SessionDetailOut(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        files_count=len(files),
        messages_count=len(messages),
        messages=messages,
        files=files,
    )


@router.patch("/sessions/{session_id}", response_model=SessionOut)
async def update_session(
    session_id: int,
    body: SessionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _get_user_session(db, session_id, current_user.id)
    session.title = body.title
    db.commit()
    db.refresh(session)
    return SessionOut(
        id=session.id,
        title=session.title,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        files_count=len(session.files),
        messages_count=len(session.messages),
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    session = _get_user_session(db, session_id, current_user.id)
    db.delete(session)
    db.commit()


# ── Helpers ────────────────────────────────────────────────

def _get_user_session(db: Session, session_id: int, user_id: int) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if not session:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return session
