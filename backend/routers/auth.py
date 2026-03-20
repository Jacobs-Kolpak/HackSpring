"""Роутер авторизации: register, login, refresh, me, logout."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.core.database import User, get_db
from backend.core.security import (
    create_tokens,
    hash_password,
    verify_password,
    verify_token,
)
from backend.schemas import Token, UserCreate, UserLogin, UserResponse, UserStatus

security = HTTPBearer()
router = APIRouter(tags=["Authentication"])


# ── DB helpers ──────────────────────────────────────────────


def _get_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()


def _get_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def _authenticate(db: Session, email: str, password: str) -> Optional[User]:
    user = _get_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):  # type: ignore[arg-type]
        return None
    return user


# ── Dependencies ────────────────────────────────────────────


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Текущий пользователь из access token."""
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = verify_token(credentials.credentials, "access")
    if email is None:
        raise exc
    user = _get_by_email(db, email)
    if user is None:
        raise exc
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


def _verify_refresh(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = verify_token(credentials.credentials, "refresh")
    if email is None:
        raise exc
    user = _get_by_email(db, email)
    if user is None:
        raise exc
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


# ── Endpoints ───────────────────────────────────────────────


@router.post("/register", response_model=UserResponse, status_code=201)
async def register(data: UserCreate, db: Session = Depends(get_db)):
    """Регистрация нового пользователя."""
    if _get_by_email(db, data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if _get_by_username(db, data.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    try:
        user = User(
            email=data.email,
            username=data.username,
            hashed_password=hash_password(data.password),
            role=data.role.value,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail="Registration failed") from exc


@router.post("/login", response_model=Token)
async def login(creds: UserLogin, db: Session = Depends(get_db)):
    """Логин — возвращает access + refresh токены."""
    user = _authenticate(db, creds.email, creds.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return create_tokens(user.email)  # type: ignore[arg-type]


@router.post("/refresh", response_model=Token)
async def refresh(current_user=Depends(_verify_refresh)):
    """Обновление access токена через refresh."""
    return create_tokens(current_user.email)


@router.get("/me", response_model=UserStatus)
async def me(current_user=Depends(get_current_user)):
    """Информация о текущем пользователе."""
    return UserStatus(user=current_user, is_authenticated=True)


@router.post("/logout", status_code=200)
async def logout():
    """Выход (клиент удаляет токены)."""
    return {"message": "Logged out. Remove tokens from client storage."}


@router.get("/status")
async def auth_status():
    """Статус системы авторизации."""
    return {"status": "Auth system is running", "version": "1.0.0"}
