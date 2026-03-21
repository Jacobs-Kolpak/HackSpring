from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    raw: Union[str, bytes] = password
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if len(raw) > 72:
        raw = raw[:72]
    result: str = pwd_context.hash(raw)
    return result


def verify_password(plain_password: str, hashed_password: str) -> bool:
    raw: Union[str, bytes] = plain_password
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if len(raw) > 72:
        raw = raw[:72]
    result: bool = pwd_context.verify(raw, hashed_password)
    return result


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire, "token_type": "access"})
    encoded_jwt: str = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta
        or timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    )
    to_encode.update({"exp": expire, "token_type": "refresh"})
    encoded_jwt: str = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_tokens(user_email: str) -> Dict[str, str]:
    return {
        "access_token": create_access_token(data={"sub": user_email}),
        "refresh_token": create_refresh_token(data={"sub": user_email}),
        "token_type": "bearer",
    }


def verify_token(
    token: str, expected_token_type: str = "access"
) -> Optional[str]:
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        email: Optional[str] = payload.get("sub")
        token_type: Optional[str] = payload.get("token_type")
        if email is None or token_type != expected_token_type:
            return None
        return email
    except JWTError:
        return None
