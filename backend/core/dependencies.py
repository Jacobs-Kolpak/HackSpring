from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.core.database import User, get_db
from backend.core.security import verify_token

security = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    email = verify_token(credentials.credentials, "access")
    if email is None:
        raise exc
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise exc
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user
