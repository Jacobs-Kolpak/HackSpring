from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.database import User, get_db
from backend.schemas import Token, UserCreate, UserLogin, UserResponse, UserStatus
from backend.security import create_tokens, hash_password, verify_password, verify_token

security = HTTPBearer()
router = APIRouter(tags=["Authentication"])


# Database utility functions
def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email from database."""
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    """Get user by username from database."""
    return db.query(User).filter(User.username == username).first()


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):  # type: ignore[arg-type]
        return None
    return user


def create_user(
    db: Session, email: str, username: str, password: str, role: str = "student"
) -> User:
    """Create a new user in the database."""
    hashed_password = hash_password(password)
    db_user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        role=role,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# Authentication dependencies
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Get the current authenticated user from the access token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    email = verify_token(token, expected_token_type="access")

    if email is None:
        raise credentials_exception

    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    return user


def verify_refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Verify the refresh token and return the associated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate refresh token",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token = credentials.credentials
    email = verify_token(token, expected_token_type="refresh")

    if email is None:
        raise credentials_exception

    user = get_user_by_email(db, email=email)
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    return user


# API Routes
@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    if get_user_by_email(db, user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    if get_user_by_username(db, user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )

    try:
        db_user = create_user(
            db=db,
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            role=user_data.role.value,
        )
        return db_user

    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User registration failed",
        ) from exc


@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user and return JWT tokens."""
    user = authenticate_user(db, user_credentials.email, user_credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )

    # Создаем токены (access token - кратковременный, refresh token - долгосрочный)
    tokens = create_tokens(user.email)  # type: ignore[arg-type]
    return tokens


@router.post("/refresh", response_model=Token)
async def refresh_token(current_user=Depends(verify_refresh_token)):
    """
    Обновление access токена с помощью refresh токена.
    Позволяет переключаться с временной на постоянную аутентификацию.
    """
    tokens = create_tokens(current_user.email)
    return tokens


@router.get("/me", response_model=UserStatus)
async def get_user_status(current_user=Depends(get_current_user)):
    """
    Получение информации о текущем пользователе и проверка авторизации.
    Основной маршрут для проверки статуса аутентификации в приложении.
    """
    return UserStatus(
        user=current_user,
        is_authenticated=True
    )


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout():
    """
    Выход из системы.
    Поскольку мы используем JWT токены без состояния на сервере,
    клиент должен удалить токены локально.
    """
    return {"message": "Successfully logged out. Please remove tokens from client storage."}


@router.get("/status")
async def auth_status():
    """Return the current status of the auth system."""
    return {
        "status": "AuthKeyHub authentication system is running",
        "version": "1.0.0"
    }
