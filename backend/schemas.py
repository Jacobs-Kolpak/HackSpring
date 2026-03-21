from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, EmailStr


class UserRoleEnum(str, Enum):
    RESEARCHER = "researcher"
    GOVERNMENT = "government"
    STUDENT = "student"


class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str
    role: UserRoleEnum = UserRoleEnum.STUDENT


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(UserBase):
    id: int
    role: UserRoleEnum
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    token_type: Optional[str] = None


class UserStatus(BaseModel):
    user: UserResponse
    is_authenticated: bool = True
