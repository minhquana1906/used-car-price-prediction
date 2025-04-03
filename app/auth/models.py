from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)
    subscription_tier: str = "Free"


class UserLogin(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)


class UserDB(UserBase):
    id: int
    subscription_tier: str
    api_key: Optional[str] = None
    created_at: datetime
    is_active: bool = True

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserDB


class TokenPayLoad(BaseModel):
    sub: str
    exp: float
    user_id: int
    subscription_tier: str
