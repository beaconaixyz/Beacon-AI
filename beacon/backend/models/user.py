"""
User model for BEACON API

This module defines the User model and related schemas.
"""

from sqlalchemy import Boolean, Column, Integer, String, ARRAY
from sqlalchemy.orm import relationship
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from beacon.backend.db.database import Base

class User(Base):
    """User database model."""
    
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    permissions = Column(ARRAY(String), default=[])

class UserBase(BaseModel):
    """Base user schema."""
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: Optional[bool] = False
    permissions: List[str] = []

class UserCreate(UserBase):
    """User creation schema."""
    password: str

class UserUpdate(UserBase):
    """User update schema."""
    password: Optional[str] = None

class UserInDB(UserBase):
    """User database schema."""
    id: int
    hashed_password: str

    class Config:
        orm_mode = True

class UserResponse(UserBase):
    """User response schema."""
    id: int

    class Config:
        orm_mode = True

class Token(BaseModel):
    """Token schema."""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None 