from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from sqlalchemy.orm import Session

from app.auth.helper import (create_user, get_user_by_email,
                             get_user_by_username, verify_password)
from app.auth.jwt import create_access_token, get_current_user
from app.auth.models import Token, UserCreate, UserDB, UserLogin
from scripts.dbmaker import User, get_db

router = APIRouter()


@router.post("/register", response_model=UserDB, status_code=status.HTTP_201_CREATED)
def register_user(user_in: UserCreate, db: Session = Depends(get_db)):
    try:
        if get_user_by_username(db, user_in.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )

        if get_user_by_email(db, user_in.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        user = create_user(
            db=db,
            username=user_in.username,
            email=user_in.email,
            password=user_in.password,
            subscription_tier=user_in.subscription_tier,
            api_key=None,
        )

        return UserDB(
            id=user.id,
            username=user.username,
            email=user.email,
            api_key=user.api_key,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at,
            is_active=user.is_active,
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


# Authentication route using OAuth2PasswordRequestForm
@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    try:
        user = get_user_by_username(db, form_data.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not verify_password(form_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token = create_access_token(
            subject=user.username,
            user_id=user.id,
            subscription_tier=user.subscription_tier,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserDB(
                id=user.id,
                username=user.username,
                email=user.email,
                subscription_tier=user.subscription_tier,
                api_key=user.api_key,
                created_at=user.created_at,
                is_active=user.is_active,
            ),
        }
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}",
        )


@router.get("/users/me", response_model=UserDB, status_code=status.HTTP_200_OK)
def get_user_me(current_user: User = Depends(get_current_user)):
    try:
        return UserDB(
            id=current_user.id,
            username=current_user.username,
            email=current_user.email,
            subscription_tier=current_user.subscription_tier,
            api_key=current_user.api_key,
            created_at=current_user.created_at,
            is_active=current_user.is_active,
        )
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}",
        )
