from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from pydantic import EmailStr
from sqlalchemy.orm import Session

from app.auth.helper import (create_user, generate_password_reset_token,
                             get_user_by_email, get_user_by_username,
                             send_reset_password_email, update_user_password,
                             verify_password, verify_password_reset_token)
from app.auth.models import (ResetPasswordRequest, Token, UserCreate, UserDB,
                             UserLogin)
from app.auth.my_jwt import create_access_token, get_current_user
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
            subscription_plan_id=user_in.subscription_plan_id,
        )

        return UserDB(
            id=user.id,
            username=user.username,
            email=user.email,
            subscription_plan_id=user.subscription_plan_id,
            created_at=user.created_at,
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


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
            subscription_plan_id=user.subscription_plan_id,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": UserDB(
                id=user.id,
                username=user.username,
                email=user.email,
                subscription_plan_id=user.subscription_plan_id,
                created_at=user.created_at,
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
            subscription_plan_id=current_user.subscription_plan_id,
            created_at=current_user.created_at,
        )
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user: {str(e)}",
        )


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    email: EmailStr, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    try:
        user = get_user_by_email(db, email)
        if not user:
            return {"message": "If the email exists, a reset link has been sent"}

        reset_token = generate_password_reset_token(email)
        reset_link = reset_token

        background_tasks.add_task(
            send_reset_password_email,
            email_to=user.email,
            username=user.username,
            reset_link=reset_link,
        )

        return {"message": "If the email exists, a reset link has been sent"}

    except Exception as e:
        logger.error(f"Forgot password error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request",
        )


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    request: ResetPasswordRequest,
    db: Session = Depends(get_db),
):
    try:
        email = verify_password_reset_token(request.token)
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )

        user = get_user_by_email(db, email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reset request"
            )

        update_user_password(db, user, request.new_password)

        return {"message": "Password has been successfully reset"}

    except Exception as e:
        logger.error(f"Reset password error: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password",
        )
