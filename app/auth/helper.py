import os
from datetime import timedelta
from typing import Optional

import bcrypt
import jwt
from jose import JWTError
from postmarker.core import PostmarkClient
from sqlalchemy.orm import Session

from scripts.dbmaker import User

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

POSTMARK_SERVER_TOKEN = os.getenv("POSTMARK_SERVER_TOKEN")
SMTP_SENDER_EMAIL = os.getenv("SMTP_SENDER_EMAIL")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter_by(username=username).first()


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter_by(email=email).first()


def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    subscription_plan_id: int = 1,
) -> User:
    password_hash = get_password_hash(password)
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        subscription_plan_id=subscription_plan_id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def generate_password_reset_token(email: str) -> str:
    from app.auth.my_jwt import create_access_token
    return create_access_token(
        subject=email,
        user_id=0,
        subscription_plan_id=1,
        purpose="reset_password",
        expires_delta=timedelta(hours=1)
    )

def verify_password_reset_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("purpose") != "reset_password":
            return None
        return payload.get("sub")
    except JWTError:
        return None

def update_user_password(db: Session, user: User, new_password: str) -> None:
    hashed_password = get_password_hash(new_password)
    user.password_hash = hashed_password
    db.commit()
    db.refresh(user)

def send_reset_password_email(email_to: str, username: str, reset_link: str):
    postmark = PostmarkClient(server_token=POSTMARK_SERVER_TOKEN)
    try:
        response = postmark.emails.send(
            From=SMTP_SENDER_EMAIL,
            To=email_to,
            Subject="Password Reset Request",
            TextBody=f"""
            Hello {username},

            You have requested to reset your password.
            Please copy thí token to reset your password:
            {reset_link}

            This token will expire in 15 minutes.
            If you did not request this, please ignore this email.
            """
        )
        print(f"Email sent to {email_to}: {response['Message']}")
    except Exception as e:
        print(f"Failed to send email to {email_to}: {str(e)}")
        raise
