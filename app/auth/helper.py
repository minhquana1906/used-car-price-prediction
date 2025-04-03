from typing import Optional

import bcrypt
from sqlalchemy.orm import Session

from scripts.dbmaker import User


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
    api_key: Optional[str] = None,
    subscription_tier: str = "Free",
) -> User:
    password_hash = get_password_hash(password)
    user = User(
        username=username,
        email=email,
        password_hash=password_hash,
        subscription_tier=subscription_tier,
        api_key=api_key,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
