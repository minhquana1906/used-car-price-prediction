from enum import PyEnum

from sqlalchemy import Column, DateTime, Enum, Integer, String, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class UserTier(str, Enum):
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"


# sqlalchemy mapping this class with table in db
class API_KEY(Base):
    id = Column(Integer, primary_key=True)
    key = Column(String(64), unique=True, index=True)
    user_id = Column(Integer)
    user_email = Column(String(100))
    tier = Column(Enum(UserTier), default=UserTier.FREE)
    created_at = Column(DateTime, default=func.now)
    last_used = Column(DateTime, nullable=True)

    def __repr__(self):
        return f
