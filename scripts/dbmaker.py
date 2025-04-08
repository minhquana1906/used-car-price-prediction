import os

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        Numeric, String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

load_dotenv()

DATABASE_URL = os.getenv("APP_DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    subscription_plan_id = Column(
        Integer, ForeignKey("subscription_plans.id"), nullable=False, default=1
    )
    created_at = Column(DateTime, default=func.now())

    subscription_plans = relationship("SubscriptionPlan", back_populates="users")
    api_usage = relationship("ApiUsage", back_populates="users")


class Car(Base):
    __tablename__ = "cars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    brand = Column(String(50), nullable=False)
    model = Column(String(50), nullable=False)
    vehicle_type = Column(String(50), nullable=False)
    fuel_type = Column(String(50), nullable=False)
    power_ps = Column(Float, nullable=False)
    kilometer = Column(Float, nullable=False)
    year_of_registration = Column(Integer, nullable=False)
    gearbox = Column(String(20), nullable=False)
    not_repaired_damage = Column(String(10), nullable=False)
    price = Column(Numeric(12, 2), nullable=False)


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20), unique=True, nullable=False)
    rate_limit_per_minute = Column(Integer, nullable=False)
    rate_limit_per_day = Column(Integer, nullable=False)
    monthly_price = Column(Numeric(10, 2), nullable=False)
    description = Column(Text, nullable=True)

    users = relationship("User", back_populates="subscription_plans")


class ApiUsage(Base):
    __tablename__ = "api_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    request_timestamp = Column(DateTime, default=func.now())
    endpoint = Column(String(100), nullable=False)
    status_code = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)
    response_time_ms = Column(Float, nullable=True)
    predicted_price = Column(Numeric(12, 2), nullable=True)
    is_cached = Column(Boolean, default=False)

    users = relationship("User", back_populates="api_usage")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.success("Database initialized successfully")


def create_subscription_plans():
    db = SessionLocal()
    try:
        existing_plans = db.query(SubscriptionPlan).count()
        if existing_plans > 0:
            logger.info("Subscription plans already exist")
            return

        plans = [
            SubscriptionPlan(
                name="Free",
                rate_limit_per_minute=10,
                rate_limit_per_day=100,
                monthly_price=0.0,
                description="Free tier with limited API calls",
            ),
            SubscriptionPlan(
                name="Basic",
                rate_limit_per_minute=50,
                rate_limit_per_day=1000,
                monthly_price=9.99,
                description="Basic tier for regular users",
            ),
            SubscriptionPlan(
                name="Premium",
                rate_limit_per_minute=100,
                rate_limit_per_day=5000,
                monthly_price=19.99,
                description="Premium tier for professional users",
            ),
        ]

        db.add_all(plans)
        db.commit()
        logger.success("Subscription plans created successfully")
    except Exception as e:
        logger.error(f"Error creating subscription plans: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    create_subscription_plans()
