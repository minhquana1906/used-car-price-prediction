import os
import random
import string
import uuid
from datetime import datetime, timedelta

import bcrypt
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, Integer,
                        Numeric, String, Text, create_engine)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

load_dotenv()

DATABASE_URL = "mysql+pymysql://root:rootadmin123@localhost:3306/used_car_price"


# SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    subscription_tier = Column(String(20), nullable=False, default="Free")
    api_key = Column(String(64), unique=True, nullable=True)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    inference_requests = relationship("InferenceRequest", back_populates="user")
    api_usages = relationship("ApiUsage", back_populates="user")


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

    inference_requests = relationship("InferenceRequest", back_populates="car")


class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(20), unique=True, nullable=False)
    rate_limit_per_minute = Column(Integer, nullable=False)
    rate_limit_per_day = Column(Integer, nullable=False)
    monthly_price = Column(Numeric(10, 2), nullable=False)
    description = Column(Text, nullable=True)


class InferenceRequest(Base):
    __tablename__ = "inference_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    car_id = Column(Integer, ForeignKey("cars.id"), nullable=False)
    request_time = Column(DateTime, default=func.now())
    predicted_price = Column(Numeric(12, 2), nullable=True)
    is_cached = Column(Boolean, default=False)

    # Relationships
    user = relationship("User", back_populates="inference_requests")
    car = relationship("Car", back_populates="inference_requests")


class ApiUsage(Base):
    __tablename__ = "api_usage"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    endpoint = Column(String(100), nullable=False)
    request_timestamp = Column(DateTime, default=func.now())
    response_time_ms = Column(Float, nullable=True)
    status_code = Column(Integer, nullable=True)
    ip_address = Column(String(45), nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_usages")


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


def generate_random_string(length=16):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def generate_hash_password(password):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_users(count=50):
    db = SessionLocal()
    try:
        # Check if users already exist
        # existing_users = db.query(User).count()
        # if existing_users > 0:
        #     logger.info(f"{existing_users} users already exist in the database")
        #     return

        # Get subscription plans
        subscription_plans = db.query(SubscriptionPlan).all()
        if not subscription_plans:
            logger.error(
                "No subscription plans found. Please run create_subscription_plans() first."
            )
            return

        plan_names = [plan.name for plan in subscription_plans]

        # Create users with weighted distribution among subscription tiers
        # Free: 60%, Basic: 30%, Premium: 10%
        weights = [0.6, 0.3, 0.1]

        users = []
        for i in range(1, count + 1):
            # Determine subscription tier based on weights
            tier = random.choices(plan_names, weights=weights, k=1)[0]

            # Generate a unique username
            username = f"user_{i}"
            while db.query(User).filter(User.username == username).first():
                username = f"user_{i}_{generate_random_string(5)}"

            # Generate email based on username
            email = f"{username}@example.com"

            # Generate API key (for some users)
            # api_key = str(uuid.uuid4()) if random.random() < 0.8 else None

            # Create user
            user = User(
                username=username,
                password_hash=generate_hash_password(f"password{i}"),
                email=email,
                subscription_tier=tier,
                # api_key=api_key,
                created_at=datetime.now() - timedelta(days=random.randint(1, 365)),
            )

            users.append(user)

        db.add_all(users)
        db.commit()
        logger.success(f"Created {count} sample users")

        # Log distribution
        for tier in plan_names:
            count = db.query(User).filter(User.subscription_tier == tier).count()
            logger.info(f"Users with {tier} subscription: {count}")

    except Exception as e:
        logger.error(f"Error creating sample users: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    create_subscription_plans()
    # create_users(10)
