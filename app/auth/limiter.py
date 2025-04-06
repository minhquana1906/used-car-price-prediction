import os
from datetime import datetime, timedelta
from typing import Dict, Tuple

from fastapi import Depends, HTTPException, Request
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.sql import func

from app.auth.my_jwt import get_current_user
from scripts.dbmaker import ApiUsage, SubscriptionPlan, User, get_db

try:
    redis_host = os.getenv("REDIS_HOST", "redis")
    redis_port = os.getenv("REDIS_PORT", 6379)

    redis_storage = f"redis://{redis_host}:{redis_port}"
    logger.debug(f"Configuring rate limiter with Redis: {redis_storage}")
except Exception as e:
    logger.error(f"Redis configuration error: {e}")
    redis_storage = None

SUBSCRIPTION_LIMITS = {
    1: {
        "name": "Free",
        "limit_per_minute": 10,
        "limit_per_day": 100,
        "monthly_price": 0.0,
    },
    2: {
        "name": "Basic",
        "limit_per_minute": 50,
        "limit_per_day": 1000,
        "monthly_price": 9.99,
    },
    3: {
        "name": "Premium",
        "limit_per_minute": 100,
        "limit_per_day": 5000,
        "monthly_price": 19.99,
    },
}


def get_user_id_key(request: Request) -> str:
    if hasattr(request.state, "user") and request.state.user:
        return f"user:{request.state.user.id}"
    return f"ip:{get_remote_address(request)}"


try:
    if redis_storage:
        limiter = Limiter(
            key_func=get_user_id_key,
            storage_uri=redis_storage,
            strategy="fixed-window",
        )
        if hasattr(limiter.storage, "ping"):
            limiter.storage.ping()
        logger.success("Rate limiter initialized with Redis storage")
    else:
        raise ValueError("Redis storage URL not configured")
except Exception as e:
    logger.warning(
        f"Failed to initialize Redis rate limiter: {e}. Falling back to in-memory storage."
    )
    limiter = Limiter(key_func=get_user_id_key)
    logger.warning(
        "Rate limiter initialized with in-memory storage - rate limits will not be shared across containers"
    )


subscription_cache: Dict[int, Tuple[str, int, int, float]] = {}
subscription_cache_expiry = datetime.now()
CACHE_TTL = timedelta(minutes=15)


def get_subscription_limits(
    subscription_plan_id: int, db: Session = Depends(get_db)
) -> Tuple[int, int]:
    global subscription_cache, subscription_cache_expiry

    if datetime.now() > subscription_cache_expiry:
        try:
            result = db.execute(select(SubscriptionPlan))
            plans = result.scalars().all()

            subscription_cache = {
                plan.id: (
                    plan.name,
                    plan.rate_limit_per_minute,
                    plan.rate_limit_per_day,
                    plan.monthly_price,
                )
                for plan in plans
            }

            subscription_cache_expiry = datetime.now() + CACHE_TTL
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    if subscription_plan_id in subscription_cache:
        name, minute_limit, daily_limit, price = subscription_cache[
            subscription_plan_id
        ]
        return minute_limit, daily_limit

    # not found in cache, return default limits
    if subscription_plan_id in SUBSCRIPTION_LIMITS:
        minute_limit = SUBSCRIPTION_LIMITS[subscription_plan_id]["limit_per_minute"]
        daily_limit = SUBSCRIPTION_LIMITS[subscription_plan_id]["limit_per_day"]
        return minute_limit, daily_limit

    # unknown => return default limits
    return (
        SUBSCRIPTION_LIMITS[1]["limit_per_minute"],
        SUBSCRIPTION_LIMITS[1]["limit_per_day"],
    )


def rate_limit_request(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    try:
        db = next(get_db())
        minute_limit, day_limit = get_subscription_limits(
            current_user.subscription_plan_id, db
        )

        today = datetime.now().date()
        today_start = datetime(today.year, today.month, today.day)

        result = db.execute(
            select(func.count()).where(
                ApiUsage.user_id == current_user.id,
                ApiUsage.request_timestamp >= today_start,
            )
        )
        day_usage = result.scalar() or 0

        request.state.user = current_user

        if day_usage >= day_limit:
            minute_limit = 0
            logger.warning(
                f"User {current_user.id} has reached daily limit ({day_usage}/{day_limit}), setting minute limit to 0"
            )

            raise HTTPException(
                status_code=429,
                detail=f"Daily request limit ({day_limit}) exceeded. Please try again tomorrow.",
            )

        request.state.minute_limit = minute_limit
        request.state.day_limit = day_limit
        request.state.day_usage = day_usage

        logger.debug(
            f"User {current_user.id} has limits: {minute_limit}/minute, {day_limit}/day, used: {day_usage}/day"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in rate_limit_request: {e}")

    return current_user
