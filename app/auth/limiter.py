from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.auth.my_jwt import get_current_user
from scripts.dbmaker import SubscriptionPlan, User, get_db

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

limiter = Limiter(key_func=get_remote_address)

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
    db: Session = Depends(get_db),
):
    """Dependency to enforce rate limits based on user's subscription tier"""
    tier_id = current_user.subscription_plan_id
    minute_limit, day_limit = get_subscription_limits(tier_id, db)

    request.state.view_rate_limit = f"{minute_limit}/minute;{day_limit}/day"

    return current_user
