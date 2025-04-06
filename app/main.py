import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pandas as pd
import redis.asyncio as redis
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from joblib import load
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy import func, select

from app.auth.limiter import (get_subscription_limits, limiter,
                              rate_limit_request)
from app.auth.my_jwt import get_current_user
from app.auth.router import router as auth_router
from app.middleware import APIUsageMiddleware, AuthMiddleware
from scripts.dbmaker import ApiUsage, User, get_db
from utils.api_helper import (CarModel, get_redis_cache_async, preprocess_data,
                              run_in_threadpool, set_redis_cache_async)

model_pipeline = None
redis_client = None
error_margin = None

FULL_PIPELINE_PATH = "./models/full_pipeline.pkl"
METRICS_PATH = "./logs/metrics/model_metrics.csv"
DEFAULT_ERROR_MARGIN = 500
DEFAULT_EXPIRATION_TIME = 60 * 60


logger.add("logs/api.log", rotation="500 MB", level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline, redis_client, error_margin

    if os.path.exists(FULL_PIPELINE_PATH):
        logger.info(f"Loading full pipeline from {FULL_PIPELINE_PATH}")
        model_pipeline = await run_in_threadpool(load, FULL_PIPELINE_PATH)
        logger.success("Model pipeline loaded successfully")
    else:
        logger.error(f"Pipeline file not found at {FULL_PIPELINE_PATH}")
        raise RuntimeError("Model pipeline not found")

    if os.path.exists(METRICS_PATH):
        metrics_df = await run_in_threadpool(pd.read_csv, METRICS_PATH)
        error_margin = (
            float(metrics_df["MAE"].iloc[0])
            if "MAE" in metrics_df.columns
            else DEFAULT_ERROR_MARGIN
        )
        logger.info(f"Using error margin: {error_margin}")
    else:
        logger.warning("Metrics file not found, using default error margin")
        error_margin = DEFAULT_ERROR_MARGIN

    try:
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = os.getenv("REDIS_PORT", 6379)
        redis_url = f"redis://{redis_host}:{redis_port}"
        redis_client = redis.from_url(redis_url, decode_responses=True)
        await redis_client.ping()
        logger.success(f"Connected to Redis at {redis_url}")
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        redis_client = None

    # endpoint to start each time the server reloads
    yield

    logger.info("Shutting down API...")
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to help users making decision to buy a used car.",
    version="0.1",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(AuthMiddleware)
app.add_middleware(APIUsageMiddleware)
app.include_router(auth_router, prefix="/auth", tags=["authentication"])


@app.get("/", tags=["health"])
def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to the Used Car Price Prediction API"},
    )


@app.get("/status", tags=["health"])
def status():
    return JSONResponse(status_code=200, content={"model": model_pipeline is not None})


@app.post("/predict", tags=["api"])
async def predict(
    request: Request, car: CarModel, current_user: User = Depends(rate_limit_request)
):
    global model_pipeline, error_margin

    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        data = car.model_dump()
        is_cached = False
        cache_key = f"car_prediction:{hash(frozenset(data.items()))}"

        if redis_client:
            cached_result = await get_redis_cache_async(redis_client, cache_key)
            if cached_result:
                logger.info("Cache hit! Returning cached result.")
                is_cached = True
                result = eval(cached_result)

                request.state.prediction_result = {
                    "predicted_price": result["predicted price"],
                    "is_cached": True,
                }

                return JSONResponse(
                    status_code=200,
                    content=result,
                )

        processed_data = preprocess_data(data)

        prediction = await run_in_threadpool(
            lambda: model_pipeline.predict(processed_data)
        )
        prediction_value = float(prediction[0])
        logger.info(f"Raw prediction: {prediction_value}")

        lower_bound = max(0, prediction_value - error_margin)
        upper_bound = prediction_value + error_margin

        result = {
            "predicted price": round(prediction_value, 2),
            "error margin": round(error_margin, 2),
            "acceptable range": f"{round(lower_bound, 2)} - {round(upper_bound, 2)}",
        }

        request.state.prediction_result = {
            "predicted_price": round(prediction_value, 2),
            "is_cached": False,
        }

        if redis_client:
            await set_redis_cache_async(
                redis_client, cache_key, str(result), DEFAULT_EXPIRATION_TIME
            )
            logger.info(f"Cached result: {result}")

        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/cached-predictions", tags=["api"])
async def get_cached_predictions(
    request: Request, current_user: User = Depends(rate_limit_request)
):
    if redis_client is None:
        return {"status": "Redis not available"}

    try:
        keys = await redis_client.keys("car_prediction:*")
        result = {}
        for key in keys:
            ttl = await redis_client.ttl(key)
            value = await redis_client.get(key)
            result[key] = {"value": eval(value), "ttl_seconds": ttl}
        return result
    except Exception as e:
        logger.error(f"Cache retrieval error: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.delete("/cached-predictions", tags=["api"])
async def delete_cached_predictions(
    request: Request, current_user: User = Depends(get_current_user)
):
    if redis_client is None:
        return {"status": "Redis not available"}

    try:
        keys = await redis_client.keys("car_prediction:*")
        if keys:
            await redis_client.delete(*keys)
            logger.info("Deleted all cached predictions")
            return {"status": "success", "message": "All cached predictions deleted"}
        else:
            return {"status": "success", "message": "No cached predictions found"}
    except Exception as e:
        logger.error(f"Cache deletion error: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/subscription-limits", tags=["api"])
def get_limits(
    request: Request,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    """Get the current user's subscription limits and usage statistics"""

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

    minute_ago = datetime.now() - timedelta(minutes=1)
    result = db.execute(
        select(func.count()).where(
            ApiUsage.user_id == current_user.id,
            ApiUsage.request_timestamp >= minute_ago,
        )
    )
    minute_usage = result.scalar() or 0

    if day_usage >= day_limit:
        minute_limit = 0
        logger.info(
            f"User {current_user.id} has reached daily limit, showing minute limit as 0"
        )

    return {
        "subscription_plan_id": current_user.subscription_plan_id,
        "limits": {
            "per_minute": minute_limit,
            "per_day": day_limit,
        },
        "current_usage": {
            "minute": minute_usage,
            "day": day_usage,
        },
        "remaining": {
            "minute": max(0, minute_limit - minute_usage),
            "day": max(0, day_limit - day_usage),
        },
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
