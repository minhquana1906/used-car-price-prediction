import os
from contextlib import asynccontextmanager

import pandas as pd
import redis.asyncio as redis
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from joblib import load
from loguru import logger

from app.auth.jwt import get_current_user
from app.auth.router import router as auth_router
from scripts.dbmaker import User
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

    # Load pipeline
    if os.path.exists(FULL_PIPELINE_PATH):
        logger.info(f"Loading full pipeline from {FULL_PIPELINE_PATH}")
        model_pipeline = await run_in_threadpool(load, FULL_PIPELINE_PATH)
        logger.success("Model pipeline loaded successfully")
    else:
        logger.error(f"Pipeline file not found at {FULL_PIPELINE_PATH}")
        raise RuntimeError("Model pipeline not found")

    # Load metrics
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

    # Connect to Redis
    try:
        redis_client = redis.from_url("redis://localhost:6379", decode_responses=True)
        await redis_client.ping()
        logger.success("Connected to Redis successfully!")
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

app.include_router(auth_router, prefix="/auth", tags=["authentication"])


@app.get("/")
def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to the Used Car Price Prediction API"},
    )


@app.get("/status")
def status():
    """Status check endpoint"""
    return JSONResponse(status_code=200, content={"model": model_pipeline is not None})


@app.post("/predict")
async def predict(car: CarModel, current_user: User = Depends(get_current_user)):
    """Predict the price of a car based on its features and return with error margin."""
    global model_pipeline, error_margin

    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess input data
        data = car.model_dump()

        cache_key = f"car_prediction:{hash(frozenset(data.items()))}"
        if redis_client:
            cached_result = await get_redis_cache_async(redis_client, cache_key)
            if cached_result:
                logger.info("Cache hit! Returning cached result.")
                return JSONResponse(
                    status_code=200,
                    content=eval(cached_result),
                )

        processed_data = preprocess_data(data)

        # Make prediction
        prediction = await run_in_threadpool(
            lambda: model_pipeline.predict(processed_data)
        )
        prediction_value = float(prediction[0])
        logger.info(f"Raw prediction: {prediction_value}")

        # Calculate range
        lower_bound = max(0, prediction_value - error_margin)
        upper_bound = prediction_value + error_margin

        result = {
            "predicted price": round(prediction_value, 2),
            "error margin": round(error_margin, 2),
            "acceptable range": f"{round(lower_bound, 2)} - {round(upper_bound, 2)}",
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


@app.get("/cached-predictions")
async def get_cached_predictions(current_user: User = Depends(get_current_user)):
    """Return all currently cached predictions."""
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


@app.delete("/cached-predictions")
async def delete_cached_predictions(current_user: User = Depends(get_current_user)):
    """Delete all cached predictions."""
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
