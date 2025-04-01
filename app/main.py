import os
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from joblib import load
from loguru import logger
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# from utils.decorators import timer

logger.add("logs/api.log", rotation="500 MB", level="INFO")

model_pipeline = None

FULL_PIPELINE_PATH = "./models/full_pipeline.pkl"

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to help users making decision to buy a used car.",
    version="0.1",
)
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class CarModel(BaseModel):
    vehicleType: str = "coupe"
    gearbox: str = "automatic"
    powerPS: float = Field(190.0, gt=0)
    model: str = "a5"
    kilometer: float = Field(125000, gt=0)
    fuelType: str = "diesel"
    brand: str = "audi"
    notRepairedDamage: str = "no"
    yearOfRegistration: int = Field(2010, ge=1885, le=datetime.now().year)


@app.on_event("startup")
# @timer
def load_model():
    """Load model pipeline and metrics at startup to avoid reloading for each request."""
    global model_pipeline
    # Load pipeline
    if os.path.exists(FULL_PIPELINE_PATH):
        logger.info(f"Loading full pipeline from {FULL_PIPELINE_PATH}")
        model_pipeline = load(FULL_PIPELINE_PATH)
        logger.success("Model pipeline loaded successfully")
    else:
        logger.error(f"Pipeline file not found at {FULL_PIPELINE_PATH}")
        raise RuntimeError("Model pipeline not found!")


# @timer
def preprocess_data(data: dict) -> pd.DataFrame:
    """Preprocess input data for prediction."""
    logger.info(f"Preprocessing input data: {data}")

    # Chuyển đổi categorical thành số
    data["notRepairedDamage"] = 1 if data["notRepairedDamage"].lower() == "yes" else 0
    data["gearbox"] = 1 if data["gearbox"].lower() == "automatic" else 0

    data["age"] = datetime.now().year - data["yearOfRegistration"]
    data.pop("yearOfRegistration")

    df = pd.DataFrame([data])

    logger.success("Data preprocessed successfully")
    return df


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


# Default endpoint with IP-based rate limiting
@app.post("/predict")
# @limiter.limit("10/minute")
async def predict(car: CarModel):
    """Predict the price of a car based on its features and return with error margin."""
    global model_pipeline, error_margin

    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Preprocess input data
        data = car.model_dump()
        processed_data = preprocess_data(data)

        # Make prediction
        prediction = model_pipeline.predict(processed_data)
        prediction_value = float(prediction[0])
        logger.info(f"Raw prediction: {prediction_value}")

        return JSONResponse(
            status_code=200,
            content={
                "predicted_price": round(prediction_value, 2),
            },
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
