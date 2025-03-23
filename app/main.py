import os
from datetime import datetime

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from joblib import load
from loguru import logger
from pydantic import BaseModel, Field

from utils.decorators import timer

logger.add("logs/api.log", rotation="500 MB", level="INFO")

model_pipeline = None
error_margin = None

FULL_PIPELINE_PATH = "./models/full_pipeline.pkl"
METRICS_PATH = "./logs/metrics/model_metrics.csv"
DEFAULT_ERROR_MARGIN = 500

app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to help users making decision to buy a used car.",
    version="0.1",
)


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
@timer
def load_model():
    """Load model pipeline and metrics at startup to avoid reloading for each request."""
    global model_pipeline, error_margin

    # Load pipeline
    if os.path.exists(FULL_PIPELINE_PATH):
        logger.info(f"Loading full pipeline from {FULL_PIPELINE_PATH}")
        model_pipeline = load(FULL_PIPELINE_PATH)
        logger.success("Model pipeline loaded successfully")
    else:
        logger.error(f"Pipeline file not found at {FULL_PIPELINE_PATH}")
        raise RuntimeError("Model pipeline not found")

    # Load metrics
    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        error_margin = (
            float(metrics_df["MAE"].iloc[0])
            if "MAE" in metrics_df.columns
            else DEFAULT_ERROR_MARGIN
        )
        logger.info(f"Using error margin: {error_margin}")
    else:
        logger.warning("Metrics file not found, using default error margin")
        error_margin = DEFAULT_ERROR_MARGIN


@timer
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


@app.post("/predict")
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

        # Calculate range
        lower_bound = max(0, prediction_value - error_margin)
        upper_bound = prediction_value + error_margin

        return JSONResponse(
            status_code=200,
            content={
                "predicted_price": round(prediction_value, 2),
                "error_margin": round(error_margin, 2),
                "acceptable_range": f"{round(lower_bound, 2)} - {round(upper_bound, 2)}",
            },
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
