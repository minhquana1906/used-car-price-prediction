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

logger.add("logs/api.log", rotation="500 MB")

FULL_PIPELINE_PATH = "./models/full_pipeline.pkl"
METRICS_PATH = "./logs/metrics/model_metrics.csv"
DEFAULT_ERROR_MARGIN = 500  # Euro


app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to help users making decision to buy an used car.",
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


@timer
def load_model_and_metrics():
    """Load model pipeline and model metrics"""
    result = {}

    # load pipeline
    try:
        if os.path.exists(FULL_PIPELINE_PATH):
            logger.info(f"Loading full pipeline from {FULL_PIPELINE_PATH}")
            pipeline = load(FULL_PIPELINE_PATH)
            logger.success("Full pipeline loaded successfully")
            result["full_pipeline"] = pipeline
        else:
            logger.error(f"Pipeline file not found at {FULL_PIPELINE_PATH}")
            return None
    except Exception as e:
        logger.error(f"Error loading pipeline: {str(e)}")
        return None

    # load metrics
    try:
        if os.path.exists(METRICS_PATH):
            logger.info(f"Loading metrics from {METRICS_PATH}")
            metrics_df = pd.read_csv(METRICS_PATH)

            if "MAE" in metrics_df.columns:
                error_margin = float(metrics_df["MAE"].iloc[0])
                logger.info(f"Using MAE as error margin: {error_margin}")
                result["error_margin"] = error_margin
            else:
                logger.warning(
                    f"MAE not found in metrics, using default error margin: {DEFAULT_ERROR_MARGIN}"
                )
                result["error_margin"] = DEFAULT_ERROR_MARGIN
        else:
            logger.warning(
                f"Metrics file not found, using default error margin: {DEFAULT_ERROR_MARGIN}"
            )
            result["error_margin"] = DEFAULT_ERROR_MARGIN
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        logger.warning(f"Using default error margin: {DEFAULT_ERROR_MARGIN}")
        result["error_margin"] = DEFAULT_ERROR_MARGIN

    return result


@timer
def preprocess_data(data):
    """Preprocess input data for prediction"""
    try:
        data_copy = data.copy()
        logger.info(f"Input data: {data_copy}")

        # manual process for notRepairedDamage
        if isinstance(data_copy["notRepairedDamage"], str):
            if data_copy["notRepairedDamage"].lower() == "yes":
                data_copy["notRepairedDamage"] = 1
            elif data_copy["notRepairedDamage"].lower() == "no":
                data_copy["notRepairedDamage"] = 0

        # manual process for gearbox
        if isinstance(data_copy["gearbox"], str):
            if data_copy["gearbox"].lower() == "manual":
                data_copy["gearbox"] = 0
            elif data_copy["gearbox"].lower() == "automatic":
                data_copy["gearbox"] = 1

        # manual process for age
        data_copy["age"] = datetime.now().year - data_copy["yearOfRegistration"]

        # Remove yearOfRegistration column
        data_copy.pop("yearOfRegistration")

        # Convert to DataFrame
        df = pd.DataFrame([data_copy])

        # Log the dataframe structure to help debug
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame types: {df.dtypes.to_dict()}")

        logger.success("Data preprocessed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during manual preprocessing: {str(e)}")
        return None


@app.get("/")
def root():
    """Root endpoint"""
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to the Used Car Price Prediction API"},
    )


@app.get("/health")
def health():
    """Health check endpoint"""
    return JSONResponse(status_code=200, content={"status": "healthy"})


@app.post("/predict")
def predict(car: CarModel):
    """Predict the price of a car based on its features and return with error margin"""
    try:
        # Load model, preprocessor, and metrics
        model_data = load_model_and_metrics()
        if model_data is None:
            raise HTTPException(
                status_code=500, detail="Failed to load model or metrics"
            )

        # Extract error margin
        error_margin = model_data.get("error_margin", DEFAULT_ERROR_MARGIN)

        # Convert Pydantic model to dict
        data = car.model_dump()

        # Preprocess data
        processed_data = preprocess_data(data)
        if processed_data is None:
            raise HTTPException(status_code=500, detail="Failed to preprocess data")

        # Make prediction
        try:
            prediction = model_data["full_pipeline"].predict(processed_data)
            prediction_value = float(prediction[0])
            logger.info(f"Raw prediction: {prediction_value}")
            logger.success(
                f"Prediction successful: {prediction_value} ± {error_margin}"
            )

            # Calculate bounds
            lower_bound = max(0, prediction_value - error_margin)
            upper_bound = prediction_value + error_margin

            # Format for response
            rounded_prediction = round(prediction_value, 2)
            rounded_error = round(error_margin, 2)
            rounded_lower = round(lower_bound, 2)
            rounded_upper = round(upper_bound, 2)

            return JSONResponse(
                status_code=200,
                content={
                    "predicted price": float(rounded_prediction),
                    "error margin": rounded_error,
                    "error in price": f"{rounded_prediction} ± {rounded_error}",
                    "acceptable range": f"{rounded_lower} - {rounded_upper}",
                },
            )
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
