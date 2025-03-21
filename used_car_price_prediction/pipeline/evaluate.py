import os
import time
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import tqdm
from joblib import load
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.decorators import timer


@timer
def load_model(cfg: DictConfig):
    """Load trained model from file

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        object: Trained model pipeline
    """
    model_path = cfg.paths.model_file_path
    logger.info(f"Loading model from {model_path}")

    try:
        model = load(model_path)
        logger.success("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


@timer
def load_test_data(cfg: DictConfig):
    """Load test data for prediction

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        tuple: X_test, y_test
    """
    test_path = cfg.paths.test_file_path
    logger.info(f"Loading test data from {test_path}")

    try:
        test_df = pd.read_csv(test_path)
        target = cfg.data.target

        X_test = test_df.drop(target, axis=1)
        y_test = test_df[target]

        logger.info(f"Test data loaded: X_test={X_test.shape}")
        return X_test, y_test
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise


@timer
def predict(model, X: pd.DataFrame) -> np.ndarray:
    """Make predictions with the model

    Args:
        model: Trained model pipeline
        X (pd.DataFrame): Input features

    Returns:
        np.ndarray: Predictions
    """
    logger.info(f"Making predictions on data with shape {X.shape}...")

    start_time = time.time()
    predictions = model.predict(X)
    prediction_time = time.time() - start_time

    logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
    return predictions


@timer
def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate predictions with various metrics

    Args:
        y_true (pd.Series): True target values
        y_pred (np.ndarray): Predicted values

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    logger.info("Evaluating predictions...")

    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    }

    logger.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    return metrics


@timer
def save_predictions(
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metrics: Dict[str, float],
    cfg: DictConfig,
):
    """Save predictions and evaluation metrics

    Args:
        y_pred (np.ndarray): Predicted values
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): True target values
        metrics (Dict[str, float]): Evaluation metrics
        cfg (DictConfig): Configuration object
    """
    predictions_dir = cfg.paths.predictions_path
    os.makedirs(predictions_dir, exist_ok=True)

    # save predictions
    predictions_df = X_test.copy()
    predictions_df[cfg.data.target + "_true"] = y_test.values
    predictions_df[cfg.data.target + "_pred"] = y_pred

    predictions_df["error"] = (
        predictions_df[cfg.data.target + "_true"]
        - predictions_df[cfg.data.target + "_pred"]
    )
    predictions_df["error_abs"] = predictions_df["error"].abs()

    # save predictions
    predictions_path = os.path.join(predictions_dir, "predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    # save metrics
    metrics_path = os.path.join(predictions_dir, "prediction_metrics.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    logger.success(f"Predictions saved to {predictions_path}")
    logger.success(f"Metrics saved to {metrics_path}")

    # Log 5 samples with highest prediction error
    logger.info("Top 5 samples with highest absolute error:")
    top_errors = predictions_df.sort_values("error_abs", ascending=False).head(5)
    for _, row in top_errors.iterrows():
        logger.info(
            f"  True: {row[cfg.data.target + '_true']:.2f}, Pred: {row[cfg.data.target + '_pred']:.2f}, Error: {row['error']:.2f}"
        )


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def evaluate_pipeline(cfg: DictConfig):
    """Main function to make and evaluate predictions"""
    logger.info("Starting prediction process...")

    try:
        stages = [
            "Load Model",
            "Load Test Data",
            "Make Predictions",
            "Evaluate Predictions",
            "Save Results",
        ]

        with tqdm.tqdm(total=len(stages), desc="Prediction Progress") as progress_bar:
            # Load model
            progress_bar.set_description("Loading Model")
            model = load_model(cfg)
            progress_bar.update(1)

            # Load test data
            progress_bar.set_description("Loading Test Data")
            X_test, y_test = load_test_data(cfg)
            progress_bar.update(1)

            # Make predictions
            progress_bar.set_description("Making Predictions")
            y_pred = predict(model, X_test)
            progress_bar.update(1)

            # Evaluate predictions
            progress_bar.set_description("Evaluating Predictions")
            metrics = evaluate_predictions(y_test, y_pred)
            progress_bar.update(1)

            # Save results
            progress_bar.set_description("Saving Results")
            save_predictions(y_pred, X_test, y_test, metrics, cfg)
            progress_bar.update(1)

        logger.success("Prediction process completed successfully!")

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        logger.exception(e)
        raise

    return True


if __name__ == "__main__":
    evaluate_pipeline()
