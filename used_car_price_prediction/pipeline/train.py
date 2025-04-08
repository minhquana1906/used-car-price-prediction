import os
import time
import warnings

import hydra
import numpy as np
import pandas as pd
import tqdm
from joblib import dump, load
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback

from utils.decorators import timer

warnings.filterwarnings("ignore")


@timer
def load_data(cfg: DictConfig):
    """Load preprocessed train and test data

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    train_path = cfg.paths.train_file_path
    test_path = cfg.paths.test_file_path

    logger.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path)

    logger.info(f"Loading testing data from {test_path}")
    test_df = pd.read_csv(test_path)

    target = cfg.data.target
    X_train = train_df.drop(target, axis=1)
    y_train = train_df[target]

    X_test = test_df.drop(target, axis=1)
    y_test = test_df[target]

    logger.info(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test


@timer
def load_preprocessor(cfg: DictConfig):
    """Load preprocessor from file

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        object: Fitted preprocessor
    """
    preprocessor_path = cfg.paths.preprocessor_file_path
    logger.info(f"Loading preprocessor from {preprocessor_path}")

    try:
        preprocessor = load(preprocessor_path)
        logger.success("Preprocessor loaded successfully")
        return preprocessor
    except Exception as e:
        logger.error(f"Error loading preprocessor: {str(e)}")
        raise


@timer
def create_xgboost_model(cfg: DictConfig):
    """Create XGBoost model with parameters from config

    Args:
        cfg (DictConfig): Configuration object

    Returns:
        XGBRegressor: Configured XGBoost model
    """
    logger.info("Creating XGBoost model...")

    model = XGBRegressor(
        n_estimators=cfg.model.n_estimators,
        max_depth=cfg.model.max_depth,
        learning_rate=cfg.model.learning_rate,
        random_state=cfg.model.random_state,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
    )

    return model


class ProgressBarCallback(TrainingCallback):
    """Custom callback to update tqdm progress bar during XGBoost training"""

    def __init__(self, pbar):
        self.pbar = pbar
        self.iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration"""
        self.iteration += 1
        self.pbar.update(1)
        return False

    def after_training(self, model):
        """Called after training"""
        self.pbar.close()
        return model


@timer
def train_model(model, preprocessor, X_train, y_train, X_test, y_test):
    """Train the XGBoost model with progress bar

    Args:
        model: XGBoost model
        preprocessor: Preprocessor pipeline
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        tuple: Trained pipeline, evaluation metrics
    """
    logger.info("Preprocessing training data...")
    X_train_processed = preprocessor.transform(X_train)
    logger.info(f"Processed training data shape: {X_train_processed.shape}")

    X_test_processed = preprocessor.transform(X_test)

    logger.info("Training XGBoost model...")
    start_time = time.time()

    eval_set = [(X_train_processed, y_train), [X_test_processed, y_test]]

    n_estimators = model.get_params()["n_estimators"]

    with tqdm.tqdm(total=n_estimators, desc="Training epochs") as pbar:
        callback = ProgressBarCallback(pbar)

        model.fit(
            X_train_processed,
            y_train,
            eval_set=eval_set,
            eval_metric="rmse",
            verbose=False,
            callbacks=[callback],
        )

    training_time = time.time() - start_time
    logger.info(f"Model training completed in {training_time:.2f} seconds")

    y_pred = model.predict(X_test_processed)

    metrics = {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "Training Time (s)": training_time,
    }

    logger.info("Model evaluation results in eval set:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    return pipeline, metrics


@timer
def save_model(model_pipeline, metrics, cfg: DictConfig):
    """Save the trained model pipeline and metrics

    Args:
        model_pipeline (Pipeline): Trained model pipeline
        metrics (dict): Evaluation metrics
        cfg (DictConfig): Configuration object
    """
    models_dir = cfg.paths.models_path
    os.makedirs(models_dir, exist_ok=True)
    metrics_dir = cfg.paths.metrics_path
    os.makedirs(metrics_dir, exist_ok=True)

    # model_path = os.path.join(models_dir, "xgboost.pkl")
    full_pipeline_path = os.path.join(models_dir, "full_pipeline.pkl")

    # logger.info(f"Saving model to {model_path}")
    logger.info(f"Saving full pipeline to {full_pipeline_path}")

    # model = model_pipeline.named_steps["model"]
    # dump(model, model_path)

    dump(model_pipeline, full_pipeline_path)

    metrics_path = os.path.join(metrics_dir, "model_metrics.csv")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_path, index=False)

    # logger.success(f"Model saved to {model_path}")
    logger.success(f"Full pipeline saved to {full_pipeline_path}")
    logger.success(f"Metrics saved to {metrics_path}")


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def train_pipline(cfg: DictConfig):
    """Main function to train and evaluate XGBoost model"""
    logger.info("Starting XGBoost model training...")

    try:
        stages = [
            "Load Data",
            "Load Preprocessor",
            "Create Model",
            "Train Model",
            "Save Model",
        ]

        with tqdm.tqdm(total=len(stages), desc="Training Progress") as progress_bar:
            progress_bar.set_description("Loading Data")
            X_train, X_test, y_train, y_test = load_data(cfg)
            progress_bar.update(1)

            progress_bar.set_description("Loading Preprocessor")
            preprocessor = load_preprocessor(cfg)
            progress_bar.update(1)

            progress_bar.set_description("Creating Model")
            xgb_model = create_xgboost_model(cfg)
            progress_bar.update(1)

            progress_bar.set_description("Training Model")
            model_pipeline, metrics = train_model(
                xgb_model, preprocessor, X_train, y_train, X_test, y_test
            )
            progress_bar.update(1)

            progress_bar.set_description("Saving Model")
            save_model(model_pipeline, metrics, cfg)
            progress_bar.update(1)

        logger.success("XGBoost model training completed successfully!")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

    return True


if __name__ == "__main__":
    train_pipline()
