import json
import os
import warnings
from datetime import datetime

import hydra
import joblib
import numpy as np
import optuna
import pandas as pd
from joblib import dump, load
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from xgboost import XGBRegressor

from utils.decorators import timer

warnings.filterwarnings("ignore")


@timer
def load_data(cfg):
    """Load the preprocessed data"""
    train_path = cfg.paths.train_file_path
    test_path = cfg.paths.test_file_path

    logger.info(f"Loading training data from {train_path}")
    train_data = pd.read_csv(train_path)

    logger.info(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path)

    target = cfg.data.target
    X_train = train_data.drop(target, axis=1)
    y_train = train_data[target]
    X_test = test_data.drop(target, axis=1)
    y_test = test_data[target]

    logger.info(
        f"Data loaded. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
    )
    return X_train, X_test, y_train, y_test


@timer
def load_preprocessor(cfg):
    """Load preprocessor from file"""
    preprocessor_path = cfg.paths.preprocessor_file_path
    logger.info(f"Loading preprocessor from {preprocessor_path}")

    try:
        preprocessor = load(preprocessor_path)
        logger.success("Preprocessor loaded successfully")
        return preprocessor
    except Exception as e:
        logger.error(f"Error loading preprocessor: {str(e)}")
        raise


def objective(trial, X, y, cv, cfg):
    """Objective function for Optuna optimization"""
    param = {
        "objective": "reg:squarederror",
        "tree_method": cfg.tuning.models.xgboost.tree_method,
        "predictor": cfg.tuning.models.xgboost.predictor,
        "grow_policy": cfg.tuning.models.xgboost.grow_policy,
        "n_estimators": trial.suggest_categorical(
            "n_estimators", cfg.tuning.models.xgboost.n_estimators
        ),
        "max_depth": trial.suggest_categorical(
            "max_depth", cfg.tuning.models.xgboost.max_depth
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", cfg.tuning.models.xgboost.learning_rate
        ),
        "subsample": trial.suggest_categorical(
            "subsample", cfg.tuning.models.xgboost.subsample
        ),
        "colsample_bytree": trial.suggest_categorical(
            "colsample_bytree", cfg.tuning.models.xgboost.colsample_bytree
        ),
        "random_state": trial.suggest_categorical(
            "random_state", cfg.tuning.models.xgboost.random_state
        ),
    }

    model = XGBRegressor(**param)

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring=cfg.tuning.scoring)

    return scores.mean()


@timer
def tune_hyperparameters(cfg, X_train, y_train):
    """Tune hyperparameters for XGBoost model"""
    logger.info("Starting XGBoost hyperparameter tuning")

    cv_folds = cfg.tuning.cv_folds
    n_trials = cfg.tuning.n_trials
    timeout = cfg.tuning.timeout

    logger.info(
        f"Optuna configuration: cv_folds={cv_folds}, n_trials={n_trials}, timeout={timeout}s"
    )

    def objective_wrapper(trial):
        score = objective(trial, X_train, y_train, cv_folds, cfg)
        logger.info(f"Trial {trial.number}: MAE = {score:.4f}, params={trial.params}")
        return score

    study = optuna.create_study(direction="maximize")

    logger.info(f"Starting optimization with {n_trials} trials (timeout: {timeout}s)")
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        timeout=timeout,
    )

    best_params = study.best_params
    best_score = study.best_value
    best_trial = study.best_trial

    logger.info(f"Optimization completed. Total trials: {len(study.trials)}")
    logger.info(f"Best trial: #{best_trial.number}")
    logger.info(f"Best MAE: {-best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")

    results_dir = os.path.join(cfg.paths.logs_path, "tuning")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_path = os.path.join(results_dir, f"xgboost_study_{timestamp}.pkl")
    params_path = os.path.join(results_dir, f"xgboost_best_params_{timestamp}.json")

    joblib.dump(study, study_path)

    import json

    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Study saved to {study_path}")
    logger.info(f"Best parameters saved to {params_path}")

    return best_params


@timer
def train_and_evaluate_model(X_train, X_test, y_train, y_test, best_params, cfg):
    """Train model with best hyperparameters and evaluate"""
    logger.info("Training XGBoost with best hyperparameters")

    params = {
        "objective": "reg:squarederror",
        "tree_method": cfg.tuning.models.xgboost.tree_method,
        "predictor": cfg.tuning.models.xgboost.predictor,
        "grow_policy": cfg.tuning.models.xgboost.grow_policy,
        "random_state": 42,
        **best_params,
    }

    logger.info(f"Model parameters: {params}")

    model = XGBRegressor(**params)

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    logger.info(
        f"Training model with {X_train_fit.shape[0]} samples, validating with {X_val.shape[0]} samples"
    )

    model.fit(
        X_train_fit,
        y_train_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=cfg.tuning.early_stopping_rounds,
        verbose=0,
    )

    logger.info(f"Training completed. Best iteration: {model.best_iteration}")

    logger.info(f"Evaluating model on test set ({X_test.shape[0]} samples)")
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    logger.info(f"Test R²: {r2:.4f}")
    logger.info(f"Test MAE: {mae:.4f}")
    logger.info(f"Test MSE: {mse:.4f}")

    metrics = {"r2": r2, "mae": mae, "mse": mse, "params": params}

    return model, metrics


@timer
def save_model_and_metrics(model, metrics, cfg):
    """Save the trained model and metrics"""
    models_dir = cfg.paths.models_path
    metrics_dir = cfg.paths.metrics_path
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(models_dir, f"xgboost_tuned_{timestamp}.pkl")

    logger.info(f"Saving model to {model_path}")
    dump(model, model_path)

    metrics_path = os.path.join(metrics_dir, f"xgboost_metrics_{timestamp}.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.success(f"Model saved to {model_path}")
    logger.success(f"Metrics saved to {metrics_path}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main function for XGBoost hyperparameter tuning"""
    logger.info("Starting XGBoost hyperparameter tuning")

    try:
        X_train, X_test, y_train, y_test = load_data(cfg)

        preprocessor = load_preprocessor(cfg)

        logger.info("Applying preprocessing to data")
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        best_params = tune_hyperparameters(cfg, X_train_processed, y_train)

        model, metrics = train_and_evaluate_model(
            X_train_processed, X_test_processed, y_train, y_test, best_params, cfg
        )

        save_model_and_metrics(model, metrics, cfg)

        logger.success("XGBoost hyperparameter tuning completed successfully!")

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

        return False

    return True


if __name__ == "__main__":
    main()
