import os
from datetime import datetime

import hydra
import pandas as pd
from loguru import logger
from omegaconf import DictConfig
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, PowerTransformer,
                                   QuantileTransformer, StandardScaler)

from utils.decorators import timer
from utils.preprocessing import OutlierClipper


@timer
def load_dataset(cfg: DictConfig) -> pd.DataFrame:
    """Load dataset from a CSV file

    Args:
        path (str): path to the dataset

    Returns:
        pd.DataFrame: Dataset loaded from the CSV file
    """
    raw_data_path = cfg.paths.dataset_file_path
    logger.info(f"Loading dataset from {raw_data_path}...")
    data = pd.read_csv(raw_data_path)
    logger.success("Dataset loaded successfully.")
    return data


@timer
def convert_to_binary(data: pd.DataFrame) -> pd.DataFrame:
    """Convert some categorical columns which have only two unique values to binary

    Args:
        data (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset with binary columns
    """
    logger.info("Converting values of 'notRepairedDamage' and 'gearbox' into binary...")
    data["notRepairedDamage"] = (
        data["notRepairedDamage"].dropna().map({"ja": 1, "nein": 0}).astype(int)
    )
    data["gearbox"] = data["gearbox"].map({"automatik": 1, "manuell": 0}).astype(int)
    logger.success("Columns converted to binary successfully.")
    return data


@timer
def drop_columns(data: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Drop redundant columns from the dataset

    Args:
        data (pd.DataFrame): Raw dataset
        columns (list): List of columns to drop

    Returns:
        pd.DataFrame: Dataset without the redundant columns
    """
    target = cfg.data.target
    col_to_drop = cfg.data.columns_cleanup

    logger.info("Creating `age` column...")
    data["age"] = datetime.now().year - data["yearOfRegistration"]
    logger.success("`age` column created successfully.")

    logger.info("Dropping redundant columns...")
    data.dropna(inplace=True)

    data = data[data[target].between(200, 20_000)]
    data = data[(data["powerPS"] > 0) & (data["powerPS"] <= 1000)]
    data = data[data["fuelType"] != "Other"]
    data = data[data["notRepairedDamage"] != "NaN"]

    data.drop(col_to_drop, axis=1, inplace=True)
    logger.success("Redundant columns dropped successfully.")
    return data


@timer
def create_preprocessor(cfg: DictConfig) -> Pipeline:
    """Create a preprocessor pipeline

    Args:
        numerical_features (list): List of numerical features
        categorical_features (list): List of categorical features

    Returns:
        Pipeline: Preprocessor pipeline
    """

    logger.info("Creating preprocessor pipeline...")
    numerical_features = cfg.data.numerical_features
    categorical_features = cfg.data.categorical_features
    numerical_features_quantile = cfg.data.numerical_features_quantile
    numerical_features_boxcox = cfg.data.numerical_features_boxcox
    outlier_features = cfg.data.outlier_features
    outlier_threshold = cfg.data.outlier_threshold
    pca_components = cfg.data.pca_components

    logger.info(f"Numerical features: {numerical_features}")
    logger.info(f"Categorical features: {categorical_features}")
    logger.info(f"Numerical features (quantile): {numerical_features_quantile}")
    logger.info(f"Numerical features (boxcox): {numerical_features_boxcox}")

    numerical_features = list(numerical_features)
    categorical_features = list(categorical_features)
    numerical_features_quantile = list(numerical_features_quantile)
    numerical_features_boxcox = list(numerical_features_boxcox)
    outlier_features = list(outlier_features)

    outlier_clipper = OutlierClipper(columns=outlier_features, factor=outlier_threshold)

    boxcox_pipeline = Pipeline(
        steps=[
            ("boxcox_transform", PowerTransformer(method="box-cox", standardize=True)),
        ]
    )

    quantile_pipeline = Pipeline(
        steps=[("quantile", QuantileTransformer(output_distribution="normal"))]
    )

    standard_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
            )
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("quantile", quantile_pipeline, numerical_features_quantile),
            ("boxcox", boxcox_pipeline, numerical_features_boxcox),
            ("standard", standard_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="passthrough",
    )

    preprocessor = Pipeline(
        steps=[
            ("outlier_clipper", outlier_clipper),
            ("transformer", column_transformer),
            ("pca", PCA(n_components=pca_components)),
        ]
    )

    logger.success("Preprocessor pipeline created successfully.")
    return preprocessor


@timer
def binary_to_category(data: pd.DataFrame) -> pd.DataFrame:
    """Convert some categorical columns from binary to categorical

    Args:
        data (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset with binary columns
    """
    logger.info(
        "Converting values of 'notRepairedDamage' and 'gearbox' from binary to categorical..."
    )
    data["notRepairedDamage"] = (
        data["notRepairedDamage"].dropna().map({1: "Yes", 0: "No"}).astype(str)
    )
    data["gearbox"] = data["gearbox"].map({1: "Automatic", 0: "Manual"}).astype(str)
    logger.success("Columns converted to categorical successfully.")
    return data


@timer
def save_datasets(data: pd.DataFrame, cfg: DictConfig):
    """Save the training and testing datasets to CSV files

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        y_train (pd.Series): Training target
        y_test (pd.Series): Testing target
        cfg (DictConfig): Configuration object
    """
    logger.info("Saving datasets to CSV files...")
    dataset_path = cfg.paths.cleaned_data_file_path

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    data.to_csv(dataset_path, index=False)

    logger.info(f"Cleaned dataset saved to {dataset_path}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def preprocess_pipeline(cfg: DictConfig):
    """Main function to process the dataset"""
    logger.info("Starting preprocessing pipeline...")

    try:
        data = load_dataset(cfg)
        data = drop_columns(data, cfg)
        data = convert_to_binary(data)

        logger.info(f"Available columns after preprocessing: {data.columns.tolist()}")
        logger.info(f"Sample data types: {data.dtypes}")

        preprocessor = create_preprocessor(cfg)
        logger.info("Fitting preprocessor to training data...")
        preprocessor.fit_transform(data)
        logger.success("Preprocessor fit and transformed successfully.")

        data = binary_to_category(data)
        save_datasets(data, cfg)

        logger.success("Preprocessing completed successfully!")

    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        raise

    return True


if __name__ == "__main__":
    preprocess_pipeline()
