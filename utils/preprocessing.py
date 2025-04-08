import numpy as np
from loguru import logger
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Transformer for clipping outliers based on IQR method"""

    def __init__(self, columns, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.bounds = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - self.factor * IQR
                upper_bound = Q3 + self.factor * IQR

                self.bounds[col] = (lower_bound, upper_bound)
                logger.info(
                    f"Outlier bounds for '{col}': [{lower_bound}, {upper_bound}]"
                )
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        for col in self.columns:
            if col in X_copy.columns and col in self.bounds:
                lower, upper = self.bounds[col]
                outliers_count = ((X_copy[col] < lower) | (X_copy[col] > upper)).sum()
                if outliers_count > 0:
                    logger.info(f"Clipping {outliers_count} outliers in '{col}'")
                X_copy[col] = X_copy[col].clip(lower=lower, upper=upper)

        return X_copy
