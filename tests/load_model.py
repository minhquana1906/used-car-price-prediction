import sys
import warnings

from joblib import load
from loguru import logger

# Bỏ qua cảnh báo
warnings.filterwarnings("ignore")

try:
    model = load("./models/xgb_88.pkl")
    logger.success("XGBoost model loaded successfully!")
except Exception as e:
    logger.exception(f"Error loading XGBoost model: {str(e)}")
    sys.exit(1)

try:
    preprocessor = load("./models/preprocessor.pkl")
    logger.success("Preprocessor loaded successfully!")
except Exception as e:
    logger.exception(f"Error loading preprocessor: {str(e)}")
    sys.exit(1)

logger.success("All components loaded successfully!")
