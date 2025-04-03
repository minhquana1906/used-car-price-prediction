# import os
# from datetime import datetime

# import pandas as pd
# import uvicorn
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from joblib import load
# from loguru import logger
# from pydantic import BaseModel, Field
# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.errors import RateLimitExceeded
# from slowapi.util import get_remote_address

# # from utils.decorators import timer

# logger.add("logs/api.log", rotation="500 MB", level="INFO")

# model_pipeline = None

# FULL_PIPELINE_PATH = "./models/full_pipeline.pkl"

# limiter = Limiter(key_func=get_remote_address)
# app = FastAPI(
#     title="Used Car Price Prediction API",
#     description="API to help users making decision to buy a used car.",
#     version="0.1",
# )
# # app.state.limiter = limiter
# # app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# class CarModel(BaseModel):
#     vehicleType: str = "coupe"
#     gearbox: str = "automatic"
#     powerPS: float = Field(190.0, gt=0)
#     model: str = "a5"
#     kilometer: float = Field(125000, gt=0)
#     fuelType: str = "diesel"
#     brand: str = "audi"
#     notRepairedDamage: str = "no"
#     yearOfRegistration: int = Field(2010, ge=1885, le=datetime.now().year)


# @app.on_event("startup")
# # @timer
# def load_model():
#     """Load model pipeline and metrics at startup to avoid reloading for each request."""
#     global model_pipeline
#     # Load pipeline
#     if os.path.exists(FULL_PIPELINE_PATH):
#         logger.info(f"Loading full pipeline from {FULL_PIPELINE_PATH}")
#         model_pipeline = load(FULL_PIPELINE_PATH)
#         logger.success("Model pipeline loaded successfully")
#     else:
#         logger.error(f"Pipeline file not found at {FULL_PIPELINE_PATH}")
#         raise RuntimeError("Model pipeline not found!")


# # @timer
# def preprocess_data(data: dict) -> pd.DataFrame:
#     """Preprocess input data for prediction."""
#     logger.info(f"Preprocessing input data: {data}")

#     # Chuyển đổi categorical thành số
#     data["notRepairedDamage"] = 1 if data["notRepairedDamage"].lower() == "yes" else 0
#     data["gearbox"] = 1 if data["gearbox"].lower() == "automatic" else 0

#     data["age"] = datetime.now().year - data["yearOfRegistration"]
#     data.pop("yearOfRegistration")

#     df = pd.DataFrame([data])

#     logger.success("Data preprocessed successfully")
#     return df


# @app.get("/")
# def root():
#     """Root endpoint"""
#     return JSONResponse(
#         status_code=200,
#         content={"message": "Welcome to the Used Car Price Prediction API"},
#     )


# @app.get("/status")
# def status():
#     """Status check endpoint"""
#     return JSONResponse(status_code=200, content={"model": model_pipeline is not None})


# # Default endpoint with IP-based rate limiting
# @app.post("/predict")
# # @limiter.limit("10/minute")
# async def predict(car: CarModel):
#     """Predict the price of a car based on its features and return with error margin."""
#     global model_pipeline, error_margin

#     if model_pipeline is None:
#         raise HTTPException(status_code=500, detail="Model not loaded")

#     try:
#         # Preprocess input data
#         data = car.model_dump()
#         processed_data = preprocess_data(data)

#         # Make prediction
#         prediction = model_pipeline.predict(processed_data)
#         prediction_value = float(prediction[0])
#         logger.info(f"Raw prediction: {prediction_value}")

#         return JSONResponse(
#             status_code=200,
#             content={
#                 "predicted_price": round(prediction_value, 2),
#             },
#         )
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Internal server error")


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="localhost", port=8000, reload=True)


import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from joblib import load
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database.dbmaker import (  # Giả sử bạn đã có file database.py
    SessionLocal, User, create_subscription_plans, init_db)

# Cấu hình
SECRET_KEY = "your-secret-key"  # Thay bằng key bí mật thực tế
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Khởi tạo
logger.add("logs/api.log", rotation="500 MB", level="INFO")
app = FastAPI(
    title="Used Car Price Prediction API",
    description="API to help users making decision to buy a used car.",
    version="0.1",
)
model_pipeline = load("./models/full_pipeline.pkl") if os.path.exists("./models/full_pipeline.pkl") else None
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Pydantic Models
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

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    subscription_tier: str = "Free"

class UserOut(BaseModel):
    username: str
    email: str
    subscription_tier: str

# Database Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Hàm hỗ trợ
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Endpoints
@app.post("/register", response_model=UserOut, status_code=201)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    db_email = db.query(User).filter(User.email == user.email).first()
    if db_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password,
        subscription_tier=user.subscription_tier
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(car: CarModel, current_user: User = Depends(get_current_user)):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data = car.dict()
        data["notRepairedDamage"] = 1 if data["notRepairedDamage"].lower() == "yes" else 0
        data["gearbox"] = 1 if data["gearbox"].lower() == "automatic" else 0
        data["age"] = datetime.now().year - data["yearOfRegistration"]
        data.pop("yearOfRegistration")
        df = pd.DataFrame([data])
        prediction = model_pipeline.predict(df)
        return {"predicted_price": round(float(prediction[0]), 2)}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/")
def root():
    return {"message": "Welcome to the Used Car Price Prediction API"}

@app.get("/status")
def status():
    return {"model": model_pipeline is not None}

if __name__ == "__main__":
    init_db()
    create_subscription_plans()
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
