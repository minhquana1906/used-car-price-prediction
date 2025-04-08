import asyncio
from datetime import datetime
from functools import partial
from typing import Any, Callable, TypeVar

import pandas as pd
from loguru import logger
from pydantic import BaseModel, Field


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


T = TypeVar("T")


async def run_in_threadpool(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a function in a thread pool and return the result."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


async def get_redis_cache_async(redis_client, key: str) -> str:
    """Get value from Redis cache asynchronously."""
    return await redis_client.get(key)


async def set_redis_cache_async(
    redis_client, key: str, value: str, expiration: int = 3600
) -> bool:
    """Set value in Redis cache asynchronously with expiration in seconds."""
    return await redis_client.set(key, value, ex=expiration)


def preprocess_data(data: dict) -> pd.DataFrame:
    """Preprocess input data for prediction."""

    logger.info(f"Preprocessing input data: {data}")

    data["notRepairedDamage"] = 1 if data["notRepairedDamage"].lower() == "yes" else 0
    data["gearbox"] = 1 if data["gearbox"].lower() == "automatic" else 0

    data["age"] = datetime.now().year - data["yearOfRegistration"]
    if data["age"] <= 0:
        data["age"] += 1
    data.pop("yearOfRegistration")

    df = pd.DataFrame([data])

    logger.success("Data preprocessed successfully")
    return df
