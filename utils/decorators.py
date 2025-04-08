import time
from functools import wraps

from loguru import logger


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.success(f"{func.__name__}() ended in {end - start}.")

        return result

    return wrapper
