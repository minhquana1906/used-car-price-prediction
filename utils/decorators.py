import time
from functools import wraps

from loguru import logger


def timer(func):
    def wrapper(*args, **kwargs):
        # logger.info(f"Starting {func.__name__}()...")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # if result:
        logger.success(f"{func.__name__}() ended in {end - start}.")
        # else:
        #     logger.error(f"{func.__name__}() failed.")

        return result

    return wrapper
