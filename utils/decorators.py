import time
from functools import wraps
from loguru import logger


def timer(func):
    def wrapper(*args, **kwargs):

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Elapsed time: {end - start}")
        return result

    return wrapper
