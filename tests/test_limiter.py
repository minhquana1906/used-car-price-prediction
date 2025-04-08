import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import requests
from loguru import logger

API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"
LOGIN_ENDPOINT = f"{API_URL}/auth/login"
REGISTER_ENDPOINT = f"{API_URL}/auth/register"
SUBSCRIPTION_LIMITS_ENDPOINT = f"{API_URL}/subscription-limits"

TEST_USERNAME = "minhquana"
TEST_PASSWORD = "123123123"
TEST_EMAIL = "test@lms.utc.edu.vn"
TEST_SUBSCRIPTION_PLAN_ID = 1

os.makedirs("logs", exist_ok=True)

logger.add("logs/test_limiter.log", rotation="100 MB", level="INFO")

CAR_DATA = {
    "vehicleType": "coupe",
    "gearbox": "automatic",
    "powerPS": 190,
    "model": "a5",
    "kilometer": 125000,
    "fuelType": "diesel",
    "brand": "audi",
    "notRepairedDamage": "no",
    "yearOfRegistration": 2010,
}


def test_register():
    logger.info("Testing registration...")
    register_data = {
        "username": TEST_USERNAME,
        "email": TEST_EMAIL,
        "password": TEST_PASSWORD,
        "subscription_plan_id": TEST_SUBSCRIPTION_PLAN_ID,
    }

    try:
        register_response = requests.post(REGISTER_ENDPOINT, json=register_data)

        assert (
            register_response.status_code == 201
        ), f"Registration failed: {register_response.text}"
        logger.success("Registration successful.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error during registration: {str(e)}")
        raise e


def test_login():
    logger.info("Testing login...")
    login_data = {
        "username": TEST_USERNAME,
        "password": TEST_PASSWORD,
    }

    try:
        login_response = requests.post(LOGIN_ENDPOINT, data=login_data)
        if login_response.status_code != 200:
            logger.error(f"Login failed: {login_response.text}")
            logger.info("Try to register first...")
            test_register()
            logger.info("Retrying login...")
            login_response = requests.post(LOGIN_ENDPOINT, data=login_data)

        assert login_response.status_code == 200, f"Login failed: {login_response.text}"
        logger.success("Login successful.")

        token_data = login_response.json()["access_token"]
        return token_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error during login: {str(e)}")
        raise e


def test_prediction():
    try:
        logger.info("Testing prediction...")
        token = test_login()

        headers = {"Authorization": f"Bearer {token}"}

        predict_response = requests.post(
            PREDICT_ENDPOINT, json=CAR_DATA, headers=headers
        )

        assert (
            predict_response.status_code == 200
        ), f"Prediction failed: {predict_response.text}"
        prediction = predict_response.json()

        logger.success(f"Prediction successful: {prediction}")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise e


def test_rate_limiter_free_tier():
    logger.info("Testing rate limiter for free tier...")
    token = test_login()

    try:
        headers = {"Authorization": f"Bearer {token}"}
        limits_response = requests.get(SUBSCRIPTION_LIMITS_ENDPOINT, headers=headers)

        assert (
            limits_response.status_code == 200
        ), f"Rate limiter test failed: {limits_response.text}"
        limits_data = limits_response.json()
        logger.info(f"Rate limits: {limits_data}")

        import random

        power = random.randint(50, 300)
        kilometer = random.randint(0, 200000)
        year = random.randint(1900, 2025)
        car_data = {
            "vehicleType": "coupe",
            "gearbox": "automatic",
            "powerPS": power,
            "model": "a5",
            "kilometer": kilometer,
            "fuelType": "diesel",
            "brand": "audi",
            "notRepairedDamage": "no",
            "yearOfRegistration": year,
        }

        logger.info("Sending consecutive requests to trigger rate limiter...")
        results = []

        for i in range(15):
            try:
                start_time = time.time()
                response = requests.post(
                    PREDICT_ENDPOINT, json=car_data, headers=headers
                )
                end_time = time.time()

                result = {
                    "request_number": i + 1,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                }

                if response.status_code == 200:
                    result["prediction"] = response.json()["predicted price"]
                    logger.success(
                        f"Request {i+1}: Success - Status {response.status_code} - Prediction: {result['prediction']}"
                    )
                else:
                    result["error"] = response.json().get("detail", "Unknown error")
                    logger.warning(
                        f"Request {i+1}: Failed - Status {response.status_code} - Error: {result['error']}"
                    )

                results.append(result)

            except Exception as e:
                logger.error(f"Request {i+1}: Error - {str(e)}")

        successful_requests = [r for r in results if r["status_code"] == 200]
        failed_requests = [r for r in results if r["status_code"] != 200]

        logger.info("=== RATE LIMITER TEST RESULTS ===")
        logger.info(f"Total requests: {len(results)}")
        logger.info(f"Successful requests: {len(successful_requests)}")
        logger.info(f"Failed requests: {len(failed_requests)}")

        minute_limit = limits_data["limits"]["per_minute"]
        expected_successful = min(15, minute_limit)
        assert len(successful_requests) <= expected_successful, (
            f"Number of successful requests {len(successful_requests)} "
            f"exceeds limit {expected_successful}/minute!"
        )

        if failed_requests:
            logger.info(
                f"Rate limiter error message: {failed_requests[0].get('error', 'No error message')}"
            )
            assert "Ratelimit" in failed_requests[0].get(
                "error", ""
            ), "Exceed number of allowed requests per minute!"

        logger.success("Rate limiter test completed successfully!")

    except Exception as e:
        logger.error(f"Error during rate limiter test: {str(e)}")
        raise e


def test_daily_limit():
    logger.info("Testing rate limiter for free tier...")
    token = test_login()

    try:
        headers = {"Authorization": f"Bearer {token}"}
        limits_response = requests.get(SUBSCRIPTION_LIMITS_ENDPOINT, headers=headers)

        assert (
            limits_response.status_code == 200
        ), f"Rate limiter test failed: {limits_response.text}"
        limits_data = limits_response.json()
        logger.info(f"Rate limits: {limits_data}")

        logger.info("Sending consecutive requests to trigger rate limiter...")
        results = []

        for i in range(83):
            try:
                if i % 10 == 0 and i > 0:
                    logger.info(f"Sleeping for 60 seconds after {i} requests...")
                    time.sleep(60)

                power = random.randint(50, 300)
                kilometer = random.randint(0, 200000)
                year = random.randint(1900, 2025)
                car_data = {
                    "vehicleType": "coupe",
                    "gearbox": "automatic",
                    "powerPS": power,
                    "model": "a5",
                    "kilometer": kilometer,
                    "fuelType": "diesel",
                    "brand": "audi",
                    "notRepairedDamage": "no",
                    "yearOfRegistration": year,
                }

                start_time = time.time()
                response = requests.post(
                    PREDICT_ENDPOINT, json=car_data, headers=headers
                )
                end_time = time.time()

                result = {
                    "request_number": i + 1,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                }

                if response.status_code == 200:
                    result["prediction"] = response.json()["predicted price"]
                    logger.success(
                        f"Request {i+1}: Success - Status {response.status_code} - Prediction: {result['prediction']}"
                    )
                else:
                    result["error"] = response.json().get("detail", "Unknown error")
                    logger.warning(
                        f"Request {i+1}: Failed - Status {response.status_code} - Error: {result['error']}"
                    )

                results.append(result)

            except Exception as e:
                logger.error(f"Request {i+1}: Error - {str(e)}")

        successful_requests = [r for r in results if r["status_code"] == 200]
        failed_requests = [r for r in results if r["status_code"] != 200]

        logger.info("=== RATE LIMITER TEST RESULTS ===")
        logger.info(f"Total requests: {len(results)}")
        logger.info(f"Successful requests: {len(successful_requests)}")
        logger.info(f"Failed requests: {len(failed_requests)}")

        day_limit = limits_data["limits"]["per_day"]
        expected_successful = min(100, day_limit)
        assert len(successful_requests) <= expected_successful, (
            f"Number of successful requests {len(successful_requests)} "
            f"exceeds limit {expected_successful}/minute!"
        )

        if failed_requests:
            logger.info(
                f"Rate limiter error message: {failed_requests[0].get('error', 'No error message')}"
            )
            assert "Ratelimit" in failed_requests[0].get(
                "error", ""
            ), "Exceed number of allowed requests per minute!"

        logger.success("Rate limiter test completed successfully!")

    except Exception as e:
        logger.error(f"Error during rate limiter test: {str(e)}")
        raise e


if __name__ == "__main__":
    choice = 1
    while choice:
        logger.info("=== RATE LIMITER TEST (FREE TIER) ===")
        logger.info("1. Test a single prediction")
        logger.info("2. Test minute limit (10 requests/minute)")
        logger.info("3. Simulate daily limit (100 requests/day)")
        logger.info("0. Exit")
        logger.info("======================================")

        choice = input("\nSelect test (0-3): ")

        if choice == "1":
            test_prediction()
            continue
        elif choice == "2":
            test_rate_limiter_free_tier()
            continue
        elif choice == "3":
            test_daily_limit()
            continue
        else:
            logger.info("Exiting program.")
            break
