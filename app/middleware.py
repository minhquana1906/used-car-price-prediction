import time
from datetime import datetime

from fastapi import Request
from jose import jwt
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.limiter import get_subscription_limits
from app.auth.my_jwt import ALGORITHM, SECRET_KEY
from scripts.dbmaker import ApiUsage, User, get_db


class APIUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request path: {request.url.path}, Method: {request.method}")

        start_time = datetime.now()
        process_start = time.time()

        response = await call_next(request)

        process_time = time.time() - process_start

        logger.info(f"Request processed in {process_time:.4f} seconds")
        logger.info(f"Response status: {response.status_code}")

        if request.method == "POST" and request.url.path.startswith("/predict"):
            try:
                user_id = None
                if hasattr(request.state, "user") and request.state.user:
                    user_id = request.state.user.id
                    logger.info(f"User authenticated with ID: {user_id}")
                else:
                    logger.warning("User not authenticated")

                forwarded_for = request.headers.get("X-Forwarded-For")
                if forwarded_for:
                    ip_address = forwarded_for.split(",")[0].strip()
                else:
                    ip_address = request.client.host if request.client else None
                logger.info(f"Request IP: {ip_address}")

                if user_id is None:
                    logger.warning("User ID is None, not logging API usage")
                    return response

                predicted_price = None
                is_cached = False

                if hasattr(request.state, "prediction_result"):
                    predicted_price = request.state.prediction_result.get(
                        "predicted_price"
                    )
                    is_cached = request.state.prediction_result.get("is_cached", False)
                    logger.info(
                        f"Extracted prediction: {predicted_price}, is_cached: {is_cached}"
                    )

                db = next(get_db())
                api_usage = ApiUsage(
                    user_id=user_id,
                    request_timestamp=start_time,
                    endpoint=f"{request.method} {request.url.path}",
                    status_code=response.status_code,
                    ip_address=ip_address,
                    response_time_ms=process_time * 1000,
                    predicted_price=predicted_price,
                    is_cached=is_cached,
                )
                db.add(api_usage)
                db.commit()
                logger.info("API usage logged successfully with prediction data")
            except Exception as e:
                logger.error(f"Failed to log API usage: {e}")
                logger.exception("Stack trace:")

        return response


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if (
            request.url.path.startswith("/auth")
            or request.url.path == "/"
            or request.url.path == "/status"
        ):
            logger.info(f"Skipping auth for path: {request.url.path}")
            return await call_next(request)

        try:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.replace("Bearer ", "")

                # Decode token
                try:
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    username = payload.get("sub")

                    if username:
                        db = next(get_db())
                        user = db.query(User).filter(User.username == username).first()

                        if user:
                            # Set user in request state
                            request.state.user = user

                            # Set rate limit info for this user based on subscription
                            try:
                                minute_limit, day_limit = get_subscription_limits(
                                    user.subscription_plan_id, db
                                )
                                # Set dynamically for the limiter to use
                                request.state.dynamic_rate_limit = (
                                    f"{minute_limit}/minute;{day_limit}/day"
                                )
                            except Exception as rate_limit_error:
                                logger.error(f"Rate limit error: {rate_limit_error}")

                            logger.info(
                                f"User authenticated: {username} (ID: {user.id})"
                            )
                        else:
                            logger.warning(f"User not found in database: {username}")
                except jwt.JWTError as jwt_error:
                    logger.error(f"JWT validation failed: {jwt_error}")
            else:
                logger.info("No valid authorization header found")
        except Exception as e:
            logger.error(f"Auth middleware error: {e}")
            logger.exception("Stack trace:")

        return await call_next(request)
