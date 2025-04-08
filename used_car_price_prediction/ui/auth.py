import os

import requests
import streamlit as st
from loguru import logger

from used_car_price_prediction.ui.utils.limiter import get_subscription_limits

API_URL = os.getenv("API_URL", "http://localhost:8000")


def login_page(cookies):
    st.title("🔐 Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(
            "Login", use_container_width=True, type="primary"
        )

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
                return False

            try:
                response = requests.post(
                    f"{API_URL}/auth/login",
                    data={"username": username, "password": password},
                )

                if response.status_code == 200:
                    token_data = response.json()
                    cookies["token"] = token_data["access_token"]
                    cookies["username"] = username
                    cookies.save()

                    st.session_state.is_authenticated = True
                    st.session_state.token = token_data["access_token"]
                    st.session_state.username = username
                    st.success("Login successful!")
                    get_subscription_limits(st.session_state.token, force_refresh=True)
                    return True
                else:
                    st.error("Invalid username or password")
                    return False
            except Exception as e:
                logger.error(f"Login failed: {str(e)}")
                st.error(f"Login failed.")
                return False
    return False


def register_page():
    st.title("📝 Register")

    with st.form("registration_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        subscription_plan_id = {
            "Free": 1,
            "Basic": 2,
            "Premium": 3,
        }

        selected_plan = st.selectbox(
            "Select Subscription Plan",
            options=list(subscription_plan_id.keys()),
            index=0,
        )

        subscription_plan_id = subscription_plan_id[selected_plan]

        submitted = st.form_submit_button(
            "Register", use_container_width=True, type="primary"
        )

        if submitted:
            if not username or not email or not password:
                st.error("Please fill in all required fields")
                return False

            if password != confirm_password:
                st.error("Passwords do not match")
                return False

            try:
                response = requests.post(
                    f"{API_URL}/auth/register",
                    json={
                        "username": username,
                        "email": email,
                        "password": password,
                        "subscription_plan_id": subscription_plan_id,
                    },
                )

                if response.status_code == 201:
                    st.success("Registration successful! Please login.")
                    return True
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    logger.error(f"Registration failed: {error_detail}")
                    st.error(f"Registration failed.")
                    return False
            except Exception as e:
                logger.error(f"Registration failed: {str(e)}")
                st.error(f"Registration failed.")
                return False

    return False


def logout(cookies):
    if "token" in st.session_state:
        del st.session_state.token
        del cookies["token"]

    if "username" in st.session_state:
        del st.session_state.username
        del cookies["username"]

    if "api_usage_data" in st.session_state:
        del st.session_state.api_usage_data

    st.session_state.is_authenticated = False

    cookies.save()
    st.success("Logged out successfully!")


def forgot_password_page():
    st.title("🔑 Forgot Password")

    with st.form("forgot_password_form"):
        email = st.text_input("Email")
        submitted = st.form_submit_button(
            "Send Reset Link", use_container_width=True, type="primary"
        )

        if submitted:
            if not email:
                st.error("Please enter your email")
                return False

            try:
                response = requests.post(
                    f"{API_URL}/auth/forgot-password",
                    params={"email": email},
                )

                if response.status_code == 202:
                    st.info(
                        f"If {email} really exists, a reset link has been sent. Please check your inbox."
                    )
                    return False
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    logger.error(f"Failed to send reset link: {error_detail}")
                    st.error("Invalid email address")
                    return False
            except Exception as e:
                logger.error(f"Failed to send reset link: {str(e)}")
                st.error("Invalid email address")
                return False

    return False


def reset_password_page():
    st.title("🔑 Reset Password")

    token = st.text_input("Paste your reset token from email", type="password")

    with st.form("reset_password_form"):
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button(
            "Reset Password", use_container_width=True, type="primary"
        )

        if submitted:
            if not token:
                st.error("Please enter your reset token")
                return False

            if not new_password or not confirm_password:
                st.error("Please fill in all fields")
                return False

            if new_password != confirm_password:
                st.error("Passwords do not match")
                return False

            try:
                response = requests.post(
                    f"{API_URL}/auth/reset-password",
                    json={
                        "token": token,
                        "new_password": new_password,
                    },
                )

                if response.status_code == 200:
                    st.success("Password reset successfully! Please login.")
                    return True
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    logger.error(f"Reset failed: {error_detail}")
                    st.error(f"Reset password failed.")
                    return False
            except Exception as e:
                logger.error(f"Reset failed: {str(e)}")
                return False
