import json
from datetime import datetime

import requests
import streamlit as st

# API endpoint
API_URL = "http://localhost:8000"


def login_page():
    st.title("🔐 Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if not username or not password:
                st.error("Please enter both username and password")
                return False

            # Make API call to FastAPI login endpoint
            try:
                response = requests.post(
                    f"{API_URL}/auth/login",
                    data={"username": username, "password": password},
                )

                if response.status_code == 200:
                    token_data = response.json()
                    st.session_state.token = token_data["access_token"]
                    st.session_state.username = username
                    st.session_state.is_authenticated = True
                    st.success("Login successful!")
                    return True
                else:
                    st.error("Invalid username or password")
                    return False
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
                return False

    return False


def register_page():
    st.title("📝 Register")

    with st.form("registration_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        subscription_tier = st.selectbox(
            "Subscription Tier", options=["Free", "Basic", "Premium"], index=0
        )

        submitted = st.form_submit_button("Register")

        if submitted:
            if not username or not email or not password:
                st.error("Please fill in all required fields")
                return False

            if password != confirm_password:
                st.error("Passwords do not match")
                return False

            # Make API call to FastAPI registration endpoint
            try:
                response = requests.post(
                    f"{API_URL}/auth/register",
                    json={
                        "username": username,
                        "email": email,
                        "password": password,
                        "subscription_tier": subscription_tier,
                    },
                )

                if response.status_code == 201:
                    st.success("Registration successful! Please login.")
                    return True
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"Registration failed: {error_detail}")
                    return False
            except Exception as e:
                st.error(f"Registration failed: {str(e)}")
                return False

    return False


def logout():
    if "token" in st.session_state:
        del st.session_state.token
    if "username" in st.session_state:
        del st.session_state.username
    st.session_state.is_authenticated = False
    st.success("Logged out successfully!")
