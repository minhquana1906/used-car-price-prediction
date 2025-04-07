import os
import time

import requests
import streamlit as st

from used_car_price_prediction.ui.utils.limiter import get_subscription_limits

API_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = API_URL + "/predict"


def predict_price(data):
    """Make a prediction request to the API"""
    try:
        token = st.session_state.get("token")

        if not token:
            st.error("Authentication token is missing.")
            return None

        headers = {
            "Authorization": f"Bearer {token}",
        }
        response = requests.post(PREDICT_ENDPOINT, json=data, headers=headers)

        get_subscription_limits(token, force_refresh=True)

        if response.status_code == 200:
            result = response.json()
            st.session_state.prediction_result = result
            return result
        elif response.status_code == 401:
            st.error("Your session has expired. Please log in again.")
            st.session_state.is_authenticated = False
            st.session_state.token = None
            st.session_state.username = None
            return None
        elif response.status_code == 429:
            error = response.json().get("detail", "Rate limit exceeded")
            return error
        else:
            st.error(f"Error from API: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
