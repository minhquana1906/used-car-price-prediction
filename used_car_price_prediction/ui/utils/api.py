import requests
import streamlit as st

API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = "http://localhost:8000/predict"


def predict_price(data):
    """Make a prediction request to the API"""
    try:
        response = requests.post(PREDICT_ENDPOINT, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error from API: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to the API: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
