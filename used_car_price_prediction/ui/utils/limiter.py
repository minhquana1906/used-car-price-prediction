import os
from datetime import datetime

import requests
import streamlit as st
from loguru import logger

API_URL = os.getenv("API_URL", "http://localhost:8000")


def get_subscription_limits(token, force_refresh=False):
    current_token = st.session_state.get("token", None)

    if current_token != token:
        force_refresh = True
        logger.info("Token changed, forcing refresh of subscription limits")

    if force_refresh or "api_usage_data" not in st.session_state:
        try:
            limit_response = requests.get(
                f"{API_URL}/subscription-limits",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5,
            )

            if limit_response.status_code == 200:
                limits = limit_response.json()
                st.session_state.api_usage_data = limits
                return limits
            else:
                logger.error(f"Error fetching limits: {limit_response.text}")
                if current_token != token and "api_usage_data" in st.session_state:
                    del st.session_state.api_usage_data
                    logger.warning("Removed stale API usage data due to token change")
                return st.session_state.get("api_usage_data", None)
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return st.session_state.get("api_usage_data", None)
    else:
        return st.session_state.get("api_usage_data", None)


def render_usage_limits():
    if "api_usage_data" not in st.session_state:
        return

    limits_data = st.session_state.api_usage_data

    if limits_data:
        plan_id = limits_data.get("subscription_plan_id")
        plan_names = {1: "Free", 2: "Basic", 3: "Premium"}
        plan_name = plan_names.get(plan_id, "Unknown Plan")

        minute_limit = limits_data.get("limits", {}).get("per_minute", 0)
        day_limit = limits_data.get("limits", {}).get("per_day", 0)

        minute_usage = limits_data.get("current_usage", {}).get("minute", 0)
        day_usage = limits_data.get("current_usage", {}).get("day", 0)

        minute_remain = limits_data.get("remaining", {}).get("minute", 0)
        day_remain = limits_data.get("remaining", {}).get("day", 0)

        st.sidebar.markdown("---")
        st.sidebar.header("⚡ API Usage")

        st.sidebar.subheader(f"**Plan:** {plan_name}")

        col1, col2 = st.sidebar.columns(2)

        col1.metric(
            "Limits Per Minute",
            f"{minute_remain}/{minute_limit}",
            f"-{minute_usage}" if minute_usage > 0 else None,
            delta_color="inverse" if minute_remain < 0 else "normal",
        )

        col2.metric(
            "Limits Per Day",
            f"{day_remain}/{day_limit}",
            f"-{day_usage}" if day_usage > 0 else None,
            delta_color="inverse" if day_remain < 0 else "normal",
        )

        minute_progress = minute_remain / minute_limit if minute_limit > 0 else 0
        st.sidebar.progress(
            min(minute_progress, 1.0),
            text=f"Minute Usage:  ({(1 - minute_progress) * 100:.1f}%)",
        )

        day_progress = day_remain / day_limit if day_remain > 0 else 0
        st.sidebar.progress(
            min(day_progress, 1.0), text=f"Day Usage:  ({(1- day_progress) * 100:.1f}%)"
        )

        if day_remain <= 0:
            st.sidebar.warning("⚠️ You have reached your daily request limit!")
