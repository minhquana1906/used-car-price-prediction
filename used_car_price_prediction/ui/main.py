import os
from datetime import datetime

import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager

from used_car_price_prediction.ui.components.footer import render_footer
from used_car_price_prediction.ui.components.sidebar import render_sidebar
from used_car_price_prediction.ui.page_modules.about import render_about_page
from used_car_price_prediction.ui.page_modules.analysis import \
    render_analysis_page
from used_car_price_prediction.ui.page_modules.home import render_home_page
from used_car_price_prediction.ui.page_modules.prediction import \
    render_prediction_page
from used_car_price_prediction.ui.utils.data_loader import load_data_from_db

st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

cookies = EncryptedCookieManager(password=os.getenv("COOKIE_PASSWORD"))
if not cookies.ready():
    st.stop()


if "session_id" not in st.session_state:
    import uuid

    st.session_state.session_id = str(uuid.uuid4())

    if "username" not in cookies or "token" not in cookies:
        if "username" in st.session_state:
            st.session_state.username = None
        if "token" in st.session_state:
            st.session_state.token = None


def restore_session():
    if "username" in cookies and "token" in cookies:
        st.session_state.username = cookies["username"]
        st.session_state.token = cookies["token"]
        st.session_state.is_authenticated = True
    else:
        st.session_state.is_authenticated = False
        st.session_state.username = None
        st.session_state.token = None


def init_session_state():
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "pages" not in st.session_state:
        st.session_state.pages = "Home"
    if "is_authenticated" not in st.session_state:
        restore_session()


def main():
    init_session_state()

    render_sidebar(cookies)

    if st.session_state.is_authenticated:
        data = load_data_from_db()
        if data is not None:
            data["age"] = datetime.now().year - data["yearOfRegistration"]

        current_page = st.session_state.pages

        if current_page == "Home":
            render_home_page(data)
        elif current_page == "Prediction":
            render_prediction_page(data)
        elif current_page == "Data Analysis":
            render_analysis_page(data)
        elif current_page == "About":
            render_about_page()

    # render_footer()


if __name__ == "__main__":
    main()
