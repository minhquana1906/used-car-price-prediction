from datetime import datetime

import streamlit as st

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


def init_session_state():
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "pages" not in st.session_state:
        st.session_state.pages = "Home"
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "token" not in st.session_state:
        st.session_state.token = None


def main():
    init_session_state()

    render_sidebar()

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

    render_footer()


if __name__ == "__main__":
    main()
