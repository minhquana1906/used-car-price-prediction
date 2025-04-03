import streamlit as st

from used_car_price_prediction.ui.auth import login_page, logout, register_page


def render_sidebar():
    """Render the sidebar with authentication and navigation components"""

    # Authentication section
    st.sidebar.title("Authentication")

    if not st.session_state.is_authenticated:
        auth_option = st.sidebar.radio("", options=["Login", "Register"])

        if auth_option == "Login":
            if login_page():
                st.rerun()
        else:
            if register_page():
                st.sidebar.info("Please log in with your new account")
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout"):
            logout()
            st.rerun()

        # Navigation section
        st.sidebar.title("Page Navigation")

        # Use session_state for persistent navigation
        if st.sidebar.button("🏠 Home", use_container_width=True):
            st.session_state.pages = "Home"
            st.rerun()
        if st.sidebar.button("🔮 Prediction", use_container_width=True):
            st.session_state.pages = "Prediction"
            st.rerun()
        if st.sidebar.button("📊 Data Analysis", use_container_width=True):
            st.session_state.pages = "Data Analysis"
            st.rerun()
        if st.sidebar.button("ℹ️ About", use_container_width=True):
            st.session_state.pages = "About"
            st.rerun()
