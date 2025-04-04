import streamlit as st

from used_car_price_prediction.ui.auth import login_page, logout, register_page


def render_sidebar(cookies):
    """Render the sidebar with authentication and navigation components"""

    st.sidebar.title("Authentication")

    if not st.session_state.is_authenticated:
        if "auth_view" not in st.session_state:
            st.session_state.auth_view = "login"

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button(
                "Login",
                use_container_width=True,
                type=(
                    "primary" if st.session_state.auth_view == "login" else "secondary"
                ),
            ):
                st.session_state.auth_view = "login"
                st.rerun()

        with col2:
            if st.button(
                "Register",
                use_container_width=True,
                type=(
                    "primary"
                    if st.session_state.auth_view == "register"
                    else "secondary"
                ),
            ):
                st.session_state.auth_view = "register"
                st.rerun()

        if st.session_state.auth_view == "login":
            if login_page(cookies):
                st.rerun()
        else:
            if register_page():
                st.session_state.auth_view = "login"
                st.sidebar.info("Please log in with your new account")
                st.rerun()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout"):
            logout(cookies)
            st.rerun()

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
