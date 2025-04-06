import streamlit as st

from used_car_price_prediction.ui.auth import (forgot_password_page,
                                               login_page, logout,
                                               register_page,
                                               reset_password_page)
from used_car_price_prediction.ui.utils.limiter import (
    get_subscription_limits, render_usage_limits)


def render_sidebar(cookies):
    """Render the sidebar with authentication and navigation components"""

    st.sidebar.title("🗝️ Authentication")

    if not st.session_state.is_authenticated:
        if "auth_view" not in st.session_state:
            st.session_state.auth_view = "login"

        if st.sidebar.button(
            "Login",
            use_container_width=True,
            type=("primary" if st.session_state.auth_view == "login" else "secondary"),
        ):
            st.session_state.auth_view = "login"
            st.rerun()

        if st.sidebar.button(
            "Register",
            use_container_width=True,
            type=(
                "primary" if st.session_state.auth_view == "register" else "secondary"
            ),
        ):
            st.session_state.auth_view = "register"
            st.rerun()

        if st.sidebar.button(
            "Forgot Password",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.auth_view == "forgot_password"
                else "secondary"
            ),
        ):
            st.session_state.auth_view = "forgot_password"
            st.rerun()

        if st.session_state.auth_view == "login":
            if login_page(cookies):
                st.session_state.pages = "Home"
                st.rerun()
        elif st.session_state.auth_view == "register":
            if register_page():
                st.session_state.auth_view = "login"
                st.sidebar.info("Please log in with your new account")
                st.rerun()
        elif st.session_state.auth_view == "forgot_password":
            if forgot_password_page():
                st.session_state.auth_view = "login"
                st.sidebar.info(
                    "Reset link sent. Check your email and return to login."
                )
                st.rerun()
            if st.sidebar.button("Have a reset token?", type="secondary"):
                st.session_state.auth_view = "reset_password"
                st.rerun()
        elif st.session_state.auth_view == "reset_password":
            if reset_password_page():
                st.session_state.auth_view = "login"
                st.sidebar.info("Password reset successful. Please login.")
                st.rerun()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.username}")

        if st.sidebar.button("Logout"):
            logout(cookies)
            st.session_state.pages = "Login"
            st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.title("📍 Page Navigation")

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

        if "token" in st.session_state:
            get_subscription_limits(st.session_state.token)

        render_usage_limits()
