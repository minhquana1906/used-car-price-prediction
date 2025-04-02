import streamlit as st


def render_footer():
    """Render the app footer"""
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center;">
        <p>© 2025 Used Car Price Prediction Project - Made with Streamlit</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
