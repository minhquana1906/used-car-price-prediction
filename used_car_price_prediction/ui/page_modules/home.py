from time import sleep

import plotly.express as px
import streamlit as st


def render_home_page(data):

    st.title("🏠 Welcome to Used Car Price Prediction")

    # Hero section with image and description
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
        ## Make better car buying decisions with AI

        Our application uses machine learning to predict the price of used cars based on their features.
        Whether you're a buyer wanting to know if a car is fairly priced, or a seller trying to set the right price,
        this tool can help you make informed decisions.

        ### 🚀 Key Features:

        - **Accurate Price Predictions**: Get estimated prices with error margins
        - **Data Visualization**: Explore trends and patterns in the used car market
        - **Similar Car Comparison**: See how your car compares to similar models
        - **Interactive Analysis**: Filter and analyze car data to find insights

        ### 📊 Based on real market data:
        - Over **{}** cars analyzed
        - **{}** different brands
        - Average price: **€{:.2f}**
        """.format(
                len(data), data["brand"].nunique(), data["price"].mean()
            )
        )

        # Call-to-action buttons
        predict_col, analyze_col = st.columns(2)
        with predict_col:
            if st.button(
                "🔮 Try Price Prediction", type="primary", use_container_width=True
            ):
                st.session_state.pages = "Prediction"
                st.rerun()

        with analyze_col:
            if st.button("📈 Explore Data Analysis", use_container_width=True):
                st.session_state.pages = "Data Analysis"
                st.rerun()

    with col2:
        # Display a representative car image or price distribution chart
        st.image(
            "https://apparelbyenemy.com/cdn/shop/articles/wp8431376_900x.jpg?v=1684366903",
            caption="Make informed decisions when buying used cars",
        )

        # Show some stats in metrics
        st.metric(label="Average Car Age", value=f"{data['age'].mean():.1f} years")
        st.metric(label="Average Power", value=f"{data['powerPS'].mean():.1f} PS")

    # Featured insights section
    st.subheader("🔍 Featured Insights")

    insight_tabs = st.tabs(
        ["Price Over Years", "Price Trends", "Popular Brands", "Power vs Price"]
    )

    with insight_tabs[0]:
        st.subheader("Price Trends by Registration Year (Real-time Visualization)")

        year_price = data.groupby("yearOfRegistration")["price"].mean().reset_index()
        year_price = year_price.sort_values("yearOfRegistration")

        year_price = year_price[year_price["yearOfRegistration"] >= 1990]

        chart_col, info_col = st.columns([3, 1])

        with chart_col:
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_container = st.empty()

            chart = chart_container.line_chart()

        with info_col:
            current_year_container = st.empty()
            current_price_container = st.empty()
            price_change_container = st.empty()

            st.caption(
                "Watch as the graph builds to show how car prices have changed over the years. Key economic events are highlighted."
            )

        total_years = len(year_price)

        step_size = max(1, total_years // 100)

        year_price_display = year_price.rename(
            columns={"yearOfRegistration": "index", "price": "Average Price"}
        ).set_index("index")

        previous_price = None

        for i in range(0, total_years, step_size):
            progress = int((i / total_years) * 100)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing price trends: {progress}%")

            current_data = year_price_display.iloc[: i + step_size]

            chart = chart_container.line_chart(current_data)

            if len(current_data) > 0:
                current_idx = min(i + step_size - 1, len(year_price) - 1)
                current_year = int(year_price.iloc[current_idx]["yearOfRegistration"])
                current_price = year_price.iloc[current_idx]["price"]

                current_year_container.metric("Year", current_year)
                current_price_container.metric("Price", f"€{current_price:,.2f}")

                if previous_price is not None and previous_price > 0:
                    price_change = (
                        (current_price - previous_price) / previous_price
                    ) * 100
                    price_change_container.metric(
                        "Change", f"{price_change:.2f}%", delta=f"{price_change:.2f}%"
                    )

                previous_price = current_price

                if current_year == 2008:
                    st.info("🔔 2008: Global Financial Crisis impacts car prices")
                elif current_year == 2020:
                    st.info("🔔 2020: COVID-19 Pandemic affects the market")

            sleep(0.05)

        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        sleep(0.5)

        progress_bar.empty()
        status_text.empty()

        st.subheader("Detailed Price Trend Analysis")

        fig = px.line(
            year_price,
            x="yearOfRegistration",
            y="price",
            title="Average Car Price by Registration Year",
            labels={
                "yearOfRegistration": "Registration Year",
                "price": "Average Price (EUR)",
            },
            markers=True,
        )

        annotations = [
            dict(
                x=2008,
                y=(
                    year_price[year_price["yearOfRegistration"] == 2008][
                        "price"
                    ].values[0]
                    if 2008 in year_price["yearOfRegistration"].values
                    else 0
                ),
                text="Financial Crisis",
                showarrow=True,
                arrowhead=1,
            ),
            dict(
                x=2020,
                y=(
                    year_price[year_price["yearOfRegistration"] == 2020][
                        "price"
                    ].values[0]
                    if 2020 in year_price["yearOfRegistration"].values
                    else 0
                ),
                text="COVID-19",
                showarrow=True,
                arrowhead=1,
            ),
        ]

        valid_annotations = [
            anno
            for anno in annotations
            if anno["x"] in year_price["yearOfRegistration"].values
        ]

        if valid_annotations:
            fig.update_layout(annotations=valid_annotations)

        fig.update_layout(
            height=500,
            hovermode="x unified",
            xaxis=dict(tickmode="linear", tick0=1990, dtick=5),
            xaxis_title="Registration Year",
            yaxis_title="Average Price (EUR)",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            """
        **Trend Analysis**:
        - Car prices generally increase for newer models
        - The 2008 financial crisis caused a noticeable dip in prices
        - There's typically a steep depreciation curve in the first few years
        - Recent models (2020+) show higher prices despite the pandemic
        """
        )

        st.button("🔄 Replay Animation", key="replay_year_price")

    with insight_tabs[1]:
        # Create a simple price by age trend
        age_price = data.groupby("age")["price"].mean().reset_index()
        fig = px.line(
            age_price,
            x="age",
            y="price",
            title="Average Price by Car Age",
            labels={"age": "Car Age (years)", "price": "Average Price (EUR)"},
            markers=True,
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "As expected, car prices generally decrease with age, with a steeper decline in the first few years. Howerver, some classic cars retain value due to brand and model popularity."
        )

    with insight_tabs[2]:
        # Top 10 popular brands
        top_brands = data["brand"].value_counts().head(10)
        fig = px.bar(
            x=top_brands.index,
            y=top_brands.values,
            title="10 Most Common Car Brands in Dataset",
            labels={"x": "Brand", "y": "Count"},
            color=top_brands.values,
            color_continuous_scale="Viridis",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"**{top_brands.index[0].title()}** is the most common brand with **{top_brands.values[0]:,}** listings."
        )

    with insight_tabs[3]:
        # Power vs Price scatter
        sample = data.sample(min(3000, len(data)))
        fig = px.scatter(
            sample,
            x="powerPS",
            y="price",
            color="vehicleType",
            opacity=0.7,
            title="Relationship Between Power and Price",
            labels={"powerPS": "Power (PS)", "price": "Price (EUR)"},
            trendline="ols",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Higher power generally correlates with higher prices, though the relationship varies by vehicle type."
        )

    # How to use section
    st.subheader("📖 How to Use This App")

    how_to_cols = st.columns(3)

    with how_to_cols[0]:
        st.markdown(
            """
        ### 1. Predict Prices

        - Go to the **Prediction** tab
        - Enter car details (brand, model, etc.)
        - Click "Predict Price"
        - Get an estimated price range
        """
        )

    with how_to_cols[1]:
        st.markdown(
            """
        ### 2. Analyze Market Data

        - Visit the **Data Analysis** tab
        - Explore interactive visualizations
        - Compare brands and models
        - Identify market trends
        """
        )

    with how_to_cols[2]:
        st.markdown(
            """
        ### 3. Make Better Decisions

        - Compare prediction with asking price
        - Check similar cars in the market
        - Understand value drivers
        - Negotiate with confidence
        """
        )

    # Recent updates or news
    with st.expander("📢 Recent Updates"):
        st.markdown(
            """
        - **March 2025**: Added similar car comparison feature
        - **March 2025**: Improved model accuracy by 12%
        - **February 2025**: Added data visualization dashboard
        - **February 2024**: Launched initial version of the app
        """
        )
