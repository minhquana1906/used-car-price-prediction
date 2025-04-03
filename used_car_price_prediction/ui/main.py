from datetime import datetime
from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from used_car_price_prediction.ui.auth import login_page, logout, register_page

API_URL = "http://localhost:8000"
PREDICT_ENDPOINT = "http://localhost:8000/predict"
DATA_PATH = "./datasets/autos_cleaned.csv"

# cache data loading
@st.cache_data()

def load_data():
    """Load and preprocess the dataset and visualization"""
    try:
        df = pd.read_csv(DATA_PATH)
        # return df
        return df.sample(30000)
    except FileNotFoundError:
        st.error("Dataset not found. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return pd.DataFrame()


#  Init session state
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

# Get current page from session state
pages = st.session_state.pages

# Authentication sidebar section
st.sidebar.title("Authentication")


if not st.session_state.is_authenticated:
    auth_option = st.sidebar.radio("", options=["Login", "Register"])

    if auth_option == "Login":
        if login_page():
            st.rerun()  # Refresh lại trang sau khi đăng nhập
    else:
        if register_page():
            st.sidebar.info("Please log in with your new account")
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        logout()  # Gửi request để xóa JWT cookie
        st.rerun()

if st.session_state.is_authenticated:
    # Page Navigation
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

# Load data for pages
data = load_data()


# Cache API request
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

if st.session_state.is_authenticated:
    # ------------------- HOME PAGE -------------------
    if pages == "Home":
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

if st.session_state.is_authenticated:
    # -----------------------Predict Page--------------------------
    if pages == "Prediction":
        st.title("🚗 Used Car Price Prediction")
        st.markdown(
            """
        This app predicts the price of a used car based on its features.
        Use the controls below to input the car details, then click the 'Predict Price' button.
        """
        )

        col1, col2 = st.columns(2)

        with col1:
            # brand, model, vehicleType, fuelType, gearbox, notRepairedDamage
            st.subheader("Car Basic Information")

            brands = sorted(data["brand"].dropna().unique())
            selected_brand = st.selectbox(
                "Brand", brands, index=brands.index("audi") if "audi" in brands else 0
            )

            models = sorted(
                data[data["brand"] == selected_brand]["model"].dropna().unique()
            )
            selected_model = st.selectbox(
                "Model", models, index=models.index("a5") if "a5" in models else 0
            )

            vehicle_types = sorted(data["vehicleType"].dropna().unique())
            selected_vehicle_type = st.selectbox(
                "Vehicle Type",
                vehicle_types,
                index=vehicle_types.index("coupe") if "coupe" in vehicle_types else 0,
            )

            fuel_types = sorted(data["fuelType"].dropna().unique())
            selected_fuel_type = st.selectbox(
                "Fuel Type",
                fuel_types,
                index=fuel_types.index("diesel") if "diesel" in fuel_types else 0,
            )

            selected_gearbox = st.radio("Gearbox", ["Automatic", "Manual"], index=0)

            selected_damaged = st.radio("Have Repaired Damage", ["No", "Yes"], index=0)

        with col2:
            st.subheader("Technical Specifications")

            # powerPS, kilometer, yearOfRegistration
            min_power = int(data["powerPS"].min())
            max_power = int(min(data["powerPS"].max(), 500))
            selected_power = st.slider(
                "Power (PS)", min_value=min_power, max_value=max_power, value=190, step=5
            )

            min_km = int(data["kilometer"].min())
            max_km = int(min(data["kilometer"].max(), 150000))
            selected_km = st.slider(
                "Kilometers", min_value=min_km, max_value=max_km, value=125000, step=1000
            )

            current_year = datetime.now().year
            selected_year = st.slider(
                "Year of Registration",
                min_value=int(data["yearOfRegistration"].min()),
                max_value=current_year,
                value=2010,
                step=1,
            )

            calculated_age = current_year - selected_year
            st.info(f"The car is approximately {calculated_age} years old")

            # Predict button
        if st.button("✨ Predict Price ✨", type="primary", use_container_width=True):
            input_data = {
                "vehicleType": selected_vehicle_type,
                "gearbox": selected_gearbox,
                "powerPS": selected_power,
                "model": selected_model,
                "kilometer": selected_km,
                "fuelType": selected_fuel_type,
                "brand": selected_brand,
                "notRepairedDamage": selected_damaged,
                "yearOfRegistration": selected_year,
            }

            with st.spinner("⏳️Predicting..."):
                result = predict_price(input_data)

                # save result to session state
                st.session_state.prediction_result = result

        if st.session_state.prediction_result:
            st.success("🎉 Prediction Complete!")
            result = st.session_state.prediction_result

            # display result and visualize in simple chart
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label="💶 Predicted Price (EUR)", value=f"€{result['predicted price']}"
                )
                st.info(f"Price Range: € {result['acceptable range']}")

            with res_col2:
                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=result["predicted price"],
                        number={"prefix": "€", "font": {"size": 24}},
                        gauge={
                            "axis": {
                                "range": [None, result["predicted price"] * 1.5],
                                "tickwidth": 1,
                            },
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {
                                    "range": [
                                        0,
                                        result["predicted price"] - result["error margin"],
                                    ],
                                    "color": "lightgray",
                                },
                                {
                                    "range": [
                                        result["predicted price"] - result["error margin"],
                                        result["predicted price"] + result["error margin"],
                                    ],
                                    "color": "lightblue",
                                },
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": result["predicted price"],
                            },
                        },
                    )
                )

                fig.update_layout(
                    height=250,
                    margin=dict(l=10, r=10, t=30, b=10),
                    title_text="Price Gauge",
                )

                st.plotly_chart(fig, use_container_width=True)
                # Add details about the car for comparison
            st.subheader("Car Details")
            details_cols = st.columns(4)

            with details_cols[0]:
                st.markdown(f"**Brand**: {selected_brand.title()}")
                st.markdown(f"**Model**: {selected_model}")

            with details_cols[1]:
                st.markdown(f"**Vehicle Type**: {selected_vehicle_type.title()}")
                st.markdown(f"**Fuel Type**: {selected_fuel_type.title()}")

            with details_cols[2]:
                st.markdown(f"**Power**: {selected_power} PS")
                st.markdown(f"**Gearbox**: {selected_gearbox.title()}")

            with details_cols[3]:
                st.markdown(f"**Age**: {calculated_age} years")
                st.markdown(f"**Kilometers**: {selected_km:,}")

            # Add similar cars section
            st.subheader("Similar Cars in the Dataset")

            # configure the search to the same brand, model and more or less than 30% of the power and kilometers
            filtered_data = data[
                (data["brand"] == selected_brand)
                & (data["model"] == selected_model)
                & (data["powerPS"].between(selected_power * 0.3, selected_power * 1.3))
                & (data["kilometer"].between(selected_km * 0.3, selected_km * 1.3))
            ]

            if len(filtered_data) > 0:
                filtered_data = filtered_data.dropna().sort_values("price")
                filtered_data_display = filtered_data[
                    [
                        "brand",
                        "model",
                        "powerPS",
                        "kilometer",
                        "vehicleType",
                        "fuelType",
                        "yearOfRegistration",
                        "price",
                    ]
                ].head(10)
                st.dataframe(filtered_data_display, use_container_width=True)

                # Add price comparison
                avg_price = filtered_data["price"].mean()
                price_diff = result["predicted price"] - avg_price
                if abs(price_diff) < 100:
                    st.info(
                        f"The predicted price (€{result['predicted price']:,.2f}) is close to the average price of similar cars (€{avg_price:,.2f})"
                    )
                elif price_diff > 0:
                    st.warning(
                        f"The predicted price (€{result['predicted price']:,.2f}) is **€{price_diff:,.2f} higher** than the average price of similar cars (€{avg_price:,.2f})"
                    )
                else:
                    st.success(
                        f"The predicted price (€{result['predicted price']:,.2f}) is **€{abs(price_diff):,.2f} lower** than the average price of similar cars (€{avg_price:,.2f})"
                    )
            else:
                st.info("No similar cars found in the dataset")

    # ------------------- DATA ANALYSIS PAGE -------------------
    elif pages == "Data Analysis":
        st.title("📊 Used Car Market Analysis")
        st.markdown(
            """
        Explore the used car market data with interactive visualizations.
        This page provides insights into the factors affecting car prices.
        """
        )

        # Data overview
        st.subheader("Dataset Overview")

        # Display basic stats
        row_count = len(data)
        brand_count = data["brand"].nunique()
        avg_price = data["price"].mean()
        avg_age = data["age"].mean()

        # Create metric cards
        col_metrics = st.columns(4)
        col_metrics[0].metric("Total Records", f"{row_count:,}")
        col_metrics[1].metric("Unique Brands", brand_count)
        col_metrics[2].metric("Average Price", f"€{avg_price:.2f}")
        col_metrics[3].metric("Average Age", f"{avg_age:.1f} years")

        # Tabs for different analysis
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Price Analysis", "Brand Analysis", "Feature Relationships", "Data Explorer"]
        )

        with tab1:
            st.subheader("Price Distribution")

            col_price1, col_price2 = st.columns(2)

            with col_price1:
                # Price histogram
                fig = px.histogram(
                    data,
                    x="price",
                    nbins=50,
                    title="Price Distribution",
                    labels={"price": "Price (EUR)", "count": "Number of Cars"},
                    color_discrete_sequence=["#3366CC"],
                )
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig, use_container_width=True)

            with col_price2:
                # Price by vehicle type
                fig = px.box(
                    data,
                    x="vehicleType",
                    y="price",
                    title="Price by Vehicle Type",
                    labels={"vehicleType": "Vehicle Type", "price": "Price (EUR)"},
                    color="vehicleType",
                )
                fig.update_layout(xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

            # Price by age
            st.subheader("Price by Age")

            # Create age bins
            data["age_group"] = pd.cut(
                data["age"],
                bins=[0, 3, 5, 10, 15, 20, 100],
                labels=["0-3", "4-5", "6-10", "11-15", "16-20", "20+"],
            )

            fig = px.box(
                data.dropna(subset=["age_group"]),
                x="age_group",
                y="price",
                color="age_group",
                title="Price Distribution by Car Age",
                labels={"age_group": "Age Group (years)", "price": "Price (EUR)"},
            )
            fig.update_layout(xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            # Price by kilometer
            st.subheader("Price vs. Kilometer")

            fig = px.scatter(
                data.sample(min(5000, len(data))),  # Sample for better performance
                x="kilometer",
                y="price",
                color="fuelType",
                opacity=0.6,
                title="Price vs. Kilometer",
                labels={
                    "kilometer": "Kilometer",
                    "price": "Price (EUR)",
                    "fuelType": "Fuel Type",
                },
                trendline="lowess",
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("Brand Analysis")

            # Get top brands by count
            top_brands = data["brand"].value_counts().head(15).index.tolist()

            # Brand selector
            selected_brands = st.multiselect(
                "Select Brands to Compare", options=top_brands, default=top_brands[:5]
            )

            if selected_brands:
                brand_data = data[data["brand"].isin(selected_brands)]

                col_brand1, col_brand2 = st.columns(2)

                with col_brand1:
                    # Average price by brand
                    brand_avg_price = (
                        brand_data.groupby("brand")["price"].mean().reset_index()
                    )
                    brand_avg_price = brand_avg_price.sort_values("price", ascending=False)

                    fig = px.bar(
                        brand_avg_price,
                        x="brand",
                        y="price",
                        title="Average Price by Brand",
                        labels={"brand": "Brand", "price": "Average Price (EUR)"},
                        color="brand",
                    )
                    fig.update_layout(xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

                with col_brand2:
                    # Count by brand
                    brand_count = brand_data["brand"].value_counts().reset_index()
                    brand_count.columns = ["brand", "count"]

                    fig = px.bar(
                        brand_count,
                        x="brand",
                        y="count",
                        title="Number of Cars by Brand",
                        labels={"brand": "Brand", "count": "Count"},
                        color="brand",
                    )
                    fig.update_layout(xaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)

                # Price distribution by brand
                fig = px.box(
                    brand_data,
                    x="brand",
                    y="price",
                    title="Price Distribution by Brand",
                    labels={"brand": "Brand", "price": "Price (EUR)"},
                    color="brand",
                )
                fig.update_layout(xaxis_title="")
                st.plotly_chart(fig, use_container_width=True)

                # Most popular models by brand
                st.subheader("Popular Models by Brand")

                for brand in selected_brands:
                    with st.expander(f"{brand.title()} Models"):
                        brand_models = (
                            data[data["brand"] == brand]["model"].value_counts().head(10)
                        )

                        fig = px.bar(
                            x=brand_models.index,
                            y=brand_models.values,
                            title=f"Top 10 {brand.title()} Models",
                            labels={"x": "Model", "y": "Count"},
                            color_discrete_sequence=["#3366CC"],
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Please select at least one brand to compare")

        with tab3:
            st.subheader("Feature Relationships")

            # Correlation heatmap
            st.subheader("Correlation between Numerical Features")

            # Select only numerical columns
            numerical_cols = ["price", "powerPS", "kilometer", "age"]
            corr = data[numerical_cols].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                title="Correlation Heatmap",
                labels={"color": "Correlation"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # Feature relationships
            feature_x = st.selectbox(
                "Select X-axis Feature",
                options=[
                    "powerPS",
                    "kilometer",
                    "age",
                    "vehicleType",
                    "fuelType",
                    "gearbox",
                ],
                index=0,
            )

            feature_y = st.selectbox(
                "Select Y-axis Feature",
                options=["price", "powerPS", "kilometer", "age"],
                index=0,
            )

            color_by = st.selectbox(
                "Color by",
                options=[
                    "vehicleType",
                    "fuelType",
                    "gearbox",
                    "brand",
                    "notRepairedDamage",
                ],
                index=0,
            )

            # Create appropriate plot based on feature types
            if feature_x in ["vehicleType", "fuelType", "gearbox"]:
                # Categorical x-axis
                fig = px.box(
                    data,
                    x=feature_x,
                    y=feature_y,
                    color=color_by,
                    title=f"{feature_y.title()} by {feature_x.title()}, Colored by {color_by.title()}",
                    labels={
                        feature_x: feature_x.title(),
                        feature_y: feature_y.title(),
                        color_by: color_by.title(),
                    },
                )
            else:
                # Numerical x-axis
                sample_size = min(5000, len(data))  # Limit sample size for performance
                fig = px.scatter(
                    data.sample(sample_size),
                    x=feature_x,
                    y=feature_y,
                    color=color_by,
                    opacity=0.6,
                    title=f"{feature_y.title()} vs. {feature_x.title()}, Colored by {color_by.title()}",
                    labels={
                        feature_x: feature_x.title(),
                        feature_y: feature_y.title(),
                        color_by: color_by.title(),
                    },
                    trendline="lowess",
                )

            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Data Explorer")

            # Add filters
            col_filter1, col_filter2, col_filter3 = st.columns(3)

            with col_filter1:
                filter_brand = st.multiselect(
                    "Brand", options=sorted(data["brand"].unique()), default=[]
                )

            with col_filter2:
                filter_fuel = st.multiselect(
                    "Fuel Type", options=sorted(data["fuelType"].unique()), default=[]
                )

            with col_filter3:
                filter_vehicle = st.multiselect(
                    "Vehicle Type", options=sorted(data["vehicleType"].unique()), default=[]
                )

            # Price range filter
            price_min, price_max = st.slider(
                "Price Range (EUR)",
                min_value=int(data["price"].min()),
                max_value=int(data["price"].max()),
                value=[int(data["price"].min()), int(data["price"].max())],
            )

            # Apply filters
            filtered_data = data.copy()

            if filter_brand:
                filtered_data = filtered_data[filtered_data["brand"].isin(filter_brand)]

            if filter_fuel:
                filtered_data = filtered_data[filtered_data["fuelType"].isin(filter_fuel)]

            if filter_vehicle:
                filtered_data = filtered_data[
                    filtered_data["vehicleType"].isin(filter_vehicle)
                ]

            filtered_data = filtered_data[
                filtered_data["price"].between(price_min, price_max)
            ]

            # Display filtered data
            st.write(f"Showing {len(filtered_data)} of {len(data)} records")

            # Select which columns to display
            display_cols = st.multiselect(
                "Select Columns to Display",
                options=data.columns.tolist(),
                default=[
                    "brand",
                    "model",
                    "vehicleType",
                    "fuelType",
                    "powerPS",
                    "kilometer",
                    "age",
                    "price",
                ],
            )

            if display_cols:
                st.dataframe(filtered_data[display_cols], use_container_width=True)
            else:
                st.info("Please select at least one column to display")

            # Download filtered data
            if len(filtered_data) > 0:
                csv = filtered_data.to_csv(index=False)
                st.download_button(
                    label="Download Filtered Data as CSV",
                    data=csv,
                    file_name="filtered_car_data.csv",
                    mime="text/csv",
                )

    # ------------------- ABOUT PAGE -------------------
    elif pages == "About":
        st.title("ℹ️ About This Project")

        st.markdown(
            """
        ## Used Car Price Prediction Project

        This application was developed as part of a data science and machine learning project focused on creating
        an accurate prediction model for used car prices. The goal is to help both buyers and sellers in the used car
        market make more informed decisions based on data-driven insights.

        ### 🧠 The Model

        Our price prediction model is built using a machine learning pipeline that includes:

        1. **Data Preprocessing**: Cleaning, handling missing values, and encoding categorical variables
        2. **Feature Engineering**: Creating new features like car age and transforming existing features
        3. **Outlier Handling**: Identifying and addressing outliers using IQR method
        4. **Feature Transformation**: Using Power Transformation (Box-Cox) for numerical features
        5. **Prediction Algorithm**: XGBoost regression model trained on thousands of car listings

        The model achieves excellent predictive performance with metrics including:
        - **R-squared**: Measures how well the model explains price variation
        - **Mean Absolute Error (MAE)**: Average prediction error in euros
        - **Root Mean Squared Error (RMSE)**: Emphasizes larger errors
        -

        ### 📊 The Data

        The dataset used in this project contains information about used car listings, including:
        - Vehicle characteristics (brand, model, vehicle type)
        - Technical specifications (power, fuel type, transmission)
        - Usage details (kilometers driven, year of registration)
        - Condition information (damage status)
        - Price in euros

        We've conducted extensive data cleaning and preprocessing to ensure the model is trained on high-quality data.

        ### 🛠️ Technical Stack

        This project leverages several technologies:

        - **Backend**: FastAPI for the prediction API
        - **Frontend**: Streamlit for the user interface
        - **Data Processing**: Pandas, NumPy, Scikit-learn
        - **Visualization**: Plotly, Matplotlib
        - **Machine Learning**: XGBoost, Scikit-learn pipelines
        - **Deployment**: Docker, GitHub Actions
        """
        )

        # Team members section
        st.subheader("👨‍💻 Project Team")

        team_col1, team_col2 = st.columns(2)

        with team_col1:
            st.markdown(
                """
            #### Lead Data Scientist
            - Responsible for model development and evaluation
            - Implemented the machine learning pipeline
            - Conducted feature engineering and selection
            """
            )

        with team_col2:
            st.markdown(
                """
            #### App Developer
            - Built the FastAPI backend
            - Developed the Streamlit frontend
            - Integrated the model with the application
            """
            )

        # Methodology section
        st.subheader("🔬 Methodology")

        methodology_tabs = st.tabs(
            ["Data Collection", "Feature Engineering", "Model Training", "Evaluation"]
        )

        with methodology_tabs[0]:
            st.markdown(
                """
            ### Data Collection

            The dataset was sourced from a public dataset on Kaggle [Used Cars](https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices) . The raw data contained over
            370,000 car listings with various attributes. We performed the following steps:

            1. Initial data loading and inspection
            2. Removal of irrelevant or redundant columns
            3. Translation of German terms to English
            4. Basic cleaning and validation

            The cleaned dataset was then split into training and testing sets for model development.
            """
            )

        with methodology_tabs[1]:
            st.markdown(
                """
            ### Feature Engineering

            Several features were created or transformed to improve model performance:

            1. **Age**: Calculated from year of registration
            2. **Binary Encoding**: For features like gearbox and damage status
            3. **Categorical Encoding**: One-hot encoding for vehicle types, fuel types, etc.
            4. **Box-Cox Transformation**: Applied to skewed numerical features
            5. **Outlier Handling**: Using IQR method to clip extreme values

            Feature importance analysis was conducted to identify the most predictive attributes.
            """
            )

        with methodology_tabs[2]:
            st.markdown(
                """
            ### Model Training

            The model training process included:

            1. Creating a preprocessing pipeline for consistent transformations
            2. Hyperparameter tuning using cross-validation
            3. Training multiple model types (KNN, Random Forest, XGBoost)
            4. Implementing a voting regressor and stacking for ensemble learning
            5. Selecting the best performing model (XGBoost)
            6. Final model training on the full training dataset

            The pipeline ensures that all transformations applied during training are consistently
            applied during prediction.
            """
            )

        with methodology_tabs[3]:
            st.markdown(
                """
            ### Evaluation

            The model was evaluated using several metrics:

            1. **R-squared**: Measures explained variance (0.8950)
            2. **MAE**: Average absolute prediction error (980.8559)
            3. **RMSE**: Root mean squared error (2370776.2883)
            4. **Feature Importance**: Which car attributes most affect price

            Error analysis was conducted to understand where the model performs well and where it struggles.
            """
            )

        # Future improvements section
        st.subheader("🚀 Future Improvements")

        st.markdown(
            """
        We're continuously working to improve this project. Future enhancements include:

        - **Time-Based Analysis**: Tracking how car prices change over time
        - **Image-Based Prediction**: Using car images to further refine predictions
        - **Market Comparison**: Comparing prices across different regions
        - **Improve Model Performance**: Improve accuracy and reduce prediction error
        - **User Feedback Loop**: Allow users to provide feedback on predictions
        - **Mobile App**: Developing a mobile version for on-the-go use
        """
        )

        # Contact information
        st.subheader("📫 Contact")

        st.markdown(
            """
        If you have questions, suggestions, or would like to contribute to this project, please feel free to reach out:

        - **GitHub Repository**: [used-car-price-prediction](https://github.com/minhquana1906/used-car-price-prediction)
        - **Email**: quann1906@gmail.com
        - **LinkedIn**: [Quan Nguyen](https://www.linkedin.com/in/quan-nguyen-427872306/)

        We welcome feedback and collaboration!
        """
        )

        # Acknowledgements
        with st.expander("🙏 Acknowledgements"):
            st.markdown(
                """
            We would like to thank the following:

            - The open-source community for providing excellent tools and libraries
            - Contributors to the dataset we used for training our model
            - Users who provided feedback to help improve the application

            This project would not have been possible without their support.
            """
            )

    # Add footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center;">
        <p>© 2025 Used Car Price Prediction Project - Made with Streamlit</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
