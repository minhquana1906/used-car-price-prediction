from datetime import datetime

import plotly.graph_objects as go
import streamlit as st

from used_car_price_prediction.ui.utils.api import predict_price


def render_prediction_page(data):
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

        filtered_data = (
            data[data["model"] == selected_model] if selected_model else data
        )

        vehicle_types = sorted(filtered_data["vehicleType"].dropna().unique())
        selected_vehicle_type = st.selectbox(
            "Vehicle Type",
            vehicle_types,
            index=vehicle_types.index("coupe") if "coupe" in vehicle_types else 0,
        )

        fuel_types = sorted(filtered_data["fuelType"].dropna().unique())
        selected_fuel_type = st.selectbox(
            "Fuel Type",
            fuel_types,
            index=fuel_types.index("diesel") if "diesel" in fuel_types else 0,
        )

        selected_gearbox = st.radio(
            "Gearbox",
            ["Automatic", "Manual"],
            index=filtered_data["gearbox"].mode()[0] == "automatic",
        )

        selected_damaged = st.radio("Have Repaired Damage", ["No", "Yes"], index=0)

    with col2:
        st.subheader("Technical Specifications")

        # powerPS, kilometer, yearOfRegistration
        min_power = int(filtered_data["powerPS"].min())
        max_power = int(min(filtered_data["powerPS"].max(), 500))
        selected_power = st.slider(
            "Power (PS)",
            min_value=min_power,
            max_value=max_power,
            value=190,
            step=5,
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

    can_make_predictions = True
    if "api_usage_data" in st.session_state:
        limits_data = st.session_state.api_usage_data
        day_remain = limits_data.get("remaining", {}).get("day", 0)
        minute_remain = limits_data.get("remaining", {}).get("minute", 0)

        if day_remain <= 0 or minute_remain <= 0:
            can_make_predictions = False

    if can_make_predictions:
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

                st.session_state.prediction_result = result

    else:
        st.button(
            "✨ Predict Price ✨",
            disabled=True,
            type="primary",
            use_container_width=True,
            help="You have reached your API usage limits",
        )
        st.warning(
            "⚠️ You have reached your API usage limit. Please try again later or upgrade your plan."
        )

    if st.session_state.prediction_result:
        st.success("🎉 Prediction Complete!")
        result = st.session_state.prediction_result

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
        top_k = 10
        st.subheader(f"Top {top_k} similar cars in the dataset")

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
            ].head(top_k)
            st.dataframe(filtered_data_display, use_container_width=True)

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
