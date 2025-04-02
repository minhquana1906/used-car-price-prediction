import pandas as pd
import plotly.express as px
import streamlit as st


def render_analysis_page(data):
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
