import streamlit as st


def render_about_page():
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
