# Used Car Price Prediction
![GitHub repo size](https://img.shields.io/github/repo-size/minhquana1906/used-car-price-prediction)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/minhquana1906/used-car-price-prediction)
[![Release](https://img.shields.io/github/v/release/minhquana1906/used-car-price-prediction)](https://img.shields.io/github/v/release/minhquana1906/used-car-price-prediction)
[![Build status](https://img.shields.io/github/actions/workflow/status/minhquana1906/used-car-price-prediction/main.yml?branch=main)](https://github.com/minhquana1906/used-car-price-prediction/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/minhquana1906/used-car-price-prediction)](https://img.shields.io/github/license/minhquana1906/used-car-price-prediction)

A machine learning project that predicts used car prices to help buyers make informed purchasing decisions, with a user-friendly web interface built using FastAPI and Streamlit.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation and Setup](#installation-and-setup)
  - [Prerequisites](#prerequisites)
  - [Setting up the Development Environment](#setting-up-the-development-environment)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Streamlit UI](#streamlit-ui)
- [Model Development](#model-development)
  - [Dataset](#dataset)
  - [Model Training](#model-training)
  - [Model Performance](#model-performance)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

The Used Car Price Prediction project uses machine learning to help users determine fair prices for used cars. By analyzing various car features (brand, model, year, mileage, etc.), the system predicts a market-appropriate price range with error margins, enabling buyers to make informed decisions and avoid overpaying.

The system consists of:
- A robust predictive model trained on a comprehensive used car dataset
- A fast, scalable REST API built with FastAPI
- An intuitive web interface created with Streamlit for easy interaction
- Comprehensive data analytics to understand market trends

This project demonstrates best practices in MLOps, including reproducible model training, model versioning, CI/CD pipeline integration, and containerized deployment.

## Project Structure

```
used-car-price-prediction/
├── app/                            # FastAPI application
├── conf/                           # Configuration files
├── data/                           # Processed data
├── datasets/                       # Raw datasets
├── logs/                           # Application logs
├── models/                         # Trained models
├── notebooks/                      # Jupyter notebooks for exploration
├── scripts/                        # Utility scripts
├── tests/                          # Test files
├── used_car_price_prediction/      # Main package
│   ├── pipeline/                   # Data processing and model training
│   └── ui/                         # Streamlit UI
└── utils/                          # Helper utilities
```

## Features

- **Price Prediction**: Estimate the fair market value of a used car based on its characteristics
- **Confidence Intervals**: Provide price ranges with error margins for better decision making
- **Data Exploration**: Interactive data visualization tools to understand market trends
- **Market Analysis**: Compare predictions with similar vehicles in the dataset
- **User-friendly Interface**: Intuitive web UI for non-technical users
- **REST API**: Programmatic access for integrations with other systems

## Technology Stack

- **Python 3.10+**: Core programming language
- **FastAPI**: High-performance API framework
- **Streamlit**: Interactive web interface
- **XGBoost**: Gradient boosting algorithm for prediction
- **Pandas/NumPy**: Data processing and analysis
- **Scikit-learn**: ML preprocessing and evaluation
- **Plotly/Matplotlib**: Data visualization
- **Hydra**: Configuration management
- **Loguru**: Logging
- **Pytest**: Testing
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- uv package manager
- Git
- Docker (optional, for containerized deployment)

### Setting up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/minhquana1906/used-car-price-prediction.git
   cd used-car-price-prediction
   ```

2. You can install `uv` by the following commands or you can read official docs to install [here](https://docs.astral.sh/uv/getting-started/installation/) . (If you already have `uv` installed, skip this step):
   ```bash
   # Using pip
   pip install uv

   # OR using pipx
   pipx install uv
   ```

3. Create and activate a virtual environment then install dependencies:
   ```bash
   # Using uv
   make install
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

### Running the Application
0. You can run the following command to quickly set up and run the application:
   ```bash
   make application
   # This command will help you set up quickly but the api and ui services will be run in the same terminal
   # If you want to run them in different terminals, please follow the instructions below
   ```

1. Trigger data prepocessing pipeline to create the processed data, model:
   ```bash
   # If you're using the provided dataset
   make pipeline
   # The preprocessed data should be placed in the data/processed/ directory
   # Similarly, the trained model should be in the models/ directory
   ```

2. Create a fast processed data for the UI:
   ```bash
   make clean-dataset
   ```

3. Start the FastAPI backend:
   ```bash
   make api
   ```

4. Start the Streamlit UI (in a separate terminal):
   ```bash
   make ui
   ```

5. Access the applications:
   - API documentation: http://localhost:8000/docs
   - Streamlit UI: http://localhost:8501

## Usage

### API Endpoints

- `GET /`: Welcome page
- `GET /health`: Health check endpoint
- `POST /predict`: Predict car price based on provided features

Example request to the prediction endpoint:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "vehicleType": "coupe",
           "gearbox": "automatic",
           "powerPS": 190.0,
           "model": "a5",
           "kilometer": 125000,
           "fuelType": "diesel",
           "brand": "audi",
           "notRepairedDamage": "no",
           "yearOfRegistration": 2010
         }'
```

### Streamlit UI

The Streamlit interface provides:

1. **Prediction Page**: Input car details and get price predictions
2. **Data Analysis Page**: Explore market trends and visualizations
3. **Home Page**: Overview of the project and quick access to features

## Model Development

### Dataset

The model is trained on a German used car dataset containing over 300,000 listings with the following features:
- Vehicle type, brand, and model
- Registration year and mileage
- Engine power and fuel type
- Gearbox type (manual/automatic)
- Damage status and more

### Model Training

To train the model:

```bash
uv run used_car_price_prediction/pipeline/train.py
```

The training pipeline consists of:
1. Data cleaning and preprocessing
2. Feature engineering and selection
3. Hyperparameter tuning using cross-validation
4. Model training and evaluation
5. Model and metrics persistence

### Model Performance

The current XGBoost model achieves:
- R² Score: ~0.89 (89,5% of price variance explained)
- RMSE: ~1,500
- MAE: ~980

## Docker Deployment

Build and run the Docker container:

```bash
# Build the image
docker build -t used-car-price-prediction .

# Run the container
docker run -p 8000:8000 -p 8501:8501 used-car-price-prediction
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`uv add -e ".[dev]"`)
4. Make your changes
5. Run tests (`pytest`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please see our Contributing Guidelines for more details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The dataset used in this project comes from a Kaggle competition
- Thanks to the open-source community for the amazing tools and libraries that made this project possible
- Special thanks to all contributors who helped improve this project

---

Made by [Quan Nguyen](https://github.com/minhquana1906).
