# Avocado Price Forecasting

This project aims to forecast avocado prices for different US regions using machine learning models. The project utilizes XGBoost for regression and leverages MLflow for experiment tracking and model management. It also includes a FastAPI application for serving the models and making predictions.

## Table of Contents

- [Installation](#installation)
- [Explainer](#explainer)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training models and running the API](#training-models)
- [Project Structure](#project-structure)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/avocado-price-forecasting.git
   cd avocado-price-forecasting
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Explainer
In the `explainer.ipynb` notebook i explain the inner workings of my functions/configs step by step while also going through my thought process behind the data processing, feature engineering, model creation, validation and optimizations.


## Usage

### Configuration

The configuration for the trainning is defined in the `configs.py` file in the `modelling` directory. This includes the target regions, lags, auxiliary regions, and features. The `explainer.ipynb` notebook gives more details. The important part to know is that the `target_regions` field specifies which regions will be trained and deployed.

### Training models and running the API

First build and run the docker containers with 

```bash
docker compose up --build
```

This will start an mlflow server, run the trainning script and launch the API to serve predictions.

The train.py script will:
- Load the configuration and data.
- Train an XGBoost model for each region.
- Log the model and metrics to MLflow.
- Register the model in the MLflow model registry.

The API will:
- Load the models from the registry
- Build the pydantic validation schemas from the mlflow logged schemas
- Offer 2 endpoints: 
- - /reload-models: For refreshing new models logged to the registry
- - /predict/{region}: For serving predictions for each of the region models in the registry

## Project Structure

```
.
├── data
│   └── ...                 # Data files
├── modelling
│   ├── configs.py          # Configuration file for training
│   ├── data_processing.py  # Data processing functions
│   ├── feature_eng.py      # Feature engineering functions
│   ├── region_best_params.json # Best parameters for each region
│   ├── train_models.py     # Script for training models
│   └── ...                 # Other modelling files
├── src
│   ├── api.py              # FastAPI application for serving models
├── requirements.txt        # Project dependencies
├── docker-compose.yml      # Docker Compose configuration file
└── README.md               # Project README file
```

