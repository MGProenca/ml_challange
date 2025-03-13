# Avocado Price Forecasting

This project aims to forecast avocado prices for different US regions using machine learning models. The project utilizes XGBoost for regression and leverages MLflow for experiment tracking and model management. It also includes a FastAPI application for serving the models and making predictions.

## Table of Contents

- [Explainer](#explainer)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training models and running the API](#training-models)
- [Project Structure](#project-structure)

## Explainer
In the `explainer.ipynb` notebook i explain the inner workings of my functions/configs step by step while also going through my thought process behind the data processing, feature engineering, model creation, validation and optimizations. 

At the end of the `explainer.ipynb` is also a section for testing the predictions of the API. Run this part after Training models and running the API.


## Instalation

1. Clone the repository:

   ```bash
   git clone https://github.com/MGProenca/ml_challenge.git
   cd ml_challenge
   ```

In this project i used astral-UV for dependency manegement. 
The following are instructions for using it to recreate the VENV. 
If you don't wish to use it you can just create a python 3.12 venv and install the requirements.

2. Install Astral-UV for managing versions(if not already installed):
    ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Or, if your system doesn't have curl, you can use wget:

   ```bash
   wget -qO- https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment and activate it:

   ```bash
   uv venv
   source .venv/bin/activate
   ```

4. Install the required dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```


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

