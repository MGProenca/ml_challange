# Avocado Price Forecasting

This project aims to forecast average avocado prices for different US regions using machine learning models. It utilizes XGBoost for regression and leverages MLflow for experiment tracking and model management. It also includes a FastAPI application for serving the models and making predictions.

My specific task will be to do a 4 week ahead forecast, and for this I'll train one separate regressor model per region.

There are out of the box time series forecasting models, i choose to implement a regression model such as XGB for a few reasons:
1) In my experience if well tuned it tends to perform better.
2) Since this is an exercise that will not be worked upon in the real world and which the objective is purely a  technical evaluation, i believe a custom solution allows me more room to express skills in machine learning engineering than models like prophet.
3) In a more time constrained environment where showing skills is not the main goal and a couple percentage points of precision are not essential I would probably go for the easier route...

<!-- ## Table of Contents
- [The data](#the-data)
- [Install and run instructions](#install-and-run-instructions)
- [How to run the explainer](#How to run the explainer)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training models and running the API](#training-models)
- [Project Structure](#project-structure) -->

## The data

The chosen dataset contains weekly avocado prices for US regions across 3 years for organic and conventional variations. 
Source: https://www.kaggle.com/datasets/neuromusic/avocado-prices/data

The columns are as follows:
 
- **Date**: The date of the observation
- **AveragePrice**: The average price of a single avocado
- **type**: Conventional or organic
- **year**: The year
- **Region**: The city or region of the observation
- **Total Volume**: Total number of avocados sold
- **4046**: Total number of avocados with PLU 4046 sold
- **4225**: Total number of avocados with PLU 4225 sold
- **4770**: Total number of avocados with PLU 4770 sold

## Install and run instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/MGProenca/ml_challenge.git
   cd ml_challenge
   ```

2. run
   ```bash
   docker compose up --build
   ```

This will run 3 separate containers, mlflow, train and API.

The mlflow container will:
- Just start the mlflow server

The train container will:
- Load the train configurations and data.
- Train an XGBoost model for each region.
- Log the models and metrics to MLflow.
- Register the models in the MLflow model registry.
- Start a jupyter server so you can run the [`explainer.ipynb`](modelling/explainer.ipynb) notebook directly inside the container.

The API container will:
- Load the models from the registry
- Build the pydantic validation schemas from the mlflow logged schemas
- Offer 2 endpoints: 
- - /reload-models: For refreshing new models logged to the registry
- - /predict/{region}: For serving predictions for each of the region models in the registry

## How to run the explainer
In the [`modelling/explainer.ipynb`](modelling/explainer.ipynb) notebook, I do a quick exploratory analysis and explain the inner workings of my functions/configs step by step while also going through my thought process behind the data processing, feature engineering, model creation, validation and optimizations. To run it, wait for the `train_model` container to launch the jupyter server, after this you can click [`here`](http://localhost:8888/notebooks/explainer.ipynb) or type `http://localhost:8888/notebooks/explainer.ipynb` in your browser to open the explainer notebook running inside the container.

## How to check the metrics
Everything is logged to the mlflow server. After the train finishes (when the jupyter server is up), you can access the mlflow UI by clicking [`here`](http://localhost:5000) or type `http://localhost:5000` in your browser to check the metrics.

## How to test the API
At the end of [`explainer.ipynb`](modelling/explainer.ipynb) is also a section for testing the the API endpoints.

## Project Structure

```
.
├── data
│   └── ...                 # Data files
├── modelling
│   ├── data                # directory with the CSV file with the avocado data
│   ├── configs.py          # Configuration file for training
│   ├── data_processing.py  # Data processing functions
│   ├── feature_eng.py      # Feature engineering functions
│   ├── region_best_params.json # Best grid searched parameters for each region
│   ├── train_models.py     # Script for training models
│   └── ...                 # Other modelling files
├── src
│   ├── api.py              # FastAPI application for serving models
├── requirements.txt        # Project dependencies
├── docker-compose.yml      # Docker Compose configuration file
├── README.md               # Project README file
│── ...               # UV and related Git files
```

