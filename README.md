# Avocado Price Forecasting

This project aims to forecast average avocado prices for different US regions using machine learning models. It utilizes XGBoost for regression and leverages MLflow for experiment tracking and model management. It also includes a FastAPI application for serving the models and making predictions.

My specific task will be to do a 4 week ahead forecast, and for this I'll train one separate regressor model per region.

There are out of the box time series forecasting models, i choose to implement a regression model such as XGB for a few reasons:
1) In my experience if well tuned it tends to perform better.
2) Since this is an exercise that will not be worked upon in the real world and which the objective is purely a  technical evaluation, i believe a custom solution allows me more room to express skills in machine learning engineering than models like prophet.
3) In a more time constrained environment where showing skills is not the main goal and a couple percentage points of precision are not essential I would probably go for the easier route...

## Table of Contents
- [The data](#the-data)
- [Explainer](#explainer)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training models and running the API](#training-models)
- [Project Structure](#project-structure)

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


## Explainer
In the `explainer.ipynb` notebook i do a quick exploratory analysis and explain the inner workings of my functions/configs step by step while also going through my thought process behind the data processing, feature engineering, model creation, validation and optimizations. 

At the end of the `explainer.ipynb` is also a section for testing the predictions of the API. Run this part after finishing training models and running the API with the instructions bellow.


## Installation

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

### Validating metrics
Everything is logged to the mlflow server, after the containers are up, in the browser access the [mlflow UI](http://localhost:5000) for visualizing the metrics

### Testing the API

At the end of the explainer.ipynb there is a section for testing the API, run it after the containers finished.

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

