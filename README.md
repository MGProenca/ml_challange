# Avocado Price Forecasting

This project aims to forecast avocado prices for different regions using machine learning models. The project utilizes XGBoost for regression and leverages MLflow for experiment tracking and model management. The project also includes a FastAPI application for serving the models and making predictions.

## Table of Contents

- [Installation](#installation)
- [Explainer](#explainer)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training Models](#training-models)
- [Serving Models](#serving-models)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

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
In the `explainer.ipynb` notebook i go step by step explaining the inner workings of my functions/configs while also going through my thought process behind the data processing, feature engineering, model creation, validation and optimizations.


## Usage

### Configuration

The configuration for the project is defined in the `load_configs` function in the `train_models.py` file. This includes the target regions, lags, auxiliary regions, and features.

### Training Models

To train the models for each region, run the `train_models.py` script:

```bash
python train_models.py
```

This script will:

- Load the configuration and data.
- Train an XGBoost model for each region.
- Log the model and metrics to MLflow.
- Register the model in the MLflow model registry.

On my machine it took about 5 minutes to run

### Serving Models

To serve the models using FastAPI, run the `api.py` script:

```bash
python api.py
```

This will start the FastAPI application, which provides endpoints for making predictions.

## API Endpoints

The FastAPI application provides the following endpoints:

### **POST /predict/{region}**

Predict avocado prices for the given region using its specific model and schema.

**Example request:**

```json
{
    "TotalVolume_conventional": {"0": 12345.67, "1": 23456.78},
    "4046_conventional": {"0": 123.45, "1": 234.56},
    "4225_conventional": {"0": 234.56, "1": 345.67},
    "4770_conventional": {"0": 345.67, "1": 456.78},
    "TotalBags_conventional": {"0": 456.78, "1": 567.89},
    "SmallBags_conventional": {"0": 567.89, "1": 678.90},
    "LargeBags_conventional": {"0": 678.90, "1": 789.01},
    "XLargeBags_conventional": {"0": 789.01, "1": 890.12},
    "type": {"0": "conventional", "1": "conventional"},
    "year": {"0": 2021, "1": 2021},
    "region": {"0": "Albany", "1": "Albany"}
}
```

**Example response:**

```json
{
    "region": "Albany",
    "prediction": [1.23, 2.34]
}
```

## Project Structure

```
.
├── api.py                  # FastAPI application for serving models
├── data_processing.py      # Data processing functions
├── feature_eng.py          # Feature engineering functions
├── train_models.py         # Script for training models
├── requirements.txt        # Project dependencies
└── README.md               # Project README file
```

