import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Path, Body
from pydantic import create_model
from mlflow.models import Model
import datetime as dt
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv
import os
from contextlib import asynccontextmanager

def make_input_schema(model_uri, model_name):
    """
    Create a Pydantic model for input data validation based on the MLflow model schema.

    Args:
        model_uri (str): The URI of the MLflow model.
        model_name (str): The name of the model.

    Returns:
        pydantic.BaseModel: A Pydantic model for input data validation.
    """
    mlflow_model = Model.load(model_uri)
    signature = mlflow_model.signature

    if signature and signature.inputs:
        input_schema = signature.inputs.to_dict() 
    else:
        raise ValueError(f"Model signature not found! Ensure input schema is logged.")

    type_mapping = {
        "string": str,
        "integer": int,
        "long": int,
        "float": float,
        "double": float,
        "boolean": bool,
        "date": dt.date,
        "timestamp": dt.datetime,
        "binary": bytes,
        "map": dict,
        "array": list,
        "struct": dict  
    }

    fields = {}
    for col in input_schema:
        col_name = col.get("name")
        mlflow_type = col.get("type", "string").lower()
        py_type = type_mapping.get(mlflow_type, str)
        fields[col_name] = (py_type, ...)

    return create_model(f"schema_{model_name}", **fields)

def load_all_models():
    """
    Load all registered models from MLflow and store them in a global dictionary.

    Returns:
        dict: A dictionary with region as keys and model details as values.
    """
    all_region_models_dict = {}
    client = mlflow.MlflowClient()
    
    registered_models = client.search_registered_models()
    
    for registered_model in registered_models:
        try:
            print('LOADING MODEL')
            model_name = registered_model.name
            print(f"Model Name: {model_name}")
            
            # Get the latest version of the model
            latest_version_info = client.get_registered_model(model_name).latest_versions[0]  # Latest version
            model_tags = latest_version_info.tags

            region = model_tags.get('region')
            print(f'Region: {region}')
            
            if not region:
                print(f"Warning: Model '{model_name}' has no 'region' tag. Skipping...")
                continue

            model_uri = latest_version_info.source
            print(f"Model URI: {model_uri}")

            all_region_models_dict[region] = {
                'model_name': model_name,
                'uri': model_uri,
                'loaded_model': mlflow.pyfunc.load_model(model_uri),
                'input_schema': make_input_schema(model_uri, model_name) 
            }
        
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            continue  # Continue with the next model if there is an error

    global all_region_models     
    all_region_models = all_region_models_dict


def validate_input_data(region: str = Path(...), data: List[dict] = Body(...)):
    """
    Validate input data based on the region-specific schema.

    Args:
        region (str): The region for which the data is being validated.
        data (List[dict]): The input data to be validated.

    Returns:
        List[dict]: The validated data.

    Raises:
        HTTPException: If the region schema is not found or data is invalid.
    """
    if region not in all_region_models:
        raise HTTPException(status_code=400, detail=f"Schema for region '{region}' not found")

    schema = all_region_models[region]['input_schema']
    try:
        validated_data = [schema(**item).dict() for item in data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    return validated_data

all_region_models = {}

# Lifespan event for FastAPI (runs before handling requests)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event for FastAPI to load models before handling requests.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None
    """
    load_dotenv()
    # Glogal var to store models in memory
    all_region_models = None
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    all_region_models = load_all_models()
    
    yield

    print("🛑 Shutting down API...")
    all_region_models.clear()


app = FastAPI(lifespan=lifespan)

@app.post("/reload-models")
def reload_models():
    """
    Reload all registered models from MLflow and update the global dictionary.

    Returns:
        dict: A message indicating the models have been reloaded.
    """
    load_all_models()
    return {"message": "Models reloaded successfully"}

# Prediction Endpoint
@app.post("/predict/{region}")
def predict(region:str, 
            validated_data: List[dict] = Depends(validate_input_data)
            ):
    """
    Predict avocado price for the given region using its specific model and schema.
    Input Data needs to be in the orient='records'

    Args:
        region (str): The region for which the prediction is made.
        validated_data (List[dict]): The validated input data.

    Returns:
        dict: The prediction results.

    Raises:
        HTTPException: If the model for the region is not found.
    """
    
    if region not in all_region_models:
        raise HTTPException(status_code=404, detail=f"Model for region '{region}' not found.")

    df = pd.DataFrame.from_dict(validated_data)
    
    # Convert int64 to int32 to avoid typing errors 
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype(np.int32)

    # fetch correct model
    model = all_region_models[region]['loaded_model']
    prediction = model.predict(df)

    return {"region": region, "prediction": prediction.tolist()}