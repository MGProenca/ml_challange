import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Path, Body
from pydantic import create_model
from mlflow.models import Model
import datetime as dt
import numpy as np
import train_models
from typing import Dict, List

# 1️⃣ Initialize FastAPI app
app = FastAPI()
mlflow.set_tracking_uri("http://localhost:5000")  # Set this if using a tracking server

configs = train_models.load_configs()
print(configs)
regions = configs['target_regions']

# Define available regions and load models dynamically
REGION_MODELS = {}
for region in configs['target_regions']:
    print(region)
    REGION_MODELS[region] = f"models:/{region}_AVOCADO_FORECAST/latest"

models = {}
schemas = {}

for region, model_uri in REGION_MODELS.items():
    print(f"Loading model for {region}...")
    models[region] = mlflow.pyfunc.load_model(model_uri)
    print('LOADED')
    # Extract mlflow schema
    mlflow_model = Model.load(model_uri)
    signature = mlflow_model.signature

    if signature and signature.inputs:
        input_schema = signature.inputs.to_dict() 
    else:
        raise ValueError(f"Model signature not found for {region}! Ensure input schema is logged.")

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
        # Expecting each 'col' is a dict like: {"name": "TotalVolume_conventional", "type": "long"}
        col_name = col.get("name")
        mlflow_type = col.get("type", "string").lower()
        py_type = type_mapping.get(mlflow_type, str)
        fields[col_name] = (py_type, ...)

    # Generate the region-specific Pydantic model
    schemas[region] = create_model(f"InputData_{region}", **fields)

print('AAA')
# Dependency to validate data based on region
def get_validated_data(region: str = Path(...), data: List[dict] = Body(...)):
    # region = region.lower()
    if region not in schemas:
        raise HTTPException(status_code=400, detail=f"Schema for region '{region}' not found")

    schema = schemas[region]
    try:
        validated_data = [schema(**item).dict() for item in data]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid data: {str(e)}")
    return validated_data


# **3️⃣ Define Prediction Endpoint**
@app.post("/predict/{region}")
def predict(region:str, 
            validated_data: List[dict] = Depends(get_validated_data)
            ):
    """Predict avocado price for the given region using its specific model and schema."""
    
    if region not in models:
        raise HTTPException(status_code=404, detail=f"Model for region '{region}' not found.")

    # # **4️⃣ Validate Input Using Region-Specific Schema**
    # InputSchema = schemas[region]  # Get the Pydantic model for the region
    # validated_data = InputSchema(**request)  # Validate input payload

    # Convert validated input to DataFrame 
    df = pd.DataFrame.from_dict(validated_data)
    
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = df[col].astype(np.int32)

    model = models[region]
    prediction = model.predict(df)

    return {"region": region, "prediction": prediction.tolist()}