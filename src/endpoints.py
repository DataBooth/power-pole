from fastapi import FastAPI, Depends, HTTPException, status, Security
from fastapi.security import APIKeyHeader, APIKeyQuery, APIKey
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import secrets
import pandas as pd
import numpy as np
from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variables
API_KEY = os.getenv("API_KEY")

# Ensure the API key is set
if not API_KEY:
    logger.error("API_KEY not found in .env file. Exiting.")
    raise ValueError("API_KEY not found in .env file.")

# Ensure the TemperatureForecaster is importable, adjust path if needed
try:
    from skforecast_eg import TemperatureForecaster
except ImportError:
    logger.error("Failed to import TemperatureForecaster. Check the file path.")
    raise

app = FastAPI()

# Instantiate the TemperatureForecaster - Adjust paths if necessary
try:
    forecaster = TemperatureForecaster(
        config_path=str(Path.cwd() / "skforecast_eg.toml"),
        db_path="data/temperature_data.duckdb",
        table_name="temperature",
    )
except Exception as e:
    logger.error(f"Failed to instantiate TemperatureForecaster: {e}")
    raise

# Define API key authentication scheme using APIKeyHeader
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Function to validate the API key provided in the header.
    """
    if api_key_header == API_KEY:
        return api_key_header
    else:
        logger.warning("Invalid API key provided.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )


class TrainData(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    split_date: Optional[str] = None


class PredictionRequest(BaseModel):
    steps: int


class PredictionResponse(BaseModel):
    predictions: list
    rmse: float


@app.get("/healthcheck")
async def healthcheck():
    """
    Endpoint for basic health check.
    """
    return {"status": "ok"}


@app.post("/train")
async def train_model(api_key: str = Depends(get_api_key)):
    """
    Endpoint to train the model. Requires a valid API key.
    """
    try:
        forecaster.train_model()
        return {"message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest, api_key: str = Depends(get_api_key)
):
    """
    Endpoint to make predictions and return RMSE. Requires a valid API key.
    """
    try:
        # Split the data
        train_data, test_data = forecaster.split_data()

        # Make predictions
        predictions = forecaster.make_predictions(steps=len(test_data))

        # Evaluate the model to get RMSE
        rmse = forecaster.evaluate_model(test_data)

        # Convert predictions to a list for JSON serialization
        predictions_list = (
            predictions.tolist()
            if isinstance(predictions, pd.Series)
            else list(predictions)
        )

        # Prepare the response
        response_data = {"predictions": predictions_list, "rmse": rmse}

        return response_data
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
