import os
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("API_KEY")

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
        train_data, _ = forecaster.split_data()
        forecaster.train_model(train_data=train_data)
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
        train_data, test_data = forecaster.split_data()

        predictions = forecaster.make_predictions(
            steps=request.steps, train_data=train_data
        )

        rmse = forecaster.evaluate_model(test_data, predictions)

        # Convert predictions to a list for JSON serialization
        predictions_list = (
            predictions.tolist()
            if isinstance(predictions, pd.Series)
            else list(predictions)
        )

        response_data = {"predictions": predictions_list, "rmse": rmse}

        return response_data
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
