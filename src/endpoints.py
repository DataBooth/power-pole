import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel

load_dotenv()

# Ensure the API key is set
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
        db_path="data/temp_forecast.duckdb",
        table_name="temperature",
        train_on_start=True,
        force_retrain=False,
    )
    logger.info("TemperatureForecaster instantiated successfully")
except Exception as e:
    logger.error(f"Failed to instantiate TemperatureForecaster: {e}")
    raise

# Split data to have test_data available for predictions
try:
    train_data, test_data = forecaster.split_data()
    logger.info("Data split successfully on startup")
except Exception as e:
    logger.error(f"Failed to split data on startup: {e}")
    train_data = None
    test_data = None

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Function to validate the API key provided in the header.
    """
    logger.info("Validating API key")
    if api_key_header == API_KEY:
        logger.info("API key validated successfully")
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


class ModelHistoryEntry(BaseModel):
    id: int
    training_start_date: str
    training_end_date: str
    split_date: str
    rmse: float
    created_at: str


@app.get("/healthcheck")
async def healthcheck():
    """
    Endpoint for basic health check.
    """
    logger.info("Health check endpoint hit")
    return {"status": "ok"}


@app.post("/train")
async def train_model(api_key: str = Depends(get_api_key)):
    """
    Endpoint to force retrain the model.
    """
    logger.info("Train endpoint hit")
    # try:
    forecaster.train()
    logger.info("Model retrained successfully")
    return {"message": "Model retrained successfully"}
    # except Exception as e:
    #     logger.error(f"Training failed: {e}")
    #     raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(
    request: PredictionRequest, api_key: str = Depends(get_api_key)
):
    """
    Endpoint to make predictions and return RMSE.
    """
    logger.info(f"Predict endpoint hit, steps={request.steps}")
    try:
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

        logger.info(f"Predictions made successfully, RMSE={rmse}")
        return response_data
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model_history", response_model=List[ModelHistoryEntry])
async def get_model_history(api_key: str = Depends(get_api_key)):
    """
    Endpoint to retrieve the model training history.
    """
    logger.info("Model history endpoint hit")
    try:
        history_df = forecaster.get_model_history()

        # Convert DataFrame to a list of dictionaries
        history = history_df.to_dict(orient="records")

        logger.info("Model history retrieved successfully")
        return history
    except Exception as e:
        logger.error(f"Failed to retrieve model history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
