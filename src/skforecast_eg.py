from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import toml
from loguru import logger
from plotly.subplots import make_subplots
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class TemperatureForecaster:
    """
    A class for generating synthetic Sydney temperature data and forecasting using skforecast.
    """

    def __init__(self, config_file: str):
        """
        Initialise the forecaster with configuration from a TOML file.

        Args:
            config_file (str): Path to the TOML configuration file.
        """
        self.config = toml.load(config_file)
        self.df = self._generate_synthetic_data()
        self.forecaster = None
        logger.info(f"Initialised TemperatureForecaster with config from {config_file}")

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic daily temperature data for Sydney.

        Returns:
            pd.DataFrame: DataFrame with date index and temperature column.
        """
        dates = pd.date_range(
            start=self.config["data"]["start_date"],
            end=self.config["data"]["end_date"],
            freq="D",
        )
        base_temp = self.config["temperature"]["base_temp"]
        annual_amplitude = self.config["temperature"]["annual_amplitude"]
        noise = np.random.normal(
            0, self.config["temperature"]["noise_scale"], len(dates)
        )
        temperatures = (
            base_temp
            + annual_amplitude * np.sin(2 * np.pi * (dates.dayofyear / 365.25))
            + noise
        )
        df = pd.DataFrame({"date": dates, "temperature": temperatures})
        df.set_index("date", inplace=True)
        logger.debug(f"Generated synthetic data with {len(df)} records")
        return df

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and test sets.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
        split_date = self.config["data"]["split_date"]
        train = self.df[:split_date]
        test = self.df[split_date:]
        logger.info(
            f"Split data at {split_date}. Train: {len(train)}, Test: {len(test)}"
        )
        return train, test

    def train_model(self, train_data: pd.DataFrame):
        """
        Train the forecaster model.

        Args:
            train_data (pd.DataFrame): Training data.
        """
        lags = self.config["model"]["lags"]
        self.forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=lags)
        self.forecaster.fit(y=train_data["temperature"])
        logger.info(f"Trained model with {lags} lags on {len(train_data)} records")

    def make_predictions(self, steps: int) -> pd.Series:
        """
        Make predictions using the trained model.

        Args:
            steps (int): Number of steps to predict.

        Returns:
            pd.Series: Predicted temperatures.
        """
        if self.forecaster is None:
            raise ValueError("Model not trained. Call train_model() first.")
        predictions = self.forecaster.predict(steps=steps)
        logger.info(f"Made predictions for {steps} steps")
        return predictions

    def evaluate_model(self, test_data: pd.DataFrame, predictions: pd.Series) -> float:
        """
        Evaluate the model using RMSE.

        Args:
            test_data (pd.DataFrame): Actual test data.
            predictions (pd.Series): Predicted values.

        Returns:
            float: Root Mean Squared Error.
        """
        mse = mean_squared_error(test_data["temperature"], predictions)
        rmse = np.sqrt(mse)
        logger.info(f"Model evaluation RMSE: {rmse:.3f}°C")
        return rmse

    def plot_results(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, predictions: pd.Series
    ):
        """
        Plot the training data, test data, and predictions using Plotly.

        Args:
            train_data (pd.DataFrame): Training data.
            test_data (pd.DataFrame): Test data.
            predictions (pd.Series): Predicted values.
        """
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=["Sydney Daily Temperature Forecast vs Actual (°C)"],
        )

        fig.add_trace(
            go.Scatter(
                x=train_data.index,
                y=train_data["temperature"],
                name="Training Data (synthetic)",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_data["temperature"],
                name="Actual Test Data (synthetic)",
                line=dict(color="green"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=predictions,
                name="Predictions",
                line=dict(color="red", width=5),
            )
        )

        fig.update_layout(
            height=600,
            width=1000,
            title_text="Sydney Daily Temperature",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            legend_title="Legend",
        )

        fig.show()
        logger.info("Plotted results using Plotly")


if __name__ == "__main__":

    log_file = "temperature_forecast.log"
    rotation = "1 MB"
    logger.add(log_file, rotation=rotation)

    config_file = Path.cwd() / "skforecast_eg.toml"
    if not config_file.exists():
        logger.error(f"Config file {config_file} not found. Exiting.")
        exit(1)

    forecaster = TemperatureForecaster(config_file)
    train, test = forecaster.split_data()
    forecaster.train_model(train)
    predictions = forecaster.make_predictions(len(test))
    rmse = forecaster.evaluate_model(test, predictions)
    forecaster.plot_results(train, test, predictions)

    logger.info("Forecasting process completed")
