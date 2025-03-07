from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import toml
from loguru import logger
from plotly.subplots import make_subplots
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import LinearRegression


class TemperatureForecaster:
    def __init__(
        self,
        config_path: str = str(Path.cwd() / "skforecast_eg.toml"),
        db_path: str = "data/temperature_data.duckdb",
        table_name: str = "temperature",
    ):
        """
        Initialise the TemperatureForecaster.

        Args:
            config_path (str): Path to the TOML configuration file.
            db_path (str): Path to the DuckDB database file.
            table_name (str): Name of the table to store temperature data.
        """
        self.config = toml.load(config_path)
        self.db_path = db_path
        self.table_name = table_name
        self.forecaster: Optional[ForecasterAutoreg] = None
        self.connection = duckdb.connect(self.db_path)

        # Configure logger
        logger.add(
            self.config["logging"]["log_file"],
            rotation=self.config["logging"]["rotation"],
        )
        logger.info("TemperatureForecaster initialised")

    def _generate_synthetic_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force: bool = False,
    ):
        """
        Generate synthetic temperature data and store it in DuckDB if not already present.

        Args:
            start_date (str, optional): Start date for synthetic data generation. Defaults to config value.
            end_date (str, optional): End date for synthetic data generation. Defaults to config value.
            force (bool): If True, regenerate and overwrite existing data.
        """
        start_date = start_date or self.config["data"]["start_date"]
        end_date = end_date or self.config["data"]["end_date"]

        logger.info(f"Generating synthetic data from {start_date} to {end_date}")

        if not force:
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            try:
                count = self.connection.execute(query).fetchone()[0]
                if count > 0:
                    logger.info(
                        f"Data already exists in {self.table_name}. Skipping generation."
                    )
                    return
            except duckdb.CatalogException:
                pass  # Table does not exist yet

        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        base_temp = self.config["temperature"]["base_temp"]
        annual_amplitude = self.config["temperature"]["annual_amplitude"]
        noise_scale = self.config["temperature"]["noise_scale"]
        noise = np.random.normal(0, noise_scale, len(dates))
        temperatures = (
            base_temp
            + annual_amplitude * np.sin(2 * np.pi * dates.dayofyear / 365.25)
            + noise
        )

        df = pd.DataFrame({"date": dates, "temperature": temperatures})

        self.connection.execute(
            f"CREATE OR REPLACE TABLE {self.table_name} AS SELECT * FROM df"
        )
        logger.info(f"Synthetic data generated and stored in {self.table_name}")

    def query_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Query temperature data from DuckDB within a specified date range.

        Args:
            start_date (str): Start date for querying.
            end_date (str): End date for querying.

        Returns:
            pd.DataFrame: Queried temperature data.
        """
        logger.info(f"Querying data from {start_date} to {end_date}")
        query = f"""
            SELECT date, temperature FROM {self.table_name}
            WHERE date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date
        """
        return self.connection.execute(query).fetchdf()

    def split_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        split_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and test sets.

        Args:
            start_date (str, optional): Start date for data. Defaults to config value.
            end_date (str, optional): End date for data. Defaults to config value.
            split_date (str, optional): Date to split train and test data. Defaults to config value.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and test DataFrames.
        """
        start_date = start_date or self.config["data"]["start_date"]
        end_date = end_date or self.config["data"]["end_date"]
        split_date = split_date or self.config["data"]["split_date"]

        logger.info(
            f"Splitting data. Train: {start_date} to {split_date}, Test: {split_date} to {end_date}"
        )

        df = self.query_data(start_date, end_date)

        # Ensure the index is in datetime format
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

        # Convert split_date to datetime
        split_datetime = datetime.strptime(split_date, "%Y-%m-%d")

        train = df[df.index < split_datetime]
        test = df[df.index >= split_datetime]

        logger.info(
            f"Data split completed. Train size: {len(train)}, Test size: {len(test)}"
        )

        return train, test

    def train_model(self, train_data: pd.DataFrame):
        """
        Train the forecasting model using the provided training data.

        Args:
            train_data (pd.DataFrame): Training data.
        """

        logger.info(
            f"Training model with data from {train_data.index[0]} to {train_data.index[-1]}"
        )

        if train_data.empty:
            logger.error("No training data available")
            raise ValueError("No training data available.")

        # Set the frequency of the DatetimeIndex
        train_data = train_data.asfreq("D")

        lags = self.config["model"]["lags"]
        self.forecaster = ForecasterAutoreg(regressor=LinearRegression(), lags=lags)

        self.forecaster.fit(y=train_data["temperature"])
        logger.info("Model trained successfully")

    def make_predictions(self, steps: int, train_data: pd.DataFrame) -> pd.Series:
        """
        Make predictions using the trained model.

        Args:
            steps (int): Number of steps ahead to predict.
            train_data (pd.DataFrame): The training data (used to get last date for prediction range).

        Returns:
            pd.Series: Predicted values with datetime index.

        Raises:
            ValueError: If no trained model exists.
        """
        if not self.forecaster:
            logger.error(
                "Model is not trained. Train the model before making predictions"
            )
            raise ValueError(
                "Model is not trained. Train the model before making predictions."
            )

        logger.info(f"Making predictions for {steps} steps ahead")

        # Get predictions
        predictions = self.forecaster.predict(steps=steps)

        # Get the last date from the training data
        last_date = train_data.index[-1]

        # Generate future dates starting from the day after last training date
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(days=1), periods=steps, freq="D"
        )

        # Create a Pandas Series with the predictions and the new dates
        return pd.Series(
            predictions.values, index=future_dates, name="predicted_temperature"
        )

    def evaluate_model(self, test_data: pd.DataFrame, predictions: pd.Series) -> float:
        """
        Evaluate the model using the provided test data and predictions.

        Args:
            test_data (pd.DataFrame): Test data.
            predictions (pd.Series): Predicted values.

        Returns:
            float: Root Mean Squared Error (RMSE).
        """
        if not self.forecaster:
            logger.error("Model is not trained. Train the model before evaluation")
            raise ValueError("Model is not trained. Train the model before evaluation.")

        logger.info(
            f"Evaluating model with data from {test_data.index[0]} to {test_data.index[-1]}"
        )

        if test_data.empty:
            logger.error("No test data available")
            raise ValueError("No test data available.")

        rmse = np.sqrt(
            np.mean((test_data["temperature"].values - predictions.values) ** 2)
        )

        logger.info(f"Model evaluation completed. RMSE: {rmse}")

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
                x=predictions.index,
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

    def plot_rmse_by_steps(self, test_data: pd.DataFrame, train_data: pd.DataFrame):
        """
        Calculate RMSE for different prediction steps and plot the results.

        Args:
            test_data (pd.DataFrame): Test data.
            train_data (pd.DataFrame): Training data.
        """
        # Store RMSE values for different steps
        rmse_values = []
        steps_range = range(1, len(test_data) + 1)

        # Calculate RMSE for each number of steps
        for steps in steps_range:
            predictions = self.make_predictions(steps=steps, train_data=train_data)

            # Align predictions with the test data for evaluation
            common_dates = test_data.index.intersection(predictions.index)
            if common_dates.empty:
                logger.warning(
                    f"No common dates between predictions and test data for {steps} steps."
                )
                rmse_values.append(np.nan)  # or some default value
                continue

            test_subset = test_data.loc[common_dates]
            predictions_subset = predictions.loc[common_dates]

            rmse = np.sqrt(
                np.mean(
                    (test_subset["temperature"].values - predictions_subset.values) ** 2
                )
            )
            rmse_values.append(rmse)

        # Create a plot
        fig = go.Figure(
            data=[go.Scatter(x=list(steps_range), y=rmse_values, mode="lines+markers")]
        )

        fig.update_layout(
            title="RMSE vs. Prediction Steps",
            xaxis_title="Prediction Steps",
            yaxis_title="RMSE",
        )

        fig.show()
        logger.info("Plotted RMSE vs. Prediction Steps")


CONFIG_FILE = Path.cwd() / "skforecast_eg.toml"

if __name__ == "__main__":
    if not CONFIG_FILE.exists():
        logger.error(f"Config file {CONFIG_FILE} not found. Exiting.")
        exit(1)

    forecaster = TemperatureForecaster()
    forecaster._generate_synthetic_data()  # Generate synthetic temperature data (if needed)
    train_data, test_data = forecaster.split_data()
    forecaster.train_model(train_data)

    predictions = forecaster.make_predictions(
        steps=len(test_data), train_data=train_data
    )
    rmse = forecaster.evaluate_model(test_data, predictions)

    forecaster.plot_results(train_data, test_data, predictions)
    forecaster.plot_rmse_by_steps(test_data, train_data)
