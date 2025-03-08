import io
import pickle
from datetime import datetime
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
from sklearn.metrics import mean_squared_error


class TemperatureForecaster:
    def __init__(
        self,
        config_path: str = str(Path.cwd() / "skforecast_eg.toml"),
        db_path: str = "data/temperature_data.duckdb",
        table_name: str = "temperature",
        train_on_start: bool = True,
        force_retrain: bool = False,
    ):
        """
        Initialise the TemperatureForecaster.

        Args:
            config_path (str): Path to the TOML configuration file.
            db_path (str): Path to the DuckDB database file.
            table_name (str): Name of the table to store temperature data.
            train_on_start (bool): Whether to train the model on startup.
            force_retrain (bool): Whether to force retraining of the model, even if a saved model exists.
        """
        self.config = toml.load(config_path)
        self.db_path = db_path
        self.table_name = table_name
        self.forecaster: Optional[ForecasterAutoreg] = None
        self.connection = duckdb.connect(self.db_path)
        self.train_on_start = train_on_start
        self.force_retrain = force_retrain
        self.model_table = "models"  # Table to store models and metadata

        # Configure logger
        logger.add(
            self.config["logging"]["log_file"],
            rotation=self.config["logging"]["rotation"],
        )
        logger.info("TemperatureForecaster initialised")

        # Generate synthetic data if the table doesn't exist
        try:
            self.connection.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
        except duckdb.CatalogException:
            logger.info(
                f"Table {self.table_name} not found. Generating synthetic data."
            )
            self._generate_synthetic_data()

        # Initialize the models table
        self._init_model_table()

        # Load model if it exists and force_retrain is False
        if not self.force_retrain:
            self.load_latest_model()
        elif self.train_on_start:
            self.train()

    def _init_model_table(self):
        """
        Initialize the table to store models and metadata.
        """
        # Create sequence
        self.connection.execute(
            f"CREATE SEQUENCE IF NOT EXISTS {self.model_table}_id_seq;"
        )

        query = f"""
            CREATE TABLE IF NOT EXISTS {self.model_table} (
                id INTEGER DEFAULT nextval('{self.model_table}_id_seq'),
                training_start_date DATE,
                training_end_date DATE,
                split_date DATE,
                rmse DOUBLE,
                model BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """
        self.connection.execute(query)
        logger.info(f"Model table {self.model_table} initialised/verified")

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

        common_dates = test_data.index.intersection(predictions.index)
        test_data_aligned = test_data.loc[common_dates]
        predictions_aligned = predictions.loc[common_dates]

        rmse = np.sqrt(
            np.mean(
                (test_data_aligned["temperature"].values - predictions_aligned.values)
                ** 2
            )
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

    def train(self):
        """
        Train the model and persist it to the database.
        """
        # Ensure synthetic data exists
        try:
            self.connection.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
        except duckdb.CatalogException:
            logger.info(
                f"Table {self.table_name} not found. Generating synthetic data."
            )
            self._generate_synthetic_data()

        train_data, test_data = self.split_data()

        self.train_model(train_data)
        predictions = self.make_predictions(
            len(test_data), train_data
        )  # need train data
        rmse = self.evaluate_model(test_data, predictions)

        # Persist the model to the database
        self.save_model(
            training_start_date=str(train_data.index[0].date()),
            training_end_date=str(train_data.index[-1].date()),
            split_date=str(test_data.index[0].date()),
            rmse=rmse,
        )

    def save_model(
        self,
        training_start_date: str,
        training_end_date: str,
        split_date: str,
        rmse: float,
    ):
        """
        Save the trained model to the DuckDB database with metadata.

        Args:
            training_start_date (str): Start date of the training data.
            training_end_date (str): End date of the training data.
            split_date (str): Date used to split the training and test sets.
            rmse (float): Root Mean Squared Error of the model on the test set.
        """
        if not self.forecaster:
            logger.error("No model to save.")
            return

        # try:
        # Serialize the model using pickle
        model_bytes = io.BytesIO()
        pickle.dump(self.forecaster, model_bytes)
        model_bytes = model_bytes.getvalue()

        # Insert the model and metadata into the database
        query = f"""
            INSERT INTO {self.model_table} (training_start_date, training_end_date, split_date, rmse, model)
            VALUES (?, ?, ?, ?, ?)
        """
        self.connection.execute(
            query,
            (training_start_date, training_end_date, split_date, rmse, model_bytes),
        )
        logger.info("Model saved to database")
        # except Exception as e:
        #     logger.error(f"Failed to save model to database: {e}")

    def load_latest_model(self):
        """
        Load the latest trained model from the DuckDB database.
        """
        try:
            # Query the database for the latest model
            query = f"""
                SELECT model FROM {self.model_table}
                ORDER BY created_at DESC
                LIMIT 1
            """
            result = self.connection.execute(query).fetchone()

            if result:
                # Deserialize the model using pickle
                model_bytes = result[0]
                model = pickle.loads(model_bytes)
                self.forecaster = model
                logger.info("Latest model loaded from database")
            else:
                logger.info("No model found in database")
        except Exception as e:
            logger.error(f"Failed to load model from database: {e}")

    def get_model_history(self) -> pd.DataFrame:
        """
        Retrieve the model training history from the DuckDB database.

        Returns:
            pd.DataFrame: DataFrame containing the model training history.
        """
        try:
            query = f"""
                SELECT id, training_start_date, training_end_date, split_date, rmse, created_at
                FROM {self.model_table}
                ORDER BY created_at DESC
            """
            history = self.connection.execute(query).fetchdf()
            logger.info("Model history retrieved from database")

            # Convert Timestamp objects to strings
            history["training_start_date"] = history["training_start_date"].astype(str)
            history["training_end_date"] = history["training_end_date"].astype(str)
            history["split_date"] = history["split_date"].astype(str)
            history["created_at"] = history["created_at"].astype(str)

            return history
        except Exception as e:
            logger.error(f"Failed to retrieve model history from database: {e}")
            return pd.DataFrame()  # Return an empty DataFrame in case of error
