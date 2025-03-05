from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import toml

from skforecast_eg import TemperatureForecaster

CONFIG_FILE = Path.cwd() / "skforecast_eg.toml"


@pytest.fixture
def forecaster():
    return TemperatureForecaster(CONFIG_FILE)


def test_init(forecaster):
    assert isinstance(forecaster, TemperatureForecaster)
    assert isinstance(forecaster.df, pd.DataFrame)
    assert "temperature" in forecaster.df.columns


def test_generate_synthetic_data(forecaster):
    df = forecaster._generate_synthetic_data()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "date"
    assert "temperature" in df.columns
    assert len(df) > 0


def test_split_data(forecaster):
    train, test = forecaster.split_data()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)

    total_days = (pd.to_datetime("2024-12-31") - pd.to_datetime("2020-01-01")).days + 1
    actual_total = len(train) + len(test) - 1

    print(f"Train start: {train.index[0]}, Train end: {train.index[-1]}")
    print(f"Test start: {test.index[0]}, Test end: {test.index[-1]}")
    print(f"Train length: {len(train)}, Test length: {len(test)}")
    print(f"Total days (expected): {total_days}, Actual total: {actual_total}")

    assert actual_total == total_days, f"Expected {total_days} days, but got {actual_total}"
    assert train.index[-2].strftime('%Y-%m-%d') == "2023-06-30"
    assert test.index[0].strftime('%Y-%m-%d') == "2023-07-01"


def test_train_model(forecaster):
    train, _ = forecaster.split_data()
    forecaster.train_model(train)
    assert forecaster.forecaster is not None


def test_make_predictions(forecaster):
    train, test = forecaster.split_data()
    forecaster.train_model(train)
    predictions = forecaster.make_predictions(len(test))
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == len(test)


def test_evaluate_model(forecaster):
    train, test = forecaster.split_data()
    forecaster.train_model(train)
    predictions = forecaster.make_predictions(len(test))
    rmse = forecaster.evaluate_model(test, predictions)
    assert isinstance(rmse, float)
    assert rmse > 0


@pytest.mark.parametrize(
    "start_date,end_date,expected_length",
    [
        ("2020-01-01", "2020-12-31", 366),  # 2020 was a leap year
        ("2021-01-01", "2021-12-31", 365),
        ("2020-01-01", "2024-12-31", 1827),  # Total days for the full range
    ],
)
def test_date_range(start_date, end_date, expected_length):
    # Temporarily modify the config file for this test
    config_data = toml.load(CONFIG_FILE)
    config_data["data"]["start_date"] = start_date
    config_data["data"]["end_date"] = end_date

    # Use a temporary file to avoid modifying the original config
    temp_config_file = Path.cwd() / "temp_config.toml"
    with open(temp_config_file, "w") as f:
        toml.dump(config_data, f)

    forecaster = TemperatureForecaster(temp_config_file)
    assert len(forecaster.df) == expected_length

    temp_config_file.unlink()  # Clean up the temporary file
