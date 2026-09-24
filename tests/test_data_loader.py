"""
Unit tests for DataLoader and Feature Engineering modules.
"""

import numpy as np
import pandas as pd
import pytest
from src.data.data_loader import DataLoader
from src.features.risk_factors import RiskFactorEngine


def test_synthetic_data_generation():
    tickers = ["SPY", "TLT", "GLD"]
    df = DataLoader.generate_synthetic_market_data(tickers, periods=100, random_state=42)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 100
    assert list(df.columns) == tickers
    assert not df.isnull().values.any()
    assert (df.values > 0).all()


def test_log_returns_computation():
    prices = pd.DataFrame({
        "A": [100.0, 105.0, 102.0, 108.0],
        "B": [50.0, 52.0, 51.0, 53.0],
    })
    returns = DataLoader.compute_log_returns(prices)

    assert len(returns) == 3
    assert np.isclose(returns.iloc[0]["A"], np.log(105.0 / 100.0))
    assert np.isclose(returns.iloc[1]["B"], np.log(51.0 / 52.0))


def test_rolling_realized_volatility():
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, 100),
    })
    vol = RiskFactorEngine.compute_rolling_realized_volatility(returns, window=21, annualize=True)
    assert len(vol) == 100 - 21 + 1
    assert (vol.values >= 0).all()


def test_drawdown_series():
    prices = pd.DataFrame({"A": [100.0, 110.0, 99.0, 88.0, 110.0]})
    dd = RiskFactorEngine.compute_drawdown_series(prices)

    assert dd.iloc[0]["A"] == 0.0
    assert dd.iloc[1]["A"] == 0.0
    assert np.isclose(dd.iloc[2]["A"], (99.0 - 110.0) / 110.0)
    assert dd.max().values[0] == 0.0
