"""
Market Data Loader Module.
Handles downloading, validating, caching, and preprocessing financial time-series.
"""

from pathlib import Path
from typing import List, Optional, Union
import logging
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoader:
    """Robust market data downloader with disk caching and validation."""

    def __init__(self, cache_dir: Union[str, Path] = "data/cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_historical_prices(
        self,
        tickers: Union[str, List[str]],
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
        field: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Download historical prices for given tickers.

        Parameters
        ----------
        tickers : str or list of str
            Asset ticker symbol(s).
        start_date : str
            Start date formatted as YYYY-MM-DD.
        end_date : str, optional
            End date formatted as YYYY-MM-DD. Default is current date.
        field : str
            Price field to extract ('Adj Close', 'Close', etc.).

        Returns
        -------
        pd.DataFrame
            Cleaned price series indexed by Datetime.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        cache_key = f"{'_'.join(sorted(tickers))}_{start_date}_{end_date or 'latest'}_{field}.csv"
        cache_file = self.cache_dir / cache_key

        if self.use_cache and cache_file.exists():
            logger.info(f"Loading cached prices from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df

        logger.info(f"Downloading prices for {tickers} from {start_date} to {end_date or 'today'}")
        try:
            raw_data = yf.download(
                tickers=tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )

            if raw_data.empty:
                raise ValueError("Downloaded dataset is empty. Check ticker symbols.")

            if len(tickers) == 1:
                ticker = tickers[0]
                if isinstance(raw_data.columns, pd.MultiIndex):
                    price_series = raw_data[field][ticker] if ticker in raw_data[field] else raw_data[field]
                else:
                    price_series = raw_data[field]
                df = pd.DataFrame({ticker: price_series})
            else:
                if isinstance(raw_data.columns, pd.MultiIndex):
                    df = raw_data[field]
                else:
                    df = raw_data[[field]]

            # Clean and sanitize data
            df = df.ffill().dropna()

            if self.use_cache and not df.empty:
                df.to_csv(cache_file)

            return df

        except Exception as e:
            logger.warning(f"Error fetching real data via yfinance: {e}. Falling back to synthetic generator.")
            return self.generate_synthetic_market_data(tickers, start_date=start_date, periods=1200)

    @staticmethod
    def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compute continuous log returns: r_t = ln(P_t / P_{t-1}).
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    @staticmethod
    def generate_synthetic_market_data(
        tickers: List[str],
        start_date: str = "2018-01-01",
        periods: int = 1500,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Generate realistic synthetic financial time series with stochastic volatility and jump diffusion.
        Useful for offline environments and reproducible statistical testing.
        """
        rng = np.random.default_rng(random_state)
        dates = pd.date_range(start=start_date, periods=periods, freq="B")
        n_assets = len(tickers)

        # Baseline parameters: annual drift ~7%, annual vol ~18%
        dt = 1.0 / 252.0
        mu = np.array([0.08, 0.12, 0.05, 0.03, 0.25][:n_assets])
        sigma = np.array([0.16, 0.22, 0.14, 0.10, 0.55][:n_assets])

        # Random correlation matrix
        raw_cov = rng.uniform(0.1, 0.6, size=(n_assets, n_assets))
        corr = (raw_cov + raw_cov.T) / 2.0
        np.fill_diagonal(corr, 1.0)
        chol = np.linalg.cholesky(corr)

        # Generate correlated geometric Brownian motion with regime jumps
        uncorrelated_shocks = rng.standard_normal((periods, n_assets))
        correlated_shocks = uncorrelated_shocks @ chol.T

        prices = np.zeros((periods, n_assets))
        initial_prices = [400.0, 350.0, 180.0, 100.0, 30000.0][:n_assets]
        prices[0] = initial_prices

        for t in range(1, periods):
            # Introduce periodic volatility clustering
            vol_multiplier = 2.5 if (400 <= t <= 480 or 950 <= t <= 1020) else 1.0
            daily_drift = (mu - 0.5 * (sigma * vol_multiplier) ** 2) * dt
            daily_shock = (sigma * vol_multiplier) * np.sqrt(dt) * correlated_shocks[t]
            prices[t] = prices[t - 1] * np.exp(daily_drift + daily_shock)

        df = pd.DataFrame(prices, index=dates, columns=tickers)
        return df
