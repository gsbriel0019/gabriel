"""
Macroeconomic Indicators and Stress Factor Loader.
Provides yield curve slopes, volatility indices (VIX), and monetary policy proxies.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MacroLoader:
    """Fetches and aligns macroeconomic indicators for stress testing and regime detection."""

    MACRO_TICKERS: Dict[str, str] = {
        "VIX": "^VIX",         # CBOE Volatility Index
        "US10Y": "^TNX",       # 10-Year Treasury Yield
        "US3M": "^IRX",        # 13-Week Treasury Bill (Proxy for Short Rate)
        "USD": "UUP",          # Invesco DB US Dollar Index Bullish Fund
        "OIL": "USO",          # United States Oil Fund (Commodity inflation shock)
    }

    def __init__(self, cache_dir: Union[str, Path] = "data/cache", use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch_macro_indicators(
        self,
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Download and compute macroeconomic features:
        - VIX level and VIX delta
        - 10Y Treasury yield
        - 10Y - 3M Yield Curve Slope (inversion indicator)
        """
        cache_file = self.cache_dir / f"macro_indicators_{start_date}_{end_date or 'latest'}.csv"

        if self.use_cache and cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

        ticker_symbols = list(self.MACRO_TICKERS.values())
        try:
            raw = yf.download(
                tickers=ticker_symbols,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
            )

            df = pd.DataFrame(index=raw.index)
            for name, sym in self.MACRO_TICKERS.items():
                if isinstance(raw.columns, pd.MultiIndex):
                    if sym in raw["Adj Close"]:
                        df[name] = raw["Adj Close"][sym]
                else:
                    df[name] = raw["Adj Close"]

            df = df.ffill().bfill()

            # Compute yield curve slope: 10Y Yield minus 3M Yield
            if "US10Y" in df.columns and "US3M" in df.columns:
                df["Yield_Curve_Slope"] = df["US10Y"] - df["US3M"]

            if "VIX" in df.columns:
                df["VIX_pct_change"] = df["VIX"].pct_change().fillna(0)

            if self.use_cache and not df.empty:
                df.to_csv(cache_file)

            return df

        except Exception as e:
            logger.warning(f"Error fetching macro indicators: {e}. Generating synthetic macro data.")
            return self._generate_synthetic_macro(start_date=start_date)

    def _generate_synthetic_macro(self, start_date: str = "2018-01-01", periods: int = 1500) -> pd.DataFrame:
        """Synthetic macro data for deterministic testing."""
        dates = pd.date_range(start=start_date, periods=periods, freq="B")
        import numpy as np
        rng = np.random.default_rng(42)

        vix = 18.0 + 8.0 * np.sin(np.linspace(0, 15, periods)) + rng.normal(0, 2, periods)
        vix = np.clip(vix, 9.0, 80.0)
        us10y = 2.5 + 1.5 * np.linspace(0, 1, periods) + rng.normal(0, 0.2, periods)
        us3m = 1.0 + 3.0 * np.linspace(0, 1, periods) + rng.normal(0, 0.1, periods)

        df = pd.DataFrame(
            {
                "VIX": vix,
                "US10Y": us10y,
                "US3M": us3m,
                "Yield_Curve_Slope": us10y - us3m,
                "VIX_pct_change": pd.Series(vix).pct_change().fillna(0).values,
            },
            index=dates,
        )
        return df
