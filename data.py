"""
data.py - Financial Data Ingestion & Technical Feature Engineering
Stock Price Prediction Application

Features:
- Robust historical market data ingestion via yfinance with error handling.
- Deterministic synthetic generator fallback for offline testing/unit tests.
- Quantitative Technical Indicators: SMA (20, 50), EMA (20), RSI (14), MACD (12, 26, 9),
  Bollinger Bands, Volatility (20-day standard deviation), and Momentum.
- Time-series lag feature extraction without lookahead data leakage.
- Chronological train-test splitting.
"""

from typing import Tuple, List, Optional
import datetime
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False


def generate_synthetic_stock_data(
    ticker: str = "SYNTH",
    days: int = 500,
    start_price: float = 150.0,
    volatility: float = 0.018,
    drift: float = 0.0004,
) -> pd.DataFrame:
    """
    Generates realistic synthetic stock market data using Geometric Brownian Motion (GBM).
    Used as a fallback for offline demonstration and unit testing.
    """
    np.random.seed(42)
    end_date = datetime.date.today()
    date_range = pd.bdate_range(end=end_date, periods=days)

    returns = np.random.normal(loc=drift, scale=volatility, size=len(date_range))
    price_series = start_price * np.exp(np.cumsum(returns))

    highs = price_series * (1 + np.abs(np.random.normal(0.005, 0.003, size=len(date_range))))
    lows = price_series * (1 - np.abs(np.random.normal(0.005, 0.003, size=len(date_range))))
    opens = (highs + lows) / 2 + np.random.normal(0, 0.3, size=len(date_range))
    volume = np.random.randint(10_000_000, 80_000_000, size=len(date_range))

    df = pd.DataFrame(
        {
            "Open": np.round(opens, 2),
            "High": np.round(highs, 2),
            "Low": np.round(lows, 2),
            "Close": np.round(price_series, 2),
            "Volume": volume,
        },
        index=date_range,
    )
    df.index.name = "Date"
    return df


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, bool]:
    """
    Fetches historical stock prices using yfinance.
    Returns:
        (df, is_synthetic_fallback: bool)
    """
    clean_ticker = ticker.strip().upper()
    if not clean_ticker:
        clean_ticker = "AAPL"

    if YFINANCE_AVAILABLE and yf is not None:
        try:
            ticker_obj = yf.Ticker(clean_ticker)
            if start_date and end_date:
                df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                df = ticker_obj.history(period=period, auto_adjust=True)

            if df is not None and not df.empty and len(df) >= 30:
                # Standardize columns
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                cols_present = [c for c in required_cols if c in df.columns]
                if len(cols_present) == len(required_cols):
                    df = df[required_cols].copy()
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df.index.name = "Date"
                    return df, False
        except Exception:
            pass

    # If yfinance is unavailable, offline, or ticker not found, fallback to synthetic data
    fallback_df = generate_synthetic_stock_data(ticker=clean_ticker, days=500)
    return fallback_df, True


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Relative Strength Index (RSI) using standard Wilder/Exponential smoothing.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder smoothing for remaining values
    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.rename(f"RSI_{period}")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends quantitative technical indicators:
    - SMA 20 & SMA 50 (Moving Averages)
    - EMA 20 (Exponential Moving Average)
    - RSI 14 (Momentum Oscillator)
    - MACD Line, Signal Line & Histogram (Trend Following)
    - Bollinger Bands (Upper, Middle, Lower, Bandwidth)
    - Daily Return (%) & 20-Day Rolling Volatility
    """
    data = df.copy()

    # Simple Moving Averages
    data["SMA_20"] = data["Close"].rolling(window=20, min_periods=1).mean()
    data["SMA_50"] = data["Close"].rolling(window=50, min_periods=1).mean()

    # Exponential Moving Average
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (14)
    data["RSI_14"] = calculate_rsi(data["Close"], period=14).fillna(50.0)

    # Moving Average Convergence Divergence (MACD)
    ema_12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema_12 - ema_26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    # Bollinger Bands (20-period, 2-std)
    data["BB_Middle"] = data["SMA_20"]
    rolling_std = data["Close"].rolling(window=20, min_periods=1).std().fillna(0.0)
    data["BB_Upper"] = data["BB_Middle"] + (2.0 * rolling_std)
    data["BB_Lower"] = data["BB_Middle"] - (2.0 * rolling_std)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / (data["BB_Middle"] + 1e-9)

    # Returns & Volatility
    data["Daily_Return"] = data["Close"].pct_change().fillna(0.0)
    data["Volatility_20"] = data["Daily_Return"].rolling(window=20, min_periods=1).std().fillna(0.0)

    return data


def prepare_features_and_target(
    df: pd.DataFrame,
    lags: List[int] = [1, 2, 3, 5, 7],
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Builds the feature matrix X and target y for time-series supervised learning:
    Target y_t = Close_t
    Features X_t = Lagged prices (Close_t-1, Close_t-2, etc.), lagged indicators,
                   lagged volume, and rolling metrics.

    Returns:
        (X, y, feature_names)
    """
    processed = add_technical_indicators(df)
    features_df = pd.DataFrame(index=processed.index)

    # 1. Price Lag Features (strictly lagged by >= 1 step to prevent lookahead leakage)
    for lag in lags:
        features_df[f"Close_Lag_{lag}"] = processed["Close"].shift(lag)

    # 2. Volume Lag
    features_df["Volume_Lag_1"] = processed["Volume"].shift(1)
    features_df["Volume_Ratio_5"] = (
        processed["Volume"].shift(1) / (processed["Volume"].shift(1).rolling(5, min_periods=1).mean() + 1e-9)
    )

    # 3. Indicator Lags (information known prior to current close)
    features_df["SMA_20_Lag_1"] = processed["SMA_20"].shift(1)
    features_df["SMA_50_Lag_1"] = processed["SMA_50"].shift(1)
    features_df["EMA_20_Lag_1"] = processed["EMA_20"].shift(1)
    features_df["RSI_14_Lag_1"] = processed["RSI_14"].shift(1)
    features_df["MACD_Lag_1"] = processed["MACD"].shift(1)
    features_df["MACD_Signal_Lag_1"] = processed["MACD_Signal"].shift(1)
    features_df["BB_Upper_Lag_1"] = processed["BB_Upper"].shift(1)
    features_df["BB_Lower_Lag_1"] = processed["BB_Lower"].shift(1)
    features_df["Volatility_20_Lag_1"] = processed["Volatility_20"].shift(1)

    # 4. Momentum / Trend Lags
    features_df["Return_Lag_1"] = processed["Daily_Return"].shift(1)
    features_df["Price_to_SMA20_Lag_1"] = (
        processed["Close"].shift(1) / (processed["SMA_20"].shift(1) + 1e-9)
    )

    # Target is the current Close price
    target_series = processed["Close"].copy()

    # Drop NaNs created by lagging
    valid_mask = ~features_df.isna().any(axis=1) & ~target_series.isna()
    X = features_df.loc[valid_mask].copy()
    y = target_series.loc[valid_mask].copy()

    feature_names = list(X.columns)
    return X, y, feature_names


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Chronological time-series split without data shuffling.
    Preserves sequential market structure.
    """
    n = len(X)
    split_idx = int(n * (1.0 - test_ratio))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    train_dates = X_train.index
    test_dates = X_test.index

    return X_train, X_test, y_train, y_test, train_dates, test_dates
