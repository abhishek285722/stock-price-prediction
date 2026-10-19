"""
test_stock.py - Automated Unit Test Suite for Stock Price Prediction Application
Tests data ingestion, quantitative technical indicators, feature engineering, model training, and recursive forecasting.
"""

import unittest
import numpy as np
import pandas as pd

from data import (
    generate_synthetic_stock_data,
    fetch_stock_data,
    add_technical_indicators,
    prepare_features_and_target,
    time_series_split,
    calculate_rsi,
)
from model import (
    train_models,
    evaluate_models,
    calculate_directional_accuracy,
    get_feature_importance,
    forecast_future_recursive,
)


class TestStockPredictionApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepare deterministic synthetic market data for fast, reliable unit testing."""
        cls.raw_data = generate_synthetic_stock_data(ticker="TEST", days=250, start_price=100.0)

    def test_synthetic_data_generator(self):
        """Verify synthetic data shape and essential market columns."""
        df = self.raw_data
        expected_cols = ["Open", "High", "Low", "Close", "Volume"]
        self.assertEqual(len(df), 250)
        for col in expected_cols:
            self.assertIn(col, df.columns)
        # Ensure High is >= Low
        self.assertTrue((df["High"] >= df["Low"]).all())
        # Ensure prices are strictly positive
        self.assertTrue((df["Close"] > 0).all())

    def test_fetch_stock_data_fallback(self):
        """Verify fetch_stock_data gracefully returns valid data even if ticker is simulated."""
        df, is_synthetic = fetch_stock_data("TEST_XYZ", period="1y")
        self.assertIsNotNone(df)
        self.assertFalse(df.empty)
        self.assertIn("Close", df.columns)

    def test_technical_indicators(self):
        """Verify computation of SMA, EMA, RSI, MACD, and Bollinger Bands."""
        df_ind = add_technical_indicators(self.raw_data)

        # Check column existence
        required_indicators = [
            "SMA_20", "SMA_50", "EMA_20", "RSI_14", "MACD", "MACD_Signal",
            "BB_Upper", "BB_Lower", "Volatility_20", "Daily_Return"
        ]
        for ind in required_indicators:
            self.assertIn(ind, df_ind.columns)

        # RSI must be strictly bounded between 0 and 100
        self.assertTrue((df_ind["RSI_14"] >= 0.0).all())
        self.assertTrue((df_ind["RSI_14"] <= 100.0).all())

        # Bollinger Bands sanity: Upper >= Middle >= Lower
        self.assertTrue((df_ind["BB_Upper"] >= df_ind["BB_Lower"]).all())

    def test_feature_engineering_and_splitting(self):
        """Verify lag features are generated without remaining NaNs and chronological split is leakage-free."""
        X, y, feature_names = prepare_features_and_target(self.raw_data)

        self.assertGreater(len(X), 100)
        self.assertEqual(len(X), len(y))
        self.assertFalse(X.isna().any().any())
        self.assertFalse(y.isna().any())

        # Verify time-series split
        X_train, X_test, y_train, y_test, train_dates, test_dates = time_series_split(
            X, y, test_ratio=0.20
        )
        self.assertGreater(len(X_train), len(X_test))
        # Ensure chronological ordering: max(train_dates) < min(test_dates)
        self.assertLess(train_dates.max(), test_dates.min())

    def test_model_training_and_evaluation(self):
        """Verify training of all 3 models and generation of test metrics."""
        X, y, feature_names = prepare_features_and_target(self.raw_data)
        X_train, X_test, y_train, y_test, _, _ = time_series_split(X, y, test_ratio=0.20)

        models, scaler = train_models(X_train, y_train)

        # Check all models trained
        expected_models = ["Linear Regression", "Ridge Regression", "Random Forest"]
        for m in expected_models:
            self.assertIn(m, models)

        # Evaluate models
        metrics_df, test_preds = evaluate_models(models, scaler, X_test, y_test)
        self.assertEqual(len(metrics_df), 3)
        self.assertIn("MAE ($)", metrics_df.columns)
        self.assertIn("RMSE ($)", metrics_df.columns)
        self.assertIn("R² Score", metrics_df.columns)
        self.assertIn("Directional Acc (%)", metrics_df.columns)

        for m in expected_models:
            self.assertIn(m, test_preds)
            self.assertEqual(len(test_preds[m]), len(y_test))

    def test_directional_accuracy_calculation(self):
        """Verify directional accuracy helper function."""
        y_true = np.array([100.0, 105.0, 102.0, 108.0])
        # Actual moves: +5, -3, +6
        # Case 1: All correct predictions
        y_pred_correct = np.array([100.0, 104.0, 101.0, 110.0])
        acc_full = calculate_directional_accuracy(y_true, y_pred_correct)
        self.assertEqual(acc_full, 100.0)

    def test_feature_importance(self):
        """Verify feature importance extraction."""
        X, y, feature_names = prepare_features_and_target(self.raw_data)
        X_train, _, y_train, _, _, _ = time_series_split(X, y, test_ratio=0.20)
        models, scaler = train_models(X_train, y_train)

        rf_model = models["Random Forest"]
        df_imp = get_feature_importance(rf_model, feature_names, top_n=5)
        self.assertEqual(len(df_imp), 5)
        self.assertIn("Feature", df_imp.columns)
        self.assertIn("Importance", df_imp.columns)

    def test_recursive_future_forecasting(self):
        """Verify recursive multi-step forecasting outputs correct future steps and confidence intervals."""
        X, y, feature_names = prepare_features_and_target(self.raw_data)
        X_train, _, y_train, _, _, _ = time_series_split(X, y, test_ratio=0.20)
        models, scaler = train_models(X_train, y_train)

        horizon = 14
        forecast_df = forecast_future_recursive(
            model=models["Random Forest"],
            scaler=scaler,
            df_raw=self.raw_data,
            feature_names=feature_names,
            horizon_days=horizon,
            historical_residual_std=2.0,
        )

        self.assertEqual(len(forecast_df), horizon)
        self.assertIn("Predicted_Close", forecast_df.columns)
        self.assertIn("Lower_Bound", forecast_df.columns)
        self.assertIn("Upper_Bound", forecast_df.columns)

        # Upper bound must be strictly >= Predicted >= Lower
        self.assertTrue((forecast_df["Upper_Bound"] >= forecast_df["Predicted_Close"]).all())
        self.assertTrue((forecast_df["Predicted_Close"] >= forecast_df["Lower_Bound"]).all())


if __name__ == "__main__":
    unittest.main()
