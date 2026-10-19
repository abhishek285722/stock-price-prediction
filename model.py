"""
model.py - Quantitative Machine Learning & Multi-Step Forecasting Engine
Stock Price Prediction Application

Features:
- Multi-model training: Linear Regression, Ridge Regression, and Random Forest Regressor.
- Standardized feature scaling (preventing data leakage).
- Quantitative performance evaluation: MAE, RMSE, R², MAPE, and Directional Accuracy.
- Feature importance analysis.
- Multi-step recursive forecasting generator with dynamic volatility-adjusted confidence intervals.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from data import add_technical_indicators


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[Dict[str, Any], StandardScaler]:
    """
    Fits feature scaler and trains 3 distinct model architectures:
    1. Linear Regression (Baseline statistical trend)
    2. Ridge Regression (L2 Regularized linear model)
    3. Random Forest (Non-linear ensemble with 100 decision trees)

    Returns:
        (models_dict, fitted_scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

    return models, scaler


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes percentage of time the predicted directional price change
    matches the actual market direction: sign(y_t - y_t-1) == sign(y_hat_t - y_t-1).
    """
    if len(y_true) < 2:
        return 50.0

    actual_diff = np.diff(y_true)
    pred_diff = y_pred[1:] - y_true[:-1]

    correct_direction = np.sign(actual_diff) == np.sign(pred_diff)
    return float(np.mean(correct_direction) * 100.0)


def evaluate_models(
    models: Dict[str, Any],
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Evaluates models on unseen test data.
    Computes MAE, RMSE, R², MAPE, and Directional Accuracy.

    Returns:
        (metrics_df, test_predictions_dict)
    """
    X_test_scaled = scaler.transform(X_test)
    y_true = y_test.values

    metrics_list = []
    predictions_dict = {}

    for name, model in models.items():
        preds = model.predict(X_test_scaled)
        predictions_dict[name] = preds

        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        mape = mean_absolute_percentage_error(y_true, preds) * 100.0
        dir_acc = calculate_directional_accuracy(y_true, preds)

        metrics_list.append(
            {
                "Model": name,
                "MAE ($)": round(mae, 2),
                "RMSE ($)": round(rmse, 2),
                "R² Score": round(r2, 4),
                "MAPE (%)": round(mape, 2),
                "Directional Acc (%)": round(dir_acc, 1),
            }
        )

    metrics_df = pd.DataFrame(metrics_list).set_index("Model")
    return metrics_df, predictions_dict


def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Extracts feature importances (for tree ensembles) or absolute coefficients (for linear models).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        importances = np.ones(len(feature_names))

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(top_n)
    return df_imp


def forecast_future_recursive(
    model: Any,
    scaler: StandardScaler,
    df_raw: pd.DataFrame,
    feature_names: List[str],
    horizon_days: int = 14,
    historical_residual_std: float = 2.5,
) -> pd.DataFrame:
    """
    Performs multi-step recursive forecasting into the future.
    For each step h in [1..horizon_days]:
      1. Uses latest window of historical + predicted prices.
      2. Computes the required lag and indicator features.
      3. Scales features with the pre-fitted scaler.
      4. Predicts Close for t+h.
      5. Appends the predicted close price to the rolling working dataframe.

    Returns:
        DataFrame with Forecast_Date, Predicted_Close, Lower_Bound, Upper_Bound
    """
    working_df = df_raw.copy()
    last_date = pd.to_datetime(working_df.index[-1])

    # Generate upcoming business trading days
    future_dates = pd.bdate_range(start=last_date + pd.DateOffset(days=1), periods=horizon_days)

    forecast_records = []

    for step_idx, f_date in enumerate(future_dates, start=1):
        # Calculate full indicators on the working historical + forecasted buffer
        processed = add_technical_indicators(working_df)

        # Build feature vector matching feature_names
        latest_idx = processed.index[-1]
        feature_vector = {}

        # 1. Price Lags (relative to upcoming day, lag 1 is the most recent close in working_df)
        for lag in [1, 2, 3, 5, 7]:
            if len(working_df) >= lag:
                feature_vector[f"Close_Lag_{lag}"] = working_df["Close"].iloc[-lag]
            else:
                feature_vector[f"Close_Lag_{lag}"] = working_df["Close"].iloc[0]

        # 2. Volume Lags
        feature_vector["Volume_Lag_1"] = working_df["Volume"].iloc[-1]
        recent_vols = working_df["Volume"].iloc[-5:]
        feature_vector["Volume_Ratio_5"] = working_df["Volume"].iloc[-1] / (recent_vols.mean() + 1e-9)

        # 3. Indicator Lags
        feature_vector["SMA_20_Lag_1"] = processed["SMA_20"].loc[latest_idx]
        feature_vector["SMA_50_Lag_1"] = processed["SMA_50"].loc[latest_idx]
        feature_vector["EMA_20_Lag_1"] = processed["EMA_20"].loc[latest_idx]
        feature_vector["RSI_14_Lag_1"] = processed["RSI_14"].loc[latest_idx]
        feature_vector["MACD_Lag_1"] = processed["MACD"].loc[latest_idx]
        feature_vector["MACD_Signal_Lag_1"] = processed["MACD_Signal"].loc[latest_idx]
        feature_vector["BB_Upper_Lag_1"] = processed["BB_Upper"].loc[latest_idx]
        feature_vector["BB_Lower_Lag_1"] = processed["BB_Lower"].loc[latest_idx]
        feature_vector["Volatility_20_Lag_1"] = processed["Volatility_20"].loc[latest_idx]

        # 4. Momentum Lags
        feature_vector["Return_Lag_1"] = processed["Daily_Return"].loc[latest_idx]
        feature_vector["Price_to_SMA20_Lag_1"] = (
            working_df["Close"].iloc[-1] / (processed["SMA_20"].loc[latest_idx] + 1e-9)
        )

        # Ensure feature alignment with training schema
        feature_row = pd.DataFrame([feature_vector])[feature_names]

        # Scale and predict
        scaled_row = scaler.transform(feature_row)
        pred_close = float(model.predict(scaled_row)[0])

        # Dynamic confidence band expanding with sqrt(time)
        vol_expansion = max(1.0, np.sqrt(step_idx))
        uncertainty = 1.96 * historical_residual_std * vol_expansion
        lower_bound = max(0.1, pred_close - uncertainty)
        upper_bound = pred_close + uncertainty

        forecast_records.append(
            {
                "Date": f_date,
                "Predicted_Close": round(pred_close, 2),
                "Lower_Bound": round(lower_bound, 2),
                "Upper_Bound": round(upper_bound, 2),
            }
        )

        # Append step prediction to working DataFrame for next recursive iteration
        new_row = pd.DataFrame(
            {
                "Open": pred_close,
                "High": pred_close * 1.002,
                "Low": pred_close * 0.998,
                "Close": pred_close,
                "Volume": int(working_df["Volume"].iloc[-5:].mean()),
            },
            index=[f_date],
        )
        new_row.index.name = "Date"
        working_df = pd.concat([working_df, new_row])

    forecast_df = pd.DataFrame(forecast_records).set_index("Date")
    return forecast_df
