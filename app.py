"""
app.py - AI Quantitative Stock Price Prediction Dashboard
Interactive Streamlit Application with Technical Candlesticks, Multi-Model Evaluation & Recursive Forecasting
"""

import os
import sys
import time
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data import (
    fetch_stock_data,
    add_technical_indicators,
    prepare_features_and_target,
    time_series_split,
)
from model import (
    train_models,
    evaluate_models,
    forecast_future_recursive,
    get_feature_importance,
)

# -------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Stock Price Prediction & Quantitative Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #0f172a 50%, #0b0f19 100%);
        color: #f8fafc;
    }

    /* Hero Banner */
    .hero-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(12px);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* KPI Cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }

    .kpi-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }

    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }

    .kpi-delta {
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 2px;
    }

    .delta-pos { color: #34d399; }
    .delta-neg { color: #f87171; }
    .delta-neutral { color: #94a3b8; }

    /* Clean Streamlit disconnect popup suppression */
    [data-testid="stConnectionStatus"],
    .stConnectionStatus,
    div[role="dialog"]:has([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# 2. Sidebar Controls
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Market Configuration")

    ticker_options = ["AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "Custom..."]
    selected_ticker_choice = st.selectbox("Select Asset Ticker", ticker_options, index=0)

    if selected_ticker_choice == "Custom...":
        ticker_input = st.text_input("Enter Ticker Symbol", value="AMZN").strip().upper()
    else:
        ticker_input = selected_ticker_choice

    period_options = {
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
    }
    selected_period_label = st.selectbox("Historical Lookback Window", list(period_options.keys()), index=2)
    selected_period = period_options[selected_period_label]

    st.divider()
    st.markdown("### 🤖 Forecasting Parameters")

    forecast_horizon = st.select_slider(
        "Forecast Horizon (Trading Days)",
        options=[7, 14, 21, 30],
        value=14,
        help="Number of future market business days to predict recursively.",
    )

    preferred_model = st.selectbox(
        "Primary Forecast Model",
        ["Random Forest", "Ridge Regression", "Linear Regression"],
        index=0,
    )

    test_split_ratio = st.slider(
        "Test Split Ratio",
        min_value=0.10,
        max_value=0.30,
        value=0.20,
        step=0.05,
        help="Proportion of latest historical data reserved for unseen testing.",
    )

    st.divider()
    st.markdown("### ⚡ Session Controls")
    if st.button("🛑 Stop Session", use_container_width=True, type="secondary", help="Stop server and close window"):
        st.info("Stopping application and closing window...")
        time.sleep(0.4)
        os._exit(0)


# -------------------------------------------------------------
# 3. Data Ingestion & Caching
# -------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Fetching financial data from market feed...")
def get_market_data(ticker: str, period: str):
    return fetch_stock_data(ticker, period=period)


df_raw, is_synthetic = get_market_data(ticker_input, selected_period)

if df_raw is None or df_raw.empty or len(df_raw) < 40:
    st.error(f"❌ Insufficient market data available for '{ticker_input}'. Please check the symbol or select another asset.")
    st.stop()

# Compute technical indicators
df_indicators = add_technical_indicators(df_raw)

# -------------------------------------------------------------
# 4. Header Banner & Live Market Metrics
# -------------------------------------------------------------
latest_row = df_raw.iloc[-1]
prev_row = df_raw.iloc[-2] if len(df_raw) >= 2 else latest_row

current_price = latest_row["Close"]
prev_close = prev_row["Close"]
price_change = current_price - prev_close
price_change_pct = (price_change / prev_close) * 100.0 if prev_close != 0 else 0.0

high_52w = df_raw["High"].max()
low_52w = df_raw["Low"].min()
vol_20d = df_indicators["Volatility_20"].iloc[-1] * 100.0
rsi_val = df_indicators["RSI_14"].iloc[-1]

st.markdown(
    f"""
    <div class="hero-card">
        <h1 class="hero-title">📈 AI Stock Price Prediction & Forecasting</h1>
        <p class="hero-subtitle">
            Quantitative Multi-Model Predictive Architecture • Technical Oscillators • Recursive Multi-Step Time-Series Forecasting
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_synthetic:
    st.warning("⚠️ Live market feed temporarily unreachable; operating in high-fidelity synthetic walk simulation mode.")

# KPI Cards Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    delta_class = "delta-pos" if price_change >= 0 else "delta-neg"
    sign = "+" if price_change >= 0 else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{ticker_input} Latest Close</div>
            <div class="kpi-value">${current_price:,.2f}</div>
            <div class="kpi-delta {delta_class}">{sign}${price_change:,.2f} ({sign}{price_change_pct:.2f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">52-Week High</div>
            <div class="kpi-value">${high_52w:,.2f}</div>
            <div class="kpi-delta delta-neutral">Peak Valuation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">52-Week Low</div>
            <div class="kpi-value">${low_52w:,.2f}</div>
            <div class="kpi-delta delta-neutral">Trough Floor</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    rsi_color = "delta-neg" if rsi_val > 70 else ("delta-pos" if rsi_val < 30 else "delta-neutral")
    rsi_status = "Overbought (>70)" if rsi_val > 70 else ("Oversold (<30)" if rsi_val < 30 else "Neutral Momentum")
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">14-Day RSI</div>
            <div class="kpi-value">{rsi_val:.1f}</div>
            <div class="kpi-delta {rsi_color}">{rsi_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">20D Volatility</div>
            <div class="kpi-value">{vol_20d:.2f}%</div>
            <div class="kpi-delta delta-neutral">Daily Std Dev</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# -------------------------------------------------------------
# 5. Core Machine Learning Pipeline
# -------------------------------------------------------------
X, y, feature_names = prepare_features_and_target(df_raw)
X_train, X_test, y_train, y_test, train_dates, test_dates = time_series_split(
    X, y, test_ratio=test_split_ratio
)

# Train all models
models, scaler = train_models(X_train, y_train)

# Evaluate on test set
metrics_df, test_predictions = evaluate_models(models, scaler, X_test, y_test)

# Compute historical test residual standard deviation for confidence intervals
primary_model = models[preferred_model]
primary_residuals = y_test.values - test_predictions[preferred_model]
residual_std = float(np.std(primary_residuals)) if len(primary_residuals) > 0 else 2.5

# Generate future recursive forecast
forecast_df = forecast_future_recursive(
    model=primary_model,
    scaler=scaler,
    df_raw=df_raw,
    feature_names=feature_names,
    horizon_days=forecast_horizon,
    historical_residual_std=residual_std,
)

# -------------------------------------------------------------
# 6. Interactive Multi-Tab Interface
# -------------------------------------------------------------
tab_market, tab_forecast, tab_backtest = st.tabs(
    [
        "📈 Live Market & Indicators",
        "🤖 ML Price Forecasting",
        "📊 Historical Backtesting & Insights",
    ]
)

# -------------------------------------------------------------
# TAB 1: Live Market & Technical Indicators
# -------------------------------------------------------------
with tab_market:
    st.subheader(f"📊 {ticker_input} Technical Analysis & Indicator Suite")

    fig_market = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
        subplot_titles=(
            f"{ticker_input} Price Action, Moving Averages & Bollinger Bands",
            "Trading Volume (Shares)",
            "Relative Strength Index (RSI 14)",
        ),
    )

    # 1. Candlestick
    fig_market.add_trace(
        go.Candlestick(
            x=df_indicators.index,
            open=df_indicators["Open"],
            high=df_indicators["High"],
            low=df_indicators["Low"],
            close=df_indicators["Close"],
            name="Candlestick",
            increasing_line_color="#22c55e",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # 2. Overlays
    fig_market.add_trace(
        go.Scatter(
            x=df_indicators.index,
            y=df_indicators["SMA_20"],
            line=dict(color="#38bdf8", width=1.5),
            name="20 SMA",
        ),
        row=1,
        col=1,
    )
    fig_market.add_trace(
        go.Scatter(
            x=df_indicators.index,
            y=df_indicators["SMA_50"],
            line=dict(color="#f59e0b", width=1.5),
            name="50 SMA",
        ),
        row=1,
        col=1,
    )
    fig_market.add_trace(
        go.Scatter(
            x=df_indicators.index,
            y=df_indicators["EMA_20"],
            line=dict(color="#c084fc", width=1.2, dash="dot"),
            name="20 EMA",
        ),
        row=1,
        col=1,
    )
    fig_market.add_trace(
        go.Scatter(
            x=df_indicators.index,
            y=df_indicators["BB_Upper"],
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
            name="Upper Bollinger Band",
        ),
        row=1,
        col=1,
    )
    fig_market.add_trace(
        go.Scatter(
            x=df_indicators.index,
            y=df_indicators["BB_Lower"],
            line=dict(color="rgba(148, 163, 184, 0.4)", width=1),
            fill="tonexty",
            fillcolor="rgba(148, 163, 184, 0.05)",
            name="Lower Bollinger Band",
        ),
        row=1,
        col=1,
    )

    # 3. Volume Bar Chart
    vol_colors = np.where(df_indicators["Close"] >= df_indicators["Open"], "#22c55e", "#ef4444")
    fig_market.add_trace(
        go.Bar(
            x=df_indicators.index,
            y=df_indicators["Volume"],
            marker_color=vol_colors,
            name="Volume",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    # 4. RSI Indicator
    fig_market.add_trace(
        go.Scatter(
            x=df_indicators.index,
            y=df_indicators["RSI_14"],
            line=dict(color="#a855f7", width=1.5),
            name="RSI 14",
        ),
        row=3,
        col=1,
    )
    # RSI Overbought/Oversold thresholds
    fig_market.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1, row=3, col=1)
    fig_market.add_hline(y=30, line_dash="dash", line_color="#22c55e", line_width=1, row=3, col=1)

    fig_market.update_layout(
        height=850,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_market, use_container_width=True)


# -------------------------------------------------------------
# TAB 2: Machine Learning Price Forecasting
# -------------------------------------------------------------
with tab_forecast:
    st.subheader("🎯 Quantitative Multi-Model Benchmark & Forecast Horizon")

    # Metrics Overview Table
    col_m1, col_m2 = st.columns([0.65, 0.35])
    with col_m1:
        st.markdown("#### 🏆 Out-of-Sample Test Set Evaluation")
        st.dataframe(
            metrics_df.style.highlight_min(
                subset=["MAE ($)", "RMSE ($)", "MAPE (%)"], color="#14532d"
            ).highlight_max(subset=["R² Score", "Directional Acc (%)"], color="#1e3a8a"),
            use_container_width=True,
        )

    with col_m2:
        st.markdown("#### 🔍 Selected Model Summary")
        best_r2 = metrics_df.loc[preferred_model, "R² Score"]
        best_mae = metrics_df.loc[preferred_model, "MAE ($)"]
        best_dir = metrics_df.loc[preferred_model, "Directional Acc (%)"]
        st.info(
            f"**Active Model**: `{preferred_model}`\n\n"
            f"- **Explained Variance ($R^2$)**: `{best_r2}`\n"
            f"- **Mean Error (MAE)**: `±${best_mae}`\n"
            f"- **Directional Accuracy**: `{best_dir}%`\n\n"
            f"Forecast extends **{forecast_horizon} business days** into the future."
        )

    # Forecast Plotly Chart
    st.markdown(f"#### 🔮 {forecast_horizon}-Day Recursive Price Forecast ({preferred_model})")

    fig_forecast = go.Figure()

    # 1. Historical Actual Price (last 120 days for clarity)
    recent_history = df_raw.tail(120)
    fig_forecast.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history["Close"],
            mode="lines",
            name="Actual Close (Historical)",
            line=dict(color="#f8fafc", width=2),
        )
    )

    # 2. Out-of-Sample Test Set Fit
    fig_forecast.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_predictions[preferred_model],
            mode="lines",
            name=f"Test Set Prediction ({preferred_model})",
            line=dict(color="#38bdf8", width=1.5, dash="dot"),
        )
    )

    # 3. Future Projected Line
    # Bridge the last actual price to the first forecast point for seamless continuity
    bridge_x = [recent_history.index[-1]] + list(forecast_df.index)
    bridge_y = [recent_history["Close"].iloc[-1]] + list(forecast_df["Predicted_Close"])

    fig_forecast.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_y,
            mode="lines+markers",
            name=f"{forecast_horizon}-Day Forecast",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=5, color="#f59e0b"),
        )
    )

    # 4. Confidence Interval Band
    bridge_upper = [recent_history["Close"].iloc[-1]] + list(forecast_df["Upper_Bound"])
    bridge_lower = [recent_history["Close"].iloc[-1]] + list(forecast_df["Lower_Bound"])

    fig_forecast.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper 95% Bound",
        )
    )
    fig_forecast.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(245, 158, 11, 0.15)",
            name="95% Volatility-Adjusted Confidence Band",
        )
    )

    fig_forecast.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis_title="Date",
        yaxis_title="Stock Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # Future Forecast Table and CSV Download
    col_t1, col_t2 = st.columns([0.7, 0.3])
    with col_t1:
        st.markdown("#### 📅 Future Day-by-Day Forecast Schedule")
        formatted_forecast = forecast_df.copy()
        formatted_forecast.index = formatted_forecast.index.strftime("%Y-%m-%d")
        st.dataframe(formatted_forecast, use_container_width=True)

    with col_t2:
        st.markdown("#### 💾 Export Forecast Data")
        csv_data = forecast_df.to_csv().encode("utf-8")
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_data,
            file_name=f"{ticker_input}_forecast_{forecast_horizon}d.csv",
            mime="text/csv",
            use_container_width=True,
        )


# -------------------------------------------------------------
# TAB 3: Historical Backtesting & Insights
# -------------------------------------------------------------
with tab_backtest:
    st.subheader(f"📊 Quantitative Backtesting Diagnostics ({preferred_model})")

    col_b1, col_b2 = st.columns(2)

    # 1. Residuals Distribution
    with col_b1:
        st.markdown("#### 📉 Residuals Error Distribution")
        fig_residuals = go.Figure()
        fig_residuals.add_trace(
            go.Histogram(
                x=primary_residuals,
                nbinsx=30,
                name="Residuals",
                marker_color="#818cf8",
                opacity=0.75,
            )
        )
        fig_residuals.add_vline(x=0, line_dash="dash", line_color="#38bdf8", line_width=1.5)
        fig_residuals.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            xaxis_title="Residual Error ($) [Actual - Predicted]",
            yaxis_title="Count Frequency",
            height=400,
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_residuals, use_container_width=True)

    # 2. Actual vs Predicted Scatter Plot
    with col_b2:
        st.markdown("#### 🎯 Actual vs. Predicted Parity Plot")
        y_test_arr = y_test.values
        pred_test_arr = test_predictions[preferred_model]

        fig_parity = go.Figure()
        fig_parity.add_trace(
            go.Scatter(
                x=y_test_arr,
                y=pred_test_arr,
                mode="markers",
                marker=dict(color="#38bdf8", size=6, opacity=0.7),
                name="Test Observations",
            )
        )
        # Ideal 45-degree reference line
        min_val = min(y_test_arr.min(), pred_test_arr.min())
        max_val = max(y_test_arr.max(), pred_test_arr.max())
        fig_parity.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=2),
                name="Ideal Fit (y = x)",
            )
        )
        fig_parity.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(15, 23, 42, 0.4)",
            plot_bgcolor="rgba(15, 23, 42, 0.6)",
            xaxis_title="Actual Price ($)",
            yaxis_title="Predicted Price ($)",
            height=400,
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_parity, use_container_width=True)

    # Feature Importance Bar Chart
    st.markdown("#### 🧬 Predictive Feature Importance Drivers")
    feat_imp_df = get_feature_importance(primary_model, feature_names, top_n=10)

    fig_importance = go.Figure(
        go.Bar(
            x=feat_imp_df["Importance"][::-1],
            y=feat_imp_df["Feature"][::-1],
            orientation="h",
            marker=dict(
                color=feat_imp_df["Importance"][::-1],
                colorscale="Viridis",
            ),
        )
    )
    fig_importance.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        xaxis_title="Relative Feature Importance / Coefficient Magnitude",
        yaxis_title="Feature Name",
        height=450,
        margin=dict(l=40, r=40, t=30, b=30),
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # Data Partition Details
    st.markdown("#### 📋 Data Partition Details")
    st.markdown(
        f"""
        - **Total Clean Trading Days**: `{len(X)}`
        - **Training Set**: `{len(X_train)}` days ({train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')})
        - **Test Evaluation Set**: `{len(X_test)}` days ({test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')})
        - **Data Leakage Safeguard**: Strict chronological partition (zero random shuffling).
        """
    )
