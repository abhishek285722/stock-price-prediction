# 📈 AI Stock Price Prediction & Quantitative Forecasting Dashboard

A production-grade, modular financial analytics and machine learning application built with **Streamlit**, **Scikit-Learn**, **Plotly**, and **yfinance**. Features real-time technical indicators, multi-model out-of-sample benchmark evaluation, and recursive multi-step price forecasting with volatility-adjusted confidence bands.

---

## 📌 Project Overview

This application bridges financial technical analysis and quantitative machine learning:
- **Market Data Engine**: Ingests historical daily OHLCV bars via `yfinance` with an automatic high-fidelity Geometric Brownian Motion (GBM) synthetic fallback for offline resilience.
- **Quantitative Indicators**: Computes 20/50 SMA, 20 EMA, 14-period RSI (Wilder smoothing), MACD (12, 26, 9), Bollinger Bands ($\pm 2\sigma$), and 20-day historical volatility.
- **Predictive ML Framework**: Compares **Linear Regression**, **Ridge Regression** ($L_2$ regularized), and **Random Forest Regressor** (100 decision trees) evaluated on an out-of-sample test split.
- **Multi-Step Recursive Forecaster**: Generates forward-looking trajectory projections for 7, 14, 21, or 30 trading days ahead with dynamic $95\%$ confidence corridors.
- **Dedicated Application Mode**: Runs in an isolated, distraction-free browser window with clean process synchronization and graceful shutdown.

---

## 🔬 Mathematical & Algorithmic Formulation

### 1. Data Leakage Prevention (Strict Chronological Partitioning)
In time-series modeling, random shuffling introduces severe lookahead bias. All feature extraction lags observations strictly by $t-1$ or greater:
$$X_t = [P_{t-1}, P_{t-2}, \dots, P_{t-7}, \text{SMA}_{20, t-1}, \text{RSI}_{14, t-1}, \text{Vol}_{20, t-1}]$$
$$y_t = P_t \quad (\text{Close Price at time } t)$$
Train-test splitting divides the chronological timeline into an 80% in-sample training horizon and a 20% unseen out-of-sample evaluation period:
$$\max(\text{Dates}_{\text{train}}) < \min(\text{Dates}_{\text{test}})$$

### 2. Quantitative Indicators
- **Relative Strength Index (RSI)**:
  $$\text{RS} = \frac{\text{EMA}_{14}(\text{Gain})}{\text{EMA}_{14}(\text{Loss})}, \quad \text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
- **Bollinger Bands**:
  $$\text{Upper} = \text{SMA}_{20}(P) + 2 \cdot \sigma_{20}(P), \quad \text{Lower} = \text{SMA}_{20}(P) - 2 \cdot \sigma_{20}(P)$$
- **Moving Average Convergence Divergence (MACD)**:
  $$\text{MACD} = \text{EMA}_{12}(P) - \text{EMA}_{26}(P), \quad \text{Signal} = \text{EMA}_9(\text{MACD})$$

### 3. Recursive Multi-Step Forecasting
To predict $H$ future trading days without future data:
1. For step $h=1$, predict $\hat{P}_{t+1}$ using features computed from historical bars up to $t$.
2. Inject $\hat{P}_{t+1}$ into the rolling price series.
3. Recompute indicators and lag vectors on the expanded series to predict $\hat{P}_{t+2}$.
4. Repeat iteratively up to horizon $H$.
5. Dynamic confidence corridors widen with time:
   $$\text{Band}_{t+h} = \hat{P}_{t+h} \pm 1.96 \cdot \sigma_{\text{residuals}} \cdot \sqrt{h}$$

---

## 📁 Project Structure

```
Stock Price Prediction/
│
├── data.py               # yfinance data ingestion, technical indicators & feature lags
├── model.py              # ML training, evaluation metrics & recursive forecaster
├── app.py                # Modern Streamlit UI with Plotly charts & KPI cards
├── launch.py             # Isolated application window supervisor (ports 8504+)
├── test_stock.py         # Automated unit test suite (8 test cases)
├── requirements.txt      # Python dependencies
├── run.bat               # One-click Windows batch launcher
├── run.sh                # One-click Linux / macOS launcher
├── .gitignore            # Git exclusion rules
├── .streamlit/
│   └── config.toml       # Theme & headless server configuration
└── README.md             # Technical documentation
```

---

## 🚀 Quickstart & Execution

### Option 1: One-Click Launcher (Recommended)
- **Windows**: Double-click [`run.bat`](file:///c:/Users/mymai/OneDrive/Desktop/projects/Stock%20Price%20Prediction/run.bat)
- **macOS / Linux**: Run `bash run.sh` in your terminal

The script verifies dependencies, starts the Streamlit background process, and opens an isolated application window without polluting your standard browser tabs.

### Option 2: Direct Command Line
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run via application launcher
python launch.py

# Or directly with Streamlit:
streamlit run app.py
```

---

## 🧪 Automated Testing

Execute the automated test suite covering indicator calculations, data splitting, model training, and future forecasting:

```bash
python test_stock.py
```

All 8 tests execute cleanly in under 5 seconds with zero failures.

---

## 🛑 Clean Session Shutdown
Clicking the **🛑 Stop Session** button in the sidebar terminates the server cleanly (`os._exit(0)`) and automatically closes the dedicated window.
