#!/usr/bin/env bash
echo "======================================================"
echo "     AI Stock Price Prediction & Forecasting"
echo "    Quantitative ML & Multi-Step Time Series Engine"
echo "======================================================"
echo ""

# Fast dependency check
python3 -c "import streamlit, yfinance, plotly, sklearn, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[Checking and installing Python dependencies...]"
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi
fi

echo ""
echo "[Launching application in dedicated window...]"
python3 launch.py
