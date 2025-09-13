"""
Stock Price Predictor (Linear Regression)

This script tries to download stock data using yfinance. If yfinance or internet is
not available, it generates a synthetic sample dataset so the script runs offline.

Usage:
    pip install -r requirements.txt
    python stock_predictor.py

Outputs:
    - Prints R2 score and MSE
    - Saves a plot to stock_plot.png
"""

import warnings
warnings.filterwarnings("ignore")

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

def get_stock_data(ticker="AAPL", start="2020-01-01", end=None):
    # Try to download via yfinance
    try:
        import yfinance as yf
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df is None or df.empty:
            raise RuntimeError("Empty data from yfinance")
        df = df.reset_index()
        return df
    except Exception as e:
        print("Could not download data via yfinance (will generate synthetic data).")
        # Generate synthetic data (random walk)
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        dates = pd.date_range(start_dt, end_dt, freq="B")  # business days
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.normal(0, 1, size=len(dates)))  # random walk
        df = pd.DataFrame({
            "Date": dates,
            "Open": prices + np.random.normal(0, 0.5, size=len(dates)),
            "High": prices + np.random.normal(0.5, 1.0, size=len(dates)),
            "Low": prices - np.random.normal(0.5, 1.0, size=len(dates)),
            "Close": prices,
            "Volume": np.random.randint(1000000, 5000000, size=len(dates))
        })
        return df

def main():
    # You can change ticker, start, end as needed
    ticker = "AAPL"
    print("Loading data for", ticker)
    df = get_stock_data(ticker=ticker, start="2020-01-01", end=None)
    df = df.sort_values("Date").reset_index(drop=True)
    df["Prev_Close"] = df["Close"].shift(1)
    df = df.dropna().reset_index(drop=True)

    X = df[["Prev_Close"]].values
    y = df["Close"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"R2 Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")

    # Plot first 100 points or full if smaller
    n_plot = min(100, len(y_test))
    plt.figure(figsize=(10,6))
    plt.plot(range(n_plot), y_test[:n_plot], label="Actual")
    plt.plot(range(n_plot), y_pred[:n_plot], label="Predicted", linestyle='--')
    plt.title(f"{ticker} - Actual vs Predicted Close Price")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    out_plot = "stock_plot.png"
    plt.savefig(out_plot)
    print(f"Saved plot to {out_plot}")

if __name__ == "__main__":
    main()
