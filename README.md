# Stock-price-predictor
Overview
This project predicts the next day's closing stock price using a simple Linear Regression model trained on historical closing prices.

Files
- `stock_predictor.py` : Main runnable script. Downloads data using `yfinance` if available, otherwise generates synthetic data so it runs offline.
- `requirements.txt` : Python dependencies.

Run
```bash
pip install -r requirements.txt
python stock_predictor.py
```

Notes
- If you have internet and `yfinance` installed, the script will fetch real stock data.
- To change ticker or date range, edit the variables inside `stock_predictor.py`.
