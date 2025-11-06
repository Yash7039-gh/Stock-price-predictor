# Stock-Price_Predictor
This project is a Stock Price Prediction App built with Python, utilizing Yahoo Finance API for fetching stock data, Streamlit for building the web application, and an LSTM (Long Short-Term Memory) model implemented using Keras for predicting stock prices.

Overview
The Stock Price Prediction App provides users with a simple interface to input a stock symbol and get predictions for its future prices based on historical data. The LSTM model is trained on historical stock data to learn patterns and make predictions.

Features
Easy-to-Use Interface: Streamlit provides a user-friendly interface for users to interact with the application.
Real-Time Data: Yahoo Finance API is used to fetch real-time stock data for predictions.
Predictive Analytics: The LSTM model predicts future stock prices based on historical data, helping users make informed decisions.
Installation
Clone the repository:

bash
Install the required dependencies:

Copy code
pip install -r requirements.txt
Run the Streamlit app:

arduino
streamlit run app.py
Access the app through your web browser at http://localhost:8501.

Usage
Enter the stock symbol of interest into the input field.
Select the desired prediction horizon (e.g., 1 day, 5 days, etc.).
Click the "Predict" button to generate predictions.
View the predicted stock prices plotted on the chart.


