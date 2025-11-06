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

Copy code
git clone https://github.com/maheshwagh2003/stock-price-predictor.git
Install the required dependencies:

Copy code
pip install -r requirements.txt
Run the Streamlit app:

arduino
Copy code
streamlit run app.py
Access the app through your web browser at http://localhost:8501.

Usage
Enter the stock symbol of interest into the input field.
Select the desired prediction horizon (e.g., 1 day, 5 days, etc.).
Click the "Predict" button to generate predictions.
View the predicted stock prices plotted on the chart.

Demo
Predictions: 
![Web 1920 – 1](https://github.com/maheshwagh2003/Stock-Price_Predictor/assets/77723262/9bddace8-ab01-4787-a690-8cfab4b8ca93)

Actual Values:
![Group 1](https://github.com/maheshwagh2003/Stock-Price_Predictor/assets/77723262/6872d2f8-2616-4970-9d1b-fb3fa91abb5d)

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you encounter any problems or have suggestions for improvements.
