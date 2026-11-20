import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime, timedelta
import ta

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables to store the trained model and latest market data
model = None
latest_data = None
backtest_results = None

def train_model():
    """
    Trains a linear regression model to predict the next day's closing price
    using Close Price, Moving Average, and RSI.
    """
    global model, latest_data
    print("Training model with richer data...")
    
    # Fetch historical NIFTY 50 data for the past 2 years to have enough data for a 50-day MA
    nifty = yf.Ticker('^NSEI')
    hist_data = nifty.history(period="2y")
    
    # Calculate a 50-day Simple Moving Average (MA)
    hist_data['ma50'] = hist_data['Close'].rolling(window=50).mean()
    
    # Calculate the Relative Strength Index (RSI)
    hist_data['rsi'] = ta.momentum.rsi(hist_data['Close'], window=14)
    
    # Create the target variable (next day's closing price)
    hist_data['Next_Day_Close'] = hist_data['Close'].shift(-1)
    
    # Drop rows with NaN values (from rolling mean and shifting)
    hist_data.dropna(inplace=True)
    
    # Define features (X) and target (y)
    X = hist_data[['Close', 'ma50', 'rsi']]
    y = hist_data['Next_Day_Close']
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Fetch the latest 60 days of data once for use in the "what-if" scenario defaults
    latest_data = nifty.history(period="60d")
    latest_data['ma50'] = latest_data['Close'].rolling(window=50).mean()
    latest_data['rsi'] = ta.momentum.rsi(latest_data['Close'], window=14)
    
    print("Model trained successfully with richer data.")
    return model

def run_backtest():
    """
    Runs a backtest on historical data and calculates performance metrics.
    Refactored to use vectorized operations for efficiency.
    """
    global backtest_results
    print("Running backtest on historical data...")

    nifty = yf.Ticker('^NSEI')
    backtest_data = nifty.history(period="1y")

    backtest_data['ma50'] = backtest_data['Close'].rolling(window=50).mean()
    backtest_data['rsi'] = ta.momentum.rsi(backtest_data['Close'], window=14)
    backtest_data.dropna(inplace=True)

    features = backtest_data[['Close', 'ma50', 'rsi']]
    backtest_data['Predicted_Close'] = model.predict(features)
    
    # Create a simple backtest strategy:
    # If the model predicts an "up" day, we assume we made a "trade"
    backtest_data['Predicted_Direction'] = backtest_data['Predicted_Close'].shift(-1) > backtest_data['Close']
    backtest_data['Actual_Direction'] = backtest_data['Close'].shift(-1) > backtest_data['Close']
    
    # Compare predicted direction to actual direction
    backtest_data['Correct_Prediction'] = backtest_data['Predicted_Direction'] == backtest_data['Actual_Direction']

    # Calculate accuracy and other metrics using vectorized operations
    accuracy = backtest_data['Correct_Prediction'].mean() * 100
    total_trades = len(backtest_data)

    # Simulate a simple portfolio strategy
    # Calculate the daily percentage change of the NIFTY 50
    backtest_data['Daily_Return'] = backtest_data['Close'].pct_change()
    
    # Calculate the returns based on the model's predictions.
    # We only "invest" when the model predicts an upward trend.
    backtest_data['Model_Return'] = backtest_data.apply(
        lambda row: row['Daily_Return'] if row['Predicted_Direction'] else 0, axis=1
    )

    # Calculate the total return of the simulated portfolio
    total_return = (backtest_data['Model_Return'].mean() * total_trades) * 100

    backtest_results = {
        "accuracy": accuracy,
        "total_trades": total_trades,
        "simulated_return": total_return
    }

    print("Backtest completed. Accuracy:", accuracy)
    return backtest_results

# Train model and run backtest on startup
model = train_model()
backtest_results = run_backtest()

@app.route('/api/predict', methods=['GET'])
def predict():
    """
    API endpoint to get the next day's price prediction.
    Accepts optional query parameters for 'close_price', 'ma50', and 'rsi'
    to enable "what-if" scenarios.
    """
    global model, latest_data
    print("Received request for prediction.")
    
    try:
        latest_row = latest_data.iloc[-1]
        
        close_price = request.args.get('close_price', type=float, default=latest_row['Close'])
        ma50 = request.args.get('ma50', type=float, default=latest_row['ma50'])
        rsi = request.args.get('rsi', type=float, default=latest_row['rsi'])

        features_df = pd.DataFrame([[close_price, ma50, rsi]], columns=['Close', 'ma50', 'rsi'])
        next_day_prediction = model.predict(features_df)[0]
        
        prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        response = {
            'status': 'success',
            'prediction': {
                'date': prediction_date,
                'price': next_day_prediction,
                'currentPrice': close_price,
                'ma50': ma50,
                'rsi': rsi
            },
            'message': 'Prediction generated using a model with custom or default features.'
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data', methods=['GET'])
def get_data():
    """
    API endpoint to get historical data for a chart, including MA.
    """
    print("Received request for historical data.")
    
    try:
        nifty = yf.Ticker('^NSEI')
        hist_data = nifty.history(period="3mo")
        hist_data['ma50'] = hist_data['Close'].rolling(window=50).mean()
        hist_data.dropna(inplace=True)
        
        if hist_data.empty:
            return jsonify({'error': 'Could not fetch historical data'}), 500
        
        data = {
            'labels': hist_data.index.strftime('%Y-%m-%d').tolist(),
            'prices': hist_data['Close'].tolist(),
            'ma50': hist_data['ma50'].tolist()
        }
        
        return jsonify(data)

    except Exception as e:
        app.logger.error("Error in /api/data: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest', methods=['GET'])
def get_backtest_results():
    """
    API endpoint to get the results of the backtest.
    """
    global backtest_results
    print("Received request for backtest results.")
    if backtest_results:
        return jsonify(backtest_results)
    else:
        return jsonify({"error": "Backtest results not available."}), 404

if __name__ == '__main__':
    app.run(debug=True)