from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import calendar

app = Flask(__name__)
model = load_model("model.h5")

companies = ["GP", "00DS30", "00DSES", "00DSEX", "00DSMEX", "1JANATAMF"]  # Add full list

def predict_stock_prices(company, days=30):
    df = pd.read_csv(f'stock_data.csv')  # Ensure data exists
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X = scaled_data[-30:].reshape(1, 30, 1)
    
    predictions = []
    for _ in range(days):
        pred = model.predict(X)
        predictions.append(pred[0, 0])
        X = np.append(X[:, 1:, :], [[pred]], axis=1)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

def save_plot(predictions, filename, days):
    plt.figure(figsize=(8, 4))
    plt.plot(predictions, marker='o', linestyle='-', label=f'Forecast ({days} days)')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig(f'static/plots/{filename}')
    plt.close()

def get_calendar():
    bd_holidays = {1: [1], 2: [21], 3: [26], 4: [14], 5: [1], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [16, 25]}
    year, month = 2025, 3  # Example: March 2025
    cal = calendar.monthcalendar(year, month)
    return cal, bd_holidays

@app.route("/")
def index():
    return render_template("index.html", companies=companies)

@app.route("/forecast", methods=["POST"])
def forecast():
    company = request.form["company"]
    predictions_7 = predict_stock_prices(company, 7)
    predictions_15 = predict_stock_prices(company, 15)
    predictions_30 = predict_stock_prices(company, 30)
    save_plot(predictions_7, "forecast_7.png", 7)
    save_plot(predictions_15, "forecast_15.png", 15)
    save_plot(predictions_30, "forecast_30.png", 30)
    return jsonify({"success": True})

if __name__ == "__main__":
    if not os.path.exists("static/plots"):
        os.makedirs("static/plots")
    app.run(debug=True)
