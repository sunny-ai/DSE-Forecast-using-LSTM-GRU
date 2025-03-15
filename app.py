from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load the saved model
model = load_model('model.h5')

# Load and preprocess data (replace with your dataset)
df = pd.read_csv('stock_data.csv')
data = df['Close'].values.reshape(-1, 1)

# Initialize and fit the scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
    return np.array(X)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the selected company (for now, using the same model)
    company = request.json['company']
    
    # Prepare data for prediction
    seq_length = 30
    last_sequence = scaled_data[-seq_length:]  # Use the global scaled_data
    last_sequence = last_sequence.reshape((1, seq_length, 1))
    
    # Predict for 7, 15, and 30 days
    predictions = []
    for days in [7, 15, 30]:
        prediction = []
        current_sequence = last_sequence.copy()
        for _ in range(days):
            next_pred = model.predict(current_sequence)
            prediction.append(next_pred[0][0])
            current_sequence = np.append(current_sequence[:, 1:, :], [[next_pred]], axis=1)
        predictions.append(scaler.inverse_transform(np.array(prediction).reshape(-1, 1)))
    
    return jsonify({
        '7_days': predictions[0].flatten().tolist(),
        '15_days': predictions[1].flatten().tolist(),
        '30_days': predictions[2].flatten().tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)