from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from credit_card_fraud_detection import load_model_and_scaler, train_model, preprocess_data, load_data
import os
import random

app = Flask(__name__)

# Load model and scaler at startup
model, scaler = load_model_and_scaler()

# If model or scaler is not loaded, train a new one
if model is None or scaler is None:
    print("Training new model...")
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    model = train_model(X_train, y_train)
    print("Model training completed!")

# Store some sample data for reference
sample_data = None
try:
    df = load_data()
    sample_data = df[['V1', 'V2', 'V3', 'Amount', 'Time']].head(100).to_dict('records')
except:
    print("Could not load sample data")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_sample', methods=['GET'])
def get_sample():
    if sample_data is None:
        return jsonify({'error': 'Sample data not available'}), 500
    
    # Return a random sample
    sample = random.choice(sample_data)
    return jsonify(sample)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not properly loaded. Please restart the application.'}), 500

        required_fields = ['V1', 'V2', 'V3', 'Amount', 'Time']

        if 'file' in request.files:
            # Handle CSV file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not file.filename.endswith('.csv'):
                return jsonify({'error': 'Please upload a CSV file'}), 400
            
            df = pd.read_csv(file)
            if not all(col in df.columns for col in required_fields):
                return jsonify({'error': 'CSV must contain V1, V2, V3, Amount, and Time columns'}), 400
            
            # Ensure correct column order
            df = df[required_fields]
            # Scale the data
            scaled_data = scaler.transform(df)
            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                results.append({
                    'transaction_id': i + 1,
                    'prediction': 'Fraud' if pred == 1 else 'Normal',
                    'confidence': float(prob[1] if pred == 1 else prob[0])
                })
            
            return jsonify({'results': results})
        else:
            # Handle single transaction prediction
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            if not all(field in data for field in required_fields):
                return jsonify({'error': 'Missing required fields'}), 400
            
            # Convert to DataFrame and ensure correct column order
            df = pd.DataFrame([data])[required_fields]
            # Scale the data
            scaled_data = scaler.transform(df)
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0]
            
            return jsonify({
                'prediction': 'Fraud' if prediction == 1 else 'Normal',
                'confidence': float(probability[1] if prediction == 1 else probability[0])
            })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 