from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from firebase_service import firebase_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Model storage paths
MODEL_DIR = Path(__file__).parent / 'models'
RES_DIR = Path(__file__).parent / 'res'
RF_MODEL_PATH = RES_DIR / 'random_forest_regression_model.pkl'
TS_MODEL_PATH = MODEL_DIR / 'timeseries_model.h5'

# Global variables for models
rf_model = None
ts_model = None
models_loaded = False

def load_models():
    """Load the trained models at startup"""
    global rf_model, ts_model, models_loaded
    
    try:
        # Load Random Forest model
        if RF_MODEL_PATH.exists():
            with open(RF_MODEL_PATH, 'rb') as f:
                rf_model = pickle.load(f)
            logger.info("Random Forest model loaded successfully")
        else:
            logger.warning(f"Random Forest model not found at {RF_MODEL_PATH}")
        
        # Load Time Series model
        if TS_MODEL_PATH.exists():
            ts_model = tf.keras.models.load_model(str(TS_MODEL_PATH))
            logger.info("Time Series model loaded successfully")
        else:
            logger.warning(f"Time Series model not found at {TS_MODEL_PATH}")
        
        models_loaded = rf_model is not None and ts_model is not None
        
        if models_loaded:
            logger.info("All models loaded successfully")
        else:
            logger.error("Failed to load one or more models")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        models_loaded = False

def get_risk_level(premise_index):
    """Determine risk level based on premise index"""
    if premise_index < 30:
        return 'low'
    elif premise_index < 60:
        return 'medium'
    else:
        return 'high'

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction using the Random Forest model"""
    try:
        if not rf_model:
            return jsonify({'error': 'Random Forest model not loaded'}), 500
        
        data = request.get_json()
        
        # Validate input data
        required_fields = ['rainfall', 'temperature', 'waterContent', 'Rainfall_7d_avg', 'WaterContent_7d_avg']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features for prediction
        features = np.array([[
            float(data['rainfall']),
            float(data['temperature']),
            float(data['waterContent']),
            float(data['Rainfall_7d_avg']),
            float(data['WaterContent_7d_avg'])
        ]])
        
        # Make prediction
        premise_index = rf_model.predict(features)[0]
        
        # Get prediction confidence (if available)
        confidence = 0.85  # Default confidence, replace with actual if your model supports it
        if hasattr(rf_model, 'predict_proba'):
            # For classification models
            probabilities = rf_model.predict_proba(features)[0]
            confidence = float(np.max(probabilities))
        
        # Determine risk level
        risk_level = get_risk_level(premise_index)
        
        response = {
            'premiseIndex': float(premise_index),
            'riskLevel': risk_level,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save prediction to Firebase
        try:
            prediction_data = {
                'id': response['timestamp'],  # Use timestamp as ID for consistency
                'timestamp': response['timestamp'],
                'premiseIndex': response['premiseIndex'],
                'rainfall': float(data['rainfall']),
                'temperature': float(data['temperature']),
                'waterContent': float(data['waterContent']),
                'Rainfall_7d_avg': float(data['Rainfall_7d_avg']),
                'WaterContent_7d_avg': float(data['WaterContent_7d_avg']),
                'riskLevel': response['riskLevel'],
                'confidence': response['confidence']
            }
            firebase_service.save_prediction(prediction_data)
        except Exception as firebase_error:
            logger.error(f"Failed to save to Firebase: {firebase_error}")
        
        logger.info(f"Prediction made: {premise_index:.2f}% ({risk_level} risk)")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def forecast():
    """Generate time series forecast"""
    try:
        if not ts_model:
            return jsonify({'error': 'Time Series model not loaded'}), 500
        
        days = int(request.args.get('days', 90))
        
        # Generate forecast dates
        start_date = datetime.now()
        dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
        
        # Prepare input for time series model
        # Note: You'll need to adjust this based on your model's input requirements
        # This is a placeholder - replace with your actual preprocessing logic
        
        # Example: If your model expects a sequence of past values
        # You might need to provide historical data or use a different approach
        
        # For demonstration, creating dummy input - REPLACE THIS
        input_sequence = np.random.random((1, 30, 3))  # Adjust shape based on your model
        
        # Make forecast prediction
        forecast_values = ts_model.predict(input_sequence)
        
        # Process forecast output based on your model's output format
        if len(forecast_values.shape) > 1:
            predictions = forecast_values.flatten()[:days].tolist()
        else:
            predictions = forecast_values[:days].tolist()
        
        # Ensure we have the right number of predictions
        if len(predictions) < days:
            # Extend with the last value if needed
            last_value = predictions[-1] if predictions else 50.0
            predictions.extend([last_value] * (days - len(predictions)))
        
        response = {
            'dates': dates,
            'predictions': [float(p) for p in predictions[:days]],
            'confidence_intervals': {
                'lower': [max(0, float(p) - 10) for p in predictions[:days]],
                'upper': [min(100, float(p) + 10) for p in predictions[:days]]
            }
        }
        
        logger.info(f"Forecast generated for {days} days")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Load models at startup
    load_models()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)