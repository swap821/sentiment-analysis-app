"""
app.py — Flask REST API for Sentiment Analysis

This API serves TWO trained ML models:
1. TF-IDF + Logistic Regression (fast, traditional)
2. LSTM Neural Network (deep learning, contextual)

Users can:
- Analyze single texts for sentiment
- Upload CSV files for batch analysis
- Get model comparison info
"""

import os
import pickle
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd

from preprocess import preprocess_text
from utils import setup_logging, validate_text, format_response, format_error_response

# TensorFlow for LSTM
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

logger = setup_logging()

# Global model variables
TFIDF_VECTORIZER = None
TFIDF_MODEL = None
LSTM_MODEL = None
LSTM_TOKENIZER = None
LABEL_ENCODER = None
MODELS_LOADED = False


def load_models():
    """Load all trained models on startup."""
    global TFIDF_VECTORIZER, TFIDF_MODEL, LSTM_MODEL, LSTM_TOKENIZER, LABEL_ENCODER, MODELS_LOADED
    
    try:
        # Load TF-IDF + Logistic Regression
        TFIDF_VECTORIZER, TFIDF_MODEL = pickle.load(open('models/tfidf_model.pkl', 'rb'))
        logger.info("Loaded TF-IDF + Logistic Regression model")
        
        # Load LSTM
        LSTM_MODEL = load_model('models/lstm_model.h5')
        LSTM_TOKENIZER = pickle.load(open('models/lstm_tokenizer.pkl', 'rb'))
        LABEL_ENCODER = pickle.load(open('models/label_encoder.pkl', 'rb'))
        logger.info("Loaded LSTM Neural Network model")
        
        MODELS_LOADED = True
        
    except FileNotFoundError as e:
        logger.warning(f"Model files not found: {e}")
        logger.warning("Run 'python train_models.py' first")


load_models()


@app.route('/', methods=['GET'])
def home():
    """Welcome endpoint."""
    return jsonify({
        'message': 'AI Sentiment Analyzer API',
        'version': '1.0.0',
        'models_loaded': MODELS_LOADED,
        'endpoints': {
            'POST /analyze': 'Analyze single text',
            'POST /analyze/batch': 'Batch CSV analysis',
            'GET /models': 'Model comparison info',
            'GET /models/plot': 'Comparison chart',
            'GET /health': 'Health check'
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze sentiment of a single text.
    
    Request body: { "text": "This movie was amazing!", "model": "lstm" }
    
    Returns:
        { sentiment, confidence, model_used, text }
    """
    if not MODELS_LOADED:
        return jsonify(format_error_response('Models not loaded', 503)), 503
    
    try:
        data = request.get_json()
        text = data.get('text', '')
        model_choice = data.get('model', 'lstm')  # Default to LSTM
        
        # Validate input
        is_valid, error = validate_text(text)
        if not is_valid:
            return jsonify(format_error_response(error, 400)), 400
        
        # Choose model
        if model_choice == 'tfidf':
            sentiment, confidence = predict_tfidf(text)
            model_used = 'TF-IDF + Logistic Regression'
        else:
            sentiment, confidence = predict_lstm(text)
            model_used = 'LSTM Neural Network'
        
        return jsonify(format_response(text, sentiment, confidence, model_used))
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify(format_error_response(str(e), 500)), 500


def predict_tfidf(text):
    """
    Predict sentiment using TF-IDF + Logistic Regression.
    
    Steps:
    1. Preprocess the text (clean, tokenize, stem)
    2. Convert to TF-IDF vector
    3. Predict with Logistic Regression
    4. Return sentiment and confidence
    """
    processed = preprocess_text(text)
    vector = TFIDF_VECTORIZER.transform([processed])
    prediction = TFIDF_MODEL.predict(vector)[0]
    confidence = float(np.max(TFIDF_MODEL.predict_proba(vector)))
    return prediction, confidence


def predict_lstm(text):
    """
    Predict sentiment using LSTM Neural Network.
    
    Steps:
    1. Preprocess the text
    2. Convert to sequence of integers
    3. Pad to fixed length
    4. Predict with LSTM
    5. Return sentiment and confidence
    """
    processed = preprocess_text(text)
    sequence = LSTM_TOKENIZER.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction_prob = float(LSTM_MODEL.predict(padded, verbose=0)[0][0])
    
    # Convert probability to sentiment
    if prediction_prob > 0.5:
        sentiment = 'positive'
        confidence = prediction_prob
    else:
        sentiment = 'negative'
        confidence = 1 - prediction_prob
    
    return sentiment, confidence


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Batch sentiment analysis from uploaded CSV.
    
    Expects multipart/form-data with a CSV file containing a 'text' column.
    """
    if not MODELS_LOADED:
        return jsonify(format_error_response('Models not loaded', 503)), 503
    
    try:
        if 'file' not in request.files:
            return jsonify(format_error_response('No file provided', 400)), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify(format_error_response('File must be CSV', 400)), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Find text column
        text_col = None
        for col in ['text', 'review', 'content', 'message']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            return jsonify(format_error_response(
                'CSV must have a text column (text/review/content/message)', 400
            )), 400
        
        # Analyze each row
        results = []
        for _, row in df.iterrows():
            text = str(row[text_col])
            sentiment, confidence = predict_lstm(text)
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'sentiment': sentiment,
                'confidence': round(confidence, 4)
            })
        
        # Summary statistics
        total = len(results)
        positive = sum(1 for r in results if r['sentiment'] == 'positive')
        negative = total - positive
        
        return jsonify({
            'status': 'success',
            'total': total,
            'positive': positive,
            'negative': negative,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        return jsonify(format_error_response(str(e), 500)), 500


@app.route('/models', methods=['GET'])
def get_models_info():
    """Get model comparison information."""
    return jsonify({
        'models': {
            'tfidf_logistic': {
                'name': 'TF-IDF + Logistic Regression',
                'type': 'Traditional ML',
                'description': 'Fast, interpretable, works well with high-dimensional text data'
            },
            'lstm': {
                'name': 'LSTM Neural Network',
                'type': 'Deep Learning',
                'description': 'Captures context and word order through sequence processing'
            }
        }
    })


@app.route('/models/plot', methods=['GET'])
def get_model_plot():
    """Serve the model comparison plot."""
    try:
        return send_file('models/comparison.png', mimetype='image/png')
    except:
        return jsonify(format_error_response('Plot not found', 404)), 404


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': MODELS_LOADED
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)