"""
app.py — FastAPI Application for Sentiment Analysis

Serves TWO trained ML models (TF-IDF + Logistic Regression and LSTM Neural Network)
with Pydantic request validation and batch CSV processing capabilities.
"""

import os
import pickle
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, status, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from preprocess import preprocess_text
from utils import setup_logging, validate_text, format_response, format_error_response

# TensorFlow for LSTM
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    HAS_TF = True
except ImportError:
    HAS_TF = False

app = FastAPI(
    title="AI Sentiment Analyzer API",
    description="Asynchronous sentiment analysis serving TF-IDF and LSTM models.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logging()

TFIDF_VECTORIZER = None
TFIDF_MODEL = None
LSTM_MODEL = None
LSTM_TOKENIZER = None
LABEL_ENCODER = None
MODELS_LOADED = False


def load_models():
    global TFIDF_VECTORIZER, TFIDF_MODEL, LSTM_MODEL, LSTM_TOKENIZER, LABEL_ENCODER, MODELS_LOADED
    try:
        if os.path.exists('models/tfidf_model.pkl'):
            TFIDF_VECTORIZER, TFIDF_MODEL = pickle.load(open('models/tfidf_model.pkl', 'rb'))
            logger.info("Loaded TF-IDF + Logistic Regression model")
        
        if HAS_TF and os.path.exists('models/lstm_model.h5'):
            LSTM_MODEL = load_model('models/lstm_model.h5')
            LSTM_TOKENIZER = pickle.load(open('models/lstm_tokenizer.pkl', 'rb'))
            LABEL_ENCODER = pickle.load(open('models/label_encoder.pkl', 'rb'))
            logger.info("Loaded LSTM Neural Network model")
        
        MODELS_LOADED = TFIDF_MODEL is not None or LSTM_MODEL is not None
    except Exception as e:
        logger.warning(f"Error loading models: {e}")


@app.on_event("startup")
async def startup_event():
    load_models()


class SingleTextAnalysisRequest(BaseModel):
    text: str = Field(..., description="Text content to analyze")
    model: str = Field(default="lstm", description="Model choice ('tfidf' or 'lstm')")


@app.get("/")
async def home():
    return {
        "message": "AI Sentiment Analyzer API (FastAPI)",
        "version": "2.0.0",
        "models_loaded": MODELS_LOADED,
        "docs": "/docs"
    }


@app.post("/analyze")
async def analyze(payload: SingleTextAnalysisRequest):
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded. Please train models first."
        )

    is_valid, error = validate_text(payload.text)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error
        )

    try:
        if payload.model == "tfidf" or LSTM_MODEL is None:
            sentiment, confidence = predict_tfidf(payload.text)
            model_used = "TF-IDF + Logistic Regression"
        else:
            sentiment, confidence = predict_lstm(payload.text)
            model_used = "LSTM Neural Network"

        return format_response(payload.text, sentiment, confidence, model_used)
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


def predict_tfidf(text: str):
    processed = preprocess_text(text)
    vector = TFIDF_VECTORIZER.transform([processed])
    prediction = TFIDF_MODEL.predict(vector)[0]
    confidence = float(np.max(TFIDF_MODEL.predict_proba(vector)))
    return prediction, confidence


def predict_lstm(text: str):
    processed = preprocess_text(text)
    sequence = LSTM_TOKENIZER.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction_prob = float(LSTM_MODEL.predict(padded, verbose=0)[0][0])
    
    if prediction_prob > 0.5:
        sentiment = 'positive'
        confidence = prediction_prob
    else:
        sentiment = 'negative'
        confidence = 1 - prediction_prob
    return sentiment, confidence


@app.post("/analyze/batch")
async def analyze_batch(file: UploadFile = File(...)):
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not loaded."
        )

    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be CSV format."
        )

    try:
        df = pd.read_csv(file.file)
        text_col = None
        for col in ['text', 'review', 'content', 'message']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="CSV must contain a column named text, review, content, or message."
            )

        results = []
        for _, row in df.iterrows():
            txt = str(row[text_col])
            sentiment, confidence = predict_tfidf(txt) if LSTM_MODEL is None else predict_lstm(txt)
            results.append({
                'text': txt[:100] + '...' if len(txt) > 100 else txt,
                'sentiment': sentiment,
                'confidence': round(confidence, 4)
            })

        total = len(results)
        positive = sum(1 for r in results if r['sentiment'] == 'positive')
        
        return {
            "status": "success",
            "total": total,
            "positive": positive,
            "negative": total - positive,
            "results": results
        }
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/models")
async def get_models_info():
    return {
        "models": {
            "tfidf_logistic": {
                "name": "TF-IDF + Logistic Regression",
                "type": "Traditional ML",
                "description": "Fast, interpretable, works well with high-dimensional text data"
            },
            "lstm": {
                "name": "LSTM Neural Network",
                "type": "Deep Learning",
                "description": "Captures context and word order through sequence processing"
            }
        }
    }


@app.get("/models/plot")
async def get_model_plot():
    plot_path = "models/comparison.png"
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Plot not found")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": MODELS_LOADED
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)