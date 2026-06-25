"""
utils.py — Utility Functions for Sentiment Analysis

Helper functions used across the application:
- Logging setup
- Text validation
- Response formatting
- Error handling
"""

import logging
import re


def setup_logging():
    """Configure logging for the Flask application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)


def validate_text(text):
    """
    Validate input text before processing.
    
    Rules:
    - Must not be empty
    - Must not exceed 5000 characters
    - Must contain at least some alphabetic characters
    
    Args:
        text (str): Input text to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Text cannot be empty"
    
    if len(text) > 5000:
        return False, f"Text too long ({len(text)} chars). Maximum is 5000."
    
    # Check if text contains at least some letters
    if not re.search(r'[a-zA-Z]', text):
        return False, "Text must contain some alphabetic characters"
    
    return True, None


def format_response(text, sentiment, confidence, model_used):
    """
    Format a prediction result as JSON.
    
    Args:
        text (str): Original input text
        sentiment (str): Predicted sentiment (positive/negative)
        confidence (float): Confidence score (0-1)
        model_used (str): Name of the model used
        
    Returns:
        dict: Formatted response
    """
    return {
        'text': text[:200] + '...' if len(text) > 200 else text,
        'sentiment': sentiment,
        'confidence': round(float(confidence), 4),
        'confidence_percentage': f"{confidence * 100:.1f}%",
        'model_used': model_used,
        'status': 'success'
    }


def format_error_response(error_message, status_code=400):
    """Format an error response."""
    return {
        'status': 'error',
        'message': error_message,
        'status_code': status_code
    }