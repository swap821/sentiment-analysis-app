"""
preprocess.py — Text Preprocessing Pipeline

This module handles all text preprocessing for sentiment analysis:
1. Text cleaning (remove HTML, URLs, special characters)
2. Tokenization (split into words)
3. Stopword removal (remove common words)
4. Stemming (reduce words to root form)

Understanding these steps:
- Cleaning: Raw text has noise (HTML tags, URLs). We remove it.
- Tokenization: ML models work with words, not strings. We split text.
- Stopword removal: Words like "the", "is" don't add meaning. We remove them.
- Stemming: "running", "runs", "ran" all mean "run". We reduce to root.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def clean_text(text):
    """
    Clean raw text by removing unwanted characters.
    
    Steps:
    1. Remove HTML tags (<br>, <div>, etc.)
    2. Remove URLs (http://, www., etc.)
    3. Remove special characters and digits
    4. Convert to lowercase
    5. Remove extra whitespace
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def remove_stopwords(tokens):
    """
    Remove common English stopwords.
    
    Stopwords are words like "the", "is", "at", "which", "on"
    that appear frequently but don't carry much meaning.
    Removing them helps the model focus on important words.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: Tokens with stopwords removed
    """
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]


def stem_tokens(tokens):
    """
    Apply Porter Stemming to reduce words to their root form.
    
    Examples:
    - "running" → "run"
    - "happiness" → "happi"
    - "better" → "better"
    - "flies" → "fli"
    
    This reduces vocabulary size and helps the model generalize.
    
    Args:
        tokens (list): List of word tokens
        
    Returns:
        list: Stemmed tokens
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]


def preprocess_text(text):
    """
    Full preprocessing pipeline for a single text.
    
    Pipeline: clean → tokenize → remove stopwords → stem → rejoin
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Preprocessed text (cleaned, tokenized, stemmed)
    """
    # Step 1: Clean
    text = clean_text(text)
    # Step 2: Tokenize
    tokens = word_tokenize(text)
    # Step 3: Remove stopwords
    tokens = remove_stopwords(tokens)
    # Step 4: Stem
    tokens = stem_tokens(tokens)
    # Step 5: Rejoin
    return ' '.join(tokens)


def preprocess_dataset(df):
    """
    Apply preprocessing to an entire DataFrame.
    
    Args:
        df (DataFrame): DataFrame with 'review' and 'sentiment' columns
        
    Returns:
        DataFrame: Preprocessed data with 'cleaned_review' column
    """
    df = df.copy()
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    df = df[df['cleaned_review'].str.len() > 0]
    return df


def save_vectorizer(vectorizer, path):
    """Save a fitted TF-IDF vectorizer to disk."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Saved vectorizer to {path}")


def load_vectorizer(path):
    """Load a saved TF-IDF vectorizer."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_label_encoder(encoder, path):
    """Save a fitted LabelEncoder to disk."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(encoder, f)
    print(f"Saved label encoder to {path}")


def load_label_encoder(path):
    """Load a saved LabelEncoder."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)