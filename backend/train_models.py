"""
train_models.py — Train and Compare Two ML Models for Sentiment Analysis

This script trains TWO different approaches to sentiment classification:
1. TF-IDF + Logistic Regression (traditional ML)
2. Neural Network with LSTM (deep learning)

TF-IDF (Term Frequency-Inverse Document Frequency):
- Term Frequency: How often a word appears in a document
- Inverse Document Frequency: Downweights common words ("the", "and")
- Result: A numerical vector representing text importance

Logistic Regression:
- A linear classifier that learns a decision boundary
- Fast, interpretable, works well with high-dimensional TF-IDF vectors

LSTM (Long Short-Term Memory):
- A type of Recurrent Neural Network (RNN)
- Can remember long-range dependencies in sequences
- Better at understanding context and word order
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Traditional ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Deep Learning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# ============================================================
# STEP 1: Text Preprocessing
# ============================================================

def clean_text(text):
    """
    Clean raw text by removing unwanted characters and formatting.
    
    Steps:
    1. Remove HTML tags (some reviews have <br> tags)
    2. Remove URLs
    3. Remove special characters and digits
    4. Convert to lowercase
    5. Remove extra whitespace
    
    Args:
        text (str): Raw text from the dataset
        
    Returns:
        str: Cleaned text
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and digits, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def remove_stopwords(tokens):
    """
    Remove common English stopwords from a list of tokens.
    
    Stopwords are words like "the", "is", "at", "which" that don't carry
    much meaning. Removing them helps focus on important words.
    
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
    
    Stemming reduces vocabulary size and helps the model generalize.
    
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
    
    Steps: clean → tokenize → remove stopwords → stem → rejoin
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Preprocessed text
    """
    # Clean the text
    text = clean_text(text)
    # Tokenize (split into words)
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    # Stem
    tokens = stem_tokens(tokens)
    # Rejoin into a string
    return ' '.join(tokens)


def preprocess_dataset(df):
    """
    Apply preprocessing to an entire DataFrame.
    
    Args:
        df (DataFrame): DataFrame with 'review' and 'sentiment' columns
        
    Returns:
        DataFrame: Preprocessed data
    """
    print("Preprocessing text data...")
    df = df.copy()
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    # Remove empty reviews after cleaning
    df = df[df['cleaned_review'].str.len() > 0]
    print(f"Preprocessing complete. {len(df)} reviews remaining.")
    return df


# ============================================================
# STEP 2: Train TF-IDF + Logistic Regression (Traditional ML)
# ============================================================

def train_tfidf_logistic(X_train, y_train, X_test, y_test):
    """
    Train a TF-IDF + Logistic Regression model.
    
    Why TF-IDF?
    - Converts text to numerical vectors that ML models can process
    - Weighs words by their importance in the document vs corpus
    - Produces sparse, high-dimensional vectors (one dimension per word)
    
    Why Logistic Regression?
    - Works well with high-dimensional sparse data (like TF-IDF)
    - Fast to train and predict
    - Provides probability scores (confidence)
    - Easy to interpret
    
    Args:
        X_train, X_test: Training and test text data
        y_train, y_test: Training and test labels
        
    Returns:
        tuple: (vectorizer, model, accuracy, predictions)
    """
    print("\n" + "="*60)
    print("Training TF-IDF + Logistic Regression...")
    print("="*60)
    
    # Create TF-IDF vectorizer
    # max_features=5000: Keep only top 5000 most important words
    # This reduces dimensionality and focuses on meaningful words
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit on training data and transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    print(f"  ({X_train_tfidf.shape[0]} documents, {X_train_tfidf.shape[1]} features)")
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTF-IDF + Logistic Regression Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return vectorizer, model, accuracy, y_pred


# ============================================================
# STEP 3: Train LSTM Neural Network (Deep Learning)
# ============================================================

def train_lstm_neural_network(X_train, y_train, X_test, y_test):
    """
    Train an LSTM Neural Network for sentiment classification.
    
    Architecture:
    1. Embedding Layer: Converts word indices to dense vectors (word embeddings)
       - Each word becomes a 128-dimensional vector
       - Similar words get similar vectors
    
    2. LSTM Layer (64 units):
       - Processes the sequence of word embeddings
       - Maintains a "memory" of previous words
       - Can capture context and word order
    
    3. Dropout (0.5): Randomly drops 50% of neurons during training
       - Prevents overfitting
    
    4. Dense Layer (1 unit with sigmoid):
       - Outputs a probability between 0 and 1
       - < 0.5 = negative, >= 0.5 = positive
    
    Why LSTM instead of simple RNN?
    - LSTM has "gates" that control information flow
    - Can remember important information from far back in the sequence
    - Doesn't suffer from vanishing gradient problem as much
    
    Args:
        X_train, X_test: Training and test text data (raw strings)
        y_train, y_test: Training and test labels
        
    Returns:
        tuple: (tokenizer, model, accuracy, predictions, history)
    """
    print("\n" + "="*60)
    print("Training LSTM Neural Network...")
    print("="*60)
    
    # Hyperparameters
    VOCAB_SIZE = 10000      # Maximum number of unique words to keep
    MAX_LENGTH = 200        # Maximum review length (in words)
    EMBEDDING_DIM = 128     # Size of word embedding vectors
    
    # Tokenize text (convert words to integers)
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences to same length
    # Shorter reviews get padded with zeros
    # Longer reviews get truncated
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    
    print(f"Sequence shape: {X_train_pad.shape}")
    print(f"  ({X_train_pad.shape[0]} documents, {MAX_LENGTH} words each)")
    print(f"Vocabulary size: {min(VOCAB_SIZE, len(tokenizer.word_index))}")
    
    # Encode labels as 0/1 integers
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    
    # Build the model
    model = Sequential([
        # Embedding layer: word index -> dense vector
        Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LENGTH),
        
        # LSTM layer: processes sequences
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        
        # Output layer: single neuron with sigmoid activation
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    # Binary crossentropy: standard loss for binary classification
    # Adam optimizer: adaptive learning rate, works well for most problems
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Early stopping: stop training if validation accuracy doesn't improve for 2 epochs
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True
    )
    
    # Train the model
    print("\nTraining... (this may take 10-15 minutes on CPU)")
    history = model.fit(
        X_train_pad, y_train_enc,
        epochs=5,              # Number of training iterations
        batch_size=64,         # Process 64 reviews at a time
        validation_split=0.2,  # Use 20% of training data for validation
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate on test set
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test_enc, y_pred)
    
    print(f"\nLSTM Neural Network Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test_enc, y_pred))
    
    return tokenizer, model, accuracy, y_pred, history, label_encoder


# ============================================================
# STEP 4: Visualization
# ============================================================

def plot_comparison(tfidf_acc, lstm_acc, history=None):
    """
    Create comparison plots between the two models.
    
    Args:
        tfidf_acc: TF-IDF model accuracy
        lstm_acc: LSTM model accuracy
        history: Keras training history (optional)
    """
    os.makedirs('models', exist_ok=True)
    
    # Accuracy comparison bar chart
    plt.figure(figsize=(8, 5))
    models = ['TF-IDF +\nLogistic Regression', 'LSTM\nNeural Network']
    accuracies = [tfidf_acc, lstm_acc]
    colors = ['#3b82f6', '#8b5cf6']
    
    bars = plt.bar(models, accuracies, color=colors, edgecolor='white', linewidth=2)
    plt.ylabel('Test Accuracy')
    plt.title('Model Comparison: Sentiment Analysis')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved to models/comparison.png")
    plt.close()
    
    # If LSTM history available, plot training curves
    if history:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('LSTM Training Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].set_ylim(0, 1)
        
        # Loss
        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('LSTM Training Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('models/lstm_training.png', dpi=150, bbox_inches='tight')
        print("LSTM training curves saved to models/lstm_training.png")
        plt.close()


def plot_confusion_matrices(y_test, tfidf_pred, lstm_pred, label_encoder=None):
    """Plot confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # TF-IDF confusion matrix
    cm1 = confusion_matrix(y_test, tfidf_pred)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('TF-IDF + Logistic Regression')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # LSTM confusion matrix
    cm2 = confusion_matrix(y_test, lstm_pred)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Purples', ax=axes[1])
    axes[1].set_title('LSTM Neural Network')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('models/confusion_matrices.png', dpi=150, bbox_inches='tight')
    print("Confusion matrices saved to models/confusion_matrices.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    """Main training pipeline."""
    print("="*60)
    print("AI SENTIMENT ANALYZER — MODEL TRAINING")
    print("="*60)
    
    # Load data
    data_path = 'data/IMDB Dataset.csv'
    if not os.path.exists(data_path):
        print(f"\nERROR: Dataset not found at {data_path}")
        print("Download from: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        return
    
    print("\nLoading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} reviews")
    
    # Use a subset for faster training (optional - remove [:10000] for full dataset)
    df = df[:10000]
    print(f"Using {len(df)} reviews for training")
    
    # Preprocess
    df = preprocess_dataset(df)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'],
        test_size=0.2, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train Model 1: TF-IDF + Logistic Regression
    vectorizer, lr_model, tfidf_acc, tfidf_pred = train_tfidf_logistic(
        X_train, y_train, X_test, y_test
    )
    
    # Train Model 2: LSTM Neural Network
    tokenizer, lstm_model, lstm_acc, lstm_pred, history, label_encoder = train_lstm_neural_network(
        X_train, y_train, X_test, y_test
    )
    
    # Compare and determine best
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"TF-IDF + Logistic Regression: {tfidf_acc:.4f}")
    print(f"LSTM Neural Network:          {lstm_acc:.4f}")
    
    if lstm_acc > tfidf_acc:
        print(f"\nWinner: LSTM Neural Network (+{lstm_acc - tfidf_acc:.4f})")
    else:
        print(f"\nWinner: TF-IDF + Logistic Regression (+{tfidf_acc - lstm_acc:.4f})")
    
    # Save models
    os.makedirs('models', exist_ok=True)
    pickle.dump((vectorizer, lr_model), open('models/tfidf_model.pkl', 'wb'))
    lstm_model.save('models/lstm_model.h5')
    pickle.dump(tokenizer, open('models/lstm_tokenizer.pkl', 'wb'))
    pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))
    print("\nModels saved to models/ directory")
    
    # Generate plots
    plot_comparison(tfidf_acc, lstm_acc, history)
    
    # Encode y_test for LSTM confusion matrix
    y_test_enc = label_encoder.transform(y_test)
    plot_confusion_matrices(y_test, tfidf_pred, lstm_pred, label_encoder)
    
    print("\nTraining complete! Start the API: python app.py")


if __name__ == "__main__":
    main()