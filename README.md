# AI Sentiment Analyzer

## Overview
Full-stack sentiment analysis application that classifies text as positive or negative using two ML approaches: TF-IDF + Logistic Regression (traditional) and Neural Network with LSTM (deep learning). Built with Flask + Scikit-learn/TensorFlow backend and React + Tailwind frontend.

## Live Demo
[Coming soon]

## Architecture
```
User Input (text)
    |
    v
React Dashboard
    |
    v
Flask API
    |
    +-- TF-IDF + Logistic Regression (fast, interpretable)
    +-- LSTM Neural Network (deep learning, contextual)
    |
    v
Sentiment Result (positive/negative + confidence)
```

## What Makes This Special
- **Two model comparison**: See traditional ML vs deep learning side-by-side
- **Real-time + batch**: Analyze single texts or upload entire CSV files
- **Educational**: Every component explains the ML concepts behind it
- **Beautiful UI**: Dark premium design with animated results

## Tech Stack
- Backend: Python, Flask, NLTK, Scikit-learn, TensorFlow/Keras
- Frontend: React, Tailwind CSS, Recharts, Framer Motion
- ML Models: TF-IDF + Logistic Regression, LSTM Neural Network

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+

### Step 1: Download Dataset
1. Go to https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Download IMDB Dataset.csv
3. Place at `backend/data/IMDB Dataset.csv`

### Step 2: Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python nltk_setup.py      # Download NLTK data
python train_models.py    # Train both models (~10-15 min)
python app.py             # Start API on port 5000
```

### Step 3: Frontend Setup
```bash
cd frontend
npm install
npm run dev               # Starts on port 5173
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /analyze | POST | Analyze single text |
| /analyze/batch | POST | Upload CSV for batch analysis |
| /models | GET | Model comparison info |
| /models/plot | GET | Comparison chart |
| /health | GET | Health check |

## Author
**Swapnil Kumar** — Full-Stack Developer & AI Enthusiast
- Portfolio: https://swapnil-kumar-portfolio016.vercel.app
- GitHub: https://github.com/swap821