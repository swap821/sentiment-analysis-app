# Setup Guide — AI Sentiment Analyzer

## Prerequisites
- Python 3.9+
- Node.js 18+

## Step 1: Download the Dataset
1. Visit: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
2. Click "Download" (free Kaggle account required)
3. Place `IMDB Dataset.csv` in `backend/data/IMDB Dataset.csv`

## Step 2: Set Up the Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python nltk_setup.py      # Downloads required NLTK data
python train_models.py    # Trains both TF-IDF and Neural Network models
python app.py             # Starts Flask API on localhost:5000
```

## Step 3: Set Up the Frontend
```bash
cd frontend
npm install
npm run dev               # Starts on localhost:5173
```

## Common Errors
- **"Resource punkt not found"**: Run `python nltk_setup.py` first
- **"No module named 'tensorflow'"**: Use `pip install tensorflow-cpu` for lighter install
- **Training takes too long**: The LSTM model can take 10-15 minutes on CPU. This is normal.

## Deployment
- Backend: Render.com (set build command to install deps + run nltk_setup.py + train_models.py)
- Frontend: Vercel.com (framework preset: Vite)