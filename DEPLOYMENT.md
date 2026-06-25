# Deployment Guide - Sentiment Analysis App

## Quick Deploy (Render + Vercel)

### Backend (Render)

1. Go to [render.com](https://render.com) → "New Web Service"
2. Connect your GitHub repo: `swap821/sentiment-analysis-app`
3. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
4. Click "Create Web Service"
5. Copy the service URL (e.g., `https://sentiment-analysis-api.onrender.com`)

### Frontend (Vercel)

1. Go to [vercel.com](https://vercel.com) → "Add New Project"
2. Import your GitHub repo: `swap821/sentiment-analysis-app`
3. Set **Framework Preset** to "Vite"
4. Set **Root Directory** to `frontend`
5. Add Environment Variable:
   - `VITE_API_URL` = your Render backend URL
6. Click "Deploy"

### One-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/swap821/sentiment-analysis-app)

---

## Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python train_models.py      # Trains TF-IDF + LSTM models
python app.py               # Starts Flask server on localhost:5000

# Frontend
cd frontend
npm install
npm run dev                 # Starts dev server on localhost:3000
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FLASK_ENV` | No | Set to `production` for deployment |
| `ALLOWED_ORIGINS` | Yes | Comma-separated CORS origins |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze` | POST | Analyze sentiment of text |
| `/api/analyze/batch` | POST | Batch analyze CSV file |
| `/api/models` | GET | List available models |
| `/api/health` | GET | Health check |

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'tensorflow'`
**Fix**: Run `pip install tensorflow` (LSTM model only). TF-IDF model works without it.

**Issue**: NLTK data not found
**Fix**: Run `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"`
