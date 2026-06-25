# Sentiment Analysis App

**Live Demo:** [https://63dcmrawskzy4.kimi.page](https://63dcmrawskzy4.kimi.page)

Full-stack sentiment analysis application that classifies text as positive or negative using two ML approaches: TF-IDF + Logistic Regression (traditional) and Neural Network with LSTM (deep learning).

## Architecture
```
User Input (text) → React Dashboard → Flask API → TF-IDF/LSTM → Sentiment Result + Confidence
```

## What Makes This Special
- **Two model comparison**: Traditional ML vs deep learning side-by-side
- **Real-time + batch**: Analyze single texts or upload entire CSV files
- **Educational**: Every component explains the ML concepts behind it

## Tech Stack
- Backend: Python, Flask, NLTK, Scikit-learn, TensorFlow/Keras
- Frontend: React, Tailwind CSS, Recharts, Framer Motion
- ML Models: TF-IDF + Logistic Regression, LSTM Neural Network

## Quick Start
```bash
cd backend
pip install -r requirements.txt
python train_models.py    # Train both models
python app.py             # API on localhost:5000
cd ../frontend
npm install && npm run dev # UI on localhost:5173
```

## Deploy
- Backend: [Deploy to Render](https://render.com/deploy?repo=https://github.com/swap821/sentiment-analysis-app)
- See `DEPLOYMENT.md` for full instructions

## Author
**Swapnil Kumar** — [Portfolio](https://swapnil-kumar-portfolio016.vercel.app) | [GitHub](https://github.com/swap821)
