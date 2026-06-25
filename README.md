# 🧠 Sentiment Analysis App

**Live Demo:** [https://sentiment-analysis-app.vercel.app](https://sentiment-analysis-app.vercel.app) *(deploy after setup)*

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=for-the-badge&logo=tailwind-css&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f5b?style=for-the-badge&logo=python&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)

A full-stack NLP application that classifies text sentiment as **Positive** or **Negative** using two different machine learning approaches side-by-side: **TF-IDF + Logistic Regression** (traditional machine learning) and **LSTM Neural Network** (deep learning). Built with a real-time React dashboard, confidence scoring, and batch CSV processing.

This project demonstrates NLP preprocessing, text vectorization, neural network architecture design, model comparison, and production deployment of text classification models.

---

# ✨ Key Features

## 🤖 Dual-Model NLP Pipeline
- **TF-IDF + Logistic Regression**: Fast, interpretable baseline (~10ms inference)
- **LSTM Neural Network**: Deep learning with word embeddings for nuanced understanding
- Side-by-side model comparison with accuracy metrics

## 💬 Real-Time Sentiment Analysis
- Type any text and get instant sentiment prediction
- Emoji feedback (😊 Positive / 😞 Negative) with confidence bar
- Animated confidence score visualization

## 📁 Batch CSV Processing
- Upload entire CSV files for bulk sentiment analysis
- Download results with predictions appended
- Progress tracking for large datasets

## 📊 Interactive Model Comparison
- Educational section explaining pros/cons of each approach
- Training accuracy/loss charts
- Model architecture diagrams

## 🎨 Beautiful React Dashboard
- Dark theme with gradient accents
- Framer Motion animations
- Fully responsive design

---

# 🛠️ Tech Stack

## Backend (`/backend`)
- Python 3.11
- Flask (REST API)
- NLTK (tokenization, stopword removal, stemming)
- scikit-learn (TF-IDF vectorizer, Logistic Regression)
- TensorFlow / Keras (LSTM neural network)
- Pandas (CSV processing)

## Frontend (`/frontend`)
- React 18
- Vite
- Tailwind CSS
- Framer Motion (animations)

---

# 📂 Project Structure

```plaintext
sentiment-analysis-app/
│
├── backend/                     # Flask API Server
│   ├── models/                  # Trained model artifacts
│   ├── train_models.py          # TF-IDF + LSTM training pipeline
│   ├── preprocess.py            # Text cleaning & NLP preprocessing
│   ├── app.py                   # Flask API endpoints
│   ├── requirements.txt         # Python dependencies
│   └── Procfile                 # Render deployment config
│
├── frontend/                    # React Application
│   ├── src/
│   │   ├── components/          # UI Components
│   │   │   ├── TextAnalyzer.jsx     # Main sentiment input/output
│   │   │   ├── ModelComparison.jsx  # Educational model comparison
│   │   │   └── BatchProcessor.jsx   # CSV upload & processing
│   │   ├── App.jsx              # Main Application
│   │   └── main.jsx             # Entry point
│   ├── package.json
│   └── vite.config.js
│
├── render.yaml                  # Render blueprint
├── vercel.json                  # Vercel deployment config
└── README.md
```

---

# 🚀 Getting Started

## 📌 Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/swap821/sentiment-analysis-app.git
cd sentiment-analysis-app
```

---

# 🔧 Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## 🏋️ Train the Models

```bash
# Train both TF-IDF and LSTM models
python train_models.py
```

**Output:** Saves `tfidf_model.pkl` and `lstm_model.h5` in `backend/models/`

**NLP Pipeline:**
```
Raw Text → Lowercase → Remove Special Chars → Tokenize → Remove Stopwords → Stem → Vectorize → Predict
```

**LSTM Architecture:**
```
Input → Embedding(5000, 128) → LSTM(64, dropout=0.2) → Dense(1, sigmoid) → Output
```

---

## ▶️ Start the Flask API

```bash
python app.py
```

API runs at `http://localhost:5000`

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/analyze` | POST | Analyze sentiment of a single text |
| `/api/analyze/batch` | POST | Batch analyze a CSV file |
| `/api/models` | GET | Get model info and metrics |
| `/api/health` | GET | Health check |

### Example Request

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I absolutely love this product, it exceeded all my expectations!", "model": "tfidf"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "emoji": "😊",
  "model_used": "TF-IDF + Logistic Regression"
}
```

---

# 🎨 Frontend Setup

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

# 🌍 Deployment

## Backend — Render

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repo: `swap821/sentiment-analysis-app`
4. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt && python train_models.py`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Add Environment Variable:
   - `ALLOWED_ORIGINS` = `https://sentiment-analysis-app.vercel.app,http://localhost:5173`
6. Click **Create Web Service**

## Frontend — Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click **Add New Project** → Import `sentiment-analysis-app`
3. Configure:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
4. Add Environment Variable:
   - `VITE_API_URL` = `https://sentiment-analysis-api.onrender.com`
5. Click **Deploy**

---

# 🧠 NLP & ML Concepts Demonstrated

This project demonstrates understanding of:

- **Text Preprocessing**: Cleaning, tokenization, stopword removal, stemming
- **TF-IDF Vectorization**: Converting text to numerical features
- **Logistic Regression**: Probabilistic classification for sentiment
- **LSTM Networks**: Sequential learning for context-aware predictions
- **Word Embeddings**: Dense vector representations of words
- **Model Comparison**: Traditional ML vs Deep Learning tradeoffs
- **REST API for NLP**: Serving text classification models
- **Batch Processing**: Handling large datasets efficiently

---

# 🚀 Future Improvements

- [ ] Add neutral sentiment class (3-way classification)
- [ ] Fine-tune BERT/BERTweet for state-of-the-art accuracy
- [ ] Add sentiment trend analysis over time
- [ ] Support multiple languages
- [ ] Real-time Twitter/social media sentiment monitoring
- [ ] Docker containerization
- [ ] Add model explainability (LIME/SHAP for text)

---

# 👨‍💻 Author

## Swapnil Kumar

- GitHub: https://github.com/swap821
- LinkedIn: https://www.linkedin.com/in/swapnil-kumar-73a68a308
- Portfolio: https://swapnil-kumar-portfolio016.vercel.app

---

# ⭐ Project Goal

This project was built to demonstrate:
- Natural Language Processing pipeline end-to-end
- Traditional ML vs Deep Learning comparison
- Text classification model deployment
- Full-stack NLP application architecture
- Real-time text analysis with confidence scoring
- Batch processing for production use cases

---

# 📜 License

This project is open-source and available for educational and learning purposes.
