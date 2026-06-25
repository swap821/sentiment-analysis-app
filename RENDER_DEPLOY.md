# Deploy to Render (Backend)

1. Go to https://dashboard.render.com/
2. Click "New +" → "Web Service"
3. Connect GitHub repo: `swap821/sentiment-analysis-app`
4. Configure:
   - **Name**: `sentiment-analysis-api`
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt && python train_models.py`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Add Environment Variable:
   - `ALLOWED_ORIGINS` = `https://sentiment-analysis.vercel.app,http://localhost:5173`
6. Click "Create Web Service"

Your backend URL will be: `https://sentiment-analysis-api.onrender.com`
