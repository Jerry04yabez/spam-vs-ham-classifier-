# SpamShield AI Platform

## Overview
SpamShield is a full‑stack machine‑learning web application that classifies SMS messages as **spam** or **ham** in real‑time. It demonstrates a classic NLP pipeline (cleaning, tokenisation, stemming, TF‑IDF vectorisation) and three classic classifiers (Multinomial Naïve Bayes, Logistic Regression, Linear SVM) with a voting‑based consensus.

## Repository Structure
```
ml model/
├─ backend/               # FastAPI server
│   ├─ data/             # Cached sms.tsv dataset
│   ├─ models/           # Serialized vectoriser & models
│   ├─ app.py            # API routes
│   ├─ train.py          # Training script
│   ├─ preprocess.py     # Text preprocessing utilities
│   └─ metrics.json     # Evaluation metrics for UI
├─ frontend/              # React (Vite) UI
│   ├─ src/App.jsx       # Main component with real‑time classification
│   ├─ src/...           # Styles, assets, etc.
│   └─ README.md        # Vite/React template information
├─ README.md             # **This document** (project overview)
└─ .gitignore
```

## Prerequisites
- **Python 3.8+** (recommended to use a virtual environment)
- **Node.js 18+** and **npm**
- Git

## Setup & Installation
### Backend (Python)
```powershell
cd "C:\Users\ASUS\OneDrive\yabez\ml model\backend"
# optional virtual env
python -m venv .venv
& .venv\Scripts\Activate.ps1   # PowerShell (or .venv\Scripts\activate.bat)
pip install -r requirements.txt   # pandas, numpy, scikit‑learn, nltk, joblib, fastapi, uvicorn
```
The first run will automatically download the NLTK stop‑words and the SMS Spam dataset.

### Frontend (React)
```powershell
cd "C:\Users\ASUS\OneDrive\yabez\ml model\frontend"
npm install   # install dependencies
npm run dev   # Vite dev server (usually http://localhost:5173)
```

## Training the Models
```powershell
python train.py
```
The script:
1. Downloads the dataset (`sms.tsv`) if missing.
2. Preprocesses the text via `preprocess_text`.
3. Vectorises with TF‑IDF.
4. Trains three classifiers and evaluates them.
5. Saves the vectoriser and model binaries (`model_nb.pkl`, `model_lr.pkl`, `model_svm.pkl`).
6. Writes `metrics.json` with accuracy, confusion matrices, and top discriminative words.

## Running the Application
1. **Start the FastAPI backend**
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```
2. **Start the React frontend** (in a separate terminal)
```powershell
npm run dev
```
3. Open the frontend URL (e.g., `http://localhost:5173`).
   - Type or paste a message – classification updates automatically after a short debounce.
   - View per‑model confidence gauges, consensus label, and a step‑by‑step NLP debug panel.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Returns backend health status and model‑load flag. |
| `/api/classify` | POST | Body `{ "message": "text" }`. Returns predictions and confidence for all three models. |
| `/api/preprocess-debug` | POST | Same payload; returns detailed preprocessing stages for UI visualization. |
| `/api/metrics` | GET | Returns `metrics.json` – dataset stats, model benchmark scores, top spam/ham words. |

## Customisation
- **Add new models** – train in `train.py`, dump with `joblib`, expose via a new route in `app.py`.
- **Swap vectoriser** – replace `TfidfVectorizer` with `CountVectorizer` or custom embeddings.
- **Modify preprocessing** – edit `backend/preprocess.py` (e.g., lemmatisation, custom stop‑words).
- **Docker** – ready‑to‑use `Dockerfile`s are provided for both backend and frontend; you can orchestrate them with `docker‑compose.yml`.

## Contribution Guidelines
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your‑feature`).
3. Ensure the backend tests pass: `pytest backend/tests` (if tests are added).
4. Run the frontend linting: `npm run lint`.
5. Submit a pull request with a clear description of changes.

## License
This project is licensed under the MIT License – see the `LICENSE` file.

---
*Enjoy exploring SpamShield! 🚀*
