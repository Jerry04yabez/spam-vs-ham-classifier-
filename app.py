import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# Import our custom NLP logic
from preprocess import preprocess_text, preprocess_step_by_step

app = FastAPI(title="Spam Classifier NLP API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
METRICS_PATH = os.path.join(BACKEND_DIR, "metrics.json")

# Global variables for models
vectorizer = None
model_nb = None
model_lr = None
model_svm = None
training_metrics = None

def load_resources():
    global vectorizer, model_nb, model_lr, model_svm, training_metrics
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        model_nb = joblib.load(os.path.join(MODEL_DIR, "model_nb.pkl"))
        model_lr = joblib.load(os.path.join(MODEL_DIR, "model_lr.pkl"))
        model_svm = joblib.load(os.path.join(MODEL_DIR, "model_svm.pkl"))
        
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                training_metrics = json.load(f)
        else:
            training_metrics = {"status": "Metrics file not found. Please run train.py first."}
        print("All machine learning models and vectorizer loaded successfully!")
    except Exception as e:
        print(f"Warning: Models could not be loaded. Please run train.py first. Error: {e}")

# Try to load resources on startup
@app.on_event("startup")
async def startup_event():
    load_resources()

class MessageRequest(BaseModel):
    message: str

@app.get("/api/health")
def health_check():
    """
    Service health check and model loading status.
    """
    models_loaded = all([vectorizer, model_nb, model_lr, model_svm])
    return {
        "status": "healthy" if models_loaded else "incomplete_setup",
        "models_loaded": models_loaded,
        "message": "API is online. Please run train.py to build model binaries if models_loaded is False."
    }

@app.get("/api/metrics")
def get_metrics():
    """
    Returns the evaluation metrics generated during training.
    """
    global training_metrics
    if not training_metrics or "status" in training_metrics:
        # Reload just in case training finished after startup
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                training_metrics = json.load(f)
        else:
            raise HTTPException(status_code=503, detail="Model metrics are not available. Run train.py first.")
    return training_metrics

@app.post("/api/preprocess-debug")
def debug_preprocess(req: MessageRequest):
    """
    Exposes full step-by-step pipeline outputs for custom visual dashboard nodes.
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    return preprocess_step_by_step(req.message)

@app.post("/api/classify")
def classify_message(req: MessageRequest):
    """
    Preprocesses, vectorizes, and queries all trained classifiers to classify 
    the input message and return predictions, probability scores, and keyword importances.
    """
    global vectorizer, model_nb, model_lr, model_svm
    
    # Reload models on-demand if they were trained after server startup
    if not all([vectorizer, model_nb, model_lr, model_svm]):
        load_resources()
        if not all([vectorizer, model_nb, model_lr, model_svm]):
            raise HTTPException(
                status_code=503, 
                detail="Machine learning models are not yet trained. Run 'python train.py' in the backend first."
            )
            
    text = req.message
    if not text.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
        
    # 1. Preprocess
    preprocessed_str = preprocess_text(text)
    
    # 2. Vectorize
    tfidf_vector = vectorizer.transform([preprocessed_str])
    
    # Get feature vocabulary mapping
    feature_names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_
    
    # Find words in the message that are in our vocabulary, and get their TF-IDF scores
    words_in_vocab = []
    # Use step-by-step stems to inspect which active tokens contributed
    stems = preprocess_step_by_step(text)["stemmed"]
    
    for stem in set(stems):
        if stem in vocab:
            idx = vocab[stem]
            tfidf_val = tfidf_vector[0, idx]
            if tfidf_val > 0:
                # Calculate simple word spam coefficient based on Logistic Regression weight
                # positive coefficient = spam, negative coefficient = ham
                lr_coef = float(model_lr.coef_[0][idx])
                words_in_vocab.append({
                    "word": stem,
                    "tfidf": float(tfidf_val),
                    "coef": lr_coef,
                    "impact": "spam" if lr_coef > 0 else "ham",
                    "weight": abs(lr_coef) * tfidf_val
                })
                
    # Sort trigger words by absolute impact
    words_in_vocab = sorted(words_in_vocab, key=lambda x: x["weight"], reverse=True)
    
    # 3. Model predictions and probabilities
    # Predictions map: 0 -> ham, 1 -> spam
    labels = {0: "ham", 1: "spam"}
    
    # Naive Bayes
    nb_pred = int(model_nb.predict(tfidf_vector)[0])
    nb_probs = model_nb.predict_proba(tfidf_vector)[0]
    
    # Logistic Regression
    lr_pred = int(model_lr.predict(tfidf_vector)[0])
    lr_probs = model_lr.predict_proba(tfidf_vector)[0]
    
    # SVM
    svm_pred = int(model_svm.predict(tfidf_vector)[0])
    svm_probs = model_svm.predict_proba(tfidf_vector)[0]
    
    return {
        "text": text,
        "preprocessed": preprocessed_str,
        "trigger_words": words_in_vocab[:10], # Top 10 words contributing to decision
        "predictions": {
            "naive_bayes": {
                "label": labels[nb_pred],
                "confidence": float(nb_probs[nb_pred]) * 100,
                "spam_probability": float(nb_probs[1]) * 100,
                "ham_probability": float(nb_probs[0]) * 100
            },
            "logistic_regression": {
                "label": labels[lr_pred],
                "confidence": float(lr_probs[lr_pred]) * 100,
                "spam_probability": float(lr_probs[1]) * 100,
                "ham_probability": float(lr_probs[0]) * 100
            },
            "svm": {
                "label": labels[svm_pred],
                "confidence": float(svm_probs[svm_pred]) * 100,
                "spam_probability": float(svm_probs[1]) * 100,
                "ham_probability": float(svm_probs[0]) * 100
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
