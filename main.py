import os
import re
import uuid
import logging
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "./models/bilstm_stemming_v1")
MODEL_NAME = os.getenv("MODEL_NAME", "bilstm_stemming")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

# (Étape 3 plus tard) Azure Application Insights : activé seulement si la variable est présente
AI_CONN = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
if AI_CONN:
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        configure_azure_monitor()
    except Exception as e:
        print("⚠️ Azure Monitor non configuré:", e)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("airparadis")

# ------------------------------------------------------------
# NLTK preprocessing (stemming + stopwords)
# ------------------------------------------------------------
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
NEGATIONS = {"no", "nor", "not", "never"}
stop_words = stop_words - NEGATIONS
stop_words |= {"rt", "amp", "im", "dont", "u", "ur", "ive", "youre", "thats"}

stemmer = PorterStemmer()
token_re = re.compile(r"[a-z]+")

def preprocess_stem(text: str, min_len: int = 2) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)

    tokens = token_re.findall(text)
    tokens = [t for t in tokens if len(t) >= min_len and t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)

# ------------------------------------------------------------
# Load model (compile=False -> pas de warning)
# ------------------------------------------------------------
model = tf.keras.models.load_model(MODEL_DIR, compile=False)
print("✅ Model loaded from", MODEL_DIR)

def predict_sentiment(text: str):
    cleaned = preprocess_stem(text)
    x = tf.constant([[cleaned]], dtype=tf.string)
    proba_pos = float(model(x, training=False).numpy().reshape(-1)[0])
    label = 1 if proba_pos >= 0.5 else 0
    return label, proba_pos

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Air Paradis - Sentiment API", version="1.0")

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    prediction_id: str
    label: int
    proba_pos: float
    model_name: str
    model_version: str

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text vide")

    pred_id = str(uuid.uuid4())
    label, proba = predict_sentiment(text)

    logger.info("prediction_made", extra={
        "prediction_id": pred_id,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predicted_label": label,
        "predicted_proba": float(proba),
        "text_len": len(text),
    })

    return PredictOut(
        prediction_id=pred_id,
        label=label,
        proba_pos=float(proba),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION
    )

class FeedbackIn(BaseModel):
    prediction_id: str
    text: str
    predicted_label: int
    predicted_proba: float
    user_validated: bool

@app.post("/feedback")
def feedback(payload: FeedbackIn):
    base = {
        "prediction_id": payload.prediction_id,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "predicted_label": int(payload.predicted_label),
        "predicted_proba": float(payload.predicted_proba),
        "user_validated": bool(payload.user_validated),
        "text_len": len(payload.text or "")
    }

    if payload.user_validated:
        logger.info("prediction_validated", extra=base)
    else:
        logger.warning("prediction_rejected", extra=base)

    return {"status": "ok"}
