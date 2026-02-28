import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

from app.inference.model import forward, load_labels, load_model, softmax
from app.schemas import PredictRequest, PredictResponse

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "outputs"

resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources["layers"] = load_model()
    resources["labels"] = load_labels()
    with open(OUTPUTS_DIR / "vectorizer.pkl", "rb") as f:
        resources["vectorizer"] = pickle.load(f)
    yield
    resources.clear()


app = FastAPI(title="GEMMServe", version="0.1.0", lifespan=lifespan)


@app.get("/")
def info():
    return {
        "service": "GEMMServe",
        "version": app.version,
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    vectorizer = resources["vectorizer"]
    layers = resources["layers"]
    labels = resources["labels"]

    features = vectorizer.transform([req.text])
    x = np.asarray(features.todense(), dtype=np.float32).squeeze(0)

    logits = forward(x, layers)
    probs = softmax(logits)

    pred_idx = int(np.argmax(probs))
    scores = {label: float(probs[i]) for i, label in enumerate(labels)}

    return PredictResponse(language=labels[pred_idx], scores=scores)
