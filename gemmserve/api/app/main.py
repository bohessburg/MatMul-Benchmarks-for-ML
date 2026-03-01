import hashlib
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db, init_db
from app.inference.model import forward, load_labels, load_model, softmax
from app.models import Prediction
from app.schemas import PredictionRecord, PredictRequest, PredictResponse

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "scripts" / "outputs"

resources: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources["layers"] = load_model()
    resources["labels"] = load_labels()
    with open(OUTPUTS_DIR / "vectorizer.pkl", "rb") as f:
        resources["vectorizer"] = pickle.load(f)
    init_db()
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
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    vectorizer = resources["vectorizer"]
    layers = resources["layers"]
    labels = resources["labels"]

    t0 = time.perf_counter()

    features = vectorizer.transform([req.text])
    x = np.asarray(features.todense(), dtype=np.float32).squeeze(0)

    logits = forward(x, layers)
    probs = softmax(logits)

    latency_ms = (time.perf_counter() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    scores = {label: float(probs[i]) for i, label in enumerate(labels)}

    input_hash = hashlib.sha256(req.text.encode()).hexdigest()

    row = Prediction(
        input_hash=input_hash,
        language=labels[pred_idx],
        scores=scores,
        latency_ms=latency_ms,
        kernel=req.kernel,
        model_version=req.model_version,
    )
    db.add(row)
    db.commit()

    return PredictResponse(language=labels[pred_idx], scores=scores, latency_ms=latency_ms, kernel=req.kernel, model_version=req.model_version)


@app.get("/predictions", response_model=list[PredictionRecord])
def list_predictions(limit: int = 50, db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).all()
    return rows
