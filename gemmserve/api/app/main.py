from fastapi import FastAPI
import time

app = FastAPI(title="GEMMServe", version="0.1.0")

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
    return {
        "status": "ok",
        "timestamp": time.time()
    }