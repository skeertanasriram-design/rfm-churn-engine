"""
main.py — Dataset Insights Engine

A standalone API: upload any transactional CSV, get back RFM segmentation
and churn risk scoring. No hardcoded dataset, no local dependency —
runs entirely on Railway once deployed.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analyze_router import router as analyze_router

app = FastAPI(
    title="Dataset Insights Engine",
    description="Upload a transactional CSV and get RFM segmentation + churn risk scoring.",
    version="1.0.0",
)

# Allow requests from any frontend (Vercel, localhost, etc.)
# Tighten this to your specific Vercel domain once the frontend is deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Dataset Insights Engine is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
