"""
analyze_router.py

Generalized "bring your own dataset" endpoint for RFM segmentation and
churn risk scoring. Designed to sit alongside your existing Zomato-specific
endpoints in main.py without touching them.

Integration (in main.py):

    from analyze_router import router as analyze_router
    app.include_router(analyze_router)

That's it — no other changes needed to your existing /insights,
/rfm-segments, /predict-churn, /restaurants endpoints.
"""

import io
import math
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

router = APIRouter()

# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------
# For each field we need, a list of common column-name variants to match
# against (case-insensitive, punctuation/underscore-insensitive).
CANDIDATE_PATTERNS = {
    "customer_id": [
        "customer_id", "custid", "cust_id", "user_id", "userid",
        "client_id", "clientid", "customerid",
    ],
    "order_date": [
        "order_date", "orderdate", "date", "purchase_date",
        "transaction_date", "order_time", "timestamp", "created_at",
    ],
    "order_value": [
        "order_value", "ordervalue", "amount", "total", "price",
        "revenue", "order_total", "value", "order_amount", "sales",
    ],
}


def _normalize(col: str) -> str:
    return col.strip().lower().replace(" ", "_").replace("-", "_")


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Returns a dict like:
        {
            "customer_id": "cust_id" | None,
            "order_date": "order_date" | None,
            "order_value": "amount" | None,
        }
    Exact normalized match first, then substring match as a fallback.
    """
    normalized_cols = {_normalize(c): c for c in df.columns}
    detected = {}

    for field, patterns in CANDIDATE_PATTERNS.items():
        match = None

        # 1. exact normalized match
        for p in patterns:
            if p in normalized_cols:
                match = normalized_cols[p]
                break

        # 2. substring match (e.g. "cust_id_hash" contains "cust_id")
        if match is None:
            for norm_col, original_col in normalized_cols.items():
                if any(p in norm_col for p in patterns):
                    match = original_col
                    break

        detected[field] = match

    return detected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _clean_for_json(obj):
    """Recursively replace NaN/inf with None so JSON serialization never breaks."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _rfm_segment(r_score: int, f_score: int, m_score: int) -> str:
    """Map R/F/M quartile scores (1-4) to a human-readable segment name."""
    total = r_score + f_score + m_score

    if r_score >= 4 and f_score >= 3:
        return "Champions"
    if r_score >= 3 and f_score >= 3:
        return "Loyal Customers"
    if r_score >= 3 and f_score <= 2:
        return "Promising"
    if r_score <= 2 and f_score >= 3:
        return "At Risk"
    if r_score <= 2 and f_score <= 2 and m_score >= 3:
        return "Cant Lose Them"
    if total <= 5:
        return "Lost"
    return "Needs Attention"


def _churn_risk(recency_days: float, recency_p75: float, recency_p50: float) -> str:
    """Rule-based churn risk off the dataset's own recency distribution."""
    if recency_days >= recency_p75:
        return "High"
    if recency_days >= recency_p50:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    customer_id_col: Optional[str] = Form(None),
    order_date_col: Optional[str] = Form(None),
    order_value_col: Optional[str] = Form(None),
):
    """
    Upload any transactional CSV and get back RFM segments + churn risk.

    If customer_id_col / order_date_col / order_value_col are provided,
    they override auto-detection (used when the frontend collects a
    manual mapping after a failed auto-detect).
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file has no rows.")

    detected = detect_columns(df)

    # Manual overrides from the frontend, if supplied
    if customer_id_col:
        detected["customer_id"] = customer_id_col
    if order_date_col:
        detected["order_date"] = order_date_col
    if order_value_col:
        detected["order_value"] = order_value_col

    missing = [f for f, col in detected.items() if col is None]
    if missing:
        # Don't fail silently — tell the frontend exactly what it needs
        # so it can render a column-mapping form.
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Could not auto-detect required columns.",
                "detected": detected,
                "missing": missing,
                "available_columns": list(df.columns),
            },
        )

    cust_col = detected["customer_id"]
    date_col = detected["order_date"]
    value_col = detected["order_value"]

    # --- Clean & coerce types ---------------------------------------------
    df = df[[cust_col, date_col, value_col]].copy()
    df.columns = ["customer_id", "order_date", "order_value"]

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")
    df = df.dropna(subset=["customer_id", "order_date", "order_value"])

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="After cleaning, no valid rows remained. Check date/amount formats.",
        )

    # --- RFM computation -----------------------------------------------------
    snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("customer_id").agg(
        recency=("order_date", lambda x: (snapshot_date - x.max()).days),
        frequency=("order_date", "count"),
        monetary=("order_value", "sum"),
    ).reset_index()

    # Quartile scores. Recency is inverted (lower recency = better = higher score).
    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4], duplicates="drop").astype(int)

    rfm["segment"] = rfm.apply(
        lambda row: _rfm_segment(row["r_score"], row["f_score"], row["m_score"]), axis=1
    )

    recency_p75 = rfm["recency"].quantile(0.75)
    recency_p50 = rfm["recency"].quantile(0.50)

    rfm["churn_risk"] = rfm["recency"].apply(
        lambda r: _churn_risk(r, recency_p75, recency_p50)
    )

    # --- Aggregate response ---------------------------------------------------
    segment_counts = rfm["segment"].value_counts().to_dict()
    churn_counts = rfm["churn_risk"].value_counts().to_dict()

    response = {
        "detected_columns": detected,
        "row_count": int(len(df)),
        "customer_count": int(rfm.shape[0]),
        "rfm_segments": {
            "summary": segment_counts,
            "customers": rfm[
                ["customer_id", "recency", "frequency", "monetary", "segment"]
            ].to_dict(orient="records"),
        },
        "churn_risk": {
            "summary": churn_counts,
            "customers": rfm[
                ["customer_id", "recency", "churn_risk"]
            ].to_dict(orient="records"),
        },
    }

    return _clean_for_json(response)
