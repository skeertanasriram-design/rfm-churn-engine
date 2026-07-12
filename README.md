# Dataset Insights Engine

Upload any transactional CSV and get automatic RFM (Recency, Frequency,
Monetary) customer segmentation and churn risk scoring — no dataset-specific
configuration required.

## How it works

- Upload a CSV with customer transaction data
- The engine auto-detects `customer_id`, `order_date`, and `order_value`
  columns from common naming variants
- If detection fails, the API returns which columns it found and asks for
  manual mapping
- Returns RFM segments (Champions, Loyal Customers, At Risk, Lost, etc.)
  and churn risk (High/Medium/Low) per customer, computed from that
  dataset's own distribution — not fixed thresholds

## Endpoints

- `POST /analyze` — upload a CSV, get RFM + churn results
- `GET /health` — health check
- `GET /docs` — interactive API docs (Swagger UI)

## Tech stack

FastAPI, pandas, deployed on Railway.
