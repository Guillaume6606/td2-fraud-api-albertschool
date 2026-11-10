"""Service de scoring : POST /predict.

Version de départ du TD2. Elle répond juste, mais trois points sont à reprendre à l'étape 0.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from fastapi import FastAPI, Request

import features as ft
import scoring
import settings

app = FastAPI(title="TD2 fraud API")


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    cfg = settings.from_env()
    model = scoring.load_model(cfg.models_dir, cfg.xgb_nthread)
    state = ft.load_state(cfg.data_dir, cfg.horizon_days)
    transactions = body["transactions"]
    ts = np.array([datetime.fromisoformat(t["tx_ts"]) for t in transactions], dtype="datetime64[s]")
    scores, _ = scoring.score(
        state,
        model,
        ts,
        [t["customer_id"] for t in transactions],
        [t["terminal_id"] for t in transactions],
        np.array([t["tx_amount"] for t in transactions], dtype=np.float64),
    )
    return {
        "scores": [
            {"tx_id": t["tx_id"], "score": float(s)}
            for t, s in zip(transactions, scores, strict=True)
        ]
    }
