import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"

needs_data = pytest.mark.skipif(
    not (DATA / "replay.parquet").exists() or not (MODELS / "model.ubj").exists(),
    reason="data/ ou models/ absents",
)


@pytest.fixture(scope="session")
def golden() -> dict:
    return json.loads((MODELS / "golden.json").read_text())


@pytest.fixture(scope="session")
def replay() -> list[dict]:
    """Les premières transactions du rejeu, au format JSON de l'API."""
    rows = pq.read_table(DATA / "replay.parquet").slice(0, 500).to_pylist()
    return [
        {
            "tx_id": r["tx_id"],
            "tx_ts": r["tx_ts"].isoformat(),
            "customer_id": r["customer_id"],
            "terminal_id": r["terminal_id"],
            "tx_amount": r["tx_amount"],
        }
        for r in rows
    ]
