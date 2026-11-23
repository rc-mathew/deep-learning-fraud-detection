"""FastAPI scoring endpoint.  Run:  uvicorn app.api:app --port 8000"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fraud_dl.serve import Scorer  # noqa: E402

app = FastAPI(title="Deep learning fraud scoring", version="1.0")
scorer = Scorer()


class ScoreRequest(BaseModel):
    transaction_id: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "transactions": len(scorer.transaction_ids())}


@app.get("/transactions")
def transactions(limit: int = 50) -> list[int]:
    return scorer.transaction_ids()[: max(1, min(limit, 1000))]


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    try:
        return scorer.score(req.transaction_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="transaction_id not in the demo sample")
