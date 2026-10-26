"""The provider's data source: a stand-in for whatever HTTP API a participant
actually contributes to the data space.

The EDC connector proxies requests to this service, so it never needs to be
exposed publicly. Its OpenAPI schema is what the connector's OpenAPI extension
reads at start-up.
"""

from __future__ import annotations

import datetime as dt
import random
from typing import Dict, List

from fastapi import FastAPI

app = FastAPI(title="Climate observations", version="1.0.0")

_STATIONS = ["ES-AST-01", "GR-ATH-04", "DE-BER-11"]


@app.get("/data", summary="Daily climate observations")
async def data(days: int = 7) -> Dict[str, List[Dict[str, object]]]:
    today = dt.date.today()
    rng = random.Random(42)
    return {
        "observations": [
            {
                "date": (today - dt.timedelta(days=offset)).isoformat(),
                "station": station,
                "temperature_c": round(rng.uniform(-4, 34), 1),
                "precipitation_mm": round(rng.uniform(0, 22), 1),
            }
            for offset in range(max(1, min(days, 90)))
            for station in _STATIONS
        ]
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
