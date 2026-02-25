"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from app.routes import router

app = FastAPI(
    title="WTI Regime Monitor",
    description="3-regime probabilistic HMM monitor for WTI Cushing daily spot returns.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
