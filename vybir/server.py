"""HTTP API (FastAPI). Run: ``vybir serve`` or ``uvicorn vybir.server:app``.

Environment: VYBIR_MODEL (default convaiinnovations/laya), VYBIR_DTYPE (default auto),
VYBIR_BACKEND (torch|ort), VYBIR_ORT_PATH (ONNX file for the ort backend).
"""

from __future__ import annotations

import os
import threading
import time

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError as error:  # pragma: no cover
    raise RuntimeError("pip install 'vybir[server]' to use the HTTP server") from error

from . import __version__
from .agent import Agent
from .router import Router

MODEL_ID = os.environ.get("VYBIR_MODEL", "convaiinnovations/laya")
DTYPE = os.environ.get("VYBIR_DTYPE", "auto")
BACKEND = os.environ.get("VYBIR_BACKEND", "torch")
ORT_PATH = os.environ.get("VYBIR_ORT_PATH")

app = FastAPI(title="vybir", version=__version__)
_lock = threading.Lock()  # concurrent first requests must not load the model twice
_agent = None
_router = None


def get_agent():
    global _agent
    with _lock:
        if _agent is None:
            _agent = Agent(MODEL_ID, dtype=DTYPE, backend=BACKEND, ort_path=ORT_PATH)
        return _agent


def get_router():
    global _router
    with _lock:
        if _router is None:
            _router = Router(dtype=DTYPE, max_loaded=2)
        return _router


class PredictBody(BaseModel):
    state: object
    questions: dict
    model: str | None = None  # english | multilingual | typed-decisions -> uses the Router
    task: str | None = None
    lang: str | None = None


class RouteBody(BaseModel):
    state: object = ""
    questions: dict | None = None
    model: str | None = None
    task: str | None = None
    lang: str | None = None


@app.get("/health")
def health():
    agent = get_agent()
    return {
        "ok": True, "version": __version__, "model": MODEL_ID, "backend": agent.backend,
        "device": str(agent.device), "dtype": agent.dtype_name,
    }


@app.post("/predict")
def predict(body: PredictBody):
    started = time.perf_counter()
    try:
        if body.model or body.task or body.lang:
            out = get_router().predict(
                body.state, body.questions, model=body.model, task=body.task, lang=body.lang
            )
        else:
            out = get_agent().predict(body.state, body.questions)
    except ValueError as error:  # malformed questions are the caller's fault, not a 500
        raise HTTPException(status_code=422, detail=str(error)) from error
    out["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return out


@app.post("/route")
def route(body: RouteBody):
    try:
        decision = Router().route(
            body.state, body.questions, model=body.model, task=body.task, lang=body.lang
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return dict(decision)
