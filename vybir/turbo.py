"""Turbo: CPU INT8 quantization + exact-match result cache.

Two independent, composable speedups that change no weights and no prompts:

1. ``quantize='int8'`` — torchao weight-only int8 for encoder/scorer/act_head
   (CPU only). Halves weight memory; speed needs ``compile=True`` with a C++
   compiler. Parity is measured, not assumed (see scripts/turbo_bench.py).
2. ``result_cache_size=N`` — LRU cache on exact (state, questions) hits.
   Inference is deterministic, so an exact hit is bit-identical to recompute.
"""

from __future__ import annotations

import copy
import json
import threading
from collections import OrderedDict

import torch

from .common import serialize_state


def quantize_int8(model: torch.nn.Module) -> torch.nn.Module:
    """INT8 weight-only quantization of encoder + scorer + act_head (in place).

    W8A16: weights per-channel int8, activations stay fp32 — immune to the
    activation outliers that destroy naive W8A8 dynamic quantization on this
    encoder (measured drift mean 0.45 / max 34 at 0.55 signal level).
    Requires torchao. The 2-layer decision ``head`` stays fp32.
    torch.compile remains compatible with the quantized layers.
    """
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("int8 weight-only quantization is CPU-only; use device='cpu'")
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("quantize='int8' needs torchao: pip install torchao") from e
    model.eval()
    with torch.no_grad():
        for name in ("encoder", "scorer", "act_head"):
            sub = getattr(model, name, None)
            if sub is not None:
                quantize_(sub, Int8WeightOnlyConfig())
    return model


class ResultCache:
    """Thread-safe exact-match LRU for full predict() payloads."""

    def __init__(self, capacity: int = 512):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("result_cache_size must be a positive integer")
        self.capacity = capacity
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(state, questions) -> str:
        # Order-preserving on purpose: the order of choice criteria is part of the
        # prompt, so {"a", "b"} and {"b", "a"} are different requests.
        return json.dumps([serialize_state(state), questions], ensure_ascii=False, default=str)

    def get(self, key: str):
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return copy.deepcopy(self._entries[key])
            self.misses += 1
            return None

    def put(self, key: str, value: dict):
        with self._lock:
            self._entries[key] = copy.deepcopy(value)
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    @property
    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "size": len(self._entries),
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
