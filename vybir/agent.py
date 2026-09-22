"""Public Torch inference runtime; prompt and result formats match upstream Laya."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    clamp_temperature,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from .model import DecisionModelTorch, EncoderConfig, remap_mlx_to_torch
from .prepared import PrefixCache
from .tokenizer import Tokenizer
from .turbo import ResultCache, quantize_int8

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def resolve_model(model_id_or_path, *, token=None, subfolder=None, revision=None):
    if subfolder:
        part = PurePosixPath(subfolder)
        if part.is_absolute() or ".." in part.parts:
            raise ValueError("subfolder must be a relative path inside the model repository")
    path = Path(model_id_or_path).expanduser()
    if not path.exists():
        value = str(model_id_or_path)
        if (
            isinstance(model_id_or_path, Path)
            or Path(value).is_absolute()
            or value.startswith((".", "~", "/", "\\"))
        ):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        prefix = subfolder.rstrip("/") + "/" if subfolder else ""
        patterns = [
            prefix + name
            for name in (
                "model.safetensors",
                "rl_agent_config.json",
                "encoder/config.json",
                "tokenizer/*",
            )
        ]
        kwargs = dict(token=token, revision=revision, allow_patterns=patterns)
        path = None
        try:  # cached snapshot first: no network round-trip, works offline
            cached = Path(snapshot_download(value, local_files_only=True, **kwargs))
            if _missing(cached / subfolder if subfolder else cached) is None:
                path = cached
        except Exception:
            pass
        if path is None:
            path = Path(snapshot_download(value, **kwargs))
    if subfolder:
        path /= subfolder
    missing = _missing(path)
    if missing is not None:
        raise FileNotFoundError(f"Not a complete Laya checkpoint: {missing} is missing")
    return path


def _missing(path):
    for name in ("model.safetensors", "rl_agent_config.json", "encoder/config.json"):
        if not (path / name).is_file():
            return path / name
    return None


def collate_items(items, pad_id, *, pad_to_multiple=None, max_length=None, device=None):
    if not items:
        raise ValueError("Cannot collate an empty batch")
    n, length = len(items), max(len(item["ids"]) for item in items)
    if pad_to_multiple:
        length = ((length + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if max_length is not None:
            length = min(length, max_length)
    count = max(2, max(len(item["markers"]) for item in items))
    input_ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, count), dtype=torch.long)
    marker_mask = torch.zeros((n, count), dtype=torch.bool)
    qtype = torch.tensor([item["qtype"] for item in items], dtype=torch.long)
    for i, item in enumerate(items):
        ln, cn = len(item["ids"]), len(item["markers"])
        input_ids[i, :ln] = torch.tensor(item["ids"], dtype=torch.long)
        attention_mask[i, :ln] = 1
        if cn:
            marker_pos[i, :cn] = torch.tensor(item["markers"], dtype=torch.long)
            marker_mask[i, :cn] = True
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "qtype": qtype,
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def _resolve_device(device):
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = str(device).lower()
    if d in ("gpu", "cuda"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "metal":
        warnings.warn("vybir: Metal is not available on this platform; using CPU.",
                      RuntimeWarning, stacklevel=3)
        return torch.device("cpu")
    if d == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be None/'auto', 'cuda'/'gpu', or 'cpu'")


class Agent:
    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        *,
        dtype="auto",
        revision=None,
        batch_size=16,
        compile=False,
        pad_to_multiple=None,
        cache_prompts=False,
        quantize=None,
        result_cache_size=0,
        backend="torch",
        ort_path=None,
    ):
        if dtype != "auto" and dtype not in DTYPES:
            raise ValueError(f"dtype must be 'auto' or one of {list(DTYPES)}")
        if quantize not in (None, "int8"):
            raise ValueError("quantize must be None or 'int8'")
        if quantize == "int8" and dtype not in ("auto", "float32"):
            raise ValueError("quantize='int8' requires dtype='float32'")
        if backend not in ("torch", "ort"):
            raise ValueError("backend must be 'torch' or 'ort'")
        if backend == "ort" and (quantize or compile):
            raise ValueError("backend='ort' cannot be combined with quantize/compile")
        if device not in (None, "auto", "gpu", "cuda", "metal", "cpu"):
            raise ValueError("device must be None/'auto', 'cuda'/'gpu', or 'cpu'")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = _resolve_device(device)
        if dtype == "auto":
            dtype = "float16" if self.device.type == "cuda" and not quantize else "float32"
        self.torch_dtype = DTYPES[dtype]
        self.dtype_name = dtype
        self.batch_size = batch_size
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.pad_to_multiple = pad_to_multiple
        self._prefix_cache = PrefixCache() if cache_prompts else None
        self.model_id = str(model_id_or_path)
        self.revision = revision
        self.model_dir = resolve_model(
            model_id_or_path, token=token, subfolder=subfolder, revision=revision
        )
        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text())
        self.encoder_cfg = json.loads((self.model_dir / "encoder/config.json").read_text())
        if "encoder" not in self.cfg or "head_layers" not in self.cfg:
            raise ValueError("Laya config must specify encoder and head_layers")
        enc_cfg = EncoderConfig(self.encoder_cfg)
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        if not 4 < head_max_len < max_len <= enc_cfg.max_position_embeddings:
            raise ValueError("Expected 4 < head_max_len < max_len <= max_position_embeddings")
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        if len(self.temperature_raw) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature_raw, *self.temperature_by_options_raw.values()]
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.temperature_by_options_raw.items()
        }
        rejected = [
            "%s=%.4g" % (k, float(v))
            for k, v in self.temperature_by_options_raw.items()
            if clamp_temperature(v) != float(v)
        ]
        rejected += [
            "temperature[%d]=%.4g" % (i, float(t))
            for i, t in enumerate(self.temperature_raw)
            if clamp_temperature(t) != float(t)
        ]
        if rejected:
            warnings.warn(
                "vybir: this checkpoint ships temperatures outside [%g, %g] which would "
                "distort confidence; clamping %s. Treat confidence from the affected buckets "
                "as uncalibrated." % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning,
                stacklevel=2,
            )
        self.tok = Tokenizer(self.model_dir / "tokenizer")
        self.backend = backend
        if backend == "torch":
            model = DecisionModelTorch(enc_cfg, self.cfg)
            weights = load_safetensors(str(self.model_dir / "model.safetensors"), device="cpu")
            if any(".layers." in k and k.startswith(("scorer.", "act_head.")) for k in weights):
                weights = remap_mlx_to_torch(weights)  # pre-converted aac6fef/*-mlx exports
            try:
                model.load_state_dict(weights, strict=True)
            except RuntimeError as error:  # missing, unexpected or misshaped tensors
                message = str(error)
            else:
                message = None
            del weights  # drop the mmap now: Windows keeps the file locked while it lives
            if message:
                raise ValueError(f"Checkpoint does not match its config: {message}")
            model.to(self.torch_dtype)
            model.to(self.device)
            model.eval()
            if quantize == "int8":
                if self.device.type != "cpu":
                    raise ValueError("quantize='int8' is CPU-only; use device='cpu'")
                model = quantize_int8(model)
            self.quantize = quantize
            self.model = torch.compile(model) if compile else model
            self._compiled = bool(compile)
            self._ort = None
        else:
            if not ort_path:
                raise ValueError("backend='ort' requires ort_path='.../laya.onnx'")
            try:
                import onnxruntime as ort
            except ImportError as e:
                raise RuntimeError("backend='ort' needs: pip install 'vybir[ort]'") from e
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = torch.get_num_threads()
            gpu = ("CUDAExecutionProvider", "DmlExecutionProvider")
            providers = [] if device == "cpu" else [
                p for p in gpu if p in ort.get_available_providers()
            ]
            if device in ("cuda", "gpu") and not providers:
                raise ValueError(
                    "backend='ort' on GPU needs onnxruntime-gpu or onnxruntime-directml"
                )
            self._ort = ort.InferenceSession(
                str(ort_path), sess_options=opts, providers=providers + ["CPUExecutionProvider"]
            )
            self.ort_provider = self._ort.get_providers()[0]
            # The exported graph is float32; report what actually runs.
            self.torch_dtype, self.dtype_name = torch.float32, "float32"
            self.quantize = None
            self.model = None
            self._compiled = False
        self._result_cache = (
            ResultCache(result_cache_size) if result_cache_size else None
        )

    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(self, state, questions):
        """Construct upstream-compatible CPU inputs, useful for parity and profiling."""
        if self._prefix_cache is not None:
            return self._prefix_cache.prepare(self, state, questions)
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def forward(self, batch):
        """Run one prepared batch; returns CPU torch tensors (logits, act)."""
        if self.backend == "ort":
            feed = {
                k: (v.numpy().astype(bool) if k == "marker_mask" else v.numpy())
                for k, v in batch.items()
            }
            logits, act = self._ort.run(None, feed)
            return torch.from_numpy(logits), torch.from_numpy(act)
        with torch.inference_mode():
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits, act = self.model(**batch)
        return logits.cpu(), act.cpu()

    def system_one(self, state, questions):
        cache_key = None
        if self._result_cache is not None:
            cache_key = ResultCache.key(state, questions)
            hit = self._result_cache.get(cache_key)
            if hit is not None:
                return hit
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start: start + self.batch_size]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
            )
            logits, act = self.forward(batch)
            logits, act = logits.numpy(), act.numpy()
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            act /= act.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid, q = question_ids[start + row], internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                z = logits[row, :k] / scale
                p = np.exp(z - z.max())
                p /= p.sum()
                answer = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        result = {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }
        if self._result_cache is not None:
            self._result_cache.put(cache_key, result)
        return result

    predict = system_one

    @property
    def cache_stats(self):
        """Hit/miss stats of the exact-match result cache (None when disabled)."""
        return self._result_cache.stats if self._result_cache is not None else None


RLAgent = Agent


def load(
    model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None, **kwargs
):
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, **kwargs)
