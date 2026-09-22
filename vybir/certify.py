"""Certified decisions: few-shot adapters + split-conformal guarantees on top of Laya.

    feats = extract(agent, texts, question)             # one forward pass per text
    cert = Certifier().fit(take(feats, tr), y[tr])       # a few labels, about a second on CPU
    cert.calibrate(take(feats, cal), y[cal], alpha=0.05) # held-out labels
    out = cert.predict(take(feats, new))                 # probs, label, set, auto

Guarantee (split conformal; calibration and new data exchangeable): the true label is
inside the prediction set with probability >= 1 - alpha. The top label is acted on
automatically when the set holds at most one label: a wrong automatic decision then means
the true label is outside the set, so P(decided automatically AND wrong) <= alpha.
Sets with two or more labels are deferred (to a human, or to a bigger model).
``fit`` is optional: without it the Certifier uses Laya's own zero-shot probabilities.
"""

import json
import math

import numpy as np
import torch

from .agent import collate_items
from .common import QTYPES, render_options, temp_bucket


def extract(agent, states, question, *, batch_size=None):
    """Zero-shot logits plus each option's marker state (what Laya's scorer reads)."""
    if agent.backend != "torch":
        raise ValueError("extract needs backend='torch' (the ONNX graph exposes no hidden states)")
    items, q = [], None
    for state in states:
        prepared, internal = agent.prepare(state, {"q": question})
        items.append(prepared[0])
        q = internal[0]
    if q is None:
        raise ValueError("extract needs at least one state")
    k = len(items[0]["markers"])
    labels = list(q["crit"]) if q["t"] == "choice" else [str(i) for i in range(k)]
    qtype = QTYPES[q["t"]]
    # Shortest-first batches waste far fewer padding tokens on real, mixed-length text.
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    logits = markers = None
    size = batch_size or agent.batch_size
    for start in range(0, len(order), size):
        rows = order[start:start + size]
        batch = collate_items(
            [items[i] for i in rows], agent.tok.pad_token_id,
            pad_to_multiple=agent.pad_to_multiple, max_length=agent.cfg.get("max_len", 512),
        )
        with torch.inference_mode():
            batch = {name: value.to(agent.device) for name, value in batch.items()}
            lg, _, m, _ = agent.model(**batch, return_hidden=True)
        if logits is None:
            logits = np.empty((len(items), k), np.float32)
            markers = np.empty((len(items), k, m.shape[-1]), np.float32)
        logits[rows] = lg[:, :k].cpu().numpy()
        markers[rows] = m[:, :k].cpu().numpy()
    temperature = agent.temperature_by_options.get(temp_bucket(qtype, k), agent.temperature[qtype])
    if len(render_options(q)) != k:
        raise ValueError("question options do not match the markers")
    return {"logits": logits, "markers": markers,
            "temperature": float(temperature), "labels": labels}


def take(feats, index):
    """Row subset of an ``extract`` result."""
    index = np.asarray(index)
    return {k: (v[index] if isinstance(v, np.ndarray) else v) for k, v in feats.items()}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_residual(markers, prior, y, lam, iters=100):
    """Minimise cross-entropy of prior + markers.w + b over options, plus lam*|w|^2.

    ``lam=inf`` keeps w = 0: only the option biases move, so Laya's ranking is untouched.
    """
    m = torch.as_tensor(markers, dtype=torch.float64)
    z0 = torch.as_tensor(prior, dtype=torch.float64)
    target = torch.as_tensor(y, dtype=torch.long)
    w = torch.zeros(m.shape[-1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(m.shape[1], dtype=torch.float64, requires_grad=True)
    bias_only = np.isinf(lam)
    opt = torch.optim.LBFGS([b] if bias_only else [w, b], max_iter=iters,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = z0 + m @ w + b
        loss = torch.nn.functional.cross_entropy(z, target) + 1e-4 * (b @ b)
        if not bias_only:
            loss = loss + lam * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), b.detach().numpy()


def _pac_rank(n, alpha, delta):
    """Smallest rank r with P(coverage < 1 - alpha) <= delta.

    With the r-th smallest of n calibration scores as threshold, coverage follows
    Beta(r, n + 1 - r), and P(Beta(r, n + 1 - r) < x) = P(Binomial(n, x) >= r).
    Returns n + 1 (never decide) when n is too small for (alpha, delta).
    """
    x = 1.0 - alpha
    log_pmf = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
               + k * math.log(x) + (n - k) * math.log1p(-x) for k in range(n + 1)]
    tail = 0.0  # P(Binomial >= r), accumulated from the top
    for r in range(n, 0, -1):
        tail += math.exp(log_pmf[r])
        if tail > delta:
            return r + 1
    return 1


def _log_loss(markers, prior, y, w, b):
    z = prior + markers @ w + b
    z = z - z.max(1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y]))


class Certifier:
    """Zero-shot or few-shot probabilities, then conformal prediction sets.

    The few-shot part is a residual marker adapter: every option's score is Laya's own
    calibrated logit plus ``w . marker_state + b_option``. ``w`` is shared by all options
    (one vector of the encoder width), so the adapter has D + K parameters whatever the
    option count, starts exactly at zero-shot Laya (w = b = 0), and the L2 strength is
    chosen by cross-validation on the labelled examples only. The candidates include
    ``inf`` (bias only), so with too few labels the adapter can keep Laya's ranking exactly
    and just move the decision threshold.
    """

    LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, np.inf)

    def __init__(self):
        self.params = None  # (mean, scale, w, b) once fitted
        self.lam = None
        self.qhat = None
        self.alpha = None
        self.labels = None
        self.question = None  # the question definition, kept with a saved certifier

    def fit(self, feats, y, *, lam=None, seed=0):
        y = np.asarray(y)
        k = feats["logits"].shape[1]
        if len(np.unique(y)) < 2 or y.min() < 0 or y.max() >= k:
            raise ValueError("fit needs labels 0..K-1 covering at least two options")
        flat = feats["markers"].reshape(-1, feats["markers"].shape[-1]).astype(np.float64)
        mean, scale = flat.mean(0), flat.std(0) + 1e-6
        markers = (feats["markers"] - mean) / scale
        prior = feats["logits"].astype(np.float64) / feats["temperature"]
        if lam is None:
            lam = self._choose_lambda(markers, prior, y, seed)
        w, b = _fit_residual(markers, prior, y, lam)
        self.params, self.lam, self.labels = (mean, scale, w, b), lam, feats["labels"]
        return self

    def _choose_lambda(self, markers, prior, y, seed):
        counts = np.bincount(y)
        folds = min(5, counts[counts > 0].min())
        if folds < 2:
            return 1.0
        order = np.random.default_rng(seed).permutation(len(y))
        fold_of = np.empty(len(y), int)
        fold_of[order] = np.arange(len(y)) % folds
        best = None
        for lam in self.LAMBDAS:
            losses = []
            for f in range(folds):
                tr, va = fold_of != f, fold_of == f
                if len(np.unique(y[tr])) < 2:
                    continue
                w, b = _fit_residual(markers[tr], prior[tr], y[tr], lam)
                losses.append(_log_loss(markers[va], prior[va], y[va], w, b))
            if losses and (best is None or np.mean(losses) < best[0]):
                best = (np.mean(losses), lam)
        return best[1] if best else 1.0

    def probabilities(self, feats):
        prior = feats["logits"].astype(np.float64) / feats["temperature"]
        if self.params is None:  # zero-shot: Laya's calibrated probabilities
            return _softmax(prior)
        mean, scale, w, b = self.params
        return _softmax(prior + ((feats["markers"] - mean) / scale) @ w + b)

    def calibrate(self, feats, y, alpha=0.05, delta=None, per_class=False):
        """Split-conformal threshold(s) from held-out labelled examples.

        ``delta=None``: coverage >= 1 - alpha on average over calibration draws.
        ``delta=0.1``: coverage >= 1 - alpha for this very calibration set, with
        probability >= 1 - delta (training-conditional / PAC; needs more examples).
        ``per_class=True`` (Mondrian): one threshold per true label, so the promise holds
        for every label separately, e.g. P(a scam is waved through automatically) <= alpha,
        not just alpha of all traffic. Needs calibration examples of every label.
        """
        if not 0 < alpha < 1 or (delta is not None and not 0 < delta < 1):
            raise ValueError("alpha and delta must be in (0, 1)")
        y = np.asarray(y)
        probs = self.probabilities(feats)
        scores = 1.0 - probs[np.arange(len(y)), y]

        def threshold(s):
            n = len(s)
            if n == 0:
                return np.inf
            rank = math.ceil((n + 1) * (1 - alpha)) if delta is None else _pac_rank(n, alpha, delta)
            # Too few points for this alpha: that label is always kept, so never decided alone.
            return np.inf if rank > n else float(np.sort(s)[rank - 1])

        if per_class:
            self.qhat = np.array([threshold(scores[y == k]) for k in range(probs.shape[1])])
        else:
            self.qhat = threshold(scores)
        self.alpha = alpha
        self.labels = self.labels or feats["labels"]
        return self

    def predict(self, feats):
        p = self.probabilities(feats)
        out = {"probabilities": p, "label": p.argmax(1)}
        if self.qhat is not None:
            sets = (1.0 - p) <= self.qhat
            single = sets.sum(1) == 1
            # A one-label set is acted on as that label (with per-class thresholds it need not
            # be the top probability); an empty set falls back to the top label. Either way a
            # wrong automatic decision implies the true label was outside the set.
            out["label"] = np.where(single, sets.argmax(1), out["label"])
            out["sets"] = sets
            out["auto"] = sets.sum(1) <= 1
        return out

    def save(self, path):
        mean, scale, w, b = self.params or (np.zeros(0),) * 4
        np.savez(path, mean=mean, scale=scale, w=w, b=b,
                 lam=np.nan if self.lam is None else self.lam,
                 qhat=np.nan if self.qhat is None else self.qhat,
                 alpha=np.nan if self.alpha is None else self.alpha,
                 labels=np.array(self.labels or [], dtype=str), fitted=self.params is not None,
                 question=json.dumps(self.question))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        cert = cls()
        if bool(data["fitted"]):
            cert.params = (data["mean"], data["scale"], data["w"], data["b"])
            cert.lam = float(data["lam"])
        qhat = data["qhat"]
        if qhat.ndim or not np.isnan(qhat):  # scalar, or one threshold per label
            cert.qhat = qhat if qhat.ndim else float(qhat)
            cert.alpha = float(data["alpha"])
        cert.labels = [str(label) for label in data["labels"]]
        cert.question = json.loads(str(data["question"])) if "question" in data else None
        return cert
