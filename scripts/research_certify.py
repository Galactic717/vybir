"""Research: certified local decisions with Laya (vybir.certify).

    python scripts/research_certify.py extract   # one forward pass per example -> research/cache
    python scripts/research_certify.py run       # experiments -> research/results.json, .png

Protocol (fixed before any result was seen): the three questions below were written once and
never tuned. Per task and seed: 40% stratified test split, 200 random calibration examples,
the rest is the pool the n labelled training examples are drawn from (at least 2 per class).
20 seeds. TF-IDF + logistic regression is only a yardstick for comparison; it is not part of
vybir.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "research" / "cache"
SEEDS = range(20)
NS = (8, 16, 32, 64, 128, 256)
ALPHAS = (0.01, 0.05, 0.10)
N_CAL = 200

TASKS = {
    "sms_scam": {
        "question": {
            "type": "choice",
            "instructions": "Is this text message legitimate, or is it spam or a scam?",
            "criteria": {
                "legitimate": "a normal personal, work or service message",
                "spam": "unsolicited advertising, a prize or lottery claim, phishing or a scam",
            },
        },
    },
    "prompt_injection": {
        "question": {
            "type": "choice",
            "instructions": "Is this input to an AI assistant a normal request, "
            "or a prompt injection attack?",
            "criteria": {
                "normal": "an ordinary question or task",
                "injection": "tries to override, ignore or reveal the assistant's instructions, "
                "or to make it break its rules",
            },
        },
    },
    "jailbreak": {
        "question": {
            "type": "choice",
            "instructions": "Is this prompt a benign request, or a jailbreak attempt?",
            "criteria": {
                "benign": "an ordinary request or role-play that respects the AI's rules",
                "jailbreak": "tries to make the AI ignore its rules, safety policies or limits",
            },
        },
    },
}


def load_task(name):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def get(repo, file):
        return hf_hub_download(repo, file, repo_type="dataset")

    if name == "sms_scam":
        df = pd.read_parquet(get("ucirvine/sms_spam", "plain_text/train-00000-of-00001.parquet"))
        return df["sms"].str.strip().tolist(), df["label"].to_numpy()
    if name == "prompt_injection":
        files = ["data/train-00000-of-00001-9564e8b05b4757ab.parquet",
                 "data/test-00000-of-00001-701d16158af87368.parquet"]
        df = pd.concat([pd.read_parquet(get("deepset/prompt-injections", f)) for f in files])
        return df["text"].tolist(), df["label"].to_numpy()
    df = pd.read_csv(get("jackhhao/jailbreak-classification",
                         "balanced/jailbreak_dataset_full_balanced.csv"))
    return df["prompt"].tolist(), (df["type"] == "jailbreak").to_numpy().astype(int)


def extract_all():
    import torch

    import vybir
    from vybir.certify import extract

    agent = vybir.Agent("convaiinnovations/laya")
    device = torch.cuda.get_device_name(0) if agent.device.type == "cuda" else "cpu"
    CACHE.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, task in TASKS.items():
        texts, y = load_task(name)
        started = time.perf_counter()
        feats = extract(agent, texts, task["question"], batch_size=32)
        seconds = time.perf_counter() - started
        np.savez(CACHE / f"{name}.npz", logits=feats["logits"], markers=feats["markers"],
                 temperature=feats["temperature"], labels=np.array(feats["labels"]), y=y)
        report[name] = {"examples": len(y), "seconds": round(seconds, 1),
                        "per_second": round(len(y) / seconds, 1), "device": device,
                        "dtype": agent.dtype_name}
        print(name, report[name], flush=True)
    (CACHE / "extract.json").write_text(json.dumps(report, indent=2))


# ---------------------------------------------------------------------- evaluation


def auroc(score, y):
    """Mann-Whitney AUROC with tie correction."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def split(y, seed):
    rng = np.random.default_rng(seed)
    test = np.concatenate([
        rng.permutation(np.flatnonzero(y == c))[: int(round(0.4 * (y == c).sum()))]
        for c in np.unique(y)
    ])
    rest = rng.permutation(np.setdiff1d(np.arange(len(y)), test))
    return test, rest[:N_CAL], rest[N_CAL:]


def draw(pool, y, n, seed):
    rng = np.random.default_rng(seed + 1000)
    picked = [rng.choice(np.intersect1d(pool, np.flatnonzero(y == c)), 2, replace=False)
              for c in np.unique(y)]
    picked = np.concatenate(picked)
    rest = rng.permutation(np.setdiff1d(pool, picked))
    return np.concatenate([picked, rest[: n - len(picked)]])


def score_model(p_test, y_test, p_cal, y_cal):
    from vybir.certify import Certifier

    label = p_test.argmax(1)
    out = {
        "accuracy": float((label == y_test).mean()),
        "balanced_accuracy": float(np.mean([(label[y_test == c] == c).mean()
                                            for c in np.unique(y_test)])),
        "auroc": auroc(p_test[:, 1], y_test),
    }
    runs = [(a, None, False, f"a{a}") for a in ALPHAS]
    runs += [(0.05, 0.1, False, "pac0.05"), (0.05, None, True, "perclass0.05")]
    for alpha, delta, per_class, key in runs:
        cert = Certifier()
        cert.probabilities = lambda feats: feats["p"]  # conformal layer over any probabilities
        cert.calibrate({"p": p_cal, "labels": None}, y_cal, alpha, delta=delta,
                       per_class=per_class)
        pred = cert.predict({"p": p_test, "labels": None})
        wrong = pred["label"] != y_test
        auto = pred["auto"]
        out[key] = {
            "coverage": float(pred["sets"][np.arange(len(y_test)), y_test].mean()),
            "auto_rate": float(auto.mean()),
            "joint_error": float((auto & wrong).mean()),  # guaranteed <= alpha in expectation
            "error_among_auto": float((auto & wrong).sum() / max(auto.sum(), 1)),
            # the harmful miss: a positive (spam / attack) decided automatically as negative
            "positive_slip": float((auto & wrong)[y_test == 1].mean()),
        }
    for tau in (0.9, 0.99):  # common practice: trust the model when it says >= tau
        auto = p_test.max(1) >= tau
        out[f"naive{tau}"] = {"auto_rate": float(auto.mean()),
                              "joint_error": float((auto & (label != y_test)).mean()),
                              "positive_slip": float((auto & (label != y_test))[y_test == 1].mean())}
    return out


def tfidf_probs(texts, y, train, others):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    x = vec.fit_transform([texts[i] for i in train])
    model = LogisticRegression(max_iter=2000).fit(x, y[train])
    return [model.predict_proba(vec.transform([texts[i] for i in idx])) for idx in others]


_TASK_DATA = {}


def _task_data(name):
    if name not in _TASK_DATA:
        data = np.load(CACHE / f"{name}.npz")
        feats = {"logits": data["logits"], "markers": data["markers"],
                 "temperature": float(data["temperature"]), "labels": list(data["labels"])}
        texts, y = load_task(name)
        assert (y == data["y"]).all()
        _TASK_DATA[name] = feats, texts, y
    return _TASK_DATA[name]


def run_seed(job):
    import torch

    from vybir.certify import Certifier, take

    torch.set_num_threads(1)  # one process per core
    name, seed = job
    feats, texts, y = _task_data(name)
    test, cal, pool = split(y, seed)
    zero = Certifier()
    rows = [{"seed": seed, "method": "laya-zero-shot", "n": 0, **score_model(
        zero.probabilities(take(feats, test)), y[test],
        zero.probabilities(take(feats, cal)), y[cal])}]
    for n in (*[n for n in NS if n <= len(pool)], len(pool)):
        train = draw(pool, y, n, seed) if n < len(pool) else pool
        started = time.perf_counter()
        cert = Certifier().fit(take(feats, train), y[train], seed=seed)
        fit_seconds = time.perf_counter() - started
        rows.append({"seed": seed, "method": "vybir-adapter", "n": int(n),
                     "fit_seconds": fit_seconds, "lambda": str(cert.lam), **score_model(
                         cert.probabilities(take(feats, test)), y[test],
                         cert.probabilities(take(feats, cal)), y[cal])})
        p_test, p_cal = tfidf_probs(texts, y, train, [test, cal])
        rows.append({"seed": seed, "method": "tfidf-yardstick", "n": int(n),
                     **score_model(p_test, y[test], p_cal, y[cal])})
    return name, rows


def run_all():
    from multiprocessing import Pool

    jobs = [(name, seed) for name in TASKS for seed in SEEDS]
    results = {name: {"rows": []} for name in TASKS}
    with Pool(16) as pool:
        for name, rows in pool.imap_unordered(run_seed, jobs):
            results[name]["rows"] += rows
            print(name, "seed", rows[0]["seed"], "done", flush=True)
    for name in TASKS:
        _, _, y = _task_data(name)
        results[name].update(examples=len(y), positives=int(y.sum()))
        results[name]["rows"].sort(key=lambda r: r["seed"])
    out = ROOT / "research" / "results.json"
    out.write_text(json.dumps(results))
    summarize(results)


def mean_std(rows, key, sub=None):
    values = [r[key][sub] if sub else r[key] for r in rows]
    return float(np.mean(values)), float(np.std(values))


def summarize(results):
    lines = []
    for name, res in results.items():
        rows = res["rows"]
        groups = {}
        for r in rows:
            n = "all" if r["method"] != "laya-zero-shot" and r["n"] > 256 else r["n"]
            groups.setdefault((r["method"], n), []).append(r)
        lines.append(f"\n### {name}  ({res['examples']} examples, {res['positives']} positive)\n")
        lines.append("| method | n labels | accuracy | balanced acc | AUROC | "
                     "auto @ a=0.05 | joint err @ a=0.05 | naive p>=0.99 auto | naive joint err |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for (method, n), g in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]).zfill(5))):
            acc, bal, auc = (mean_std(g, k) for k in ("accuracy", "balanced_accuracy", "auroc"))
            auto, joint = mean_std(g, "a0.05", "auto_rate"), mean_std(g, "a0.05", "joint_error")
            nauto, njoint = mean_std(g, "naive0.99", "auto_rate"), mean_std(g, "naive0.99", "joint_error")
            lines.append(
                f"| {method} | {n} | {acc[0]:.3f}±{acc[1]:.3f} | {bal[0]:.3f} | {auc[0]:.3f} | "
                f"{auto[0]:.1%} | {joint[0]:.2%} | {nauto[0]:.1%} | {njoint[0]:.2%} |")
    lines.append("\n## Guarantee check: P(automatic and wrong) must stay <= alpha\n")
    lines.append("| task | method | n | alpha | coverage | auto | joint err (mean ± s.e.) | "
                 "seeds over alpha |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, res in results.items():
        rows = res["rows"]
        full = max(r["n"] for r in rows)
        for method, n in (("laya-zero-shot", 0), ("vybir-adapter", 64), ("vybir-adapter", full),
                          ("tfidf-yardstick", full)):
            g = [r for r in rows if r["method"] == method and r["n"] == n]
            for key in [f"a{a}" for a in ALPHAS] + ["pac0.05", "perclass0.05"]:
                if key not in g[0]:
                    continue
                alpha = float(key.removeprefix("perclass").removeprefix("pac").removeprefix("a"))
                label = (f"{alpha} (PAC, delta=0.1)" if key.startswith("pac")
                         else f"{alpha} (per class)" if key.startswith("perclass") else alpha)
                joint = np.array([r[key]["joint_error"] for r in g])
                cover = np.mean([r[key]["coverage"] for r in g])
                auto = np.mean([r[key]["auto_rate"] for r in g])
                lines.append(
                    f"| {name} | {method} | {n} | {label} | {cover:.3f} | {auto:.1%} | "
                    f"{joint.mean():.2%} ± {joint.std(ddof=1) / np.sqrt(len(joint)):.2%} | "
                    f"{(joint > alpha).sum()}/{len(joint)} |")
    text = "\n".join(lines)
    (ROOT / "research" / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    plot(results)


def plot(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.6), sharey=False)
    for ax, (name, res) in zip(np.atleast_1d(axes), results.items()):
        rows = res["rows"]
        zero = np.mean([r["balanced_accuracy"] for r in rows if r["method"] == "laya-zero-shot"])
        ax.axhline(zero, color="#888", ls="--", label="Laya zero-shot")
        for method, color in (("vybir-adapter", "#1f9d74"), ("tfidf-yardstick", "#c2703d")):
            ns = sorted({r["n"] for r in rows if r["method"] == method})[:-1]
            means = [np.mean([r["balanced_accuracy"] for r in rows
                              if r["method"] == method and r["n"] == n]) for n in ns]
            ax.plot(ns, means, marker="o", color=color, label=method)
        ax.set_xscale("log", base=2)
        ax.set_title(name)
        ax.set_xlabel("labelled examples")
        ax.set_ylabel("balanced accuracy")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "research" / "fewshot.png", dpi=130)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    {"extract": extract_all, "run": run_all}[sys.argv[1]]()
