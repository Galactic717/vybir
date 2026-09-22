"""README figures from recorded results (no model needed).

    python scripts/figures.py   -> docs/assets/speed.png, docs/assets/certified.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
GREEN, ORANGE, GREY, RED = "#1f9d74", "#c2703d", "#8b949e", "#d64545"
TITLES = {"sms_scam": "SMS spam / scam", "prompt_injection": "Prompt injection",
          "jailbreak": "Jailbreak prompts"}


def speed():
    data = json.loads((ROOT / "research" / "benchmarks.json").read_text())
    rows = data["rows"]
    labels = [f"{r['backend']} · {r['device']}\n{r['dtype']}" for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    colors = [GREY, GREY, GREEN, GREEN]
    a.barh(labels, [r["p50_ms"] for r in rows], color=colors)
    a.set_xlabel("ms per question (P50, lower is better)")
    b.barh(labels, [r["batch_qps"] for r in rows], color=colors)
    b.set_xlabel("questions per second, batched (higher is better)")
    b.set_yticklabels([])
    for ax, key, fmt in ((a, "p50_ms", "{:.0f} ms"), (b, "batch_qps", "{:.1f}/s")):
        for i, r in enumerate(rows):
            ax.text(r[key], i, " " + fmt.format(r[key]), va="center", fontsize=9)
        ax.invert_yaxis()
        ax.margins(x=0.25)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Laya 421M end to end on one laptop (i7-12700H / RTX 3060 Laptop)")
    fig.tight_layout()
    fig.savefig(ASSETS / "speed.png", dpi=140)


def certified():
    results = json.loads((ROOT / "research" / "results.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
    for ax, (task, res) in zip(axes, results.items()):
        rows = res["rows"]

        def pick(method, n, key, sub):
            g = [r for r in rows if r["method"] == method and r["n"] == n]
            return (np.mean([r[key]["auto_rate"] for r in g]) * 100,
                    np.mean([r[key][sub] for r in g]) * 100)

        points = [
            ("naive p>=0.9, Laya", pick("laya-zero-shot", 0, "naive0.9", "joint_error"), RED, "x"),
            ("naive p>=0.9, 8 labels", pick("vybir-adapter", 8, "naive0.9", "joint_error"),
             RED, "^"),
            ("certified, Laya zero-shot", pick("laya-zero-shot", 0, "a0.05", "joint_error"),
             GREEN, "o"),
            ("certified, 64 labels", pick("vybir-adapter", 64, "a0.05", "joint_error"),
             GREEN, "s"),
            ("certified PAC, 64 labels", pick("vybir-adapter", 64, "pac0.05", "joint_error"),
             "#0b6e4f", "D"),
        ]
        for label, (auto, err), color, marker in points:
            ax.scatter(auto, err, color=color, marker=marker, s=60, label=label, zorder=3)
        ax.axhline(5, color=GREY, ls="--", lw=1)
        ax.text(1, 5.3, "promised limit: 5 %", color=GREY, fontsize=8)
        ax.set_title(TITLES.get(task, task))
        ax.set_xlabel("decided automatically (%)")
        ax.set_xlim(0, 105)
        ax.set_ylim(0, max(12, ax.get_ylim()[1]))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("automatic AND wrong (% of all inputs)")
    axes[-1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Conformal decisions land on the promised 5 % on average and PAC ones below it "
                 "with 90 % confidence;\nfixed confidence thresholds are either wasteful or over "
                 "the limit (mean of 20 random splits)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(ASSETS / "certified.png", dpi=140)


if __name__ == "__main__":
    speed()
    certified()
