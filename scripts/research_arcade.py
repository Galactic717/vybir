"""Score the shipped arcade brains per generation on fresh games; plot learning curves.

    python scripts/research_arcade.py              -> research/arcade_eval.json + figure
    python scripts/research_arcade.py --plot-only  -> docs/assets/arcade_learning.png
"""

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vybir.arcade.learn import GAMES, load_brain, new_game, step  # noqa: E402

GENERATIONS = (0, 1, 2, 3, 5, 8, 12, 16, 20)
GAMES_PER_POINT = 8
CAPS = {"tetris": 250_000, "snake": 100_000, "2048": 100_000}  # none is reached
OUT = ROOT / "research" / "arcade_eval.json"


def run(job):
    name, weights, seed = job
    game = new_game(name, seed)
    started = time.perf_counter()
    while not game.over and game.steps < CAPS[name]:
        step(game, np.array(weights))
    extra = {"max_tile": int(2 ** game.board.max())} if name == "2048" else {}
    return {"score": int(game.score), "steps": game.steps, "capped": not game.over,
            "seconds": time.perf_counter() - started, **extra}


def main():
    out = {}
    with Pool(16) as pool:
        for name in GAMES:
            _, _, brain = load_brain(name)
            rows = []
            for gen in GENERATIONS:
                weights = brain["history"][gen]["weights"]
                games = pool.map(run, [(name, weights, 777_000 + k) for k in range(GAMES_PER_POINT)])
                scores = [g["score"] for g in games]
                row = {"generation": gen, "scores": scores, "mean": float(np.mean(scores)),
                       "median": float(np.median(scores)), "capped": sum(g["capped"] for g in games)}
                if name == "2048":
                    row["max_tiles"] = [g["max_tile"] for g in games]
                rows.append(row)
                print(name, json.dumps(row), flush=True)
            out[name] = {"training_seconds": brain["seconds"], "rows": rows}
    OUT.write_text(json.dumps(out, indent=1))
    plot()


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(OUT.read_text())
    labels = {"tetris": "Tetris: lines cleared", "snake": "Snake (12x12): food eaten",
              "2048": "2048: game score"}
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.5))
    for ax, (name, res) in zip(np.atleast_1d(axes), data.items()):
        rows = res["rows"]
        gens = [r["generation"] for r in rows]
        for r in rows:
            ax.scatter([r["generation"]] * len(r["scores"]), np.maximum(r["scores"], 0.5), s=10,
                       color="#1f9d74", alpha=0.35)
        ax.plot(gens, [max(r["mean"], 0.5) for r in rows], marker="o", color="#1f9d74")
        if name != "snake":
            ax.set_yscale("log")
        ax.set_xticks(gens)
        ax.set_title(f"{labels[name]}\nlearned in {res['training_seconds']:.0f} s", fontsize=10)
        ax.set_xlabel("generation")
        ax.grid(alpha=0.3, which="both")
    np.atleast_1d(axes)[0].set_ylabel("fresh games (dots) and their mean")
    fig.suptitle("One learner, three games, zero prior knowledge: weights start at 0, "
                 "feedback = the game's score (laptop CPU)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "assets" / "arcade_learning.png", dpi=140)


if __name__ == "__main__":
    plot() if "--plot-only" in sys.argv else main()
