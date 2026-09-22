"""One learner for several games. It starts knowing nothing and learns from the score.

Every game exposes the same thing: for each move it can make now, a row of measurements
of the position that move leads to. The brain is one weight per measurement and plays the
move with the highest weighted sum. All weights start at zero. The cross-entropy method
(a population of brains, the best-scoring ones become the parents of the next generation)
learns them from the game's own score and nothing else.

    vybir arcade watch tetris              # the shipped brain, live in the terminal
    vybir arcade watch snake --generation 0   # the same brain before learning
    vybir arcade train 2048                # learn your own from zero
    vybir arcade record snake --out snake.gif
"""

import argparse
import importlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

GAMES = {"tetris": "tetris", "snake": "snake", "2048": "g2048"}
# Per game: training budget (max steps per game, games per brain) and the step cap used to
# score the shipped brains, chosen so that none of the evaluation games hit it.
SETTINGS = {
    "tetris": {"cap": 5000, "games": 1, "generations": 20},
    "snake": {"cap": 3000, "games": 2, "generations": 20},
    "2048": {"cap": 20000, "games": 3, "generations": 20},
}
BRAINS = Path(__file__).with_name("brains")


def game_module(name):
    if name not in GAMES:
        raise ValueError(f"unknown game {name!r}; choose from {sorted(GAMES)}")
    return importlib.import_module(f"vybir.arcade.{GAMES[name]}")


def new_game(name, seed):
    game = game_module(name).Game(seed)
    game.ties = np.random.default_rng(seed + 7919)  # separate stream: game dice stay the same
    return game


def step(game, weights):
    feats = game.options()
    if feats is None or not len(feats):
        game.over = True
        return
    values = feats @ weights
    best = np.flatnonzero(values >= values.max() - 1e-12)
    # Ties are broken at random, so an all-zero brain really plays blind instead of
    # accidentally following "always take the first move".
    game.apply(int(best[game.ties.integers(len(best))]) if len(best) > 1 else int(best[0]))


def play(name, weights, seed, cap):
    game = new_game(name, seed)
    while not game.over and game.steps < cap:
        step(game, weights)
    return game.score, game.steps


def _fitness(job):
    name, weights, seeds, cap = job
    results = [play(name, weights, s, cap) for s in seeds]
    # score first; lasting longer only breaks ties between equal scores
    return float(np.mean([score + steps / 1e7 for score, steps in results]))


def train(name, generations=None, population=64, elite=10, cap=None, games=None, seed=0,
          workers=None, log=print):
    """Cross-entropy method from all-zero weights; returns the per-generation history."""
    settings = SETTINGS[name]
    generations = generations or settings["generations"]
    cap, games = cap or settings["cap"], games or settings["games"]
    size = len(game_module(name).FEATURES)
    rng = np.random.default_rng(seed)
    mean, var = np.zeros(size), np.full(size, 100.0)
    history = [{"generation": 0, "weights": mean.tolist()}]
    with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        for gen in range(1, generations + 1):
            started = time.perf_counter()
            pop = mean + np.sqrt(var) * rng.standard_normal((population, size))
            seeds = [seed * 100_000 + gen * 10 + k for k in range(games)]  # shared by all
            scores = np.array(pool.map(_fitness, [(name, w, seeds, cap) for w in pop]))
            top = pop[np.argsort(scores)[::-1][:elite]]
            mean = top.mean(0)
            var = top.var(0) + max(5.0 - gen / 10, 0.0)  # extra noise against early collapse
            test = pool.map(_fitness, [(name, mean, [10**9 + gen * 10 + k], cap) for k in range(4)])
            entry = {
                "generation": gen, "weights": mean.tolist(),
                "population_best": round(float(scores.max()), 1),
                "population_mean": round(float(scores.mean()), 1),
                "test_mean": round(float(np.mean(test)), 1),
                "seconds": round(time.perf_counter() - started, 1),
            }
            history.append(entry)
            log(json.dumps({k: v for k, v in entry.items() if k != "weights"}))
    return history


def load_brain(name, path=None, generation=None):
    data = json.loads(Path(path or BRAINS / f"{name}.json").read_text())
    entry = data["history"][-1 if generation is None else generation]
    return entry["generation"], np.array(entry["weights"]), data


def preferences(name, weights, count=5):
    features = game_module(name).FEATURES
    order = np.argsort(-np.abs(weights))[:count]
    return [(("avoid " if weights[i] < 0 else "seek  ") + f"{features[i]:<20}{weights[i]:+8.2f}",
             "#e05c5c" if weights[i] < 0 else "#5cd65c") for i in order if weights[i]]


# ----------------------------------------------------------------------------- showing it


def _font(size):
    from PIL import ImageFont

    for path in ("C:/Windows/Fonts/consola.ttf", "/System/Library/Fonts/Menlo.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def record(name, out, steps=300, seed=12345, fps=15, path=None):
    """GIF: the untrained brain (generation 0) and the learned one, same random seed."""
    from PIL import Image, ImageDraw

    module = game_module(name)
    sides = []
    for generation in (0, None):
        gen, weights, _ = load_brain(name, path, generation)
        sides.append((gen, weights, new_game(name, seed)))
    grid = sides[0][2].cells()
    cell, pad, top = module.Game.CELL, 24, 54
    width = 2 * grid.shape[1] * cell + 3 * pad
    height = top + grid.shape[0] * cell + 58
    font, small = _font(13), _font(max(10, cell // 4))
    frames = []
    for _ in range(steps):
        image = Image.new("RGB", (width, height), "#0d1117")
        draw = ImageDraw.Draw(image)
        for i, (gen, weights, game) in enumerate(sides):
            x0 = pad + i * (grid.shape[1] * cell + pad)
            cells = game.cells()
            draw.text((x0, 12), "untrained (generation 0)" if gen == 0
                      else f"after {gen} generations", fill="#e6edf3", font=font)
            draw.text((x0, 30), "all weights zero" if gen == 0 else "learned from score only",
                      fill="#8b949e", font=font)
            draw.rectangle([x0 - 2, top - 2, x0 + cells.shape[1] * cell + 1,
                            top + cells.shape[0] * cell + 1], outline="#30363d")
            for (r, c), value in np.ndenumerate(cells):
                if not value and not module.Game.NUMBERS:
                    continue
                x, y = x0 + c * cell, top + r * cell
                draw.rectangle([x + 1, y + 1, x + cell - 2, y + cell - 2],
                               fill=module.Game.PALETTE[min(value, len(module.Game.PALETTE) - 1)])
                if module.Game.NUMBERS and value:
                    draw.text((x + cell / 2, y + cell / 2), module.Game.label(value),
                              fill="#ffffff", font=small, anchor="mm")
            base = top + cells.shape[0] * cell
            draw.text((x0, base + 10), f"{module.Game.SCORE:<7}{game.score}", fill="#e6edf3",
                      font=font)
            draw.text((x0, base + 28), f"{module.Game.STEP:<7}{game.steps}", fill="#8b949e",
                      font=font)
            if game.over:
                y = top + cells.shape[0] * cell // 2
                draw.rectangle([x0 + 10, y - 8, x0 + cells.shape[1] * cell - 10, y + 22],
                               fill="#0d1117", outline="#e05c5c")
                draw.text((x0 + cells.shape[1] * cell / 2, y + 7), "GAME OVER", fill="#e05c5c",
                          font=font, anchor="mm")
            else:
                step(game, weights)
        frames.append(image)
    frames += [frames[-1]] * fps
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps),
                   loop=0, optimize=True)
    return out


def _render(name, game, info):
    from rich.text import Text

    module = game_module(name)
    cells = game.cells()
    side = [
        (f"VYBIR ARCADE · {module.Game.TITLE}", "bold #5fe0b0"),
        ("learns from the score only", "#8b949e"), ("", ""),
        (f"brain generation  {info['generation']}", "#e6edf3"),
        (f"{module.Game.SCORE:<18}{game.score}", "bold #e6edf3"),
        (f"{module.Game.STEP:<18}{game.steps}", "#e6edf3"),
        (f"speed             {info['speed']:.0f} moves/s", "#8b949e"), ("", ""),
        ("strongest learned preferences", "#8b949e"), *info["prefs"], ("", ""),
        ("SPACE pause  +/- speed  Q quit", "#8b949e"),
    ]
    text = Text()
    rows = max(cells.shape[0], len(side))
    for r in range(rows):
        text.append("  ")
        if r < cells.shape[0]:
            text.append("│", "#30363d")
            for value in cells[r]:
                color = module.Game.PALETTE[min(value, len(module.Game.PALETTE) - 1)]
                if module.Game.NUMBERS:
                    text.append(f"{module.Game.label(value):>5} ", f"bold #ffffff on {color}")
                else:
                    text.append("██" if value else " ·", color if value else "#30363d")
            text.append("│", "#30363d")
        else:
            text.append(" " * (cells.shape[1] * (6 if module.Game.NUMBERS else 2) + 2))
        text.append("    ")
        if r < len(side):
            text.append(*side[r])
        text.append("\n")
    if game.over:
        text.append("  GAME OVER - R restarts", "bold #e05c5c")
    return text


def watch(name, generation=None, seed=12345, fps=20, path=None):
    from rich.console import Console
    from rich.live import Live

    from ..snake.cli import Keyboard

    gen, weights, _ = load_brain(name, path, generation)
    info = {"generation": gen, "speed": 0.0, "prefs": preferences(name, weights)}
    game, paused, shown = new_game(name, seed), False, []
    with Keyboard() as keys, Live(console=Console(), auto_refresh=False, screen=True) as live:
        while True:
            pressed = keys.read().lower()
            if "q" in pressed or "\x03" in pressed:
                break
            if "r" in pressed:
                seed += 1
                game = new_game(name, seed)
            paused ^= " " in pressed
            fps = min(1000, fps * 2) if "+" in pressed else max(1, fps / 2) if "-" in pressed else fps
            if not paused and not game.over:
                step(game, weights)
                shown = (shown + [time.perf_counter()])[-30:]
                if len(shown) > 1:
                    info["speed"] = (len(shown) - 1) / (shown[-1] - shown[0])
            live.update(_render(name, game, info), refresh=True)
            time.sleep(1 / fps if not game.over else 0.1)
    print(json.dumps({"game": name, "generation": gen, "score": game.score,
                      "steps": game.steps, "game_over": game.over}))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vybir arcade", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("train", help="Learn a brain from zero")
    sub.add_argument("game", choices=sorted(GAMES))
    sub.add_argument("--generations", type=int)
    sub.add_argument("--population", type=int, default=64)
    sub.add_argument("--cap", type=int, help="Max steps per training game")
    sub.add_argument("--games", type=int, help="Games per brain per generation")
    sub.add_argument("--seed", type=int, default=0)
    sub.add_argument("--out", type=Path, help="Default: models/<game>_brain.json")
    for command, help_text in (("watch", "Watch a brain play live"),
                               ("record", "GIF: untrained vs trained, same seed")):
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument("game", choices=sorted(GAMES))
        sub.add_argument("--brain", type=Path, help="Default: the shipped brain")
        sub.add_argument("--seed", type=int, default=12345)
        if command == "watch":
            sub.add_argument("--generation", type=int, help="0 = before any learning")
            sub.add_argument("--fps", type=float, default=20)
        else:
            sub.add_argument("--out", type=Path, required=True)
            sub.add_argument("--steps", type=int, default=300)
    args = parser.parse_args(argv)
    if args.command == "train":
        started = time.perf_counter()
        history = train(args.game, args.generations, args.population, cap=args.cap,
                        games=args.games, seed=args.seed)
        out = args.out or Path("models") / f"{args.game}_brain.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "game": args.game, "features": list(game_module(args.game).FEATURES),
            "history": history, "seconds": round(time.perf_counter() - started, 1),
            "method": "cross-entropy method from zero weights; fitness = the game's score",
        }, indent=1))
        print(f"saved {out}")
    elif args.command == "record":
        print(record(args.game, args.out, args.steps, args.seed, path=args.brain))
    else:
        watch(args.game, args.generation, args.seed, args.fps, path=args.brain)
    return 0


if __name__ == "__main__":
    sys.exit(main())
