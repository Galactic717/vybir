"""Real Laya predictions, with an explicit optional deterministic safety shield."""

import math
import os
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .game import DIRECTIONS

DEFAULT_MODEL = "convaiinnovations/laya"


def local_checkpoint(value=None):
    """Resolve a directory or an already cached Hub snapshot without network access."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    value = DEFAULT_MODEL if value is None else value
    path = Path(value).expanduser()
    if path.is_dir():
        return path
    if isinstance(value, Path) or path.is_absolute() or str(value).startswith((".", "~", "\\")):
        raise FileNotFoundError(f"Local checkpoint does not exist: {value}")
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(str(value), local_files_only=True))
    except Exception as error:
        raise FileNotFoundError(
            f"{value} is not cached. Download it before starting the offline demo:\n"
            f"  hf download {value}\n"
            "  vybir snake"
        ) from error


def hardware_name(device="cpu"):
    """Short human CPU/GPU name for the demo panel."""
    if device == "cuda":
        import torch

        return torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "")
    name = ""
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        elif platform.system() == "Darwin":
            name = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout
        else:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    name = line.split(":", 1)[1]
                    break
    except OSError:
        pass
    name = re.sub(r"\((R|TM)\)|\d+(st|nd|rd|th) Gen |CPU @.*| with .*", "", name)
    return " ".join(name.split()) or platform.machine()


@dataclass
class Decision:
    probabilities: dict
    proposed: str
    executed: str
    safe_directions: list
    intervened: bool
    dead_end_risk: float
    food_reachable: float
    inference_ms: float
    decision_ms: float
    input_tokens: int
    output_tokens: int
    safe_count: int
    planner_best: str

    def to_dict(self):
        return asdict(self)


class LayaPolicy:
    def __init__(
        self, model=None, *, subfolder=None, guarded=True, prompt="compact", optimize=False,
        backend="torch", ort_path=None, device=None,
    ):
        from vybir import Agent

        self.path = local_checkpoint(model)
        self.agent = Agent(
            self.path,
            subfolder=subfolder,
            device=device,
            batch_size=3,
            backend=backend,
            ort_path=ort_path,
            # Prefix reuse + 16-token buckets; torch.compile is left out because on
            # Windows it needs a C++ compiler (MSVC cl) on PATH.
            pad_to_multiple=16 if optimize else None,
            cache_prompts=optimize,
        )
        self.guarded = guarded
        if prompt not in ("compact", "detailed"):
            raise ValueError("prompt must be compact or detailed")
        self.prompt = prompt
        agent = self.agent
        device_name = "cuda" if agent.device.type == "cuda" and backend == "torch" else "cpu"
        self.engine = " · ".join(
            (backend.upper(), device_name.upper(), agent.dtype_name.replace("float", "FP"))
        )
        self.metadata = {
            "name": str(model or DEFAULT_MODEL) + (f"/{subfolder}" if subfolder else ""),
            "hardware": hardware_name(device_name),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "engine": self.engine,
            "network": "offline",
            "policy": "Laya probabilities over planner features; optional cycle safety shield",
            "prompt": prompt,
            "optimization": "prefix cache + 16-token buckets" if optimize else "eager",
        }

    def decide(self, game):
        started = time.perf_counter()
        moves = game.moves()
        safe = [m for m in moves if m.safe]
        if not safe and self.guarded:
            raise RuntimeError("Cycle safety invariant violated: no safe action")
        preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
        reachable, space = game.food_reachability()
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }
        if self.prompt == "compact":
            state = (
                f"Safe route: {'yes' if safe else 'no'}. "
                f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
            )
            questions["move"]["instructions"] = "Choose the best safe move toward food."
            questions["move"]["criteria"] = {
                m.direction: (
                    "Blocked. Collision."
                    if not m.legal
                    else "Unsafe. Traps the snake."
                    if not m.safe
                    else "Safe. Eat food now. Best."
                    if m.eats
                    else "Safe. Best route to food."
                    if m.direction == preferred
                    else "Safe. Slower route."
                )
                for m in moves
            }
            questions["risk"]["instructions"] = "Is a safe route available?"
            questions["food"]["instructions"] = "Is food reachable through empty cells?"
        inference_start = time.perf_counter()
        output = self.agent.predict(state, questions)
        inference_ms = (time.perf_counter() - inference_start) * 1000
        answers = output["answers"]
        probabilities = answers["move"]["probabilities"]
        scores = [*probabilities.values(), answers["risk"]["noul"], answers["food"]["noul"]]
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
            raise ValueError("Model returned an invalid probability; no move executed")
        proposed = max(DIRECTIONS, key=probabilities.__getitem__)
        allowed = [m.direction for m in safe]
        executed = (
            max(allowed, key=probabilities.__getitem__)
            if self.guarded and proposed not in allowed
            else proposed
        )
        return Decision(
            probabilities=probabilities,
            proposed=proposed,
            executed=executed,
            safe_directions=allowed,
            intervened=proposed != executed,
            dead_end_risk=1 - answers["risk"]["noul"],
            food_reachable=answers["food"]["noul"],
            inference_ms=inference_ms,
            decision_ms=(time.perf_counter() - started) * 1000,
            input_tokens=output["usage"]["input_tokens"],
            output_tokens=output["usage"].get("output_tokens", 0),
            safe_count=len(safe),
            planner_best=preferred,
        )
