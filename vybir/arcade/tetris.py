"""Tetris: every placement (rotation x column) of the current piece is simulated and the
resulting board is measured. The brain never sees a rule, only these measurements."""

import numpy as np

H, W = 20, 10
PIECES = "IOTSZJL"
SPAWN = {
    "I": [(0, 0), (0, 1), (0, 2), (0, 3)],
    "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
    "T": [(0, 0), (0, 1), (0, 2), (1, 1)],
    "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
    "Z": [(0, 0), (0, 1), (1, 1), (1, 2)],
    "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
    "L": [(0, 2), (1, 0), (1, 1), (1, 2)],
}
FEATURES = (
    [f"height_{c}" for c in range(W)] + [f"step_{c}" for c in range(W - 1)]
    + [f"holes_{c}" for c in range(W)]
    + ["max_height", "lines_cleared", "row_transitions", "column_transitions"]
)


def _rotations(cells):
    out, current = [], cells
    for _ in range(4):
        r0, c0 = min(r for r, _ in current), min(c for _, c in current)
        shape = tuple(sorted((r - r0, c - c0) for r, c in current))
        if shape not in out:
            out.append(shape)
        current = [(c, -r) for r, c in current]
    return out


def _placements(piece):
    """Per placement: cell rows/cols relative to the landing row, and a column profile."""
    rows, cols, prof_cols, prof_bottom = [], [], [], []
    for shape in _rotations(SPAWN[piece]):
        width = max(c for _, c in shape) + 1
        bottom = {c: max(r for r, cc in shape if cc == c) for c in range(width)}
        for x in range(W - width + 1):
            rows.append([r for r, _ in shape])
            cols.append([c + x for _, c in shape])
            prof_cols.append([x + c for c in range(width)] + [x] * (4 - width))
            prof_bottom.append([bottom[c] for c in range(width)] + [bottom[0]] * (4 - width))
    return tuple(np.array(a) for a in (rows, cols, prof_cols, prof_bottom))


PLACEMENTS = [_placements(p) for p in PIECES]


def measure(boards, lines):
    """The 33 perception numbers for a stack of boards (B, H, W)."""
    filled = boards > 0
    has = filled.any(1)
    heights = np.where(has, H - filled.argmax(1), 0)
    holes = ((np.cumsum(filled, 1) > 0) & ~filled).sum(1)
    steps = np.abs(np.diff(heights, axis=1))
    walls = np.ones((len(boards), H, 1), bool)
    across = np.concatenate([walls, filled, walls], 2)
    floor = np.concatenate([filled, np.ones((len(boards), 1, W), bool)], 1)
    return np.column_stack([
        heights, steps, holes, heights.max(1), lines,
        (across[:, :, 1:] != across[:, :, :-1]).sum((1, 2)),
        (floor[:, 1:] != floor[:, :-1]).sum((1, 2)),
    ]).astype(np.float64)


def options(board, piece):
    """Every legal resulting board for ``piece`` (index), with lines cleared and features."""
    rows, cols, prof_cols, prof_bottom = PLACEMENTS[piece]
    filled = board > 0
    top = np.where(filled.any(0), filled.argmax(0), H)
    landing = (top[prof_cols] - prof_bottom - 1).min(1)
    ok = landing >= 0
    if not ok.any():
        return None
    rows, cols, landing = rows[ok], cols[ok], landing[ok]
    boards = np.repeat(board[None], len(landing), 0)
    boards[np.arange(len(landing))[:, None], rows + landing[:, None], cols] = piece + 1
    full = (boards > 0).all(2)
    lines = full.sum(1)
    order = np.argsort(~full, axis=1, kind="stable")  # full rows first, rest keep order
    boards = np.take_along_axis(boards, order[:, :, None], 1)
    boards[np.arange(H)[None, :] < lines[:, None]] = 0
    return boards, lines, measure(boards, lines)


class Game:
    """Standard 10 x 20 Tetris, seven pieces drawn uniformly, hard drops."""

    TITLE, SCORE, STEP = "TETRIS", "lines", "pieces"
    CELL, NUMBERS = 16, False
    PALETTE = ["#1c2128", "#3ec7e0", "#f2d544", "#b36be0", "#5cd65c", "#e05c5c", "#4a78e0",
               "#f09a3e"]

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((H, W), np.int8)
        self.piece, self.next = self.rng.integers(7), self.rng.integers(7)
        self.score = self.steps = 0
        self.over = False
        self._found = None

    def options(self):
        self._found = options(self.board, self.piece)
        return None if self._found is None else self._found[2]

    def apply(self, index):
        boards, lines, _ = self._found
        self.board, self.score = boards[index], self.score + int(lines[index])
        self.steps += 1
        self.piece, self.next = self.next, self.rng.integers(7)

    def cells(self):
        return self.board
