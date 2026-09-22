"""2048: each of the four slides is simulated (before the random new tile) and the board is
measured. The brain is never told that corners, order or empty space matter."""

import numpy as np

N = 4
FEATURES = (
    "empty_cells", "merge_gain", "max_tile", "max_in_corner", "monotonicity", "smoothness",
    "adjacent_pairs",
)


def _slide_left(board):
    """Slide every row left; returns the new board (exponents) and the score gained."""
    out = np.zeros_like(board)
    gained = 0
    for r in range(N):
        tiles = [v for v in board[r] if v]
        merged, i = [], 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                merged.append(tiles[i] + 1)
                gained += 2 ** (tiles[i] + 1)
                i += 2
            else:
                merged.append(tiles[i])
                i += 1
        out[r, : len(merged)] = merged
    return out, gained


def slide(board, direction):
    """direction 0..3 = left, up, right, down."""
    rotated = np.rot90(board, direction)
    moved, gained = _slide_left(rotated)
    return np.rot90(moved, -direction), gained


def measure(board, gained):
    rows_cols = list(board) + list(board.T)
    mono = 0.0
    for line in rows_cols:
        diffs = np.diff(line.astype(float))
        mono += max(diffs[diffs > 0].sum(), -diffs[diffs < 0].sum())
    horizontal = board[:, :-1], board[:, 1:]
    vertical = board[:-1, :], board[1:, :]
    smooth = pairs = 0
    for a, b in (horizontal, vertical):
        both = (a > 0) & (b > 0)
        smooth -= np.abs(a - b)[both].sum()
        pairs += ((a == b) & both).sum()
    top = board.max()
    corners = (board[0, 0], board[0, -1], board[-1, 0], board[-1, -1])
    return [(board == 0).sum(), np.log2(gained + 1), top, float(top in corners), mono, smooth,
            pairs]


class Game:
    """4 x 4 2048. New tile: 2 (90 %) or 4 (10 %). Score: sum of merged tiles, as in the
    original game."""

    TITLE, SCORE, STEP = "2048", "score", "moves"
    CELL, NUMBERS = 52, True
    PALETTE = ["#1c2128", "#3b4250", "#4a5568", "#8a5a2b", "#b8652a", "#cf5d3a", "#d9453b",
               "#c9a227", "#d4b030", "#dcbc3a", "#e6c645", "#f2d544", "#5fd0a0", "#3ec7e0",
               "#4a78e0", "#b36be0", "#e05c9c", "#e05c5c"]

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.board = np.zeros((N, N), np.int64)
        self.score = self.steps = 0
        self.over = False
        self._spawn()
        self._spawn()
        self._next = []

    def _spawn(self):
        empty = np.argwhere(self.board == 0)
        if len(empty):
            r, c = empty[self.rng.integers(len(empty))]
            self.board[r, c] = 1 if self.rng.random() < 0.9 else 2

    def options(self):
        feats, self._next = [], []
        for direction in range(4):
            board, gained = slide(self.board, direction)
            if not np.array_equal(board, self.board):
                self._next.append((board, gained))
                feats.append(measure(board, gained))
        return np.array(feats, dtype=np.float64) if feats else None

    def apply(self, index):
        self.board, gained = self._next[index]
        self.score += int(gained)
        self.steps += 1
        self._spawn()

    def cells(self):
        return self.board

    @staticmethod
    def label(value):
        return str(2 ** int(value)) if value else ""
