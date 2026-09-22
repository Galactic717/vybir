"""Snake: each of the three possible turns is simulated and the result is measured.
No path planner and no hints: the brain only sees these numbers and learns what they mean."""

from collections import deque

import numpy as np

H = W = 12
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
FEATURES = (
    "dies", "eats", "distance_to_food", "reachable_space", "tail_reachable", "free_neighbours",
)


def _flood(start, blocked, target=None):
    """Cells reachable from ``start``; also whether ``target`` is adjacent to that region."""
    seen, queue, touches = {start}, deque([start]), False
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES:
            cell = (r + dr, c + dc)
            if cell == target:
                touches = True
            if 0 <= cell[0] < H and 0 <= cell[1] < W and cell not in blocked and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen), touches


class Game:
    """12 x 12 Snake. Eating grows the snake; hitting a wall or itself, or starving for
    W * H steps without food, ends the game. Score: food eaten."""

    TITLE, SCORE, STEP = "SNAKE", "food", "steps"
    CELL, NUMBERS = 20, False
    PALETTE = ["#1c2128", "#2f9e6e", "#7ee2b8", "#f2b544"]  # empty, body, head, food

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.body = deque([(H // 2, W // 2 - i) for i in range(3)])  # head first, facing right
        self.direction = (0, 1)
        self.food = self._spawn()
        self.score = self.steps = self.hunger = 0
        self.over = False
        self._moves = []

    def _spawn(self):
        taken = set(self.body)
        free = [(r, c) for r in range(H) for c in range(W) if (r, c) not in taken]
        return free[self.rng.integers(len(free))] if free else None

    def options(self):
        head, tail = self.body[0], self.body[-1]
        taken = set(self.body)
        feats, self._moves = [], []
        for move in MOVES:
            if move == (-self.direction[0], -self.direction[1]):
                continue
            cell = (head[0] + move[0], head[1] + move[1])
            eats = cell == self.food
            inside = 0 <= cell[0] < H and 0 <= cell[1] < W
            dies = not inside or (cell in taken and not (cell == tail and not eats))
            self._moves.append(move)
            if dies:
                feats.append([1, 0, 1, 0, 0, 0])
                continue
            after = (taken if eats else taken - {tail}) | {cell}
            new_tail = tail if eats else self.body[-2]
            space, tail_ok = _flood(cell, after - {cell}, target=new_tail)
            free_cells = H * W - len(after)
            neighbours = sum(
                0 <= cell[0] + dr < H and 0 <= cell[1] + dc < W and (cell[0] + dr, cell[1] + dc)
                not in after for dr, dc in MOVES
            )
            distance = 0 if eats else (abs(cell[0] - self.food[0]) + abs(cell[1] - self.food[1]))
            feats.append([0, int(eats), distance / (H + W), (space - 1) / max(free_cells, 1),
                          int(tail_ok), neighbours / 3])
        return np.array(feats, dtype=np.float64)

    def apply(self, index):
        move = self._moves[index]
        head, tail = self.body[0], self.body[-1]
        cell = (head[0] + move[0], head[1] + move[1])
        eats = cell == self.food
        taken = set(self.body)
        self.steps += 1
        if not (0 <= cell[0] < H and 0 <= cell[1] < W) or (
            cell in taken and not (cell == tail and not eats)
        ):
            self.over = True
            return
        self.direction = move
        self.body.appendleft(cell)
        if eats:
            self.score += 1
            self.hunger = 0
            self.food = self._spawn()
            self.over = self.food is None  # board full: won
        else:
            self.body.pop()
            self.hunger += 1
            self.over = self.hunger > H * W

    def cells(self):
        grid = np.zeros((H, W), np.int8)
        for r, c in self.body:
            grid[r, c] = 1
        grid[self.body[0]] = 2
        if self.food is not None:
            grid[self.food] = 3
        return grid
