import numpy as np
import pytest

from vybir.arcade import g2048, snake, tetris
from vybir.arcade.learn import GAMES, game_module, load_brain, new_game, play, record, step, train


def test_tetris_placement_counts_on_empty_board():
    empty = np.zeros((tetris.H, tetris.W), np.int8)
    counts = {p: len(tetris.options(empty, i)[1]) for i, p in enumerate(tetris.PIECES)}
    assert counts["I"] == 7 + 10 and counts["O"] == 9 and counts["T"] == 8 + 8 + 9 + 9


def test_tetris_line_clear_drops_rows_above():
    board = np.zeros((tetris.H, tetris.W), np.int8)
    board[-1, 4:] = 1
    board[-2, 9] = 2
    boards, lines, feats = tetris.options(board, tetris.PIECES.index("I"))
    best = int(np.argmax(lines))
    assert lines[best] == 1
    assert (boards[best][-1] > 0).sum() == 1 and boards[best][-1, 9] == 2
    assert feats[best, tetris.FEATURES.index("lines_cleared")] == 1


def test_tetris_measure_heights_holes_and_steps():
    board = np.zeros((1, tetris.H, tetris.W), np.int8)
    board[0, -3, 0] = 1
    board[0, -1, 1] = 1
    f = dict(zip(tetris.FEATURES, tetris.measure(board, np.zeros(1))[0]))
    assert f["height_0"] == 3 and f["holes_0"] == 2 and f["step_0"] == 2


def test_snake_rules_death_growth_and_tail_chasing():
    game = snake.Game(seed=0)
    game.food = (game.body[0][0], game.body[0][1] + 1)  # right in front
    feats = game.options()
    moves = game._moves
    ahead = moves.index((0, 1))
    assert feats[ahead, snake.FEATURES.index("eats")] == 1
    game.apply(ahead)
    assert game.score == 1 and len(game.body) == 4
    game = snake.Game(seed=0)
    game.body = snake.deque([(0, 0), (0, 1), (1, 1), (1, 0)])  # head at a corner, tail below
    game.direction = (0, -1)
    feats = game.options()
    down = game._moves.index((1, 0))
    assert feats[down, 0] == 0  # moving into the tail cell is legal: the tail moves away
    up = game._moves.index((-1, 0))
    assert feats[up, 0] == 1  # the wall kills


def test_2048_slide_merges_once_and_scores():
    board = np.array([[1, 1, 2, 2], [1, 0, 1, 0], [3, 3, 3, 0], [0, 0, 0, 0]])
    left, gained = g2048.slide(board, 0)
    assert left[0].tolist() == [2, 3, 0, 0] and left[1].tolist() == [2, 0, 0, 0]
    assert left[2].tolist() == [4, 3, 0, 0]
    assert gained == 4 + 8 + 4 + 16
    right, _ = g2048.slide(board, 2)
    assert right[0].tolist() == [0, 0, 2, 3]


@pytest.mark.parametrize("name", sorted(GAMES))
def test_every_game_is_seeded_and_an_untrained_brain_does_poorly(name):
    zeros = np.zeros(len(game_module(name).FEATURES))
    assert play(name, zeros, 3, 3000) == play(name, zeros, 3, 3000)
    game = new_game(name, 3)
    while not game.over and game.steps < 3000:
        step(game, zeros)
    assert game.over


@pytest.mark.parametrize("name", sorted(GAMES))
def test_shipped_brain_beats_the_untrained_one(name):
    _, zero, _ = load_brain(name, generation=0)
    _, learned, _ = load_brain(name)
    assert not zero.any()
    assert np.mean([play(name, learned, s, 2000)[0] for s in (1, 2)]) > np.mean(
        [play(name, zero, s, 2000)[0] for s in (1, 2)])


def test_training_improves_on_a_tiny_budget():
    history = train("snake", generations=3, population=12, elite=3, cap=150, games=1,
                    workers=2, log=lambda *_: None)
    assert history[0]["weights"] == [0.0] * len(snake.FEATURES)
    assert history[-1]["population_best"] > 0


def test_record_writes_a_gif(tmp_path):
    out = record("2048", tmp_path / "g.gif", steps=5)
    assert out.stat().st_size > 1000
