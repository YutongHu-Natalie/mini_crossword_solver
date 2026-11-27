"""ToT-style evaluation: letter-, word- and game-level accuracy.

letter = fraction of the 25 cells that match the solution
word   = fraction of the 10 answers (5 across + 5 down) that match exactly
game   = 1 if the whole grid is correct

ToT's reported numbers (78 / 60 / 4 of 20 for ToT+GPT-4) use the FIRST DEEPEST state
the DFS reached as the output. `select_output` implements that rule, and
tests/test_metrics.py checks it reproduces the paper from ToT's released logs.
"""
from __future__ import annotations

from .data import answers_from_board


def score_board(board: list[str], solution: list[str] | tuple[str, ...]) -> dict:
    board = [c.upper() for c in board]
    solution = list(solution)
    letter = sum(a == b for a, b in zip(board, solution)) / 25
    word = sum(a == b for a, b in zip(answers_from_board(board), answers_from_board(solution))) / 10
    return {"letter": letter, "word": word, "game": float(board == solution)}


def aggregate(scores: list[dict]) -> dict:
    """Mean letter/word/game accuracy in %, plus the count of fully solved games."""
    n = len(scores)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "letter_%": round(100 * sum(s["letter"] for s in scores) / n, 1),
        "word_%": round(100 * sum(s["word"] for s in scores) / n, 1),
        "game_%": round(100 * sum(s["game"] for s in scores) / n, 1),
        "games_solved": int(sum(s["game"] for s in scores)),
    }


def select_output(states: list[dict], rule: str = "deepest_first") -> dict | None:
    """Pick the DFS state used as the final answer.

    deepest_first: first state with the most words placed (ToT's paper metric)
    oracle_best:   state with the best word accuracy (ToT's "+best state" upper bound;
                   uses the answer key, so report it only as an oracle)
    last:          last state explored
    """
    if not states:
        return None
    if rule == "deepest_first":
        return max(states, key=lambda s: s["env_step"])  # max() keeps the first maximum
    if rule == "oracle_best":
        return max(states, key=lambda s: s["info"]["r_word"])
    if rule == "last":
        return states[-1]
    raise ValueError(f"unknown rule {rule!r}")
