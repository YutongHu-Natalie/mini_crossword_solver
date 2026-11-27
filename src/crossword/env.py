"""Crossword board environment, ported from Tree of Thoughts (MIT license).

This mirrors tot/tasks/crosswords.py::MiniCrosswordsEnv as closely as possible so the
ToT baseline sees exactly the same observation text (the `render()` output is pasted
into the propose prompt). The LLM calls that lived inside the original env
(`prompt_status`) were moved to crossword/tot.py so the env has no model dependency.

Slot status codes (from ToT): 0 = unfilled, 1 = filled, 2 = filled, then overwritten
by a crossing word (a conflict).
"""
from __future__ import annotations

from .data import Puzzle, answers_from_board
from .metrics import score_board


class CrosswordEnv:
    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle
        self.reset()

    # --- state -----------------------------------------------------------------
    def reset(self, board=None, status=None, steps=None) -> str:
        self.data = list(self.puzzle.clues)
        self.board_gt = list(self.puzzle.solution)
        self.ans_gt = answers_from_board(self.board_gt)
        self.board = ["_"] * 25
        self.ans = ["_____"] * 10
        self.steps = 0
        self.status = [0] * 10
        if board is not None:
            self.board = list(board)
            self.ans = answers_from_board(self.board)
        if status is not None:
            self.status = list(status)
        if steps is not None:
            self.steps = steps
        return self.render()

    def snapshot(self):
        return self.board.copy(), self.status.copy(), self.steps

    def restore(self, snap) -> None:
        board, status, steps = snap
        self.reset(board=board.copy(), status=status.copy(), steps=steps)

    # --- rendering (text must match ToT exactly) --------------------------------
    def render_board(self) -> str:
        s = "Current Board:\n"
        for i in range(5):
            s += "".join(self.board[i * 5:(i + 1) * 5]) + "\n"
        return s

    def render_clues(self, status=None) -> str:
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += "h" + str(i + 1) + ". " + self.data[i] + "\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += "v" + str(i - 5 + 1) + ". " + self.data[i] + "\n"
        return s

    def render_ans(self, status=None) -> str:
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += "h" + str(i + 1) + ". " + self.data[i] + ": " + self.ans[i] + "\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += "v" + str(i - 5 + 1) + ". " + self.data[i] + ": " + self.ans[i] + "\n"
        return s

    def render(self, status: bool = True) -> str:
        if status:
            return (self.render_board() + "\nUnfilled:\n" + self.render_ans(status=0)
                    + "\nFilled:\n" + self.render_ans(status=1)
                    + "\nChanged:\n" + self.render_ans(status=2))
        return self.render_board() + "\n" + self.render_ans()

    # --- actions ----------------------------------------------------------------
    def step(self, action: str):
        """Place a word, e.g. "h1. apple". Returns (obs, solved, done, info) like ToT."""
        self.steps += 1
        action = action.split("\n")[-1]
        parts = action.split(". ")
        if len(parts) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = parts
        if len(word) != 5:
            return "Invalid! Word should have 5 letters.", 0, False, {}
        if pos.startswith("h"):
            idx = int(pos[1:]) - 1
            self.board[idx * 5:(idx + 1) * 5] = list(word.upper())
        elif pos.startswith("v"):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.upper())
            idx += 5
        else:
            return "Invalid! Position should be h1-h5 or v1-v5", 0, False, {}

        new_ans = answers_from_board(self.board)
        self.status = [
            2 if any(l != nl and l != "_" for l, nl in zip(ans, n_ans)) else st
            for st, ans, n_ans in zip(self.status, self.ans, new_ans)
        ]
        self.status[idx] = 1
        self.ans = new_ans
        sc = score_board(self.board, self.board_gt)
        solved = sc["game"] == 1.0
        info = {"r_letter": sc["letter"], "r_word": sc["word"], "r_game": solved}
        return self.render(), solved, (solved or self.steps >= 20), info

    def has_conflict(self) -> bool:
        return any(s == 2 for s in self.status)
