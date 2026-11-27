"""Loading the ToT mini-crossword dataset and the fixed splits.

Dataset: data/mini0505.json from Tree of Thoughts (Yao et al., NeurIPS 2023), MIT license.
156 puzzles; each is [10 clues (h1-h5, v1-v5), 25 solution letters (row-major)].

Splits (fixed; do not change without telling the team):
  test    = indices 0, 5, ..., 95   (the 20 puzzles ToT reports on)
  fewshot = indices 100, 105, ..., 120 (the 5 worked examples inside ToT's prompts;
            excluded from dev so we never tune on puzzles the prompt already solves)
  dev     = everything else (131 puzzles), for tuning k, prompts, etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PUZZLE_FILE = DATA_DIR / "mini0505.json"

SLOTS = ["h1", "h2", "h3", "h4", "h5", "v1", "v2", "v3", "v4", "v5"]

TEST_IDS = list(range(0, 100, 5))
FEWSHOT_IDS = [100, 105, 110, 115, 120]
DEV_IDS = [i for i in range(156) if i not in set(TEST_IDS) | set(FEWSHOT_IDS)]


def answers_from_board(board: list[str]) -> list[str]:
    """25 cells (row-major) -> 10 words in SLOTS order (5 rows, then 5 columns)."""
    rows = ["".join(board[i * 5:(i + 1) * 5]) for i in range(5)]
    cols = ["".join(board[i::5]) for i in range(5)]
    return rows + cols


@dataclass(frozen=True)
class Puzzle:
    idx: int
    clues: tuple[str, ...]      # 10 clues, SLOTS order
    solution: tuple[str, ...]   # 25 uppercase letters, row-major

    @property
    def answers(self) -> list[str]:
        """The 10 gold answers in SLOTS order."""
        return answers_from_board(list(self.solution))

    def clue(self, slot: str) -> str:
        return self.clues[SLOTS.index(slot)]

    def render(self, board: list[str] | None = None) -> str:
        """Clues plus a grid (the solution by default)."""
        board = list(self.solution) if board is None else board
        grid = "\n".join(" ".join(board[i * 5:(i + 1) * 5]) for i in range(5))
        clues = "\n".join(f"{s}. {c}" for s, c in zip(SLOTS, self.clues))
        return f"Puzzle {self.idx}\n{grid}\n\n{clues}"


@lru_cache(maxsize=1)
def load_puzzles(path: str | Path = PUZZLE_FILE) -> tuple[Puzzle, ...]:
    raw = json.loads(Path(path).read_text())
    return tuple(Puzzle(i, tuple(clues), tuple(sol)) for i, (clues, sol) in enumerate(raw))


def get_split(name: str) -> list[Puzzle]:
    ids = {"test": TEST_IDS, "fewshot": FEWSHOT_IDS, "dev": DEV_IDS}[name]
    puzzles = load_puzzles()
    return [puzzles[i] for i in ids]
