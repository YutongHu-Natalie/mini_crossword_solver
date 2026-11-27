"""Mini-crossword solver: LLM candidate generation + classical search (CS 557 project)."""

from .data import Puzzle, load_puzzles, get_split, SLOTS, TEST_IDS, FEWSHOT_IDS, DEV_IDS
from .metrics import score_board, aggregate

__all__ = [
    "Puzzle", "load_puzzles", "get_split", "SLOTS", "TEST_IDS", "FEWSHOT_IDS", "DEV_IDS",
    "score_board", "aggregate",
]
