"""Our method: the LLM proposes candidates, A* search fills the grid.

1. Candidates. ONE LLM call per puzzle asks for up to k answers per clue, each with a
   confidence 0-100. (ToT asks the LLM at every search step; we ask once.)
2. A* search over partial grids.
     state  : a set of placed words (slot -> word), all letters consistent
     action : place a candidate in the unfilled slot with the FEWEST consistent candidates
              (most-constrained-slot ordering)
     cost   : -log(confidence) of the placed word, so the cheapest full grid is the one
              the LLM finds most likely overall
     h      : for every unfilled slot, the cheapest candidate still consistent with the
              letters already placed, summed. It ignores conflicts between unfilled slots,
              so it never overestimates (admissible).
   Forward checking: candidates that no longer fit the placed letters are dropped
   automatically when a slot's options are computed.
   A slot whose candidates are all ruled out may be left UNRESOLVED at a fixed penalty
   cost (-log 0.05 ~ 3). Its cells are then filled by the crossing words. So A* always
   returns the cheapest full grid, possibly with a few unresolved slots.
3. Re-prompting. For each unresolved slot we ask the LLM again, now giving the letter
   pattern its crossing words fixed (e.g. "S _ A _ E"). New candidates are added and the
   search restarts, so h is recomputed with the new candidates and stays admissible.
   This repeats until no slot is unresolved, nothing new comes back, or the budget is used.
"""
from __future__ import annotations

import heapq
import itertools
import math
import re
from dataclasses import dataclass, asdict

from .data import SLOTS, Puzzle
from .llm import LLM
from .metrics import score_board

# cells (0-24, row-major) covered by each slot
CELLS = {**{f"h{i+1}": [i * 5 + j for j in range(5)] for i in range(5)},
         **{f"v{j+1}": [i * 5 + j for i in range(5)] for j in range(5)}}

CANDIDATE_PROMPT = """You are solving a 5x5 mini crossword. Every answer is exactly 5 letters.
For each clue below, list up to {k} possible 5-letter answers, each with your confidence (0-100) that it is correct.
Use one line per answer, in exactly this format:
h1. APPLE 80
List answers for every clue (h1-h5 across, v1-v5 down). Output only the lines.

Clues:
{clues}
"""

PATTERN_PROMPT = """Crossword clue: {clue}
The answer has exactly 5 letters and must match this pattern, where _ is an unknown letter: {pattern}
List up to {k} possible answers, each with your confidence (0-100) that it is correct.
Use one line per answer, in exactly this format:
APPLE 80
Output only the lines.
"""

_SLOT_LINE = re.compile(r"^\W*([hv][1-5])\W+([A-Za-z]{5})\b\D*?(\d{1,3}(?:\.\d+)?)", re.I | re.M)
_WORD_LINE = re.compile(r"^\W*([A-Za-z]{5})\b\D*?(\d{1,3}(?:\.\d+)?)", re.M)


def parse_candidates(text: str) -> dict[str, dict[str, float]]:
    """"h1. APPLE 80" lines -> {slot: {WORD: confidence in (0, 1]}}; keeps the max per word."""
    out: dict[str, dict[str, float]] = {}
    for slot, word, conf in _SLOT_LINE.findall(text):
        c = min(max(float(conf) / 100, 0.01), 1.0)
        d = out.setdefault(slot.lower(), {})
        d[word.upper()] = max(d.get(word.upper(), 0), c)
    return out


def parse_pattern_candidates(text: str, pattern: str) -> dict[str, float]:
    """"APPLE 80" lines -> {WORD: conf}, keeping only words that match the pattern."""
    out: dict[str, float] = {}
    for word, conf in _WORD_LINE.findall(text):
        w = word.upper()
        if all(p in ("_", ch) for p, ch in zip(pattern, w)):
            out[w] = max(out.get(w, 0), min(max(float(conf) / 100, 0.01), 1.0))
    return out


@dataclass
class AStarConfig:
    k: int = 5                    # candidates per clue in the first call
    k_reprompt: int = 5           # candidates per re-prompt
    reprompt: bool = True         # pattern-conditioned re-prompting on/off
    max_reprompts: int = 10       # re-prompt calls per puzzle
    reprompts_per_round: int = 3  # re-prompts between search restarts
    unresolved_penalty: float = 0.05  # a slot may stay unresolved at cost -log(0.05) ~ 3.0
    max_expansions: int = 50_000  # per search round (no LLM calls, so this is cheap)
    temperature: float = 0.0
    max_tokens: int = 2000


UNRESOLVED = "?"   # a slot left without a word; its cells come from the crossing words


def _board(assign: dict[str, str]) -> list:
    b = [None] * 25
    for s, w in assign.items():
        if w == UNRESOLVED:
            continue
        for c, ch in zip(CELLS[s], w):
            b[c] = ch
    return b


def _pattern(board, slot) -> str:
    return "".join(board[c] or "_" for c in CELLS[slot])


def _fits(word, board, slot) -> bool:
    return all(board[c] in (None, ch) for c, ch in zip(CELLS[slot], word))


class AStarSolver:
    name = "astar"

    def __init__(self, llm: LLM, config: AStarConfig | None = None, verbose: bool = False):
        self.llm = llm
        self.cfg = config or AStarConfig()
        self.verbose = verbose

    # --- LLM ------------------------------------------------------------------
    def propose(self, puzzle: Puzzle) -> dict[str, dict[str, float]]:
        clues = "\n".join(f"{s}. {c}" for s, c in zip(SLOTS, puzzle.clues))
        text = self.llm.generate(CANDIDATE_PROMPT.format(k=self.cfg.k, clues=clues), n=1,
                                 temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)[0]
        cands = parse_candidates(text)
        return {s: cands.get(s, {}) for s in SLOTS}

    def propose_pattern(self, puzzle: Puzzle, slot: str, pattern: str) -> dict[str, float]:
        prompt = PATTERN_PROMPT.format(clue=puzzle.clue(slot), pattern=" ".join(pattern),
                                       k=self.cfg.k_reprompt)
        text = self.llm.generate(prompt, n=1, temperature=self.cfg.temperature,
                                 max_tokens=self.cfg.max_tokens)[0]
        return parse_pattern_candidates(text, pattern)

    # --- search ---------------------------------------------------------------
    @property
    def wild_cost(self) -> float:
        return -math.log(self.cfg.unresolved_penalty)

    def _options(self, board, slot, cands):
        return [(w, -math.log(p)) for w, p in cands[slot].items() if _fits(w, board, slot)]

    def _h(self, board, assign, cands) -> float:
        """Admissible: each unfilled slot's cheapest still-consistent option (or leaving it
        unresolved), ignoring conflicts between unfilled slots."""
        return sum(min([c for _, c in self._options(board, s, cands)] + [self.wild_cost])
                   for s in SLOTS if s not in assign)

    def search(self, cands):
        """One A* run. Returns (cheapest full assignment, nodes expanded).

        Every slot gets either a candidate word or UNRESOLVED (cost = wild_cost). An
        unresolved slot places no letters; its cells are filled by the crossing words."""
        tie = itertools.count()
        frontier = [(self._h(_board({}), {}, cands), next(tie), 0.0, {})]
        expanded = 0
        best = None
        while frontier:
            f, _, g, assign = heapq.heappop(frontier)
            if len(assign) == 10:
                return assign, expanded
            if expanded >= self.cfg.max_expansions:
                best = best or assign
                break
            expanded += 1
            board = _board(assign)
            options = {s: self._options(board, s, cands) for s in SLOTS if s not in assign}
            slot = min(options, key=lambda s: (len(options[s]), SLOTS.index(s)))   # most constrained
            for w, c in options[slot] + [(UNRESOLVED, self.wild_cost)]:
                child = {**assign, slot: w}
                gc = g + c
                heapq.heappush(frontier, (gc + self._h(_board(child), child, cands), next(tie), gc, child))
        # expansion budget hit: fall back to the deepest state seen, rest unresolved
        assign = {s: (best or {}).get(s, UNRESOLVED) for s in SLOTS}
        return assign, expanded

    # --- full solve -------------------------------------------------------------
    def solve(self, puzzle: Puzzle) -> dict:
        before = self.llm.usage.copy()
        cands = self.propose(puzzle)
        n_initial = sum(len(v) for v in cands.values())
        gold_initial = sum(a in cands[s] for s, a in zip(SLOTS, puzzle.answers))
        asked: set[tuple[str, str]] = set()
        reprompts, nodes, rounds = [], 0, 0
        while True:
            rounds += 1
            goal, expanded = self.search(cands)
            nodes += expanded
            unresolved = [s for s in SLOTS if goal[s] == UNRESOLVED]
            if not unresolved or not self.cfg.reprompt or len(reprompts) >= self.cfg.max_reprompts:
                break
            # re-prompt unresolved slots with the letters their crossing words fixed
            board = _board(goal)
            todo = [(s, _pattern(board, s)) for s in unresolved
                    if (s, _pattern(board, s)) not in asked or (s, "_____") not in asked]
            todo = todo[:min(self.cfg.reprompts_per_round, self.cfg.max_reprompts - len(reprompts))]
            if not todo:
                break
            added = 0
            for slot, pat in todo:
                # If nothing fits the pattern, the crossing letters are probably wrong:
                # ask again with the clue alone ("_____").
                for p in [pat] if pat == "_____" else [pat, "_____"]:
                    if (slot, p) in asked or len(reprompts) >= self.cfg.max_reprompts:
                        continue
                    asked.add((slot, p))
                    new = self.propose_pattern(puzzle, slot, p)
                    fresh = {w: q for w, q in new.items() if w not in cands[slot]}
                    cands[slot].update(fresh)
                    added += len(fresh)
                    reprompts.append({"slot": slot, "pattern": p, "new": sorted(fresh)})
                    if self.verbose:
                        print(f"  re-prompt {slot} {' '.join(p)} -> {sorted(fresh) or 'nothing new'}")
                    if fresh:
                        break
            if added == 0:
                break
        assign = goal
        complete = not any(w == UNRESOLVED for w in assign.values())
        board = [c or "_" for c in _board(assign)]
        used = self.llm.usage.minus(before)
        if self.verbose:
            print(f"  {'complete grid' if complete else 'grid with unresolved slots'} after {rounds} "
                  f"search round(s), {nodes} nodes, {len(reprompts)} re-prompts")
        return {
            "method": self.name, "model": self.llm.model, "config": asdict(self.cfg),
            "idx": puzzle.idx, "board": "".join(board), "complete": complete,
            "scores": score_board(board, puzzle.solution),
            "nodes": nodes, "rounds": rounds,
            "n_candidates_initial": n_initial,
            # diagnostics: was the right answer ever available to the search?
            "gold_in_initial": gold_initial,
            "gold_in_candidates": sum(a in cands[s] for s, a in zip(SLOTS, puzzle.answers)),
            "reprompts": reprompts,
            "assignment": assign,
            "usage": used.as_dict(),
        }
