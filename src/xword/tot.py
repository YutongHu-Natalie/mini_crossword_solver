"""Tree of Thoughts DFS baseline for mini crosswords.

A faithful port of ToT's scripts/crosswords/search_crosswords-dfs.ipynb (`dfs` with
prune=True, max_per_state=3, time_limit=100), with the model calls routed through
xword.llm so we can (a) swap models and (b) count calls and tokens.

Per DFS node ToT makes:
  * 1 propose call (n=8 samples): "list all possible answers ... with confidence".
    Confidences are summed across samples: certain=1, high=0.5, medium=0.2, low=0.1.
  * 1 value call (n=1) per partially filled word, asking sure/maybe/impossible.
    With prune=True a node is not expanded if any word is judged "impossible".
Both are memoised within a run (same as ToT), so repeated states cost nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .data import Puzzle
from .env import CrosswordEnv
from .llm import LLM
from .metrics import score_board, select_output
from .tot_prompts import propose_prompt, value_prompt

CONFIDENCE_TO_VALUE = {"certain": 1, "high": 0.5, "medium": 0.2, "low": 0.1}
_LINE = re.compile(r"^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$")


def parse_response(response: str) -> list[tuple[str, float]]:
    """Parse lines like "h1. apple (medium)" -> [("h1. apple", 0.2), ...]."""
    out = []
    for line in response.split("\n"):
        m = _LINE.match(line)
        if m:
            out.append((m.group(1).lower() + ". " + m.group(2).lower(),
                        CONFIDENCE_TO_VALUE.get(m.group(3), 0)))
    return out


@dataclass
class ToTConfig:
    time_limit: int = 100        # max number of states recorded (ToT: 100)
    prune: bool = True           # skip states with any "impossible" word (ToT: True)
    max_per_state: int = 3       # children explored per state (ToT: 3)
    n_propose: int = 8           # samples per propose call (ToT: 8)
    temperature: float = 0.7     # ToT: 0.7
    max_tokens: int = 1000       # ToT: 1000
    output_rule: str = "deepest_first"  # how the final answer is picked (ToT's metric)


class ToTSolver:
    name = "tot_dfs"

    def __init__(self, llm: LLM, config: ToTConfig | None = None, verbose: bool = False):
        self.llm = llm
        self.cfg = config or ToTConfig()
        self.verbose = verbose
        self._propose_memo: dict[str, dict[str, float]] = {}
        self._value_memo: dict[str, str] = {}

    # --- LLM steps ----------------------------------------------------------------
    def candidates(self, env: CrosswordEnv) -> dict[str, float]:
        obs = env.render()
        if obs in self._propose_memo:
            return self._propose_memo[obs]
        responses = self.llm.generate(propose_prompt.format(input=obs), n=self.cfg.n_propose,
                                      temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
        scores: dict[str, float] = {}
        for r in responses:
            for cand, s in parse_response(r):
                scores[cand] = scores.get(cand, 0) + s
        self._propose_memo[obs] = scores
        return scores

    def prompt_status(self, env: CrosswordEnv) -> dict[str, int]:
        count = {"sure": 0, "maybe": 0, "impossible": 0}
        for ans, clue in zip(env.ans, env.data):
            if ans.count("_") >= 4:
                continue
            line = f"{clue}: {' '.join(ans.lower())}"
            prompt = value_prompt.format(input=line)
            if prompt not in self._value_memo:
                self._value_memo[prompt] = self.llm.generate(
                    prompt, n=1, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)[0]
            res = self._value_memo[prompt].split("\n")[-1].strip()
            if res in count:
                count[res] += 1
        return count

    # --- search -------------------------------------------------------------------
    def _dfs(self, env: CrosswordEnv, actions: list[str], states: list[dict]) -> None:
        cands = self.candidates(env)
        if not cands:
            return
        snap = env.snapshot()
        explored_here = 0
        for action in sorted(cands, key=cands.get, reverse=True):
            _, _, _, info = env.step(action)
            if len(states) < self.cfg.time_limit and env.steps < 10 and not env.has_conflict():
                explored_here += 1
                if explored_here > self.cfg.max_per_state:
                    break
                count = self.prompt_status(env)
                actions.append(action)
                states.append({"total_step": len(states), "env_step": env.steps,
                               "actions": actions.copy(), "board": env.board.copy(),
                               "info": info, "count": count})
                if self.verbose:
                    print(len(states) - 1, actions, info["r_word"], count)
                if not self.cfg.prune or count["impossible"] < 1:
                    self._dfs(env, actions, states)
                actions.pop()
            env.restore(snap)

    def solve(self, puzzle: Puzzle) -> dict:
        """Run DFS on one puzzle; returns a JSON-serialisable record."""
        before = self.llm.usage.copy()
        env = CrosswordEnv(puzzle)
        states: list[dict] = []
        self._dfs(env, [], states)
        chosen = select_output(states, self.cfg.output_rule)
        board = chosen["board"] if chosen else ["_"] * 25
        used = self.llm.usage.minus(before)
        return {
            "method": self.name,
            "model": self.llm.model,
            "config": asdict(self.cfg),
            "idx": puzzle.idx,
            "board": "".join(board),
            "scores": score_board(board, puzzle.solution),
            "oracle_scores": (score_board(select_output(states, "oracle_best")["board"], puzzle.solution)
                              if states else score_board(board, puzzle.solution)),
            "nodes": len(states),
            "usage": used.as_dict(),
            "states": states,
        }
