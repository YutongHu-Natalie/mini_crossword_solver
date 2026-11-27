"""A fake LLM for dry runs and tests. It knows the answers, and it costs nothing.

For each clue it proposes the right answer plus random distractors with random
confidences, and with probability `miss_rate` it leaves the right answer out, so the
search and re-prompting have real work to do. Scores from it mean nothing about LLMs.
"""
from __future__ import annotations

import random
import re

from .data import SLOTS, load_puzzles
from .llm import ScriptedLLM


def fake_crossword_llm(miss_rate: float = 0.2, n_distract: int = 3, seed: int = 0) -> ScriptedLLM:
    puzzles = load_puzzles()
    pool = sorted({w for p in puzzles for w in p.answers})
    clue_to_answer = {}
    for p in puzzles:
        for s, c, a in zip(SLOTS, p.clues, p.answers):
            clue_to_answer.setdefault(c, a)
    rnd = random.Random(seed)

    def fits(w, pat):
        return all(p in ("_", ch) for p, ch in zip(pat, w))

    def fn(prompt: str, n: int) -> list[str]:
        if prompt.startswith("Crossword clue:"):           # pattern re-prompt
            clue = re.search(r"Crossword clue: (.*)", prompt).group(1)
            pat = re.search(r"unknown letter: (.*)", prompt).group(1).replace(" ", "")
            gold = clue_to_answer.get(clue, "")
            words = ([gold] if gold and fits(gold, pat) else []) + [w for w in pool if fits(w, pat)][:2]
            return ["\n".join(f"{w} {rnd.randint(20, 90)}" for w in words)] * n
        lines = []                                          # first call: all 10 clues
        for slot, clue in re.findall(r"^([hv][1-5])\. (.*)$", prompt, flags=re.M):
            gold = clue_to_answer.get(clue)
            if gold and rnd.random() >= miss_rate:
                lines.append(f"{slot}. {gold} {rnd.randint(30, 95)}")
            for w in rnd.sample(pool, n_distract):
                lines.append(f"{slot}. {w} {rnd.randint(10, 80)}")
        return ["\n".join(lines)] * n

    return ScriptedLLM(fn, model=f"fake-miss{miss_rate}")
