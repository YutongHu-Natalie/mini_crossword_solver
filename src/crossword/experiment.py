"""Running a solver over a split, saving results, and summarising them.

Results go to results/<method>__<model>__<split>[__<tag>].jsonl, one JSON record per
puzzle, appended as each puzzle finishes. Re-running skips puzzles already in the file,
so a crashed or interrupted Colab run resumes where it stopped. Commit these files: they
are how teammates share results without re-running (and re-paying for) experiments.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .data import REPO_ROOT, Puzzle, load_puzzles
from .llm import PRICES_PER_M
from .metrics import aggregate, score_board, select_output

RESULTS_DIR = REPO_ROOT / "results"
TOT_LOG_DIR = REPO_ROOT / "data" / "tot_logs"


def results_path(method: str, model: str, split: str, tag: str = "") -> Path:
    safe = lambda s: re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")
    name = "__".join(safe(x) for x in [method, model, split] + ([tag] if tag else []))
    return RESULTS_DIR / f"{name}.jsonl"


def load_results(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(solver, puzzles: list[Puzzle], path: Path | str, keep_states: bool = True) -> list[dict]:
    """Solve each puzzle not already in `path`, appending one record per puzzle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["idx"] for r in load_results(path)}
    todo = [p for p in puzzles if p.idx not in done]
    print(f"{len(done)} puzzles already in {path.name}; running {len(todo)}")
    for i, p in enumerate(todo, 1):
        rec = solver.solve(p)
        if not keep_states:
            rec.pop("states", None)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        s, u = rec["scores"], rec["usage"]
        print(f"[{i}/{len(todo)}] puzzle {p.idx:3d}  word={s['word']:.1f} letter={s['letter']:.2f} "
              f"game={int(s['game'])}  nodes={rec['nodes']}  calls={u['calls']}  "
              f"tokens={u['prompt_tokens'] + u['completion_tokens']}")
    return load_results(path)


def summarize(records: list[dict], model: str | None = None) -> dict:
    """Accuracy (ToT metric), oracle accuracy, and per-puzzle cost averages."""
    if not records:
        return {"n": 0}
    out = aggregate([r["scores"] for r in records])
    if all("oracle_scores" in r for r in records):
        orc = aggregate([r["oracle_scores"] for r in records])
        out.update({f"oracle_{k}": v for k, v in orc.items() if k != "n"})
    n = len(records)
    if all("usage" in r for r in records):
        tot = lambda k: sum(r["usage"][k] for r in records)
        out.update({
            "llm_calls/puzzle": round(tot("calls") / n, 1),
            "input_tokens/puzzle": round(tot("prompt_tokens") / n),
            "output_tokens/puzzle": round(tot("completion_tokens") / n),
            "nodes/puzzle": round(sum(r["nodes"] for r in records) / n, 1),
        })
        model = model or records[0].get("model")
        if model in PRICES_PER_M:
            pin, pout = PRICES_PER_M[model]
            out["usd/puzzle"] = round((tot("prompt_tokens") * pin + tot("completion_tokens") * pout) / 1e6 / n, 4)
    return out


def count_tokens_gpt4():
    """Token counter for GPT-4 (tiktoken). Falls back to ~4 characters per token."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4")
        return (lambda s: len(enc.encode(s))), "exact (tiktoken, gpt-4)"
    except Exception:
        return (lambda s: max(1, round(len(s) / 4))), "approx (chars/4: pip install tiktoken for exact)"


def tot_paper_usage(count_tokens=None) -> dict[int, dict]:
    """LLM usage of ToT's paper run (GPT-4, prune=True), per test puzzle.

    data/tot_logs/tot_paper_calls.json lists every call that run needed, rebuilt from the
    released trajectories (every call was verified to exist in ToT's released caches):
      * propose calls: prompt text is exact; the n=8 answers were not saved, only the
        parsed scores, so their output tokens are a LOWER BOUND (~7 tokens per answer line,
        at least as many lines as the scores imply). Input is billed once per request.
      * value calls: prompt and answer both exact.
    """
    from .tot_prompts import propose_prompt, value_prompt
    count, _ = count_tokens or count_tokens_gpt4()
    calls = json.loads((TOT_LOG_DIR / "tot_paper_calls.json").read_text())
    out = {}
    for rec in calls:
        p_in = sum(count(propose_prompt.format(input=o)) for o in rec["propose_obs"])
        p_out = 7 * sum(rec["propose_mentions_lb"])
        v_in = sum(count(value_prompt.format(input=l)) for l in rec["value_lines"])
        v_out = sum(count(r) for r in rec["value_responses"])
        out[rec["idx"]] = {
            "calls": len(rec["propose_obs"]) + len(rec["value_lines"]),
            "propose_calls": len(rec["propose_obs"]), "value_calls": len(rec["value_lines"]),
            "prompt_tokens": p_in + v_in, "completion_tokens": p_out + v_out,
        }
    return out


def tot_paper_records(prune: bool = True, rule: str = "deepest_first", with_usage: bool = False,
                      count_tokens=None) -> list[dict]:
    """ToT + GPT-4 results rebuilt from the trajectories ToT released (no API calls).

    The logs store each DFS state's actions, so we replay them to get the board.
    No token counts are logged, so these records have no "usage".
    """
    from .env import CrosswordEnv
    fname = "infoss_dfs_prune.json" if prune else "infoss_dfs_no_prune.json"
    logs = json.loads((TOT_LOG_DIR / fname).read_text())
    puzzles = load_puzzles()
    records = []
    for idx, states in zip(range(0, 100, 5), logs):
        p = puzzles[idx]

        def board_of(state):
            env = CrosswordEnv(p)
            for a in state["actions"]:
                env.step(a)
            return env.board

        chosen = select_output(states, rule)
        best = select_output(states, "oracle_best")
        records.append({
            "method": "tot_dfs_paper" + ("" if prune else "_noprune"), "model": "gpt-4", "idx": idx,
            "board": "".join(board_of(chosen)), "scores": score_board(board_of(chosen), p.solution),
            "oracle_scores": score_board(board_of(best), p.solution), "nodes": len(states),
        })
    if with_usage:
        if not prune:
            raise ValueError("usage is only reconstructed for the prune=True run")
        usage = tot_paper_usage(count_tokens)
        for r in records:
            r["usage"] = usage[r["idx"]]
    return records
