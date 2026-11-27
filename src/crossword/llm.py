"""LLM backends with a shared on-disk cache and token/cost accounting.

Every method in the project (ToT baseline, our A* solver) talks to models only through
`LLM.generate(prompt, n, temperature, max_tokens) -> list[str]`, so swapping GPT-4-class
models for an 8B open model is a one-line change in the notebook.

Backends
  openai : any OpenAI-compatible chat API. Set base_url to use other providers or a
           local vLLM server (e.g. an 8B Llama served on Colab).
  hf     : Hugging Face transformers, running locally on a Colab GPU.
  scripted: a Python function instead of a model (tests / dry runs, costs nothing).

Cost accounting
  `usage` counts every request the method makes, INCLUDING ones answered from the
  disk cache, because the cache only exists to avoid paying twice for the same run.
  The real spend of this session is in `usage.billed_*`.

Cache
  cache/llm_cache.sqlite (git-ignored). Key = backend, model, prompt, n, temperature,
  max_tokens, cache_tag. Re-running an experiment therefore replays the same samples
  (free and reproducible). Change `cache_tag` to draw fresh samples.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from .data import REPO_ROOT

# Override with the CROSSWORD_CACHE env var, e.g. a Google Drive path so the cache survives
# Colab restarts: os.environ["CROSSWORD_CACHE"] = "/content/drive/MyDrive/crossword_cache.sqlite"
DEFAULT_CACHE = Path(os.environ.get("CROSSWORD_CACHE", REPO_ROOT / "cache" / "llm_cache.sqlite"))

# USD per 1M tokens (input, output). Check current pricing before reporting dollars;
# these are list prices as of when this file was written and may be out of date.
PRICES_PER_M = {
    "gpt-4": (30.0, 60.0),          # the model ToT used (gpt-4-0613)
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
}


@dataclass
class Usage:
    calls: int = 0              # requests the method made (cached or not)
    samples: int = 0            # completions returned (n per call)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    billed_calls: int = 0       # requests that actually hit the model this session
    billed_prompt_tokens: int = 0
    billed_completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        for k, v in asdict(other).items():
            setattr(self, k, getattr(self, k) + v)

    def minus(self, other: "Usage") -> "Usage":
        return Usage(**{k: v - getattr(other, k) for k, v in asdict(self).items()})

    def copy(self) -> "Usage":
        return Usage(**asdict(self))

    def cost_usd(self, model: str, billed: bool = False) -> float | None:
        price = PRICES_PER_M.get(model)
        if price is None:
            return None
        p = self.billed_prompt_tokens if billed else self.prompt_tokens
        c = self.billed_completion_tokens if billed else self.completion_tokens
        return (p * price[0] + c * price[1]) / 1e6

    def as_dict(self) -> dict:
        return asdict(self)


class _Cache:
    def __init__(self, path: Path | str | None):
        self.path = None if path is None else Path(path)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as con:
                con.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key: str):
        if self.path is None:
            return None
        with sqlite3.connect(self.path) as con:
            row = con.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put(self, key: str, value: dict) -> None:
        if self.path is None:
            return
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value)))


class LLM:
    backend = "base"

    def __init__(self, model: str, cache_path: Path | str | None = DEFAULT_CACHE, cache_tag: str = ""):
        self.model = model
        self.cache = _Cache(cache_path)
        self.cache_tag = cache_tag
        self.usage = Usage()

    def _key(self, prompt, n, temperature, max_tokens) -> str:
        blob = json.dumps([self.backend, self.model, prompt, n, temperature, max_tokens, self.cache_tag])
        return hashlib.sha256(blob.encode()).hexdigest()

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7, max_tokens: int = 1000) -> list[str]:
        key = self._key(prompt, n, temperature, max_tokens)
        hit = self.cache.get(key)
        if hit is None:
            outputs, p_tok, c_tok = self._generate(prompt, n, temperature, max_tokens)
            hit = {"outputs": outputs, "prompt_tokens": p_tok, "completion_tokens": c_tok}
            self.cache.put(key, hit)
            self.usage.billed_calls += 1
            self.usage.billed_prompt_tokens += p_tok
            self.usage.billed_completion_tokens += c_tok
        self.usage.calls += 1
        self.usage.samples += len(hit["outputs"])
        self.usage.prompt_tokens += hit["prompt_tokens"]
        self.usage.completion_tokens += hit["completion_tokens"]
        return list(hit["outputs"])

    def _generate(self, prompt, n, temperature, max_tokens) -> tuple[list[str], int, int]:
        raise NotImplementedError


class OpenAIChat(LLM):
    backend = "openai"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 8, n_per_request: int = 20, request_kwargs: dict | None = None, **kw):
        super().__init__(model, **kw)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                             base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        self.max_retries = max_retries
        self.n_per_request = n_per_request          # 1 = one request per sample
        self.request_kwargs = request_kwargs or {}  # extra API options, e.g. reasoning_effort

    def _generate(self, prompt, n, temperature, max_tokens):
        import openai
        outputs, p_tok, c_tok = [], 0, 0
        remaining = n
        while remaining > 0:
            batch = min(remaining, self.n_per_request)
            for attempt in range(self.max_retries):
                try:
                    kwargs = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                                  temperature=temperature, max_tokens=max_tokens, **self.request_kwargs)
                    if batch > 1:
                        kwargs["n"] = batch
                    res = self.client.chat.completions.create(**kwargs)
                    break
                except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                        openai.InternalServerError):
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(min(60, 2 ** attempt))
            outputs += [c.message.content or "" for c in res.choices]
            p_tok += res.usage.prompt_tokens
            c_tok += res.usage.completion_tokens
            remaining -= batch
        return outputs, p_tok, c_tok


class HFChat(LLM):
    """Local transformers model (e.g. meta-llama/Llama-3.1-8B-Instruct on a Colab GPU)."""
    backend = "hf"

    def __init__(self, model: str, load_in_4bit: bool = False, **kw):
        super().__init__(model, **kw)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4: fp16
        else:
            dtype = torch.float32
        load_kw = {"device_map": "auto", "dtype": dtype}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        try:
            self.hf_model = AutoModelForCausalLM.from_pretrained(model, **load_kw)
        except TypeError:  # older transformers call it torch_dtype
            load_kw["torch_dtype"] = load_kw.pop("dtype")
            self.hf_model = AutoModelForCausalLM.from_pretrained(model, **load_kw)
        self.hf_model.eval()

    def _generate(self, prompt, n, temperature, max_tokens):
        text = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                            tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False).to(self.hf_model.device)
        with self.torch.no_grad():
            out = self.hf_model.generate(
                **enc, max_new_tokens=max_tokens, do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                num_return_sequences=n, pad_token_id=self.tok.eos_token_id)
        prompt_len = enc["input_ids"].shape[1]
        gen = out[:, prompt_len:]
        outputs = self.tok.batch_decode(gen, skip_special_tokens=True)
        c_tok = int((gen != self.tok.pad_token_id).sum()) if self.tok.pad_token_id is not None else gen.numel()
        return outputs, prompt_len, c_tok


class ScriptedLLM(LLM):
    """Answers with a Python function: fn(prompt, n) -> list[str]. For tests and dry runs."""
    backend = "scripted"

    def __init__(self, fn: Callable[[str, int], list[str]], model: str = "scripted", **kw):
        kw.setdefault("cache_path", None)
        super().__init__(model, **kw)
        self.fn = fn

    def _generate(self, prompt, n, temperature, max_tokens):
        outputs = self.fn(prompt, n)
        approx = lambda s: max(1, len(s) // 4)  # rough token estimate
        return outputs, approx(prompt), sum(approx(o) for o in outputs)


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_llm(backend: str, model: str, **kw) -> LLM:
    """backend: 'gemini' (free tier, key from Google AI Studio in GEMINI_API_KEY),
    'hf' (free, local GPU), or 'openai' (paid, OPENAI_API_KEY)."""
    if backend == "openai":
        return OpenAIChat(model, **kw)
    if backend == "gemini":
        kw.setdefault("api_key", os.environ.get("GEMINI_API_KEY"))
        kw.setdefault("base_url", GEMINI_BASE_URL)
        kw.setdefault("n_per_request", 1)   # one request per sample: safest on Gemini's endpoint
        return OpenAIChat(model, **kw)
    if backend == "hf":
        return HFChat(model, **kw)
    raise ValueError(f"unknown backend {backend!r} (use 'gemini', 'hf' or 'openai')")
