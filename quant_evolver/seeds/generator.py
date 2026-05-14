from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import yaml

from quant_evolver.dsl import ALL_OPS
from quant_evolver.prompts.templates import build_seed_generation_prompt
from quant_evolver.scenarios.schema import ScenarioConfig
from .models import SeedCandidate


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7, max_tokens: int = 4096) -> str: ...


@dataclass
class SeedGeneratorConfig:
    model: str = "gpt-4o-mini"
    num_candidates: int = 50
    batch_size: int = 32
    temperature: float = 0.8
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: dict | None) -> "SeedGeneratorConfig":
        data = data or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StrongModelSeedGenerator:
    def __init__(self, client: ChatClient, config: SeedGeneratorConfig | None = None):
        self.client = client
        self.config = config or SeedGeneratorConfig()

    def generate(self, scenario: ScenarioConfig, reward_config: dict, dsl_ops: list[str] | None = None) -> list[SeedCandidate]:
        if self.config.num_candidates > self.config.batch_size > 0:
            all_seeds: list[SeedCandidate] = []
            remaining = self.config.num_candidates
            batch_idx = 0
            while remaining > 0:
                n = min(self.config.batch_size, remaining)
                sub_cfg = SeedGeneratorConfig(
                    model=self.config.model,
                    num_candidates=n,
                    batch_size=n,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                sub = StrongModelSeedGenerator(self.client, sub_cfg).generate(scenario, reward_config, dsl_ops)
                for seed in sub:
                    seed.metadata.setdefault("generation_batch", batch_idx)
                    seed.id = f"seed_{len(all_seeds):04d}"
                all_seeds.extend(sub)
                remaining -= n
                batch_idx += 1
            return all_seeds
        prompt = build_seed_generation_prompt(
            scenario=scenario,
            reward_config=reward_config,
            dsl_ops=dsl_ops or sorted(ALL_OPS),
            n=self.config.num_candidates,
        )
        prompt += "\nReturn ONLY YAML. Do not wrap in markdown fences."
        content = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return parse_seed_yaml(content)



@dataclass
class SeedRepairConfig:
    max_repairs: int = 1
    temperature: float = 0.2
    max_tokens: int = 2048


class SeedRepairer:
    def __init__(self, client: ChatClient, config: SeedRepairConfig | None = None):
        self.client = client
        self.config = config or SeedRepairConfig()

    def repair(self, seed: SeedCandidate, error: str, scenario: ScenarioConfig, reward_config: dict) -> SeedCandidate:
        prompt = f"""You are repairing a quantitative factor DSL expression.

Scenario:
{scenario.to_prompt_context()}

Reward configuration:
{reward_config}

Original seed:
name: {seed.name}
expr: {seed.expr}
intuition: {seed.intuition}
category: {seed.category}

Validation/backtest error:
{error}

Return ONLY YAML for one corrected seed with fields: name, expr, intuition, category.
Follow DSL syntax strictly:
- Field accessors require integer windows: close(60), volume(120), returns(60).
- returns takes integer only.
- Arrays must be converted to scalar for scalar ops using last/ts_mean/ts_std/ts_skew/ts_kurt/corr/cov.
- ema_arr(arr, span) returns a scalar EMA value.
"""
        content = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        seeds = parse_seed_yaml(content)
        repaired = seeds[0]
        repaired.id = seed.id + "_repair"
        repaired.metadata.update({"repaired_from": seed.id, "repair_error": error})
        return repaired

def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:yaml|yml)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def parse_seed_yaml(text: str) -> list[SeedCandidate]:
    raw = _strip_fence(text)
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return _parse_seed_loose(raw)
    if isinstance(data, dict):
        seeds = data.get("seeds", [])
    elif isinstance(data, list):
        seeds = data
    else:
        raise ValueError("Seed generator output is not YAML list/dict")
    out: list[SeedCandidate] = []
    for i, item in enumerate(seeds):
        if not isinstance(item, dict) or "expr" not in item:
            continue
        out.append(SeedCandidate.from_dict(item, i))
    if not out:
        raise ValueError("No valid seed items found in model output")
    return out


def _parse_seed_loose(text: str) -> list[SeedCandidate]:
    """Best-effort parser for model YAML with unquoted colons in text fields."""
    items: list[dict] = []
    cur: dict[str, str] | None = None
    key_re = re.compile(r"^\s*(?:-\s*)?(name|expr|intuition|category)\s*:\s*(.*)\s*$")
    for line in text.splitlines():
        m = key_re.match(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
        if line.lstrip().startswith("-") or (key == "name" and cur and "expr" in cur):
            if cur:
                items.append(cur)
            cur = {}
        if cur is None:
            cur = {}
        cur[key] = val
    if cur:
        items.append(cur)
    out = [SeedCandidate.from_dict(x, i) for i, x in enumerate(items) if x.get("expr")]
    if not out:
        raise ValueError("No valid seed items found in model output")
    return out
