from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Protocol

import yaml

from .schema import ScenarioConfig


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.0, max_tokens: int = 2048) -> str: ...


SCENARIO_REFINER_SYSTEM_PROMPT = """You are a quantitative research assistant.
Rewrite a user's raw alpha-factor mining request into a structured scenario config.

Return ONLY valid YAML with exactly these fields:
- raw: original user request
- market: broad market or asset class, e.g. equity, futures, crypto, fx, generic
- universe: list of tradable symbols or universe names; use [ASSET] if unspecified
- frequency: bar frequency, e.g. 1min, 5min, 1h, 1d
- horizon: integer prediction horizon in bars
- target: prediction target, e.g. direction, ic, rank_ic, return
- factor_type: e.g. single_asset_timing, cross_sectional, event_driven
- fields: available OHLCV or market fields
- constraints: mapping of additional constraints
- preferred_signals: list of preferred signal families

Do not invent private data paths. Use conservative defaults when details are missing.
"""


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.search(r"```(?:yaml|yml|json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


class ScenarioRefiner:
    """LLM-based scenario refiner for turning raw user requests into structured configs."""

    def __init__(self, client: ChatClient | None = None, *, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        if client is None:
            from quant_evolver.llm import OpenAICompatibleClient

            client = OpenAICompatibleClient(model=model or "gpt-4o-mini", base_url=base_url, api_key=api_key)
        self.client = client

    def refine(self, raw: str, overrides: dict | None = None) -> ScenarioConfig:
        content = self.client.chat(
            [
                {"role": "system", "content": SCENARIO_REFINER_SYSTEM_PROMPT},
                {"role": "user", "content": raw},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        cfg = self._parse_response(content, raw)
        if overrides:
            for k, v in overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg

    @staticmethod
    def _parse_response(content: str, raw: str) -> ScenarioConfig:
        text = _strip_fence(content)
        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Scenario refiner output must be a YAML/JSON object")
        data.setdefault("raw", raw)
        cfg = ScenarioConfig.from_dict(data)
        if not cfg.raw:
            cfg.raw = raw
        return cfg

    @staticmethod
    def to_dict(cfg: ScenarioConfig) -> dict:
        return asdict(cfg)
