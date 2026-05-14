from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from quant_evolver.scenarios.schema import ScenarioConfig
from .templates import build_seed_generation_prompt, build_mutation_prompt


class StrongModelClient(Protocol):
    def complete(self, prompt: str) -> str: ...


@dataclass
class PromptPack:
    system_prompt: str
    seed_generation_prompt: str
    mutation_prompt_template: str


class PromptOptimizer:
    """Prompt optimizer entrypoint.

    If a strong-model client is provided, it can rewrite prompts. Otherwise we
    return deterministic templates, which makes the pipeline fully offline.
    """

    def __init__(self, client: StrongModelClient | None = None):
        self.client = client

    def build(self, scenario: ScenarioConfig, reward_config: dict, dsl_ops: list[str]) -> PromptPack:
        system = f"Quant factor discovery task.\n{scenario.to_prompt_context()}\nReward: {reward_config}"
        seed_prompt = build_seed_generation_prompt(scenario, reward_config, dsl_ops)
        mutation = build_mutation_prompt(scenario, "{seed_expr}", {"placeholder": "seed_metrics"})
        if self.client:
            system = self.client.complete("Rewrite this as a concise system prompt:\n" + system)
            seed_prompt = self.client.complete("Optimize this seed-generation prompt:\n" + seed_prompt)
            mutation = self.client.complete("Optimize this mutation prompt template:\n" + mutation)
        return PromptPack(system, seed_prompt, mutation)
