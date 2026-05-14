from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .models import SeedEvaluation


@dataclass
class SeedSamplerConfig:
    temperature: float = 1.0
    exploration_ratio: float = 0.2
    include_contrarian: bool = True


class SeedSampler:
    def __init__(self, config: SeedSamplerConfig | None = None, rng: random.Random | None = None):
        self.config = config or SeedSamplerConfig()
        self.rng = rng or random.Random()

    def sample(self, evals: list[SeedEvaluation], k: int = 1) -> list[SeedEvaluation]:
        pool = [ev for ev in evals if ev.valid]
        if self.config.include_contrarian:
            pool += [ev for ev in evals if ev.invert_recommended and ev not in pool]
        if not pool:
            return []
        elite = [ev for ev in pool if ev.status == "elite_candidate"] or pool
        explore = [ev for ev in pool if ev not in elite] or pool
        out = []
        for _ in range(k):
            source = explore if self.rng.random() < self.config.exploration_ratio else elite
            out.append(self._softmax_one(source))
        return out

    def _softmax_one(self, pool: list[SeedEvaluation]) -> SeedEvaluation:
        t = max(self.config.temperature, 1e-6)
        m = max(ev.seed_score for ev in pool)
        weights = [math.exp((ev.seed_score - m) / t) for ev in pool]
        return self.rng.choices(pool, weights=weights, k=1)[0]
