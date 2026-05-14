from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import SeedEvaluation


@dataclass
class SeedScoringConfig:
    """User-configurable seed scoring.

    `metric_weights` names should match evaluation metrics. Metrics can come
    from any reward profile, so seed scoring follows the user's objective.
    """

    metric_weights: dict[str, float] = field(default_factory=lambda: {
        "reward": 1.0,
        "robustness": 0.2,
        "novelty": 0.2,
        "coverage": 0.1,
        "complexity": -0.05,
    })
    hard_filters: dict[str, Any] = field(default_factory=lambda: {
        "min_coverage": 0.6,
        "min_factor_std": 0.01,
    })
    contrarian_metric: str = "direction_acc"
    contrarian_threshold: float = 0.49

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SeedScoringConfig":
        data = data or {}
        return cls(
            metric_weights=dict(data.get("metric_weights", cls().metric_weights)),
            hard_filters=dict(data.get("hard_filters", cls().hard_filters)),
            contrarian_metric=str(data.get("contrarian_metric", "direction_acc")),
            contrarian_threshold=float(data.get("contrarian_threshold", 0.49)),
        )


class SeedScorer:
    def __init__(self, config: SeedScoringConfig | None = None):
        self.config = config or SeedScoringConfig()

    def score(self, ev: SeedEvaluation) -> SeedEvaluation:
        metrics = ev.metrics
        reasons = list(ev.reasons)
        for key, threshold in self.config.hard_filters.items():
            if key.startswith("min_"):
                metric = key[4:]
                if metric in metrics and metrics[metric] < float(threshold):
                    reasons.append(f"{metric}<{threshold}")
            elif key.startswith("max_"):
                metric = key[4:]
                if metric in metrics and metrics[metric] > float(threshold):
                    reasons.append(f"{metric}>{threshold}")
        score = 0.0
        for name, weight in self.config.metric_weights.items():
            if name == "complexity":
                value = metrics.get("expr_nodes", 0.0)
            else:
                value = metrics.get(name, metrics.get(f"composite/{name}", 0.0))
            score += float(weight) * float(value)
        ev.seed_score = score
        ev.reasons = reasons
        ev.valid = ev.valid and not reasons
        ev.status = "elite_candidate" if ev.valid and score > 0 else ("valid_low_score" if ev.valid else "filtered")
        if metrics.get(self.config.contrarian_metric, 1.0) < self.config.contrarian_threshold:
            ev.invert_recommended = True
        return ev
