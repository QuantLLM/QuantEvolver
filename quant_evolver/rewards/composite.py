from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import RewardContext, RewardResult
from .registry import get_reward


@dataclass
class RewardProfile:
    primary: str
    components: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewardProfile":
        data = data or {}
        return cls(primary=data.get("primary", "direction_accuracy"), components=data.get("components", {}))

    def evaluate(self, ctx: RewardContext) -> RewardResult:
        total = 0.0
        metrics: dict[str, float] = {}
        passed = True
        reasons = []
        components = self.components or {self.primary: {"weight": 1.0}}
        for name, cfg in components.items():
            weight = float(cfg.get("weight", 1.0 if name == self.primary else 0.0))
            subctx = RewardContext(factor=ctx.factor, prices=ctx.prices, horizon=int(cfg.get("horizon", ctx.horizon)), config=cfg)
            res = get_reward(name).evaluate(subctx)
            total += weight * float(res.value)
            metrics.update({f"{name}/{k}": v for k, v in res.metrics.items()})
            metrics[f"{name}/value"] = float(res.value)
            passed = passed and res.passed
            if res.reason:
                reasons.append(f"{name}:{res.reason}")
        return RewardResult("composite", total, metrics, passed, ";".join(reasons))
