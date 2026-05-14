from __future__ import annotations

from dataclasses import dataclass

from quant_evolver.dsl import compile_expr
from quant_evolver.evaluation import BacktestConfig, run_factor_backtrader
from quant_evolver.rewards import RewardContext, RewardProfile
from .models import SeedEvaluation


@dataclass
class SeedEvaluatorConfig:
    backtest: BacktestConfig
    reward_profile: RewardProfile

    @classmethod
    def from_dict(cls, data: dict | None, reward_cfg: dict | None = None) -> "SeedEvaluatorConfig":
        data = data or {}
        return cls(
            backtest=BacktestConfig.from_dict(data.get("backtest")),
            reward_profile=RewardProfile.from_dict(reward_cfg or data.get("reward")),
        )


class SeedEvaluator:
    def __init__(self, config: SeedEvaluatorConfig):
        self.config = config

    def evaluate(self, ev: SeedEvaluation) -> SeedEvaluation:
        if not ev.valid:
            return ev
        try:
            code = compile_expr(ev.seed.expr)
            run = run_factor_backtrader(code, self.config.backtest)
            reward = self.config.reward_profile.evaluate(
                RewardContext(
                    factor=run.factor,
                    prices=run.prices,
                    horizon=self.config.backtest.bar_minutes and self.config.reward_profile.components.get(self.config.reward_profile.primary, {}).get("horizon", 1),
                    config={},
                )
            )
            ev.metrics.update(run.metrics)
            ev.metrics.update(reward.metrics)
            ev.metrics["reward"] = float(reward.value)
            ev.valid = ev.valid and reward.passed
            if not reward.passed and reward.reason:
                ev.reasons.append(reward.reason)
            # convenience aliases for common seed scoring filters
            if "direction_accuracy/direction_acc" in reward.metrics:
                ev.metrics["direction_acc"] = reward.metrics["direction_accuracy/direction_acc"]
            if "direction_accuracy/coverage" in reward.metrics:
                ev.metrics["coverage"] = reward.metrics["direction_accuracy/coverage"]
            ev.status = "evaluated" if ev.valid else "eval_filtered"
        except Exception as exc:
            ev.valid = False
            ev.status = "eval_error"
            ev.reasons.append(f"eval_error={exc}")
        return ev
