from __future__ import annotations

import numpy as np

from .base import RewardContext, RewardResult, finite_align


class DirectionAccuracyReward:
    name = "direction_accuracy"

    def evaluate(self, ctx: RewardContext) -> RewardResult:
        fwd = ctx.future_return()
        factor, fwd = finite_align(ctx.factor, fwd)
        if len(factor) < int(ctx.config.get("min_samples", 50)):
            return RewardResult(self.name, -100.0, {"samples": float(len(factor))}, False, "too_few_samples")
        pred = np.sign(factor.to_numpy(dtype=float))
        label = np.sign(fwd.to_numpy(dtype=float))
        mask = pred != 0
        if mask.sum() == 0:
            return RewardResult(self.name, -100.0, {"coverage": 0.0}, False, "zero_signal")
        acc = float(np.mean(pred[mask] == label[mask]))
        coverage = float(mask.mean())
        score = (acc - 0.5) * 200.0
        return RewardResult(self.name, score, {"direction_acc": acc, "coverage": coverage, "samples": float(mask.sum())})
