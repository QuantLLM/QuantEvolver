from __future__ import annotations

import numpy as np

from .base import RewardContext, RewardResult, finite_align


class SignalReturnReward:
    name = "signal_return"

    def evaluate(self, ctx: RewardContext) -> RewardResult:
        factor, fwd = finite_align(ctx.factor, ctx.future_return())
        pos = np.sign(factor.to_numpy(dtype=float))
        pnl = pos * fwd.to_numpy(dtype=float)
        if len(pnl) < int(ctx.config.get("min_samples", 50)):
            return RewardResult(self.name, -1.0, {"samples": float(len(pnl))}, False, "too_few_samples")
        mean = float(np.mean(pnl))
        std = float(np.std(pnl) + 1e-12)
        sharpe_like = mean / std * np.sqrt(float(ctx.config.get("annualization", 365 * 24 * 12)))
        return RewardResult(self.name, sharpe_like, {"mean_pnl": mean, "pnl_std": std, "sharpe_like": float(sharpe_like)})
