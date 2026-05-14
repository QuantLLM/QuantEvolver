from __future__ import annotations

from .base import RewardContext, RewardResult, finite_align


class ICReward:
    name = "ic"
    rank = False

    def evaluate(self, ctx: RewardContext) -> RewardResult:
        factor, fwd = finite_align(ctx.factor, ctx.future_return())
        if len(factor) < int(ctx.config.get("min_samples", 50)):
            return RewardResult(self.name, -1.0, {"samples": float(len(factor))}, False, "too_few_samples")
        value = float(factor.corr(fwd, method="spearman" if self.rank else "pearson"))
        return RewardResult(self.name, value, {self.name: value, "samples": float(len(factor))})


class RankICReward(ICReward):
    name = "rank_ic"
    rank = True
