from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd


@dataclass
class RewardResult:
    name: str
    value: float
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    reason: str = ""


@dataclass
class RewardContext:
    factor: pd.Series
    prices: pd.DataFrame
    horizon: int = 1
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def close(self) -> pd.Series:
        if "close" not in self.prices:
            raise KeyError("prices must contain close column")
        return self.prices["close"]

    def future_return(self) -> pd.Series:
        return self.close.shift(-self.horizon) / self.close - 1.0


def finite_align(*series: pd.Series) -> list[pd.Series]:
    df = pd.concat(series, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    return [df.iloc[:, i] for i in range(df.shape[1])]


class Reward(Protocol):
    name: str
    def evaluate(self, ctx: RewardContext) -> RewardResult: ...
