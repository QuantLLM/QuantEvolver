from __future__ import annotations

from .direction import DirectionAccuracyReward
from .ic import ICReward, RankICReward
from .signal_return import SignalReturnReward

REGISTRY = {
    "direction_accuracy": DirectionAccuracyReward,
    "ic": ICReward,
    "rank_ic": RankICReward,
    "signal_return": SignalReturnReward,
}


def get_reward(name: str):
    if name not in REGISTRY:
        raise KeyError(f"Unknown reward {name!r}; available={sorted(REGISTRY)}")
    return REGISTRY[name]()
