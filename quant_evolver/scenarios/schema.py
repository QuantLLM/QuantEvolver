from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ScenarioConfig:
    raw: str = ""
    market: str = "generic"
    universe: list[str] = field(default_factory=lambda: ["ASSET"])
    frequency: str = "5min"
    horizon: int = 1
    target: str = "direction"
    factor_type: str = "single_asset_timing"
    fields: list[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume"])
    constraints: dict[str, Any] = field(default_factory=dict)
    preferred_signals: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ScenarioConfig":
        data = dict(data or {})
        for key in ("universe", "fields", "preferred_signals"):
            value = data.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                data[key] = [x.strip() for x in value.split(",") if x.strip()]
            elif not isinstance(value, list):
                data[key] = list(value) if isinstance(value, tuple) else [str(value)]
        if "horizon" in data:
            try:
                data["horizon"] = int(data["horizon"])
            except Exception:
                data["horizon"] = 1
        if not isinstance(data.get("constraints", {}), dict):
            data["constraints"] = {"notes": data.get("constraints")}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt_context(self) -> str:
        return (
            f"Market: {self.market}\n"
            f"Universe: {', '.join(self.universe)}\n"
            f"Frequency: {self.frequency}\n"
            f"Prediction horizon: {self.horizon} bar(s)\n"
            f"Target: {self.target}\n"
            f"Factor type: {self.factor_type}\n"
            f"Available fields: {', '.join(self.fields)}\n"
            f"Constraints: {self.constraints}\n"
            f"Preferred signal families: {', '.join(self.preferred_signals)}"
        )
