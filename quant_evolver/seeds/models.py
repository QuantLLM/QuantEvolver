from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SeedCandidate:
    id: str
    expr: str
    name: str = ""
    intuition: str = ""
    category: str = "general"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], idx: int = 0) -> "SeedCandidate":
        return cls(
            id=str(data.get("id", f"seed_{idx:04d}")),
            expr=str(data["expr"]),
            name=str(data.get("name", data.get("id", f"seed_{idx:04d}"))),
            intuition=str(data.get("intuition", "")),
            category=str(data.get("category", "general")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "expr": self.expr,
            "intuition": self.intuition,
            "category": self.category,
            "metadata": self.metadata,
        }


@dataclass
class SeedEvaluation:
    seed: SeedCandidate
    valid: bool
    status: str
    seed_score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    canonical_hash: str = ""
    invert_recommended: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.seed.to_dict(),
            "valid": self.valid,
            "status": self.status,
            "seed_score": self.seed_score,
            "metrics": self.metrics,
            "reasons": self.reasons,
            "canonical_hash": self.canonical_hash,
            "invert_recommended": self.invert_recommended,
        }
