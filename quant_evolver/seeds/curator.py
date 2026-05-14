from __future__ import annotations

from dataclasses import dataclass, field

from .dedup import DedupConfig, SeedDeduplicator
from .models import SeedEvaluation


@dataclass
class SeedLibrary:
    elite: list[SeedEvaluation]
    diverse: list[SeedEvaluation]
    contrarian: list[SeedEvaluation]
    rejected: list[SeedEvaluation]

    def to_dict(self):
        return {
            "elite": [x.to_dict() for x in self.elite],
            "diverse": [x.to_dict() for x in self.diverse],
            "contrarian": [x.to_dict() for x in self.contrarian],
            "rejected": [x.to_dict() for x in self.rejected],
        }


@dataclass
class SeedLibraryConfig:
    elite_top_k: int = 10
    diverse_top_k: int = 20
    contrarian_top_k: int = 10
    elite_min_score: float = 0.0
    dedup: DedupConfig = field(default_factory=DedupConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "SeedLibraryConfig":
        data = data or {}
        return cls(
            elite_top_k=int(data.get("elite_top_k", 10)),
            diverse_top_k=int(data.get("diverse_top_k", 20)),
            contrarian_top_k=int(data.get("contrarian_top_k", 10)),
            elite_min_score=float(data.get("elite_min_score", 0.0)),
            dedup=DedupConfig.from_dict(data.get("dedup")),
        )


class SeedLibraryBuilder:
    def __init__(self, config: SeedLibraryConfig | None = None):
        self.config = config or SeedLibraryConfig()

    def build(self, evals: list[SeedEvaluation]) -> SeedLibrary:
        valid = [ev for ev in evals if ev.valid]
        valid = SeedDeduplicator(self.config.dedup).dedup(valid)
        ordered = sorted(valid, key=lambda ev: ev.seed_score, reverse=True)
        elite = [ev for ev in ordered if ev.seed_score >= self.config.elite_min_score][: self.config.elite_top_k]
        diverse = ordered[: self.config.diverse_top_k]
        contrarian = [ev for ev in ordered if ev.invert_recommended][: self.config.contrarian_top_k]
        selected = set(id(x) for x in elite + diverse + contrarian)
        rejected = [ev for ev in evals if (not ev.valid) or id(ev) not in selected]
        return SeedLibrary(elite=elite, diverse=diverse, contrarian=contrarian, rejected=rejected)
