from __future__ import annotations

from dataclasses import dataclass

from .models import SeedEvaluation


@dataclass
class DedupConfig:
    ast_hash: bool = True
    max_per_category: int | None = None
    metric_sort: str = "seed_score"

    @classmethod
    def from_dict(cls, data: dict | None) -> "DedupConfig":
        data = data or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SeedDeduplicator:
    def __init__(self, config: DedupConfig | None = None):
        self.config = config or DedupConfig()

    def dedup(self, evals: list[SeedEvaluation]) -> list[SeedEvaluation]:
        ordered = sorted(evals, key=lambda ev: getattr(ev, self.config.metric_sort, ev.seed_score), reverse=True)
        seen_hashes: set[str] = set()
        per_cat: dict[str, int] = {}
        out: list[SeedEvaluation] = []
        for ev in ordered:
            if self.config.ast_hash and ev.canonical_hash and ev.canonical_hash in seen_hashes:
                ev.status = "duplicate_ast"
                ev.valid = False
                ev.reasons.append("duplicate_ast")
                continue
            cat = ev.seed.category or "general"
            if self.config.max_per_category is not None and per_cat.get(cat, 0) >= self.config.max_per_category:
                ev.status = "duplicate_category_quota"
                ev.valid = False
                ev.reasons.append("category_quota")
                continue
            out.append(ev)
            if ev.canonical_hash:
                seen_hashes.add(ev.canonical_hash)
            per_cat[cat] = per_cat.get(cat, 0) + 1
        return out
