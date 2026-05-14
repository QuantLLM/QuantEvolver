from __future__ import annotations

from dataclasses import dataclass

from quant_evolver.dsl import ALL_OPS, analyse_expr, compile_expr
from .models import SeedCandidate, SeedEvaluation


@dataclass
class SeedValidationConfig:
    max_depth: int = 18
    max_nodes: int = 120
    max_lookback: int = 2000

    @classmethod
    def from_dict(cls, data: dict | None) -> "SeedValidationConfig":
        data = data or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SeedValidator:
    def __init__(self, config: SeedValidationConfig | None = None):
        self.config = config or SeedValidationConfig()

    def validate(self, seed: SeedCandidate) -> SeedEvaluation:
        reasons: list[str] = []
        stats = analyse_expr(seed.expr)
        if not stats.valid_syntax:
            return SeedEvaluation(seed, False, "invalid_syntax", reasons=[stats.error])
        unknown_ops = [op for op in stats.operators if op not in ALL_OPS and op not in {"Add", "Sub", "Mult", "Div"}]
        if unknown_ops:
            reasons.append(f"unknown_ops={unknown_ops}")
        if stats.depth > self.config.max_depth:
            reasons.append(f"depth>{self.config.max_depth}")
        if stats.nodes > self.config.max_nodes:
            reasons.append(f"nodes>{self.config.max_nodes}")
        try:
            compile_expr(seed.expr)
        except Exception as exc:
            reasons.append(f"compile_error={exc}")
        valid = not reasons
        return SeedEvaluation(
            seed=seed,
            valid=valid,
            status="static_valid" if valid else "static_invalid",
            reasons=reasons,
            canonical_hash=stats.ast_hash,
            metrics={"expr_depth": float(stats.depth), "expr_nodes": float(stats.nodes)},
        )
