from __future__ import annotations

from pathlib import Path
from typing import Iterable

from quant_evolver.utils.config import load_config, save_config
from .models import SeedCandidate, SeedEvaluation


def load_seed_candidates(path: str | Path) -> list[SeedCandidate]:
    data = load_config(path)
    seeds = data.get("seeds", data if isinstance(data, list) else [])
    return [SeedCandidate.from_dict(item, i) for i, item in enumerate(seeds)]


def save_seed_evaluations(evals: Iterable[SeedEvaluation], path: str | Path, header: dict | None = None) -> None:
    save_config({**(header or {}), "seeds": [ev.to_dict() for ev in evals]}, path)
