from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import random

from quant_evolver.seeds.models import SeedCandidate, SeedEvaluation
from quant_evolver.seeds.library import load_seed_candidates
from quant_evolver.utils.config import load_config, save_config


@dataclass
class TimeSplit:
    name: str
    start_date: str
    end_date: str
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimeSplit":
        return cls(
            name=str(data["name"]),
            start_date=str(data["start_date"]),
            end_date=str(data["end_date"]),
            weight=float(data.get("weight", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "start_date": self.start_date, "end_date": self.end_date, "weight": self.weight}


@dataclass
class TaskSpec:
    id: str
    seed_id: str
    seed_expr: str
    seed_name: str = ""
    family: str = "general"
    time_split: str = "full"
    start_date: str = "2026-02-23"
    end_date: str = "2026-03-25"
    bar_minutes: int = 5
    scenario: str = ""
    mutation_hints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            id=str(data["id"]),
            seed_id=str(data["seed_id"]),
            seed_expr=str(data["seed_expr"]),
            seed_name=str(data.get("seed_name", "")),
            family=str(data.get("family", data.get("category", "general"))),
            time_split=str(data.get("time_split", "full")),
            start_date=str(data.get("start_date", "2026-02-23")),
            end_date=str(data.get("end_date", "2026-03-25")),
            bar_minutes=int(data.get("bar_minutes", 5)),
            scenario=str(data.get("scenario", "")),
            mutation_hints=list(data.get("mutation_hints", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed_id": self.seed_id,
            "seed_name": self.seed_name,
            "family": self.family,
            "seed_expr": self.seed_expr,
            "time_split": self.time_split,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "bar_minutes": self.bar_minutes,
            "scenario": self.scenario,
            "mutation_hints": self.mutation_hints,
            "metadata": self.metadata,
        }


@dataclass
class TaskBank:
    tasks: list[TaskSpec]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskBank":
        return cls(tasks=[TaskSpec.from_dict(x) for x in data.get("tasks", [])], metadata=dict(data.get("metadata", {})))

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata, "tasks": [t.to_dict() for t in self.tasks]}


def load_task_bank(path: str | Path) -> TaskBank:
    return TaskBank.from_dict(load_config(path))


def save_task_bank(bank: TaskBank, path: str | Path) -> None:
    save_config(bank.to_dict(), path)


def _load_seed_evals(path: str | Path) -> list[SeedEvaluation]:
    data = load_config(path)
    raw = data.get("seeds", data if isinstance(data, list) else [])
    out: list[SeedEvaluation] = []
    for i, item in enumerate(raw):
        seed = SeedCandidate.from_dict(item, i)
        out.append(SeedEvaluation(
            seed=seed,
            valid=bool(item.get("valid", True)),
            status=str(item.get("status", "loaded")),
            seed_score=float(item.get("seed_score", 0.0)),
            metrics=dict(item.get("metrics", {})),
            reasons=list(item.get("reasons", [])),
            canonical_hash=str(item.get("canonical_hash", "")),
            invert_recommended=bool(item.get("invert_recommended", False)),
        ))
    return out


@dataclass
class TaskBuildConfig:
    max_seeds: int = 64
    include_invalid: bool = False
    sort_metric: str = "seed_score"
    bar_minutes: int = 5
    scenario: str = ""
    time_splits: list[TimeSplit] = field(default_factory=lambda: [
        TimeSplit("q1", "2026-02-23", "2026-03-03"),
        TimeSplit("q2", "2026-03-03", "2026-03-10"),
        TimeSplit("q3", "2026-03-10", "2026-03-18"),
        TimeSplit("q4", "2026-03-18", "2026-03-25"),
    ])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TaskBuildConfig":
        data = data or {}
        splits = data.get("time_splits")
        return cls(
            max_seeds=int(data.get("max_seeds", 64)),
            include_invalid=bool(data.get("include_invalid", False)),
            sort_metric=str(data.get("sort_metric", "seed_score")),
            bar_minutes=int(data.get("bar_minutes", 5)),
            scenario=str(data.get("scenario", "")),
            time_splits=[TimeSplit.from_dict(x) for x in splits] if splits else cls().time_splits,
        )


class TaskBankBuilder:
    def __init__(self, config: TaskBuildConfig | None = None):
        self.config = config or TaskBuildConfig()

    def build_from_seed_evaluations(self, evals: list[SeedEvaluation]) -> TaskBank:
        seeds = [ev for ev in evals if self.config.include_invalid or ev.valid]
        def key(ev: SeedEvaluation):
            if self.config.sort_metric == "seed_score":
                return ev.seed_score
            return float(ev.metrics.get(self.config.sort_metric, ev.seed_score))
        ordered = sorted(seeds, key=key, reverse=True)
        seeds = []
        seen: set[str] = set()
        for ev in ordered:
            dedup_key = ev.canonical_hash or " ".join(ev.seed.expr.split())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            seeds.append(ev)
            if len(seeds) >= self.config.max_seeds:
                break
        tasks: list[TaskSpec] = []
        for ev in seeds:
            seed = ev.seed
            for split in self.config.time_splits:
                tid = f"{seed.id}__{split.name}"
                tasks.append(TaskSpec(
                    id=tid,
                    seed_id=seed.id,
                    seed_name=seed.name,
                    family=seed.category or "general",
                    seed_expr=seed.expr,
                    time_split=split.name,
                    start_date=split.start_date,
                    end_date=split.end_date,
                    bar_minutes=self.config.bar_minutes,
                    scenario=self.config.scenario,
                    mutation_hints=list(seed.metadata.get("mutation_hints", [])),
                    metadata={"seed_score": ev.seed_score, "seed_metrics": ev.metrics, "canonical_hash": ev.canonical_hash},
                ))
        return TaskBank(tasks=tasks, metadata={"max_seeds": self.config.max_seeds, "num_tasks": len(tasks), "time_splits": [s.to_dict() for s in self.config.time_splits]})


def build_task_bank_from_file(seed_eval_path: str | Path, config: TaskBuildConfig | None = None) -> TaskBank:
    return TaskBankBuilder(config).build_from_seed_evaluations(_load_seed_evals(seed_eval_path))


class StratifiedTaskSampler:
    def __init__(self, tasks: list[TaskSpec], seed: int = 0):
        self.tasks = list(tasks)
        self.rng = random.Random(seed)
        self.by_family: dict[str, list[TaskSpec]] = {}
        for t in self.tasks:
            self.by_family.setdefault(t.family or "general", []).append(t)

    def sample(self, n: int) -> list[TaskSpec]:
        if not self.tasks or n <= 0:
            return []
        families = list(self.by_family.keys())
        out: list[TaskSpec] = []
        while len(out) < n:
            self.rng.shuffle(families)
            for fam in families:
                out.append(self.rng.choice(self.by_family[fam]))
                if len(out) >= n:
                    break
        self.rng.shuffle(out)
        return out
