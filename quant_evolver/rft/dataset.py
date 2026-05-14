from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import RFTConfig
from .prompts import build_rft_messages
from .task_bank import StratifiedTaskSampler, TaskSpec, TaskBuildConfig, load_task_bank


def write_prompt_parquet(config: RFTConfig, path: str | Path, split: str = "train", n_tasks: int = 8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    task_bank = load_task_bank(config.task_bank_path) if config.task_bank_path else None
    if task_bank and task_bank.tasks:
        if n_tasks <= 0:
            selected = task_bank.tasks
        else:
            selected = StratifiedTaskSampler(task_bank.tasks, seed=config.task_sampler_seed + (0 if split == "train" else 100000)).sample(n_tasks)
        for task_id, task in enumerate(selected):
            rows.append(_row_from_task(config, task, task_id, split))
    else:
        # No-seed modes should still have real tasks: at minimum time_split
        # variation.  Otherwise all prompt rows are identical and reward is
        # always computed on the same interval, which is not a fair comparison
        # with seeded task-bank training.
        time_splits = TaskBuildConfig().time_splits
        for task_id in range(n_tasks):
            ts = time_splits[task_id % len(time_splits)]
            rows.append({
                "data_source": f"quant_evolver/{'free_code' if config.mode == 'free_code' else 'no_seed'}/{ts.name}",
                "prompt": build_rft_messages(
                    config.mode,
                    config.seed_expr,
                    config.bar_minutes,
                    family="free_code" if config.mode == "free_code" else "no_seed",
                    time_split=ts.name,
                    start_date=ts.start_date,
                    end_date=ts.end_date,
                    reward_kind=config.reward_kind,
                    horizon_bars=config.horizon_bars,
                    scenario=(
                        "from-scratch Python factor; no seed expression and no DSL is provided"
                        if config.mode == "free_code"
                        else (
                            "from-scratch multi-asset cross-sectional RankIC factor; no seed expression is provided"
                            if config.reward_kind == "cross_sectional_rankic"
                            else "from-scratch single-asset next-bar direction; no seed expression is provided"
                        )
                    ),
                ),
                "reward_model": {"ground_truth": [], "style": "rule"},
                "extra_info": {
                    "index": task_id,
                    "task_id": f"{'free_code' if config.mode == 'free_code' else 'no_seed'}__{ts.name}__{task_id}",
                    "seed_id": "",
                    "seed_name": "",
                    "seed_expr": "",
                    "family": "free_code" if config.mode == "free_code" else "no_seed",
                    "time_split": ts.name,
                    "start_date": ts.start_date,
                    "end_date": ts.end_date,
                    "split": split,
                    "bar_minutes": config.bar_minutes,
                    "reward_kind": config.reward_kind,
                    "horizon_bars": config.horizon_bars,
                },
            })
    pd.DataFrame(rows).to_parquet(path)
    return path


def _row_from_task(config: RFTConfig, task: TaskSpec, task_id: int, split: str) -> dict:
    messages = build_rft_messages(
        config.mode,
        task.seed_expr,
        task.bar_minutes or config.bar_minutes,
        seed_name=task.seed_name,
        family=task.family,
        time_split=task.time_split,
        start_date=task.start_date,
        end_date=task.end_date,
        scenario=task.scenario,
        mutation_hints=task.mutation_hints,
        reward_kind=config.reward_kind,
        horizon_bars=config.horizon_bars,
    )
    return {
        "data_source": f"quant_evolver/{task.family}/{task.time_split}",
        "prompt": messages,
        "reward_model": {"ground_truth": [], "style": "rule"},
        "extra_info": {
            "index": task_id,
            "task_id": task.id,
            "seed_id": task.seed_id,
            "seed_name": task.seed_name,
            "seed_expr": task.seed_expr,
            "family": task.family,
            "time_split": task.time_split,
            "start_date": task.start_date,
            "end_date": task.end_date,
            "bar_minutes": task.bar_minutes or config.bar_minutes,
            "reward_kind": config.reward_kind,
            "horizon_bars": config.horizon_bars,
            "split": split,
        },
    }
