from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RFTBackendName(str, Enum):
    """Supported RFT execution backends."""

    LOCAL_SMOKE = "local_smoke"
    PURE_VERL = "pure_verl"


@dataclass
class RFTConfig:
    backend: RFTBackendName = RFTBackendName.PURE_VERL
    mode: str = "seeded_ocd"
    seed_expr: str | None = None
    seed_index: int = 0
    bar_minutes: int = 5
    reward_kind: str = "direction"
    horizon_bars: int = 1
    universe_symbols: list[str] = field(default_factory=list)
    single_asset_symbol: str = "ASSET"
    data_dir: str = "data"
    data_file_template: str = "{symbol_safe}_1m.parquet"
    min_symbols_per_time: int = 8
    min_times: int = 20

    experiment_name: str = "quantevolver_seeded_ocd_smoke"
    project_name: str = "quant_evolver"
    output_dir: str = "runs/generated_factors"
    run_root: str = "runs/rft"

    model_path: str = "path/to/base-model"
    n_gpus: int = 4
    total_epochs: int = 2
    train_batch_size: int = 8
    rollout_n: int = 8
    val_n: int = 2
    save_freq: int = 999999
    test_freq: int = 5
    max_prompt_length: int = 3500
    max_response_length: int = 2000
    max_model_len: int = 6000
    gpu_memory_utilization: float = 0.70
    temperature: float = 0.9
    reward_workers: int = 16
    free_code_eval_timeout_s: int = 120
    task_bank_path: str | None = None
    train_prompt_rows: int = 0
    val_prompt_rows: int = 8
    task_sampler_seed: int = 0
    novelty_archive_path: str | None = None
    novelty_penalty: float = 0.25
    relative_reward: bool = False
    improvement_bonus: float = 0.5
    underperform_penalty: float = 0.0
    diversity_reward: bool = False
    exact_repeat_penalty: float = 0.15
    family_archive_path: str | None = None
    family_free_quota: int = 8
    family_low_quality_threshold: float = 0.08
    family_repeat_penalty: float = 0.02
    family_good_new_threshold: float = 0.10
    family_new_bonus: float = 0.02
    residual_response_reward: bool = False
    residual_archive_path: str | None = None
    residual_archive_top_k: int = 128
    residual_good_score_threshold: float = 0.16
    residual_bonus: float = 0.04
    residual_corr_penalty: float = 0.03
    residual_corr_threshold: float = 0.92
    residual_min_samples: int = 200
    residual_record_full: bool = True
    residual_full_start_date: str = "2026-02-23"
    residual_full_end_date: str = "2026-03-25"
    xsec_behavior_diversity: bool = False
    xsec_behavior_archive_path: str | None = None
    xsec_behavior_archive_top_k: int = 128
    xsec_behavior_good_score_threshold: float = 0.12
    xsec_behavior_corr_threshold: float = 0.85
    xsec_behavior_corr_penalty: float = 0.08
    xsec_behavior_low_corr_bonus_threshold: float = 0.50
    xsec_behavior_low_corr_bonus: float = 0.02
    xsec_behavior_min_shared_times: int = 200
    xsec_behavior_record_full: bool = True
    enable_swanlab: bool = False
    lr: str = "1e-6"

    conda_sh: str = "~/miniconda3/etc/profile.d/conda.sh"
    train_conda_env: str = "quantevolver-rft"

    extra_env: dict[str, str] = field(default_factory=dict)
    extra_overrides: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "RFTConfig":
        raw = dict(raw or {})
        if "backend" in raw and not isinstance(raw["backend"], RFTBackendName):
            raw["backend"] = RFTBackendName(raw["backend"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["backend"] = self.backend.value
        return d

    @property
    def run_root_path(self) -> Path:
        return Path(self.run_root).expanduser().resolve()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()
