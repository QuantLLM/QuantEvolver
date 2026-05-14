from __future__ import annotations

import argparse
from pathlib import Path

import ray

from quant_evolver.rft.config import RFTConfig
from quant_evolver.rft.dataset import write_prompt_parquet
from quant_evolver.rft.reward_bridge import QuantRewardBridge
from quant_evolver.utils.config import load_config


def build_verl_config(rft: RFTConfig):
    from omegaconf import OmegaConf

    base_path = Path(__file__).resolve().parents[2] / "configs" / "verl_ppo_trainer_base.yaml"
    base = OmegaConf.load(base_path)
    data_dir = rft.run_root_path / "pure_verl_data" / rft.experiment_name
    train_rows = rft.train_prompt_rows if rft.train_prompt_rows > 0 else rft.train_batch_size
    val_rows = rft.val_prompt_rows if rft.val_prompt_rows > 0 else rft.train_batch_size
    train_file = write_prompt_parquet(rft, data_dir / "train.parquet", "train", n_tasks=train_rows)
    val_file = write_prompt_parquet(rft, data_dir / "val.parquet", "val", n_tasks=val_rows)
    loggers = ["console", "swanlab"] if rft.enable_swanlab else ["console"]
    overrides = OmegaConf.create({
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
        "data": {
            "train_files": [str(train_file)],
            "val_files": [str(val_file)],
            "train_batch_size": rft.train_batch_size,
            "val_batch_size": rft.train_batch_size,
            "max_prompt_length": rft.max_prompt_length,
            "max_response_length": rft.max_response_length,
            "return_raw_chat": False,
            "filter_overlong_prompts": True,
            "truncation": "error",
            "dataloader_num_workers": 0,
            "validation_shuffle": False,
            "custom_cls": {"path": str(Path(__file__).resolve().parent / "no_think_dataset.py"), "name": "NoThinkRLHFDataset"},
        },
        "actor_rollout_ref": {
            "model": {
                "path": rft.model_path,
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
                "trust_remote_code": True,
            },
            "rollout": {
                "name": "vllm",
                "mode": "sync",
                "n": rft.rollout_n,
                "temperature": rft.temperature,
                "prompt_length": rft.max_prompt_length,
                "response_length": rft.max_response_length,
                "max_model_len": rft.max_model_len,
                "gpu_memory_utilization": rft.gpu_memory_utilization,
                "tensor_model_parallel_size": 1,
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 8000,
                "val_kwargs": {"n": rft.val_n, "do_sample": False},
            },
            "actor": {
                "ppo_mini_batch_size": rft.train_batch_size,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_kl_loss": True,
                "kl_loss_coef": 0.001,
                "kl_loss_type": "low_var_kl",
                "entropy_coeff": 0,
                "optim": {"lr": float(rft.lr)},
                "fsdp_config": {"param_offload": False, "optimizer_offload": True},
                "ppo_max_token_len_per_gpu": 8000,
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "log_prob_max_token_len_per_gpu": 8000,
                "fsdp_config": {"param_offload": True},
            },
        },
        "critic": {
            "strategy": "fsdp",
            "ppo_max_token_len_per_gpu": 8000,
            "forward_max_token_len_per_gpu": 8000,
        },
        "reward_model": {"enable": False, "launch_reward_fn_async": False},
        "trainer": {
            "project_name": rft.project_name,
            "experiment_name": rft.experiment_name,
            "logger": loggers,
            "nnodes": 1,
            "n_gpus_per_node": rft.n_gpus,
            "total_epochs": rft.total_epochs,
            "save_freq": rft.save_freq,
            "test_freq": rft.test_freq,
            "val_before_train": False,
            "critic_warmup": 0,
            "default_local_dir": str(rft.run_root_path / "pure_verl" / "checkpoints" / rft.experiment_name),
            "validation_data_dir": str(rft.run_root_path / "pure_verl" / "experiments" / rft.experiment_name / "validation_log"),
            "rollout_data_dir": str(rft.run_root_path / "pure_verl" / "experiments" / rft.experiment_name / "rollout_log"),
        },
        "ray_init": {"num_cpus": None},
    })
    cfg = OmegaConf.merge(base, overrides)
    OmegaConf.resolve(cfg)
    return cfg


def run(rft: RFTConfig):
    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.utils.fs import copy_to_local
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

    cfg = build_verl_config(rft)
    print(OmegaConf.to_yaml(cfg))
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_USE_V1": "1"}}, num_cpus=cfg.ray_init.num_cpus)

    local_path = copy_to_local(cfg.actor_rollout_ref.model.path, use_shm=cfg.actor_rollout_ref.model.get("use_shm", False))
    tokenizer = hf_tokenizer(local_path, trust_remote_code=cfg.data.get("trust_remote_code", True))
    processor = hf_processor(local_path, trust_remote_code=cfg.data.get("trust_remote_code", True), use_fast=True)

    if cfg.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError(cfg.actor_rollout_ref.actor.strategy)

    role_worker_mapping = {Role.ActorRollout: ray.remote(actor_rollout_cls)}
    mapping = {Role.ActorRollout: "global_pool"}
    if cfg.actor_rollout_ref.actor.use_kl_loss or cfg.algorithm.use_kl_in_reward:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = "global_pool"
    # GRPO does not use critic; keep Critic out of role mapping.
    resource_pool_manager = ResourcePoolManager(resource_pool_spec={"global_pool": [cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes}, mapping=mapping)

    train_dataset = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer, processor)
    val_dataset = create_rl_dataset(cfg.data.val_files, cfg.data, tokenizer, processor)
    train_sampler = create_rl_sampler(cfg.data, train_dataset)
    reward_fn = QuantRewardBridge(
        tokenizer,
        bar_minutes=rft.bar_minutes,
        output_dir=rft.output_dir,
        max_workers=rft.reward_workers,
        free_code_eval_timeout_s=rft.free_code_eval_timeout_s,
        archive_path=rft.novelty_archive_path,
        novelty_penalty=rft.novelty_penalty,
        mode=rft.mode,
        relative_reward=rft.relative_reward,
        improvement_bonus=rft.improvement_bonus,
        underperform_penalty=rft.underperform_penalty,
        diversity_reward=rft.diversity_reward,
        exact_repeat_penalty=rft.exact_repeat_penalty,
        family_archive_path=rft.family_archive_path,
        family_free_quota=rft.family_free_quota,
        family_low_quality_threshold=rft.family_low_quality_threshold,
        family_repeat_penalty=rft.family_repeat_penalty,
        family_good_new_threshold=rft.family_good_new_threshold,
        family_new_bonus=rft.family_new_bonus,
        residual_response_reward=rft.residual_response_reward,
        residual_archive_path=rft.residual_archive_path,
        residual_archive_top_k=rft.residual_archive_top_k,
        residual_good_score_threshold=rft.residual_good_score_threshold,
        residual_bonus=rft.residual_bonus,
        residual_corr_penalty=rft.residual_corr_penalty,
        residual_corr_threshold=rft.residual_corr_threshold,
        residual_min_samples=rft.residual_min_samples,
        residual_record_full=rft.residual_record_full,
        residual_full_start_date=rft.residual_full_start_date,
        residual_full_end_date=rft.residual_full_end_date,
        xsec_behavior_diversity=rft.xsec_behavior_diversity,
        xsec_behavior_archive_path=rft.xsec_behavior_archive_path,
        xsec_behavior_archive_top_k=rft.xsec_behavior_archive_top_k,
        xsec_behavior_good_score_threshold=rft.xsec_behavior_good_score_threshold,
        xsec_behavior_corr_threshold=rft.xsec_behavior_corr_threshold,
        xsec_behavior_corr_penalty=rft.xsec_behavior_corr_penalty,
        xsec_behavior_low_corr_bonus_threshold=rft.xsec_behavior_low_corr_bonus_threshold,
        xsec_behavior_low_corr_bonus=rft.xsec_behavior_low_corr_bonus,
        xsec_behavior_min_shared_times=rft.xsec_behavior_min_shared_times,
        xsec_behavior_record_full=rft.xsec_behavior_record_full,
        reward_kind=rft.reward_kind,
        horizon_bars=rft.horizon_bars,
        universe_symbols=rft.universe_symbols,
        single_asset_symbol=rft.single_asset_symbol,
        data_dir=rft.data_dir,
        data_file_template=rft.data_file_template,
        min_symbols_per_time=rft.min_symbols_per_time,
        min_times=rft.min_times,
    )
    trainer = RayPPOTrainer(
        config=cfg,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        device_name=cfg.trainer.device,
    )
    trainer.init_workers()
    trainer.fit()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args(argv)
    raw = load_config(args.config)
    rft = RFTConfig.from_dict(raw.get("rft", raw))
    run(rft)


if __name__ == "__main__":
    main()
