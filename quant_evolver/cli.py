from __future__ import annotations

import argparse
from pathlib import Path

from quant_evolver.dsl import ALL_OPS
from quant_evolver.prompts.optimizer import PromptOptimizer
from quant_evolver.scenarios.refiner import ScenarioRefiner
from quant_evolver.scenarios.schema import ScenarioConfig
from quant_evolver.seeds.curator import SeedLibraryBuilder, SeedLibraryConfig
from quant_evolver.seeds.generator import SeedGeneratorConfig, StrongModelSeedGenerator
from quant_evolver.seeds.library import load_seed_candidates, save_seed_evaluations
from quant_evolver.llm import OpenAICompatibleClient
from quant_evolver.rft import RFTConfig, RFTRunner
from quant_evolver.rft.local_smoke import evaluate_expr
from quant_evolver.evaluation.cross_sectional_rankic import CrossSectionalConfig, evaluate_cross_sectional_expr
from quant_evolver.evaluation.data import LocalDataConfig
from quant_evolver.rft.task_bank import TaskBuildConfig, build_task_bank_from_file, save_task_bank
from quant_evolver.seeds.evaluator import SeedEvaluator, SeedEvaluatorConfig
from quant_evolver.seeds.pipeline import SeedPipeline
from quant_evolver.seeds.scorer import SeedScoringConfig, SeedScorer
from quant_evolver.seeds.validator import SeedValidationConfig, SeedValidator
from quant_evolver.utils.config import load_config, save_config


def cmd_refine_scenario(args):
    cfg = ScenarioRefiner(model=args.model, base_url=args.base_url, api_key=args.api_key).refine(args.raw)
    save_config({"scenario": cfg.to_dict()}, args.output)
    print(f"wrote {args.output}")


def cmd_build_prompts(args):
    cfg = load_config(args.config)
    scenario = ScenarioConfig.from_dict(cfg.get("scenario"))
    reward = cfg.get("reward", {})
    pack = PromptOptimizer().build(scenario, reward, sorted(ALL_OPS))
    save_config({"system_prompt": pack.system_prompt, "seed_generation_prompt": pack.seed_generation_prompt, "mutation_prompt_template": pack.mutation_prompt_template}, args.output)
    print(f"wrote {args.output}")


def cmd_generate_seeds(args):
    cfg = load_config(args.config)
    scenario = ScenarioConfig.from_dict(cfg.get("scenario"))
    reward = cfg.get("reward", {})
    gen_cfg = SeedGeneratorConfig.from_dict(cfg.get("seed", {}).get("generator"))
    if args.num_candidates is not None:
        gen_cfg.num_candidates = args.num_candidates
    if args.model is not None:
        gen_cfg.model = args.model
    client = OpenAICompatibleClient(model=gen_cfg.model)
    seeds = StrongModelSeedGenerator(client, gen_cfg).generate(scenario, reward)
    save_config({"scenario_id": cfg.get("experiment", {}).get("name", ""), "generated_by": gen_cfg.model, "seeds": [s.to_dict() for s in seeds]}, args.output)
    print(f"generated {len(seeds)} seeds -> {args.output}")


def cmd_validate_seeds(args):
    cfg = load_config(args.config) if args.config else {}
    seeds = load_seed_candidates(args.seeds)
    validator = SeedValidator(SeedValidationConfig.from_dict(cfg.get("seed", {}).get("validation")))
    scorer = SeedScorer(SeedScoringConfig.from_dict(cfg.get("seed", {}).get("scoring")))
    evaluator = None
    if args.backtest:
        evaluator = SeedEvaluator(SeedEvaluatorConfig.from_dict(cfg.get("evaluation"), reward_cfg=cfg.get("reward")))
    pipeline = SeedPipeline(validator, scorer, evaluator)
    evals = pipeline.run(seeds) if args.backtest else pipeline.run_static(seeds)
    if args.library:
        lib_cfg = SeedLibraryConfig.from_dict(cfg.get("seed", {}).get("library"))
        lib = SeedLibraryBuilder(lib_cfg).build(evals)
        save_config({"source": str(args.seeds), "library": lib.to_dict()}, args.library)
        print(f"wrote seed library -> {args.library}")
    save_seed_evaluations(evals, args.output, header={"source": str(args.seeds), "config": cfg.get("seed", {}), "backtest": bool(args.backtest)})
    print(f"validated {len(evals)} seeds -> {args.output}")
    print(f"valid: {sum(ev.valid for ev in evals)} / {len(evals)}")


def cmd_build_task_bank(args):
    cfg = load_config(args.config) if args.config else {}
    task_cfg_raw = cfg.get("task_bank", cfg.get("tasks", {}))
    if args.max_seeds is not None:
        task_cfg_raw = {**task_cfg_raw, "max_seeds": args.max_seeds}
    if args.bar_minutes is not None:
        task_cfg_raw = {**task_cfg_raw, "bar_minutes": args.bar_minutes}
    task_cfg = TaskBuildConfig.from_dict(task_cfg_raw)
    bank = build_task_bank_from_file(args.seed_evaluations, task_cfg)
    save_task_bank(bank, args.output)
    print(f"built {len(bank.tasks)} tasks -> {args.output}")


def cmd_prepare_task_bank(args):
    cfg = load_config(args.config)
    scenario = ScenarioConfig.from_dict(cfg.get("scenario"))
    reward = cfg.get("reward", {})
    gen_cfg = SeedGeneratorConfig.from_dict(cfg.get("seed", {}).get("generator"))
    if args.model:
        gen_cfg.model = args.model
    if args.num_candidates is not None:
        gen_cfg.num_candidates = args.num_candidates
    client = OpenAICompatibleClient(model=gen_cfg.model)
    seeds = StrongModelSeedGenerator(client, gen_cfg).generate(scenario, reward)
    generated_path = Path(args.generated_output)
    save_config({"scenario_id": cfg.get("experiment", {}).get("name", ""), "generated_by": gen_cfg.model, "seeds": [s.to_dict() for s in seeds]}, generated_path)
    print(f"generated {len(seeds)} seeds -> {generated_path}")

    validator = SeedValidator(SeedValidationConfig.from_dict(cfg.get("seed", {}).get("validation")))
    scorer = SeedScorer(SeedScoringConfig.from_dict(cfg.get("seed", {}).get("scoring")))
    evaluator = SeedEvaluator(SeedEvaluatorConfig.from_dict(cfg.get("evaluation"), reward_cfg=reward)) if args.backtest else None
    evals = SeedPipeline(validator, scorer, evaluator).run(seeds) if args.backtest else SeedPipeline(validator, scorer, None).run_static(seeds)
    eval_path = Path(args.evaluated_output)
    save_seed_evaluations(evals, eval_path, header={"source": str(generated_path), "config": cfg.get("seed", {}), "backtest": bool(args.backtest)})
    print(f"validated {len(evals)} seeds -> {eval_path}; valid={sum(ev.valid for ev in evals)}")

    task_cfg = TaskBuildConfig.from_dict(cfg.get("task_bank", cfg.get("tasks", {})))
    bank = build_task_bank_from_file(eval_path, task_cfg)
    save_task_bank(bank, args.task_output)
    print(f"built {len(bank.tasks)} tasks -> {args.task_output}")


def cmd_rft_launch(args):
    cfg = load_config(args.config)
    rft_cfg = RFTConfig.from_dict(cfg.get("rft", cfg))
    if args.mode:
        rft_cfg.mode = args.mode
    if args.seed_expr:
        rft_cfg.seed_expr = args.seed_expr
    if args.experiment_name:
        rft_cfg.experiment_name = args.experiment_name
    if args.total_epochs is not None:
        rft_cfg.total_epochs = args.total_epochs
    if args.save_freq is not None:
        rft_cfg.save_freq = args.save_freq
    result = RFTRunner(rft_cfg).launch(dry_run=args.dry_run)
    print(f"backend: {result.backend}")
    if result.pid is not None:
        print(f"pid: {result.pid}")
    if result.log_path:
        print(f"log: {result.log_path}")
    if result.pid_path:
        print(f"pid_file: {result.pid_path}")
    if result.notes:
        for note in result.notes:
            print(note)
    if args.print_command or args.dry_run:
        print("\n--- command ---")
        print(result.command)


def cmd_rft_eval_expr(args):
    expr = args.expr
    if expr.startswith("<expr>"):
        expr = expr.replace("<expr>", "").replace("</expr>", "").strip()
    data_cfg = LocalDataConfig(data_dir=args.data_dir, file_template=args.data_file_template)
    if args.reward_kind == "cross_sectional_rankic":
        result = evaluate_cross_sectional_expr(
            expr,
            CrossSectionalConfig(
                start_date=args.start_date,
                end_date=args.end_date,
                bar_minutes=args.bar_minutes,
                horizon_bars=args.horizon_bars,
                symbols=args.symbols or [],
                min_symbols_per_time=args.min_symbols_per_time,
                min_times=args.min_times,
                data=data_cfg,
            ),
        )
        print({"score": result.score, "metrics": result.metrics})
        return
    result = evaluate_expr(
        expr,
        symbol=args.symbol,
        bar_minutes=args.bar_minutes,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        data_file_template=args.data_file_template,
    )
    print(result.__dict__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quantevolver")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("refine-scenario", help="Use an OpenAI-compatible LLM to rewrite a raw request as a structured scenario")
    s.add_argument("raw")
    s.add_argument("-o", "--output", default="scenario.refined.yaml")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--base-url", default=None, help="OpenAI-compatible base URL; defaults to OPENAI_BASE_URL or OpenAI")
    s.add_argument("--api-key", default=None, help="API key; defaults to OPENAI_API_KEY or DASHSCOPE_API_KEY")
    s.set_defaults(func=cmd_refine_scenario)

    s = sub.add_parser("build-prompts")
    s.add_argument("config")
    s.add_argument("-o", "--output", default="prompts.yaml")
    s.set_defaults(func=cmd_build_prompts)

    s = sub.add_parser("generate-seeds")
    s.add_argument("config")
    s.add_argument("-o", "--output", default="seed_candidates.generated.yaml")
    s.add_argument("--model", default=None)
    s.add_argument("-n", "--num-candidates", type=int, default=None)
    s.set_defaults(func=cmd_generate_seeds)

    s = sub.add_parser("validate-seeds")
    s.add_argument("seeds")
    s.add_argument("-c", "--config", default=None)
    s.add_argument("-o", "--output", default="seed_evaluations.yaml")
    s.add_argument("--backtest", action="store_true", help="Run empirical backtest after static validation")
    s.add_argument("--library", default=None, help="Also write curated elite/diverse/contrarian seed library")
    s.set_defaults(func=cmd_validate_seeds)

    s = sub.add_parser("build-task-bank", help="Expand evaluated/generated seeds into seed × time_split RFT tasks")
    s.add_argument("seed_evaluations")
    s.add_argument("-c", "--config", default=None)
    s.add_argument("-o", "--output", default="task_bank.yaml")
    s.add_argument("--max-seeds", type=int, default=None)
    s.add_argument("--bar-minutes", type=int, default=None)
    s.set_defaults(func=cmd_build_task_bank)

    s = sub.add_parser("prepare-task-bank", help="Generate seeds with a strong model, validate/evaluate them, then build seed × time_split task bank")
    s.add_argument("config")
    s.add_argument("--model", default=None)
    s.add_argument("-n", "--num-candidates", type=int, default=None)
    s.add_argument("--backtest", action="store_true")
    s.add_argument("--generated-output", default="runs/quantevolver_generated_seeds.yaml")
    s.add_argument("--evaluated-output", default="runs/quantevolver_seed_evaluations.yaml")
    s.add_argument("--task-output", default="runs/quantevolver_task_bank.yaml")
    s.set_defaults(func=cmd_prepare_task_bank)

    s = sub.add_parser("rft-launch", help="Launch RFT through a QuantEvolver-owned backend config")
    s.add_argument("config")
    s.add_argument("--mode", default=None)
    s.add_argument("--seed-expr", default=None)
    s.add_argument("--experiment-name", default=None)
    s.add_argument("--total-epochs", type=int, default=None)
    s.add_argument("--save-freq", type=int, default=None)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--print-command", action="store_true")
    s.set_defaults(func=cmd_rft_launch)

    s = sub.add_parser("rft-eval-expr", help="Evaluate one DSL expression locally")
    s.add_argument("expr")
    s.add_argument("--symbol", default="ASSET", help="Single-asset symbol for direction/IC evaluation")
    s.add_argument("--bar-minutes", type=int, default=5)
    s.add_argument("--start-date", default="2024-01-01")
    s.add_argument("--end-date", default="2024-02-01")
    s.add_argument("--data-dir", default="data")
    s.add_argument("--data-file-template", default="{symbol_safe}_1m.parquet")
    s.add_argument("--reward-kind", default="direction", choices=["direction", "cross_sectional_rankic"])
    s.add_argument("--horizon-bars", type=int, default=6)
    s.add_argument("--symbols", nargs="*", default=None)
    s.add_argument("--min-symbols-per-time", type=int, default=8)
    s.add_argument("--min-times", type=int, default=20)
    s.set_defaults(func=cmd_rft_eval_expr)
    return p


def main(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
