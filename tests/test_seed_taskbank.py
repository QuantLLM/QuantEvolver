from quant_evolver.seeds.library import load_seed_candidates
from quant_evolver.seeds.pipeline import SeedPipeline
from quant_evolver.rft.task_bank import TaskBankBuilder, TaskBuildConfig, TimeSplit


def test_static_seed_validation_and_task_bank():
    seeds = load_seed_candidates("examples/seed_candidates.yaml")
    evals = SeedPipeline().run_static(seeds)
    assert any(ev.valid for ev in evals)
    assert any(not ev.valid for ev in evals)

    cfg = TaskBuildConfig(
        max_seeds=2,
        time_splits=[TimeSplit("a", "2024-01-01", "2024-01-10"), TimeSplit("b", "2024-01-10", "2024-01-20")],
    )
    bank = TaskBankBuilder(cfg).build_from_seed_evaluations(evals)
    assert len(bank.tasks) == 4
    assert {t.time_split for t in bank.tasks} == {"a", "b"}
