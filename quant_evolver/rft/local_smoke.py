from __future__ import annotations

from dataclasses import dataclass

from quant_evolver.dsl.compiler import compile_expr, extract_expr
from quant_evolver.evaluation.backtrader_signal import BacktestConfig, run_factor_backtrader
from quant_evolver.rewards import RewardContext, RewardProfile


@dataclass
class LocalSmokeResult:
    expr: str
    valid: bool
    score: float = 0.0
    direction_acc: float = 0.0
    coverage: float = 0.0
    error: str | None = None


def evaluate_expr(
    expr: str,
    *,
    symbol: str = "ASSET",
    bar_minutes: int = 5,
    start_date: str = "2024-01-01",
    end_date: str = "2024-02-01",
    data_dir: str = "data",
    data_file_template: str = "{symbol_safe}_1m.parquet",
) -> LocalSmokeResult:
    try:
        from quant_evolver.evaluation.data import LocalDataConfig

        code = compile_expr(expr, warmup_param=240 if bar_minutes == 5 else 96)
        run = run_factor_backtrader(
            code,
            BacktestConfig(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                bar_minutes=bar_minutes,
                data=LocalDataConfig(data_dir=data_dir, file_template=data_file_template),
            ),
        )
        reward = RewardProfile.from_dict({"primary": "direction_accuracy", "components": {"direction_accuracy": {"weight": 1.0, "horizon": 1}}}).evaluate(
            RewardContext(factor=run.factor, prices=run.prices, horizon=1, config={})
        )
        acc = float(reward.metrics.get("direction_accuracy/direction_acc", 0.0))
        coverage = float(reward.metrics.get("direction_accuracy/coverage", 0.0))
        return LocalSmokeResult(expr=expr, valid=reward.passed, score=float(reward.value), direction_acc=acc, coverage=coverage, error=reward.reason or None)
    except Exception as exc:
        return LocalSmokeResult(expr=expr, valid=False, error=str(exc))
