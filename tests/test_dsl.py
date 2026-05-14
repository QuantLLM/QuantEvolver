import pandas as pd

from quant_evolver.dsl import analyse_expr, compile_expr
from quant_evolver.dsl.evaluator import evaluate_expr_series


def test_dsl_compile_and_evaluate_series():
    expr = "div(ts_mean(returns(5)), ts_std(returns(5)))"
    stats = analyse_expr(expr)
    assert stats.valid_syntax
    code = compile_expr(expr)
    assert "GeneratedAlphaFactor" in code

    idx = pd.date_range("2024-01-01", periods=40, freq="min")
    bars = pd.DataFrame(
        {
            "open": range(40),
            "high": range(1, 41),
            "low": range(40),
            "close": [100 + i * 0.1 for i in range(40)],
            "volume": [1000 + i for i in range(40)],
        },
        index=idx,
    )
    out = evaluate_expr_series(expr, bars, warmup_bars=10)
    assert len(out) > 0
    assert out.notna().sum() > 0
    assert out.index.min() >= bars.index.min()
