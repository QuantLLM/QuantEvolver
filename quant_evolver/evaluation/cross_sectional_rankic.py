from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant_evolver.dsl.compiler import compile_expr
from quant_evolver.dsl.evaluator import evaluate_expr_series
from quant_evolver.evaluation.data import LocalDataConfig, LocalDataStore


@dataclass
class CrossSectionalConfig:
    symbols: list[str] = field(default_factory=list)
    start_date: str = "2026-02-23"
    end_date: str = "2026-03-25"
    bar_minutes: int = 5
    horizon_bars: int = 6
    min_symbols_per_time: int = 8
    min_times: int = 20
    warmup_days: int = 10
    extra_buffer_days: int = 2
    data: LocalDataConfig = field(default_factory=LocalDataConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CrossSectionalConfig":
        data = dict(data or {})
        data_cfg = LocalDataConfig.from_dict(data.get("data"))
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "data"}
        return cls(**kwargs, data=data_cfg)


@dataclass
class CrossSectionalResult:
    score: float
    metrics: dict[str, float]
    profile: dict[str, Any] | None
    code: str


def _spearman_corr(x: pd.Series, y: pd.Series) -> float:
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    if float(xr.std(ddof=0)) <= 1e-12 or float(yr.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(xr.corr(yr))


def _bars_for_symbol(symbol: str, cfg: CrossSectionalConfig) -> pd.DataFrame:
    actual_start = datetime.strptime(cfg.start_date, "%Y-%m-%d")
    warmup_start = actual_start - timedelta(days=cfg.warmup_days + cfg.extra_buffer_days)
    df_1m = LocalDataStore(cfg.data).load(symbol, warmup_start, cfg.end_date)
    if df_1m.empty:
        return pd.DataFrame()
    rule = f"{int(cfg.bar_minutes)}min"
    return pd.DataFrame({
        "open": df_1m["open"].astype(float).resample(rule, label="right", closed="right").first(),
        "high": df_1m["high"].astype(float).resample(rule, label="right", closed="right").max(),
        "low": df_1m["low"].astype(float).resample(rule, label="right", closed="right").min(),
        "close": df_1m["close"].astype(float).resample(rule, label="right", closed="right").last(),
        "volume": df_1m["volume"].astype(float).resample(rule, label="right", closed="right").sum(),
    }).dropna()


def _future_return_from_bars(bars: pd.DataFrame, cfg: CrossSectionalConfig) -> pd.Series:
    if bars.empty or "close" not in bars:
        return pd.Series(dtype=float)
    fwd = bars["close"].shift(-int(cfg.horizon_bars)) / bars["close"] - 1.0
    return fwd[fwd.index >= pd.Timestamp(cfg.start_date)]


def default_symbols_from_data(data_cfg: LocalDataConfig | None = None, limit: int = 32) -> list[str]:
    """Infer a universe from local parquet files when symbols are not configured."""
    symbols = LocalDataStore(data_cfg).available_symbols()
    return symbols[: int(limit)]


def evaluate_cross_sectional_expr(expr: str, cfg: CrossSectionalConfig) -> CrossSectionalResult:
    symbols = cfg.symbols or default_symbols_from_data(cfg.data, limit=32)
    if len(symbols) < int(cfg.min_symbols_per_time):
        return CrossSectionalResult(-1.0, {"error": 1.0, "symbols": float(len(symbols))}, None, "")

    try:
        warmup = 240 if int(cfg.bar_minutes) == 5 else 96
        code = compile_expr(expr, warmup_param=warmup)
    except Exception:
        return CrossSectionalResult(-1.0, {"format_error": 1.0}, None, "")

    factor_cols: dict[str, pd.Series] = {}
    fwd_cols: dict[str, pd.Series] = {}
    valid_symbols = 0
    for sym in symbols:
        try:
            bars = _bars_for_symbol(sym, cfg)
            if bars.empty:
                continue
            factor = evaluate_expr_series(expr, bars, warmup_bars=warmup)
            factor = factor[factor.index >= pd.Timestamp(cfg.start_date)]
            fwd = _future_return_from_bars(bars, cfg)
            if factor.empty or fwd.empty:
                continue
            factor_cols[sym] = factor.astype(float)
            fwd_cols[sym] = fwd.astype(float)
            valid_symbols += 1
        except Exception:
            continue

    if valid_symbols < int(cfg.min_symbols_per_time):
        return CrossSectionalResult(-1.0, {"error": 1.0, "valid_symbols": float(valid_symbols)}, None, code)

    factor = pd.DataFrame(factor_cols).replace([np.inf, -np.inf], np.nan)
    fwd = pd.DataFrame(fwd_cols).replace([np.inf, -np.inf], np.nan)
    common_index = factor.index.intersection(fwd.index)
    factor = factor.loc[common_index]
    fwd = fwd.loc[common_index]
    common_cols = [c for c in factor.columns if c in fwd.columns]
    factor = factor[common_cols]
    fwd = fwd[common_cols]

    rank_ics: list[float] = []
    ls_returns: list[float] = []
    n_symbols_by_time: list[int] = []
    ts_values: list[str] = []
    factor_rank_rows: list[list[float | None]] = []
    for ts in factor.index:
        pair = pd.concat([factor.loc[ts], fwd.loc[ts]], axis=1, keys=["factor", "fwd"]).dropna()
        if len(pair) < int(cfg.min_symbols_per_time):
            continue
        ric = _spearman_corr(pair["factor"], pair["fwd"])
        if not np.isfinite(ric):
            continue
        q = max(1, int(len(pair) * 0.2))
        ordered = pair.sort_values("factor")
        ls = float(ordered["fwd"].tail(q).mean() - ordered["fwd"].head(q).mean())
        rank_ics.append(float(ric))
        ls_returns.append(ls)
        n_symbols_by_time.append(int(len(pair)))
        ts_values.append(pd.Timestamp(ts).isoformat())
        ranks = pair["factor"].rank(method="average", pct=True)
        factor_rank_rows.append([
            float(ranks.loc[sym]) if sym in ranks.index and np.isfinite(float(ranks.loc[sym])) else None
            for sym in common_cols
        ])

    if len(rank_ics) < int(cfg.min_times):
        return CrossSectionalResult(
            -1.0,
            {"error": 1.0, "valid_symbols": float(valid_symbols), "valid_times": float(len(rank_ics))},
            None,
            code,
        )

    arr = np.asarray(rank_ics, dtype=float)
    ls_arr = np.asarray(ls_returns, dtype=float)
    mean_ric = float(np.mean(arr))
    std_ric = float(np.std(arr) + 1e-8)
    icir = float(mean_ric / std_ric * np.sqrt(len(arr)))
    mean_ls = float(np.mean(ls_arr))
    ls_sharpe = float(mean_ls / (np.std(ls_arr) + 1e-8) * np.sqrt(len(ls_arr)))
    coverage = float(np.mean(np.asarray(n_symbols_by_time, dtype=float) / max(1, len(common_cols))))
    # Keep reward in roughly the old [-1, 1] scale while preserving sign.
    score = float(np.clip(mean_ric * 5.0 + 0.02 * np.tanh(icir), -1.0, 1.0))
    metrics = {
        "cross_sectional/rank_ic_mean": mean_ric,
        "cross_sectional/rank_ic_std": float(np.std(arr)),
        "cross_sectional/rank_ic_ir": icir,
        "cross_sectional/long_short_mean": mean_ls,
        "cross_sectional/long_short_sharpe": ls_sharpe,
        "cross_sectional/valid_times": float(len(arr)),
        "cross_sectional/valid_symbols": float(valid_symbols),
        "cross_sectional/mean_symbols_per_time": float(np.mean(n_symbols_by_time)),
        "cross_sectional/coverage": coverage,
        "predictive_score": mean_ric * 100.0,
    }
    profile = {
        "ts": ts_values,
        "symbols": list(common_cols),
        "factor_rank": factor_rank_rows,
        "rank_ic": [float(x) for x in rank_ics],
        "ls_return": [float(x) for x in ls_returns],
        "n_symbols": n_symbols_by_time,
    }
    return CrossSectionalResult(score, metrics, profile, code)
