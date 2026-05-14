from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from quant_evolver.evaluation.data import LocalDataConfig, LocalDataStore
from quant_evolver.evaluation.factor_loader import load_factor_class_from_code


@dataclass
class BacktestConfig:
    symbol: str = "ASSET"
    start_date: str = "2026-02-23"
    end_date: str = "2026-03-25"
    bar_minutes: int = 5
    warmup_days: int = 10
    extra_buffer_days: int = 2
    data: LocalDataConfig = field(default_factory=LocalDataConfig)

    @classmethod
    def from_dict(cls, data: dict | None) -> "BacktestConfig":
        data = data or {}
        data_cfg = LocalDataConfig.from_dict(data.get("data"))
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "data"}
        return cls(**kwargs, data=data_cfg)


@dataclass
class FactorRunResult:
    factor: pd.Series
    prices: pd.DataFrame
    report: str
    metrics: dict[str, float]


class FactorSignalCollectStrategy:  # actual subclass is created lazily to keep import optional
    pass


def run_factor_backtrader(factor_code: str, config: BacktestConfig) -> FactorRunResult:
    try:
        import backtrader as bt
    except ImportError as exc:
        raise ImportError("backtrader is required. Install quant-evolver[backtest].") from exc

    factor_cls = load_factor_class_from_code(factor_code)
    actual_start = datetime.strptime(config.start_date, "%Y-%m-%d")
    warmup_start = actual_start - timedelta(days=config.warmup_days + config.extra_buffer_days)
    df_1m = LocalDataStore(config.data).load(config.symbol, warmup_start, config.end_date)
    if df_1m.empty:
        raise ValueError(f"Data is empty for {config.symbol} in {warmup_start}~{config.end_date}")

    is_daily_like = int(config.bar_minutes) >= 1440
    daily_ts_map: dict[pd.Timestamp, pd.Timestamp] | None = None
    if is_daily_like:
        original_index = pd.to_datetime(df_1m.index)
        synthetic_index = pd.date_range("2000-01-01", periods=len(df_1m), freq=f"{int(config.bar_minutes)}min")
        daily_ts_map = {pd.Timestamp(s): pd.Timestamp(o) for s, o in zip(synthetic_index, original_index)}
        df_1m = df_1m.copy()
        df_1m.index = synthetic_index

    class _CollectStrategy(bt.Strategy):
        params = (("factor_cls", None), ("warmup_bars", 0))

        def __init__(self):
            if len(self.datas) == 1:
                self.data_target = self.datas[0]
            else:
                candidates = [d for d in self.datas if d._timeframe == bt.TimeFrame.Minutes and d._compression > 1]
                if not candidates:
                    candidates = [d for d in self.datas if d._timeframe == bt.TimeFrame.Days]
                if not candidates:
                    raise ValueError("No resampled data feed found")
                self.data_target = candidates[-1]
            self.ind = self.p.factor_cls(self.data_target)
            self.ind.plotinfo.plot = False
            self.last_bar_time = None
            self.bar_count = 0
            self.nan_count = 0
            self.rows = []

        def next(self):
            current_time = self.data_target.datetime.datetime(0)
            if self.last_bar_time == current_time:
                return
            self.last_bar_time = current_time
            self.bar_count += 1
            if self.bar_count < int(self.p.warmup_bars):
                return
            try:
                score = float(self.ind.score[0])
                close_now = float(self.data_target.close[0])
            except Exception:
                return
            if not np.isfinite(score) or not np.isfinite(close_now) or close_now <= 0:
                self.nan_count += 1
                return
            self.rows.append((current_time, score, close_now))

    cerebro = bt.Cerebro(stdstats=False)
    if is_daily_like:
        data_base = bt.feeds.PandasData(
            dataname=df_1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
        )
        data_base.plotinfo.plot = False
        cerebro.adddata(data_base, name=config.symbol)
    else:
        data_1m = bt.feeds.PandasData(dataname=df_1m, timeframe=bt.TimeFrame.Minutes, compression=1)
        data_1m.plotinfo.plot = False
        cerebro.adddata(data_1m, name=config.symbol)
        data_resampled = cerebro.resampledata(data_1m, timeframe=bt.TimeFrame.Minutes, compression=config.bar_minutes)
        data_resampled.plotinfo.plot = False
    warmup_bars = 1 if is_daily_like else int(config.warmup_days) * (1440 // config.bar_minutes)
    cerebro.addstrategy(_CollectStrategy, factor_cls=factor_cls, warmup_bars=warmup_bars)
    strat = cerebro.run()[0]
    if not strat.rows:
        err_count = getattr(strat.ind, "_error_count", None)
        err_text = getattr(strat.ind, "_last_error", "") or ""
        raise ValueError(
            "No valid factor scores collected. "
            f"bars={max(strat.bar_count - warmup_bars, 0)} nan={strat.nan_count} "
            f"errors={err_count}; {err_text[-800:]}"
        )
    out = pd.DataFrame(strat.rows, columns=["dt", "score", "close"]).sort_values("dt")
    out["dt"] = pd.to_datetime(out["dt"])
    if daily_ts_map is not None:
        out["dt"] = out["dt"].map(lambda x: daily_ts_map.get(pd.Timestamp(x), pd.Timestamp(x)))
    out = out.set_index("dt")
    out = out[out.index >= pd.Timestamp(config.start_date)]
    factor = out["score"].astype(float)
    prices = pd.DataFrame({"close": out["close"].astype(float)}, index=out.index)
    coverage = float(len(out) / max(1, len(out)))
    factor_std = float(np.std(factor.to_numpy())) if len(factor) else 0.0
    report = f"symbol={config.symbol}\nrange={config.start_date}~{config.end_date}\nbar_minutes={config.bar_minutes}\nsamples={len(out)}\nfactor_std={factor_std:.6f}"
    return FactorRunResult(factor=factor, prices=prices, report=report, metrics={"samples": float(len(out)), "coverage": coverage, "factor_std": factor_std})
