from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import ast
import hashlib
import json
import math
import multiprocessing as mp
import os
import threading
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from quant_evolver.dsl.compiler import compile_expr, extract_expr
from quant_evolver.evaluation.backtrader_signal import BacktestConfig, run_factor_backtrader
from quant_evolver.evaluation.data import LocalDataConfig
from quant_evolver.evaluation.cross_sectional_rankic import CrossSectionalConfig, evaluate_cross_sectional_expr
from quant_evolver.rewards import RewardContext, RewardProfile
from quant_evolver.rewards.base import finite_align
from quant_evolver.rft.novelty import NoveltyArchive

_EXPR_RE = re.compile(r"<expr>\s*(.*?)\s*</expr>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.DOTALL | re.IGNORECASE)


def _window_bucket(x: int | float) -> str:
    try:
        v = float(x)
    except Exception:
        return "N"
    if v <= 5:
        return "W_1_5"
    if v <= 20:
        return "W_6_20"
    if v <= 60:
        return "W_21_60"
    if v <= 120:
        return "W_61_120"
    if v <= 240:
        return "W_121_240"
    return "W_241_480"


def expr_family_hash(expr: str) -> tuple[str, str]:
    """Structural family hash for soft diversity shaping.

    Keeps operator/field structure but buckets numeric constants so local
    window refinements remain in the same family. This is intentionally coarse:
    exact expressions are handled separately by NoveltyArchive.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except Exception:
        h = hashlib.sha1(("invalid:" + " ".join(expr.split())).encode("utf-8")).hexdigest()[:16]
        return h, "invalid"

    class Canon(ast.NodeTransformer):
        def visit_Constant(self, node):  # noqa: N802
            if isinstance(node.value, (int, float)):
                return ast.copy_location(ast.Constant(value=_window_bucket(node.value)), node)
            return node

    tree = Canon().visit(tree)
    ast.fix_missing_locations(tree)
    canonical = ast.dump(tree, annotate_fields=False, include_attributes=False)
    h = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]
    return h, canonical


class FamilyArchive:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self.counts: dict[str, int] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    h = str(item.get("family_hash", ""))
                    c = int(item.get("count", 1))
                    if h:
                        self.counts[h] = max(self.counts.get(h, 0), c)
                except Exception:
                    continue
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def seen_count(self, expr: str) -> tuple[int, str, str]:
        h, canonical = expr_family_hash(expr)
        with self._lock:
            return self.counts.get(h, 0), h, canonical

    def record(self, expr: str, *, task_id: str = "") -> tuple[int, str]:
        h, canonical = expr_family_hash(expr)
        with self._lock:
            new = self.counts.get(h, 0) + 1
            self.counts[h] = new
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"family_hash": h, "count": new, "expr": expr, "canonical": canonical, "task_id": task_id}, ensure_ascii=False) + "\n")
        return new, h


def _direction_profile(factor: pd.Series, prices: pd.DataFrame) -> dict[str, Any]:
    fwd = prices["close"].shift(-1) / prices["close"] - 1.0
    f2, y = finite_align(factor, fwd)
    pred = np.sign(f2.to_numpy(dtype=float)).astype(np.int8)
    label = np.sign(y.to_numpy(dtype=float)).astype(np.int8)
    mask = (pred != 0).astype(np.uint8)
    correct = ((pred == label) & (mask.astype(bool))).astype(np.uint8)
    # Timestamp strings make slicing across overlapping/non-overlapping task
    # intervals exact while keeping the archive independent of row offsets.
    return {
        "ts": [x.isoformat() for x in f2.index],
        "pred": pred.tolist(),
        "label": label.tolist(),
        "mask": mask.tolist(),
        "correct": correct.tolist(),
    }


def _single_asset_reward_profile(reward_kind: str, horizon_bars: int) -> RewardProfile:
    reward_name = str(reward_kind or "direction").strip().lower()
    if reward_name == "direction":
        reward_name = "direction_accuracy"
    if reward_name not in {"direction_accuracy", "ic", "rank_ic", "signal_return"}:
        reward_name = "direction_accuracy"
    min_samples = 30 if reward_name in {"ic", "rank_ic"} else 50
    return RewardProfile.from_dict(
        {
            "primary": reward_name,
            "components": {
                reward_name: {
                    "weight": 1.0,
                    "horizon": int(max(1, horizon_bars)),
                    "min_samples": min_samples,
                }
            },
        }
    )


def _single_asset_warmup_bars(bars: int) -> int:
    bars = int(max(1, bars))
    if bars >= 1440:
        return 240
    if bars <= 5:
        return 240
    if bars <= 60:
        return 96
    if bars <= 240:
        return 64
    return 32


def _single_asset_warmup_days(bars: int, warmup_bars: int) -> int:
    bars = int(max(1, bars))
    warmup_bars = int(max(1, warmup_bars))
    if bars >= 1440:
        # Equity daily data uses trading days, so calendar-day buffers must be
        # materially larger than the bar count to cover weekends/holidays.
        return max(180, warmup_bars * 2 + 30)
    days = int(math.ceil((bars * warmup_bars) / 1440.0))
    return max(10, days + 5)


def _single_asset_score_from_reward(reward_kind: str, reward) -> float:
    reward_name = str(reward_kind or "direction").strip().lower()
    if reward_name == "direction":
        reward_name = "direction_accuracy"
    if not reward.passed:
        return -1.0
    value = float(reward.value)
    if reward_name == "direction_accuracy":
        return value / 100.0 + 0.1
    return value


def _single_asset_profile(reward_kind: str, factor: pd.Series, prices: pd.DataFrame) -> dict[str, Any] | None:
    reward_name = str(reward_kind or "direction").strip().lower()
    if reward_name == "direction":
        reward_name = "direction_accuracy"
    if reward_name == "direction_accuracy":
        return _direction_profile(factor, prices)
    return None


def _score_expr_worker(
    expr: str,
    start: str,
    end: str,
    bars: int,
    symbol: str = "ASSET",
    data_dir: str = "data",
    data_file_template: str = "{symbol_safe}_1m.parquet",
    reward_kind: str = "direction",
    horizon_bars: int = 1,
) -> tuple[float, dict[str, float], str, float, dict[str, Any] | None]:
    try:
        warmup_bars = _single_asset_warmup_bars(bars)
        code = compile_expr(expr, warmup_param=warmup_bars)
        run = run_factor_backtrader(
            code,
            BacktestConfig(
                symbol=symbol,
                start_date=start,
                end_date=end,
                bar_minutes=bars,
                warmup_days=_single_asset_warmup_days(bars, warmup_bars),
                data=LocalDataConfig(data_dir=data_dir, file_template=data_file_template),
            ),
        )
        reward_profile = _single_asset_reward_profile(reward_kind, horizon_bars)
        reward = reward_profile.evaluate(RewardContext(factor=run.factor, prices=run.prices, horizon=int(max(1, horizon_bars)), config={}))
        score = _single_asset_score_from_reward(reward_kind, reward)
        metrics = {k: float(v) for k, v in reward.metrics.items()}
        metrics["predictive_score"] = float(reward.value)
        return score, metrics, code, float(reward.value), _single_asset_profile(reward_kind, run.factor, run.prices)
    except Exception:
        return -1.0, {"error": 1.0}, "", 0.0, None


def _score_code_worker(
    code: str,
    start: str,
    end: str,
    bars: int,
    symbol: str = "ASSET",
    data_dir: str = "data",
    data_file_template: str = "{symbol_safe}_1m.parquet",
    reward_kind: str = "direction",
    horizon_bars: int = 1,
) -> tuple[float, dict[str, float], str, float, dict[str, Any] | None]:
    try:
        warmup_bars = _single_asset_warmup_bars(bars)
        run = run_factor_backtrader(
            code,
            BacktestConfig(
                symbol=symbol,
                start_date=start,
                end_date=end,
                bar_minutes=bars,
                warmup_days=_single_asset_warmup_days(bars, warmup_bars),
                data=LocalDataConfig(data_dir=data_dir, file_template=data_file_template),
            ),
        )
        reward_profile = _single_asset_reward_profile(reward_kind, horizon_bars)
        reward = reward_profile.evaluate(RewardContext(factor=run.factor, prices=run.prices, horizon=int(max(1, horizon_bars)), config={}))
        score = _single_asset_score_from_reward(reward_kind, reward)
        metrics = {k: float(v) for k, v in reward.metrics.items()}
        metrics["predictive_score"] = float(reward.value)
        return score, metrics, code, float(reward.value), _single_asset_profile(reward_kind, run.factor, run.prices)
    except Exception:
        return -1.0, {"error": 1.0}, "", 0.0, None


def _run_factor_backtrader_target(
    out_q: mp.queues.Queue,
    factor_code: str,
    config: BacktestConfig,
) -> None:
    try:
        run = run_factor_backtrader(factor_code, config)
        out_q.put(("ok", run))
    except Exception as exc:
        out_q.put(("err", repr(exc)))


def _run_factor_backtrader_with_timeout(
    factor_code: str,
    config: BacktestConfig,
    timeout_s: int | float | None = None,
) -> FactorRunResult:
    timeout = float(timeout_s or 0)
    if timeout <= 0:
        return run_factor_backtrader(factor_code, config)

    ctx = mp.get_context("fork")
    out_q = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_run_factor_backtrader_target, args=(out_q, factor_code, config))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(2.0)
        raise TimeoutError(
            f"run_factor_backtrader timed out after {timeout:.1f}s "
            f"for {config.symbol} {config.start_date}~{config.end_date}"
        )
    if not out_q.empty():
        status, payload = out_q.get()
        if status == "ok":
            return payload
        raise RuntimeError(str(payload))
    raise RuntimeError(f"run_factor_backtrader subprocess exited without result (exitcode={proc.exitcode})")


def _score_cross_sectional_expr_worker(
    expr: str,
    start: str,
    end: str,
    bars: int,
    horizon_bars: int,
    symbols: tuple[str, ...],
    min_symbols_per_time: int,
    min_times: int,
) -> tuple[float, dict[str, float], str, float, dict[str, Any] | None]:
    try:
        result = evaluate_cross_sectional_expr(
            expr,
            CrossSectionalConfig(
                symbols=list(symbols),
                start_date=start,
                end_date=end,
                bar_minutes=bars,
                horizon_bars=horizon_bars,
                min_symbols_per_time=min_symbols_per_time,
                min_times=min_times,
            ),
        )
        pred = float(result.metrics.get("predictive_score", 0.0))
        return result.score, result.metrics, result.code, pred, result.profile
    except Exception:
        return -1.0, {"error": 1.0}, "", 0.0, None


def _score_cross_sectional_code_worker(
    code: str,
    start: str,
    end: str,
    bars: int,
    horizon_bars: int,
    symbols: tuple[str, ...],
    min_symbols_per_time: int,
    min_times: int,
    eval_timeout_s: int | float = 0,
) -> tuple[float, dict[str, float], str, float, dict[str, Any] | None]:
    """Score a free-form Backtrader factor class on the cross-sectional task.

    This is intentionally only used for the no-DSL ablation.  It mirrors
    evaluate_cross_sectional_expr, but obtains each symbol's factor series by
    replaying the generated Backtrader code instead of compiling a DSL expr.
    """
    try:
        factor_cols: dict[str, pd.Series] = {}
        fwd_cols: dict[str, pd.Series] = {}
        valid_symbols = 0
        timeout_symbols = 0
        
        def _eval_one_symbol(sym: str) -> tuple[str, str, pd.Series | None, pd.Series | None]:
            try:
                run = _run_factor_backtrader_with_timeout(
                    code,
                    BacktestConfig(symbol=sym, start_date=start, end_date=end, bar_minutes=bars),
                    timeout_s=eval_timeout_s,
                )
                factor = run.factor[run.factor.index >= pd.Timestamp(start)].astype(float)
                fwd = (run.prices["close"].shift(-int(horizon_bars)) / run.prices["close"] - 1.0)
                fwd = fwd[fwd.index >= pd.Timestamp(start)].astype(float)
                if factor.empty or fwd.empty:
                    return sym, "empty", None, None
                return sym, "ok", factor, fwd
            except TimeoutError:
                return sym, "timeout", None, None
            except Exception:
                return sym, "error", None, None

        # Default to full symbol-level parallelism. Each thread coordinates one
        # symbol job, while the heavy backtest itself still runs in a dedicated
        # subprocess via _run_factor_backtrader_with_timeout, so this scales
        # with available local CPU without adding a new config parameter.
        with ThreadPoolExecutor(max_workers=max(1, len(symbols))) as ex:
            futs = [ex.submit(_eval_one_symbol, sym) for sym in symbols]
            for fut in as_completed(futs):
                sym, status, factor, fwd = fut.result()
                if status == "timeout":
                    timeout_symbols += 1
                    continue
                if status != "ok" or factor is None or fwd is None:
                    continue
                factor_cols[sym] = factor
                fwd_cols[sym] = fwd
                valid_symbols += 1
        if valid_symbols < int(min_symbols_per_time):
            return -1.0, {"error": 1.0, "valid_symbols": float(valid_symbols), "timeout_symbols": float(timeout_symbols)}, code, 0.0, None

        factor = pd.DataFrame(factor_cols).replace([np.inf, -np.inf], np.nan)
        fwd = pd.DataFrame(fwd_cols).replace([np.inf, -np.inf], np.nan)
        common_index = factor.index.intersection(fwd.index)
        common_cols = [c for c in factor.columns if c in fwd.columns]
        factor = factor.loc[common_index, common_cols]
        fwd = fwd.loc[common_index, common_cols]

        rank_ics: list[float] = []
        ls_returns: list[float] = []
        n_symbols_by_time: list[int] = []
        ts_values: list[str] = []
        factor_rank_rows: list[list[float | None]] = []
        for ts in factor.index:
            pair = pd.concat([factor.loc[ts], fwd.loc[ts]], axis=1, keys=["factor", "fwd"]).dropna()
            if len(pair) < int(min_symbols_per_time):
                continue
            xr = pair["factor"].rank(method="average")
            yr = pair["fwd"].rank(method="average")
            if float(xr.std(ddof=0)) <= 1e-12 or float(yr.std(ddof=0)) <= 1e-12:
                continue
            ric = float(xr.corr(yr))
            if not np.isfinite(ric):
                continue
            q = max(1, int(len(pair) * 0.2))
            ordered = pair.sort_values("factor")
            ls = float(ordered["fwd"].tail(q).mean() - ordered["fwd"].head(q).mean())
            rank_ics.append(ric)
            ls_returns.append(ls)
            n_symbols_by_time.append(int(len(pair)))
            ts_values.append(pd.Timestamp(ts).isoformat())
            ranks = pair["factor"].rank(method="average", pct=True)
            factor_rank_rows.append([
                float(ranks.loc[sym]) if sym in ranks.index and np.isfinite(float(ranks.loc[sym])) else None
                for sym in common_cols
            ])
        if len(rank_ics) < int(min_times):
            return -1.0, {"error": 1.0, "valid_symbols": float(valid_symbols), "valid_times": float(len(rank_ics))}, code, 0.0, None
        arr = np.asarray(rank_ics, dtype=float)
        ls_arr = np.asarray(ls_returns, dtype=float)
        mean_ric = float(np.mean(arr))
        std_ric = float(np.std(arr) + 1e-8)
        icir = float(mean_ric / std_ric * np.sqrt(len(arr)))
        mean_ls = float(np.mean(ls_arr))
        ls_sharpe = float(mean_ls / (np.std(ls_arr) + 1e-8) * np.sqrt(len(ls_arr)))
        coverage = float(np.mean(np.asarray(n_symbols_by_time, dtype=float) / max(1, len(common_cols))))
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
            "cross_sectional/timeout_symbols": float(timeout_symbols),
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
        return score, metrics, code, float(metrics["predictive_score"]), profile
    except Exception:
        return -1.0, {"error": 1.0}, code, 0.0, None


class ResidualResponseArchive:
    """Sample-level residual archive for behavior/ensemble complementarity.

    Stores compact int8/uint8 prediction/correctness vectors.  The archive is
    keyed by exact timestamps, so a factor recorded on the canonical full
    interval can be sliced for any later non-overlapping 4-way time split.
    """

    def __init__(self, path: str | Path | None, *, top_k: int = 128):
        self.path = Path(path) if path else None
        self.top_k = int(top_k)
        self._lock = threading.Lock()
        self.items: dict[str, dict[str, Any]] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    h = str(item.get("expr_hash", ""))
                    if h:
                        old = self.items.get(h)
                        if old is None or float(item.get("raw_score", -9)) >= float(old.get("raw_score", -9)):
                            self.items[h] = item
                except Exception:
                    continue
            self._trim_locked()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _trim_locked(self) -> None:
        ordered = sorted(self.items.values(), key=lambda x: float(x.get("raw_score", -9)), reverse=True)[: self.top_k]
        self.items = {str(x.get("expr_hash")): x for x in ordered if x.get("expr_hash")}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.items.values())

    def record(self, expr: str, raw_score: float, profile: dict[str, Any] | None, *, task_id: str = "", start: str = "", end: str = "") -> None:
        if not profile:
            return
        h = hashlib.sha1(expr.encode("utf-8")).hexdigest()[:12]
        item = {"expr_hash": h, "expr": expr, "raw_score": float(raw_score), "task_id": task_id, "start": start, "end": end, "profile": profile}
        with self._lock:
            old = self.items.get(h)
            if old is not None and float(old.get("raw_score", -9)) > raw_score:
                return
            self.items[h] = item
            self._trim_locked()
            self._compiled_dirty = True
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _profile_map(profile: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
        ts = profile.get("ts") or []
        correct = profile.get("correct") or []
        mask = profile.get("mask") or []
        pred = profile.get("pred") or []
        n = min(len(ts), len(correct), len(mask), len(pred))
        return {str(ts[i]): (int(correct[i]), int(mask[i]), int(pred[i])) for i in range(n)}

    def residual_stats(self, profile: dict[str, Any] | None) -> dict[str, float]:
        if not profile:
            return {"residual_samples": 0.0, "residual_score": 0.0, "behavior_max_corr": 0.0}
        archives = self.snapshot()
        if not archives:
            return {"residual_samples": 0.0, "residual_score": 0.0, "behavior_max_corr": 0.0}
        cand = self._profile_map(profile)
        if not cand:
            return {"residual_samples": 0.0, "residual_score": 0.0, "behavior_max_corr": 0.0}

        # Approximate elite ensemble by majority vote of archived pred signs on
        # timestamps shared with the candidate.  Wrong-mask is then exact at
        # sample level for those timestamps.
        correct_on_wrong = []
        cand_pred_vec = []
        max_corr = 0.0
        for item in archives:
            amap = self._profile_map(item.get("profile") or {})
            shared = sorted(set(cand).intersection(amap))
            if len(shared) < 20:
                continue
            cp = np.asarray([cand[t][2] for t in shared], dtype=float)
            ap = np.asarray([amap[t][2] for t in shared], dtype=float)
            if np.std(cp) > 1e-8 and np.std(ap) > 1e-8:
                c = float(np.corrcoef(cp, ap)[0, 1])
                if np.isfinite(c):
                    max_corr = max(max_corr, abs(c))

        all_ts = sorted(cand.keys())
        archive_maps = [self._profile_map(x.get("profile") or {}) for x in archives]
        label = {str(ts): int(lb) for ts, lb in zip(profile.get("ts") or [], profile.get("label") or [])}
        for t in all_ts:
            c_correct, c_mask, _c_pred = cand[t]
            if not c_mask:
                continue
            votes = [m[t][2] for m in archive_maps if t in m and m[t][1]]
            if not votes:
                continue
            ens_pred = int(np.sign(np.sum(votes)))
            if ens_pred == 0 or t not in label:
                continue
            if ens_pred != int(label[t]):
                correct_on_wrong.append(float(c_correct))
        if not correct_on_wrong:
            return {"residual_samples": 0.0, "residual_score": 0.0, "behavior_max_corr": float(max_corr)}
        acc = float(np.mean(correct_on_wrong))
        return {"residual_samples": float(len(correct_on_wrong)), "residual_acc": acc, "residual_score": (acc - 0.5) * 200.0, "behavior_max_corr": float(max_corr)}


class CrossSectionalBehaviorArchive:
    """Behavior archive for cross-sectional factors.

    Stores compact per-timestamp factor rank vectors over the evaluated
    universe.  Diversity is measured as Spearman/Pearson correlation of these
    rank vectors versus archived high-quality factors on shared timestamps and
    symbols.  This penalizes behaviorally identical variants even when their
    DSL AST differs.
    """

    def __init__(self, path: str | Path | None, *, top_k: int = 128, min_shared_times: int = 200):
        self.path = Path(path) if path else None
        self.top_k = int(top_k)
        self.min_shared_times = int(min_shared_times)
        self._lock = threading.Lock()
        self.items: dict[str, dict[str, Any]] = {}
        self._compiled_dirty = True
        self._compiled_items: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    h = str(item.get("expr_hash", ""))
                    if h:
                        old = self.items.get(h)
                        if old is None or float(item.get("raw_score", -9)) >= float(old.get("raw_score", -9)):
                            self.items[h] = item
                except Exception:
                    continue
            self._trim_locked()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def _trim_locked(self) -> None:
        ordered = sorted(self.items.values(), key=lambda x: float(x.get("raw_score", -9)), reverse=True)[: self.top_k]
        self.items = {str(x.get("expr_hash")): x for x in ordered if x.get("expr_hash")}
        self._compiled_dirty = True

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.items.values())

    @staticmethod
    def _profile_map(profile: dict[str, Any] | None) -> tuple[list[str], dict[str, np.ndarray]]:
        if not profile:
            return [], {}
        symbols = [str(x) for x in (profile.get("symbols") or [])]
        ts = profile.get("ts") or []
        ranks = profile.get("factor_rank") or []
        n = min(len(ts), len(ranks))
        out: dict[str, np.ndarray] = {}
        for i in range(n):
            try:
                arr = np.asarray([np.nan if x is None else float(x) for x in ranks[i]], dtype=float)
                if len(arr) == len(symbols):
                    out[str(ts[i])] = arr
            except Exception:
                continue
        return symbols, out

    @staticmethod
    def _row_corr(a: np.ndarray, b: np.ndarray) -> float:
        mask = np.isfinite(a) & np.isfinite(b)
        if int(mask.sum()) < 4:
            return float("nan")
        x = a[mask]
        y = b[mask]
        if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _normalize_rows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(mat)
        counts = finite.sum(axis=1).astype(np.float32)
        safe_counts = np.maximum(counts, 1.0)
        filled = np.where(finite, mat, 0.0).astype(np.float32, copy=False)
        mean = filled.sum(axis=1, keepdims=True) / safe_counts[:, None]
        centered = np.where(finite, filled - mean, 0.0).astype(np.float32, copy=False)
        ss = np.square(centered).sum(axis=1)
        valid = (counts >= 4.0) & (ss > 1e-12)
        denom = np.sqrt(np.maximum(ss, 1e-12)).astype(np.float32)
        z = centered / denom[:, None]
        z[~valid, :] = 0.0
        return z.astype(np.float32, copy=False), valid

    @staticmethod
    def _profile_matrix(profile: dict[str, Any] | None) -> dict[str, Any] | None:
        if not profile:
            return None
        symbols = [str(x) for x in (profile.get("symbols") or [])]
        ts_raw = profile.get("ts") or []
        ranks_raw = profile.get("factor_rank") or []
        n = min(len(ts_raw), len(ranks_raw))
        if not symbols or n <= 0:
            return None
        rows: list[np.ndarray] = []
        ts: list[str] = []
        width = len(symbols)
        for i in range(n):
            try:
                arr = np.asarray([np.nan if x is None else float(x) for x in ranks_raw[i]], dtype=np.float32)
            except Exception:
                continue
            if arr.shape[0] != width:
                continue
            rows.append(arr)
            ts.append(str(ts_raw[i]))
        if not rows:
            return None
        mat = np.vstack(rows).astype(np.float32, copy=False)
        z, valid = CrossSectionalBehaviorArchive._normalize_rows(mat)
        return {
            "symbols": symbols,
            "sym_pos": {s: i for i, s in enumerate(symbols)},
            "ts": ts,
            "ts_pos": {t: i for i, t in enumerate(ts)},
            "rank": mat,
            "z": z,
            "valid": valid,
        }

    def _compiled_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            if self._compiled_dirty:
                compiled: list[dict[str, Any]] = []
                for item in self.items.values():
                    cm = self._profile_matrix(item.get("profile") or {})
                    if cm is not None:
                        cm["expr_hash"] = item.get("expr_hash", "")
                        cm["raw_score"] = float(item.get("raw_score", -9))
                        compiled.append(cm)
                self._compiled_items = compiled
                self._compiled_dirty = False
            return list(self._compiled_items)

    def behavior_stats(self, profile: dict[str, Any] | None) -> dict[str, float]:
        cand = self._profile_matrix(profile)
        if cand is None:
            return {"xsec_behavior_samples": 0.0, "xsec_behavior_max_corr": 0.0, "xsec_behavior_mean_corr": 0.0}
        corrs_by_item: list[float] = []
        samples_by_item: list[int] = []
        cand_symbols = cand["symbols"]
        cand_ts_pos = cand["ts_pos"]
        for arch in self._compiled_snapshot():
            arch_symbols = arch["symbols"]
            pairs = [(cand["sym_pos"][s], j) for j, s in enumerate(arch_symbols) if s in cand["sym_pos"]]
            if len(pairs) < 4:
                continue
            shared = [(cand_ts_pos[t], j) for t, j in arch["ts_pos"].items() if t in cand_ts_pos]
            if len(shared) < self.min_shared_times:
                continue
            cti = np.asarray([x[0] for x in shared], dtype=np.int64)
            ati = np.asarray([x[1] for x in shared], dtype=np.int64)
            ci = np.asarray([x[0] for x in pairs], dtype=np.int64)
            ai = np.asarray([x[1] for x in pairs], dtype=np.int64)

            same_symbols = len(pairs) == len(cand_symbols) == len(arch_symbols) and all(
                cand_symbols[k] == arch_symbols[k] for k in range(len(cand_symbols))
            )
            if same_symbols:
                dots = np.einsum("tn,tn->t", cand["z"][cti], arch["z"][ati], optimize=True)
                valid = cand["valid"][cti] & arch["valid"][ati] & np.isfinite(dots)
            else:
                cz, cv = self._normalize_rows(cand["rank"][cti][:, ci])
                az, av = self._normalize_rows(arch["rank"][ati][:, ai])
                dots = np.einsum("tn,tn->t", cz, az, optimize=True)
                valid = cv & av & np.isfinite(dots)
            vals = np.abs(dots[valid])
            if vals.size >= self.min_shared_times:
                corrs_by_item.append(float(np.mean(vals)))
                samples_by_item.append(int(vals.size))
        if not corrs_by_item:
            return {"xsec_behavior_samples": 0.0, "xsec_behavior_max_corr": 0.0, "xsec_behavior_mean_corr": 0.0}
        return {
            "xsec_behavior_samples": float(max(samples_by_item)),
            "xsec_behavior_max_corr": float(max(corrs_by_item)),
            "xsec_behavior_mean_corr": float(np.mean(corrs_by_item)),
        }

    def record(self, expr: str, raw_score: float, profile: dict[str, Any] | None, *, task_id: str = "", start: str = "", end: str = "") -> None:
        if not profile:
            return
        # Require the profile to actually contain cross-sectional rank vectors.
        if not profile.get("symbols") or not profile.get("factor_rank"):
            return
        h = hashlib.sha1(expr.encode("utf-8")).hexdigest()[:12]
        item = {"expr_hash": h, "expr": expr, "raw_score": float(raw_score), "task_id": task_id, "start": start, "end": end, "profile": profile}
        with self._lock:
            old = self.items.get(h)
            if old is not None and float(old.get("raw_score", -9)) > raw_score:
                return
            self.items[h] = item
            self._trim_locked()
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")


class QuantRewardBridge:
    """Computes token-level Verl rewards from generated quant DSL expressions."""

    def __init__(
        self,
        tokenizer,
        bar_minutes: int = 5,
        output_dir: str | None = None,
        start_date: str = "2026-02-23",
        end_date: str = "2026-03-25",
        max_workers: int = 16,
        free_code_eval_timeout_s: int = 120,
        archive_path: str | None = None,
        novelty_penalty: float = 0.25,
        executor: str = "process",
        mode: str = "ocd",
        relative_reward: bool = False,
        improvement_bonus: float = 0.5,
        underperform_penalty: float = 0.0,
        diversity_reward: bool = False,
        exact_repeat_penalty: float = 0.15,
        family_archive_path: str | None = None,
        family_free_quota: int = 8,
        family_low_quality_threshold: float = 0.08,
        family_repeat_penalty: float = 0.02,
        family_good_new_threshold: float = 0.10,
        family_new_bonus: float = 0.02,
        residual_response_reward: bool = False,
        residual_archive_path: str | None = None,
        residual_archive_top_k: int = 128,
        residual_good_score_threshold: float = 0.16,
        residual_bonus: float = 0.04,
        residual_corr_penalty: float = 0.03,
        residual_corr_threshold: float = 0.92,
        residual_min_samples: int = 200,
        residual_record_full: bool = True,
        residual_full_start_date: str = "2026-02-23",
        residual_full_end_date: str = "2026-03-25",
        xsec_behavior_diversity: bool = False,
        xsec_behavior_archive_path: str | None = None,
        xsec_behavior_archive_top_k: int = 128,
        xsec_behavior_good_score_threshold: float = 0.12,
        xsec_behavior_corr_threshold: float = 0.85,
        xsec_behavior_corr_penalty: float = 0.08,
        xsec_behavior_low_corr_bonus_threshold: float = 0.50,
        xsec_behavior_low_corr_bonus: float = 0.02,
        xsec_behavior_min_shared_times: int = 200,
        xsec_behavior_record_full: bool = True,
        reward_kind: str = "direction",
        horizon_bars: int = 1,
        universe_symbols: list[str] | None = None,
        single_asset_symbol: str = "ASSET",
        data_dir: str = "data",
        data_file_template: str = "{symbol_safe}_1m.parquet",
        min_symbols_per_time: int = 8,
        min_times: int = 20,
    ):
        self.tokenizer = tokenizer
        self.bar_minutes = bar_minutes
        self.output_dir = Path(output_dir) if output_dir else None
        self.start_date = start_date
        self.end_date = end_date
        self.cache: dict[tuple[str, str, str, int], tuple[float, dict[str, float]]] = {}
        self.profile_cache: dict[tuple[str, str, str, int], dict[str, Any] | None] = {}
        self._cache_lock = threading.Lock()
        self.max_workers = max(1, int(max_workers))
        self.free_code_eval_timeout_s = max(0, int(free_code_eval_timeout_s))
        self.executor = executor
        self.mode = mode
        self.relative_reward = bool(relative_reward)
        self.improvement_bonus = float(improvement_bonus)
        self.underperform_penalty = float(underperform_penalty)
        self.diversity_reward = bool(diversity_reward)
        self.exact_repeat_penalty = float(exact_repeat_penalty)
        self.family_free_quota = int(family_free_quota)
        self.family_low_quality_threshold = float(family_low_quality_threshold)
        self.family_repeat_penalty = float(family_repeat_penalty)
        self.family_good_new_threshold = float(family_good_new_threshold)
        self.family_new_bonus = float(family_new_bonus)
        self.residual_response_reward = bool(residual_response_reward)
        self.residual_good_score_threshold = float(residual_good_score_threshold)
        self.residual_bonus = float(residual_bonus)
        self.residual_corr_penalty = float(residual_corr_penalty)
        self.residual_corr_threshold = float(residual_corr_threshold)
        self.residual_min_samples = int(residual_min_samples)
        self.residual_record_full = bool(residual_record_full)
        self.residual_full_start_date = str(residual_full_start_date)
        self.residual_full_end_date = str(residual_full_end_date)
        self.xsec_behavior_diversity = bool(xsec_behavior_diversity)
        self.xsec_behavior_good_score_threshold = float(xsec_behavior_good_score_threshold)
        self.xsec_behavior_corr_threshold = float(xsec_behavior_corr_threshold)
        self.xsec_behavior_corr_penalty = float(xsec_behavior_corr_penalty)
        self.xsec_behavior_low_corr_bonus_threshold = float(xsec_behavior_low_corr_bonus_threshold)
        self.xsec_behavior_low_corr_bonus = float(xsec_behavior_low_corr_bonus)
        self.xsec_behavior_min_shared_times = int(xsec_behavior_min_shared_times)
        self.xsec_behavior_record_full = bool(xsec_behavior_record_full)
        self.reward_kind = str(reward_kind)
        self.horizon_bars = int(horizon_bars)
        self.universe_symbols = tuple(str(x) for x in (universe_symbols or []) if str(x).strip())
        self.single_asset_symbol = str(single_asset_symbol)
        self.data_dir = str(data_dir)
        self.data_file_template = str(data_file_template)
        self.min_symbols_per_time = int(min_symbols_per_time)
        self.min_times = int(min_times)
        exact_penalty = self.exact_repeat_penalty if self.diversity_reward else novelty_penalty
        self.archive = NoveltyArchive(archive_path, penalty=exact_penalty) if archive_path and exact_penalty > 0 else None
        if family_archive_path is None and archive_path and self.diversity_reward:
            family_archive_path = str(Path(archive_path).with_suffix(Path(archive_path).suffix + ".family.jsonl"))
        self.family_archive = FamilyArchive(family_archive_path) if self.diversity_reward and family_archive_path else None
        if residual_archive_path is None and archive_path and self.residual_response_reward:
            residual_archive_path = str(Path(archive_path).with_suffix(Path(archive_path).suffix + ".residual.jsonl"))
        self.residual_archive = ResidualResponseArchive(residual_archive_path, top_k=residual_archive_top_k) if self.residual_response_reward and residual_archive_path else None
        if xsec_behavior_archive_path is None and archive_path and self.xsec_behavior_diversity:
            xsec_behavior_archive_path = str(Path(archive_path).with_suffix(Path(archive_path).suffix + ".xsec_behavior.jsonl"))
        self.xsec_behavior_archive = (
            CrossSectionalBehaviorArchive(
                xsec_behavior_archive_path,
                top_k=xsec_behavior_archive_top_k,
                min_shared_times=self.xsec_behavior_min_shared_times,
            )
            if self.xsec_behavior_diversity and xsec_behavior_archive_path
            else None
        )
        self.reward_profile = _single_asset_reward_profile(self.reward_kind, self.horizon_bars)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / ".seen").mkdir(exist_ok=True)

    def _decode_responses(self, data) -> list[str]:
        responses = data.batch["responses"]
        attention_mask = data.batch.get("response_mask", None)
        outs = []
        for i in range(responses.shape[0]):
            ids = responses[i]
            if attention_mask is not None:
                ids = ids[attention_mask[i].bool()]
            else:
                # strip pad tokens best-effort
                ids = ids[ids != self.tokenizer.pad_token_id]
            outs.append(self.tokenizer.decode(ids.tolist(), skip_special_tokens=True))
        return outs

    def _task_values(self, data: Any, key: str, n: int, default: Any = None) -> list[Any]:
        nb = getattr(data, "non_tensor_batch", {}) or {}
        extra = nb.get("extra_info")
        values: list[Any] = []
        for i in range(n):
            val = default
            try:
                if key in nb:
                    val = nb[key][i]
                elif extra is not None and isinstance(extra[i], dict):
                    val = extra[i].get(key, default)
            except Exception:
                val = default
            values.append(val)
        return values

    def _score_expr(self, expr: str, start_date: str | None = None, end_date: str | None = None, bar_minutes: int | None = None) -> tuple[float, dict[str, float]]:
        start = start_date or self.start_date
        end = end_date or self.end_date
        bars = int(bar_minutes or self.bar_minutes)
        key = (expr, start, end, bars)
        with self._cache_lock:
            if key in self.cache:
                return self.cache[key]
        try:
            if self.reward_kind == "cross_sectional_rankic":
                score, metrics, code, predictive_score, _profile = _score_cross_sectional_expr_worker(
                    expr,
                    start,
                    end,
                    bars,
                    self.horizon_bars,
                    self.universe_symbols,
                    self.min_symbols_per_time,
                    self.min_times,
                )
            else:
                score, metrics, code, predictive_score, _profile = _score_expr_worker(
                    expr,
                    start,
                    end,
                    bars,
                    self.single_asset_symbol,
                    self.data_dir,
                    self.data_file_template,
                    self.reward_kind,
                    self.horizon_bars,
                )
            self._save_if_positive(expr, code, predictive_score, metrics)
        except Exception as exc:
            score = -1.0
            metrics = {"error": 1.0}
        with self._cache_lock:
            self.cache[key] = (score, metrics)
        return score, metrics

    def _save_if_positive(self, expr: str, code: str, predictive_score: float, metrics: dict[str, float]) -> None:
        if not self.output_dir:
            return
        if self.reward_kind == "cross_sectional_rankic":
            rank_ic = float(metrics.get("cross_sectional/rank_ic_mean", 0.0))
            coverage = float(metrics.get("cross_sectional/coverage", 0.0))
            if rank_ic < 0.01 or coverage < 0.60:
                return
        else:
            if self.reward_kind in {"direction", "direction_accuracy"}:
                if predictive_score < 2.0:
                    return
                acc = metrics.get("direction_accuracy/direction_acc", 0.0)
                cov = metrics.get("direction_accuracy/coverage", 0.0)
                if acc < 0.51 or cov < 0.60:
                    return
            elif self.reward_kind == "ic":
                ic = float(metrics.get("ic/ic", metrics.get("ic", 0.0)))
                samples = float(metrics.get("ic/samples", metrics.get("samples", 0.0)))
                if ic < 0.01 or samples < 50:
                    return
            elif self.reward_kind == "rank_ic":
                rank_ic = float(metrics.get("rank_ic/rank_ic", metrics.get("rank_ic", 0.0)))
                samples = float(metrics.get("rank_ic/samples", metrics.get("samples", 0.0)))
                if rank_ic < 0.01 or samples < 50:
                    return
        h = hashlib.sha1(expr.encode("utf-8")).hexdigest()[:12]
        marker = self.output_dir / ".seen" / f"{h}.marker"
        try:
            fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            return
        fname = self.output_dir / f"factor_pure_verl_{predictive_score:.2f}_{h}.py"
        label = "Python code hash" if self.mode == "free_code" else "DSL expr"
        fname.write_text(f"# mode: pure_verl\n# {label}: {expr}\n\n{code}", encoding="utf-8")

    def _full_profile_for_expr(self, expr: str, bars: int) -> dict[str, Any] | None:
        key = (expr, self.residual_full_start_date, self.residual_full_end_date, int(bars))
        with self._cache_lock:
            if key in self.profile_cache:
                return self.profile_cache[key]
        try:
            _score, _metrics, _code, _pred, profile = _score_expr_worker(
                expr,
                self.residual_full_start_date,
                self.residual_full_end_date,
                int(bars),
                self.single_asset_symbol,
                self.data_dir,
                self.data_file_template,
                self.reward_kind,
                self.horizon_bars,
            )
        except Exception:
            profile = None
        with self._cache_lock:
            self.profile_cache[key] = profile
        return profile

    def _full_xsec_profile_for_expr(self, expr: str, bars: int, horizon_bars: int) -> dict[str, Any] | None:
        key = (expr, self.residual_full_start_date, self.residual_full_end_date, int(bars), int(horizon_bars), "xsec")
        with self._cache_lock:
            if key in self.profile_cache:
                return self.profile_cache[key]
        try:
            _score, _metrics, _code, _pred, profile = _score_cross_sectional_expr_worker(
                expr,
                self.residual_full_start_date,
                self.residual_full_end_date,
                int(bars),
                int(horizon_bars),
                self.universe_symbols,
                self.min_symbols_per_time,
                self.min_times,
            )
        except Exception:
            profile = None
        with self._cache_lock:
            self.profile_cache[key] = profile
        return profile

    def __call__(self, data, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        prompt_len = data.batch["prompts"].shape[-1]
        response_lengths = data.batch["attention_mask"][:, prompt_len:].sum(dim=1)
        extra = defaultdict(list)
        texts = self._decode_responses(data)
        scores = []
        for text in texts:
            if self.mode == "free_code":
                m = _CODE_RE.search(text)
                expr = m.group(1).strip() if m else ""
                if not expr and "class GeneratedAlphaFactor" in text:
                    expr = text.strip()
            else:
                expr = extract_expr(text)
                if not expr:
                    m = _EXPR_RE.search(text)
                    expr = m.group(1).strip() if m else ""
                if not expr:
                    # Fallback: allow bare DSL line without tags if it looks like op(...).
                    stripped = text.strip().splitlines()[-1].strip() if text.strip() else ""
                    if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*\(.*\)$", stripped):
                        expr = stripped
            scores.append(None)
            extra["expr"].append(expr)

        n_items = len(extra["expr"])
        task_ids = self._task_values(data, "task_id", n_items, "")
        start_dates = self._task_values(data, "start_date", n_items, self.start_date)
        end_dates = self._task_values(data, "end_date", n_items, self.end_date)
        bar_minutes_values = self._task_values(data, "bar_minutes", n_items, self.bar_minutes)
        families = self._task_values(data, "family", n_items, "")
        time_splits = self._task_values(data, "time_split", n_items, "")
        seed_exprs = self._task_values(data, "seed_expr", n_items, "")
        horizon_bars_values = self._task_values(data, "horizon_bars", n_items, self.horizon_bars)

        candidate_jobs = {
            (
                e,
                str(start_dates[i]),
                str(end_dates[i]),
                int(bar_minutes_values[i] or self.bar_minutes),
                int(horizon_bars_values[i] or self.horizon_bars),
            )
            for i, e in enumerate(extra["expr"]) if e
        }
        seed_jobs = set()
        if self.relative_reward and self.mode != "free_code":
            seed_jobs = {
                (
                    str(seed_exprs[i]),
                    str(start_dates[i]),
                    str(end_dates[i]),
                    int(bar_minutes_values[i] or self.bar_minutes),
                    int(horizon_bars_values[i] or self.horizon_bars),
                )
                for i in range(n_items) if str(seed_exprs[i] or "").strip()
            }
        unique_jobs = sorted(candidate_jobs | seed_jobs)
        scored: dict[tuple[str, str, str, int, int], tuple[float, dict[str, float], dict[str, Any] | None, str, float]] = {}
        executor_cls = ProcessPoolExecutor if self.executor == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=min(self.max_workers, max(1, len(unique_jobs)))) as ex:
            if self.executor == "process":
                if self.reward_kind == "cross_sectional_rankic":
                    worker = _score_cross_sectional_code_worker if self.mode == "free_code" else _score_cross_sectional_expr_worker
                    futs = {
                        ex.submit(
                            worker,
                            e,
                            s,
                            en,
                            b,
                            h,
                            self.universe_symbols,
                            self.min_symbols_per_time,
                            self.min_times,
                            self.free_code_eval_timeout_s if self.mode == "free_code" else 0,
                        ): (e, s, en, b, h)
                        for e, s, en, b, h in unique_jobs
                    }
                else:
                    worker = _score_code_worker if self.mode == "free_code" else _score_expr_worker
                    futs = {
                        ex.submit(
                            worker,
                            e,
                            s,
                            en,
                            b,
                            self.single_asset_symbol,
                            self.data_dir,
                            self.data_file_template,
                            self.reward_kind,
                            self.horizon_bars,
                        ): (e, s, en, b, h)
                        for e, s, en, b, h in unique_jobs
                    }
            else:
                futs = {ex.submit(self._score_expr, e, s, en, b): (e, s, en, b, h) for e, s, en, b, h in unique_jobs}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    if self.executor == "process":
                        score, metrics, code, pred, profile = fut.result()
                        self._save_if_positive(key[0], code, pred, metrics)
                        scored[key] = (score, metrics, profile, code, pred)
                        with self._cache_lock:
                            self.cache[key] = (score, metrics)
                    else:
                        score, metrics = fut.result()
                        scored[key] = (score, metrics, None, "", float(metrics.get("predictive_score", 0.0)))
                except Exception:
                    scored[key] = (-1.0, {"error": 1.0}, None, "", 0.0)

        per_sample_metrics: list[dict[str, float]] = []
        all_metric_keys: set[str] = set()
        archive_seen: dict[str, tuple[int, str]] = {}
        if self.archive is not None:
            for expr in set(e for e in extra["expr"] if e):
                archive_seen[expr] = self.archive.seen_count(expr)
        family_seen: dict[str, tuple[int, str, str]] = {}
        if self.family_archive is not None:
            for expr in set(e for e in extra["expr"] if e):
                family_seen[expr] = self.family_archive.seen_count(expr)
        for i, expr in enumerate(extra["expr"]):
            if not expr:
                score, metrics = -1.0, {"format_error": 1.0}
            else:
                key = (
                    expr,
                    str(start_dates[i]),
                    str(end_dates[i]),
                    int(bar_minutes_values[i] or self.bar_minutes),
                    int(horizon_bars_values[i] or self.horizon_bars),
                )
                score, metrics, profile, code, predictive_score = scored.get(key, (-1.0, {"error": 1.0}, None, "", 0.0))
            raw_score = float(score)
            baseline_score = 0.0
            improvement = 0.0
            # Hard failures (format/runtime/coverage errors) should remain a clean
            # -1 reward.  Diversity/family/residual shaping is only meaningful for
            # successfully evaluated candidates; otherwise repeat penalties can
            # push invalid samples below the intended GRPO reward floor.
            shape_candidate = bool(expr) and raw_score > -1.0
            if self.relative_reward and shape_candidate and self.mode != "free_code":
                seed_expr = str(seed_exprs[i] or "").strip()
                if seed_expr:
                    seed_key = (
                        seed_expr,
                        str(start_dates[i]),
                        str(end_dates[i]),
                        int(bar_minutes_values[i] or self.bar_minutes),
                        int(horizon_bars_values[i] or self.horizon_bars),
                    )
                    baseline_score = float(scored.get(seed_key, (0.0, {}, None, "", 0.0))[0])
                    improvement = raw_score - baseline_score
                    score = raw_score + self.improvement_bonus * max(0.0, improvement) - self.underperform_penalty * max(0.0, -improvement)
            seen_count = 0
            exact_penalty = 0.0
            family_penalty = 0.0
            family_bonus = 0.0
            residual_bonus_value = 0.0
            residual_corr_penalty_value = 0.0
            residual_score = 0.0
            residual_samples = 0.0
            behavior_max_corr = 0.0
            xsec_behavior_penalty_value = 0.0
            xsec_behavior_bonus_value = 0.0
            xsec_behavior_samples = 0.0
            xsec_behavior_max_corr = 0.0
            xsec_behavior_mean_corr = 0.0
            expr_hash = ""
            family_count = 0
            family_hash = ""
            if shape_candidate and self.archive is not None:
                seen_count, expr_hash = archive_seen.get(expr, (0, ""))
                exact_penalty = self.archive.penalty if seen_count > 0 else 0.0
                score = float(score) - exact_penalty
            if shape_candidate and self.family_archive is not None:
                family_count, family_hash, _canonical = family_seen.get(expr, (0, "", ""))
                if family_count == 0 and raw_score >= self.family_good_new_threshold:
                    family_bonus = self.family_new_bonus
                elif family_count > self.family_free_quota and raw_score < self.family_low_quality_threshold:
                    family_penalty = self.family_repeat_penalty
                score = float(score) + family_bonus - family_penalty
            if shape_candidate and self.residual_archive is not None:
                rstats = self.residual_archive.residual_stats(profile)
                residual_score = float(rstats.get("residual_score", 0.0))
                residual_samples = float(rstats.get("residual_samples", 0.0))
                behavior_max_corr = float(rstats.get("behavior_max_corr", 0.0))
                if raw_score >= self.residual_good_score_threshold and residual_samples >= self.residual_min_samples:
                    # residual_score is in predictive-score points, e.g. 8.0.
                    # Convert to reward units (divide by 100) and keep shaping
                    # bounded to avoid overwhelming raw GRPO reward.
                    residual_bonus_value = self.residual_bonus * max(0.0, min(1.0, residual_score / 10.0))
                if behavior_max_corr > self.residual_corr_threshold:
                    residual_corr_penalty_value = self.residual_corr_penalty * (behavior_max_corr - self.residual_corr_threshold)
                score = float(score) + residual_bonus_value - residual_corr_penalty_value
            if (
                shape_candidate
                and self.xsec_behavior_archive is not None
                and raw_score >= self.xsec_behavior_good_score_threshold
            ):
                xstats = self.xsec_behavior_archive.behavior_stats(profile)
                xsec_behavior_samples = float(xstats.get("xsec_behavior_samples", 0.0))
                xsec_behavior_max_corr = float(xstats.get("xsec_behavior_max_corr", 0.0))
                xsec_behavior_mean_corr = float(xstats.get("xsec_behavior_mean_corr", 0.0))
                if xsec_behavior_samples >= self.xsec_behavior_min_shared_times and xsec_behavior_max_corr > self.xsec_behavior_corr_threshold:
                    xsec_behavior_penalty_value = self.xsec_behavior_corr_penalty * (xsec_behavior_max_corr - self.xsec_behavior_corr_threshold)
                elif xsec_behavior_samples > 0 and xsec_behavior_max_corr < self.xsec_behavior_low_corr_bonus_threshold:
                    xsec_behavior_bonus_value = self.xsec_behavior_low_corr_bonus
                score = float(score) + xsec_behavior_bonus_value - xsec_behavior_penalty_value
            score = float(np.clip(float(score), -1.0, 1.0))
            scores[i] = score
            enriched = dict(metrics)
            enriched.update({
                "raw_score": raw_score,
                "seed_baseline_score": float(baseline_score),
                "seed_improvement": float(improvement),
                "train_score": float(score),
                "novelty_penalty": float(exact_penalty + family_penalty - family_bonus),
                "exact_repeat_penalty": float(exact_penalty),
                "family_repeat_penalty": float(family_penalty),
                "family_new_bonus": float(family_bonus),
                "residual_bonus": float(residual_bonus_value),
                "residual_corr_penalty": float(residual_corr_penalty_value),
                "residual_score": float(residual_score),
                "residual_samples": float(residual_samples),
                "behavior_max_corr": float(behavior_max_corr),
                "xsec_behavior_penalty": float(xsec_behavior_penalty_value),
                "xsec_behavior_bonus": float(xsec_behavior_bonus_value),
                "xsec_behavior_samples": float(xsec_behavior_samples),
                "xsec_behavior_max_corr": float(xsec_behavior_max_corr),
                "xsec_behavior_mean_corr": float(xsec_behavior_mean_corr),
                "expr_seen_count": float(seen_count),
                "family_seen_count": float(family_count),
            })
            per_sample_metrics.append(enriched)
            all_metric_keys.update(enriched.keys())
            extra["task_id"].append(str(task_ids[i]))
            extra["family"].append(str(families[i]))
            extra["time_split"].append(str(time_splits[i]))
            extra["start_date"].append(str(start_dates[i]))
            extra["end_date"].append(str(end_dates[i]))
            extra["horizon_bars"].append(str(horizon_bars_values[i]))
            extra["expr_hash"].append(str(expr_hash))
            extra["family_hash"].append(str(family_hash))
        if self.archive is not None:
            for i, expr in enumerate(extra["expr"]):
                if expr:
                    self.archive.record(expr, task_id=str(task_ids[i]))
        if self.family_archive is not None:
            for i, expr in enumerate(extra["expr"]):
                if expr:
                    self.family_archive.record(expr, task_id=str(task_ids[i]))
        if self.residual_archive is not None:
            for i, expr in enumerate(extra["expr"]):
                if not expr:
                    continue
                raw = float(per_sample_metrics[i].get("raw_score", -1.0))
                if raw < self.residual_good_score_threshold:
                    continue
                key = (
                    expr,
                    str(start_dates[i]),
                    str(end_dates[i]),
                    int(bar_minutes_values[i] or self.bar_minutes),
                    int(horizon_bars_values[i] or self.horizon_bars),
                )
                _score, _metrics, profile, _code, _pred = scored.get(key, (-1.0, {}, None, "", 0.0))
                if self.residual_record_full:
                    full_profile = self._full_profile_for_expr(expr, int(bar_minutes_values[i] or self.bar_minutes))
                    profile_to_store = full_profile or profile
                    record_start, record_end = self.residual_full_start_date, self.residual_full_end_date
                else:
                    profile_to_store = profile
                    record_start, record_end = str(start_dates[i]), str(end_dates[i])
                self.residual_archive.record(expr, raw, profile_to_store, task_id=str(task_ids[i]), start=record_start, end=record_end)
        if self.xsec_behavior_archive is not None:
            for i, expr in enumerate(extra["expr"]):
                if not expr:
                    continue
                raw = float(per_sample_metrics[i].get("raw_score", -1.0))
                if raw < self.xsec_behavior_good_score_threshold:
                    continue
                key = (
                    expr,
                    str(start_dates[i]),
                    str(end_dates[i]),
                    int(bar_minutes_values[i] or self.bar_minutes),
                    int(horizon_bars_values[i] or self.horizon_bars),
                )
                _score, _metrics, profile, _code, _pred = scored.get(key, (-1.0, {}, None, "", 0.0))
                if self.xsec_behavior_record_full:
                    full_profile = self._full_xsec_profile_for_expr(
                        expr,
                        int(bar_minutes_values[i] or self.bar_minutes),
                        int(horizon_bars_values[i] or self.horizon_bars),
                    )
                    profile_to_store = full_profile or profile
                    record_start, record_end = self.residual_full_start_date, self.residual_full_end_date
                else:
                    profile_to_store = profile
                    record_start, record_end = str(start_dates[i]), str(end_dates[i])
                self.xsec_behavior_archive.record(expr, raw, profile_to_store, task_id=str(task_ids[i]), start=record_start, end=record_end)
        # Verl requires every non-tensor extra_info key to have exactly batch
        # length. Different failure/success paths expose different metric keys,
        # so densify them here instead of appending sparsely.
        for metrics in per_sample_metrics:
            for k in all_metric_keys:
                extra[k].append(float(metrics.get(k, 0.0)))
        reward_scores = torch.tensor(scores, device=reward_tensor.device, dtype=torch.float32)
        reward_tensor[torch.arange(len(data)), response_lengths - 1] = reward_scores
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": extra}
        return reward_tensor
