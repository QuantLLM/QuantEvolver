from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as _scipy_stats


@dataclass
class _Ctx:
    close: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray
    i: int

    def arr(self, name: str, n: int) -> np.ndarray:
        values = getattr(self, name)
        n = max(1, min(int(n), self.i + 1))
        return values[self.i - n + 1 : self.i + 1].astype(float)


def _safe_num(x):
    if isinstance(x, np.ndarray):
        return np.nan_to_num(x.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        y = float(x)
    except Exception:
        return 0.0
    return y if np.isfinite(y) else 0.0


class _Eval:
    def __init__(self, tree: ast.AST):
        self.tree = tree

    def eval_at(self, ctx: _Ctx) -> float:
        return float(_safe_num(self._visit(self.tree.body, ctx)))

    def _visit(self, node: ast.AST, ctx: _Ctx):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self._visit(node.operand, ctx)
        if isinstance(node, ast.BinOp):
            a = self._visit(node.left, ctx)
            b = self._visit(node.right, ctx)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / (np.abs(b) + 1e-8) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else float(a) / (abs(float(b)) + 1e-8)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            args = [self._visit(a, ctx) for a in node.args]
            return self._call(fn, args, ctx)
        raise ValueError(f"Unsupported DSL node: {ast.dump(node)}")

    def _call(self, fn: str, args: list, ctx: _Ctx):
        if fn in ("close", "open", "high", "low", "volume"):
            return ctx.arr(fn, int(args[0]))
        if fn == "returns":
            closes = ctx.arr("close", int(args[0]) + 1)
            return np.diff(closes) / (closes[:-1] + 1e-8)
        if fn == "vwap":
            c = ctx.arr("close", int(args[0]))
            v = ctx.arr("volume", int(args[0]))
            return float(np.sum(c * v) / (np.sum(v) + 1e-8))
        if fn == "diff":
            return np.diff(np.asarray(args[0], dtype=float))
        if fn == "log_arr":
            return np.log(np.abs(np.asarray(args[0], dtype=float)) + 1e-8)
        if fn == "normalize":
            a = np.asarray(args[0], dtype=float)
            return (a - np.mean(a)) / (np.std(a) + 1e-8)
        if fn == "ema_arr":
            a = np.asarray(args[0], dtype=float)
            span = int(args[1])
            alpha = 2.0 / (span + 1)
            w = (1 - alpha) ** np.arange(len(a) - 1, -1, -1)
            w /= w.sum() + 1e-8
            return float(np.dot(w, a))
        if fn == "slice_arr":
            return np.asarray(args[0], dtype=float)[-int(args[1]) :]
        if fn in ("add_arr", "sub_arr", "mul_arr", "div_arr"):
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            n = min(len(a), len(b))
            a, b = a[-n:], b[-n:]
            if fn == "add_arr":
                return a + b
            if fn == "sub_arr":
                return a - b
            if fn == "mul_arr":
                return a * b
            return a / (np.abs(b) + 1e-8)
        if fn in ("ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum", "ts_skew", "ts_kurt", "last", "first"):
            a = np.asarray(args[0], dtype=float)
            if len(a) == 0:
                return 0.0
            if fn == "ts_mean":
                return float(np.mean(a))
            if fn == "ts_std":
                return float(np.std(a) + 1e-8)
            if fn == "ts_max":
                return float(np.max(a))
            if fn == "ts_min":
                return float(np.min(a))
            if fn == "ts_sum":
                return float(np.sum(a))
            if fn == "ts_skew":
                return float(_safe_num(_scipy_stats.skew(a)))
            if fn == "ts_kurt":
                return float(_safe_num(_scipy_stats.kurtosis(a)))
            if fn == "last":
                return float(a[-1])
            return float(a[0])
        if fn in ("corr", "cov"):
            a = np.asarray(args[0], dtype=float)
            b = np.asarray(args[1], dtype=float)
            n = min(len(a), len(b))
            if n < 2:
                return 0.0
            a, b = a[-n:], b[-n:]
            if fn == "corr":
                if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
                    return 0.0
                return float(_safe_num(np.corrcoef(a, b)[0, 1]))
            return float(_safe_num(np.cov(a, b)[0, 1]))
        if fn == "zscore":
            s = float(args[0])
            a = np.asarray(args[1], dtype=float)
            return (s - float(np.mean(a))) / (float(np.std(a)) + 1e-8)
        if fn == "rank":
            s = float(args[0])
            a = np.asarray(args[1], dtype=float)
            return float(np.sum(a <= s)) / float(len(a) + 1)
        if fn == "neg":
            return -float(args[0])
        if fn == "abs_s":
            return abs(float(args[0]))
        if fn == "log_s":
            return float(np.log(abs(float(args[0])) + 1e-8))
        if fn == "tanh_s":
            return float(np.tanh(float(args[0])))
        if fn == "sign_s":
            return float(np.sign(float(args[0])))
        if fn in ("add", "sub", "mul", "div"):
            a, b = float(args[0]), float(args[1])
            if fn == "add":
                return a + b
            if fn == "sub":
                return a - b
            if fn == "mul":
                return a * b
            return a / (abs(b) + 1e-8)
        raise ValueError(f"Unknown DSL operator: {fn}")


def evaluate_expr_series(expr: str, bars: pd.DataFrame, *, warmup_bars: int = 240) -> pd.Series:
    tree = ast.parse(expr.strip(), mode="eval")
    ev = _Eval(tree)
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in bars]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = bars[required].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    arrays = {c: df[c].to_numpy(dtype=float) for c in required}
    out = []
    idx = []
    start = max(0, int(warmup_bars) - 1)
    for i in range(start, len(df)):
        ctx = _Ctx(arrays["close"], arrays["open"], arrays["high"], arrays["low"], arrays["volume"], i)
        try:
            v = float(_safe_num(ev.eval_at(ctx)))
        except Exception:
            v = float("nan")
        out.append(v)
        idx.append(df.index[i])
    return pd.Series(out, index=pd.DatetimeIndex(idx), dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

