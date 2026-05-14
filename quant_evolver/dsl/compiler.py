"""
Operator-Constrained Decoding (OCD) — DSL Expression Compiler

Compiles a single-line DSL expression into a valid GeneratedAlphaFactor
Python class, eliminating format errors, runtime loops, and future-bias bugs
by construction.

DSL Grammar (valid Python call syntax):
  Field accessors (→ array):  close(w), open(w), high(w), low(w), volume(w),
                               returns(w), vwap(w)
  Array → Array:  diff, log_arr, normalize, ema_arr, slice_arr,
                  add_arr, sub_arr, mul_arr, div_arr
  Array → Scalar: ts_mean, ts_std, ts_max, ts_min, ts_sum,
                  ts_skew, ts_kurt, last, first, corr, cov
  Scalar → Scalar: neg, abs_s, log_s, tanh_s, sign_s,
                   add, sub, mul, div
  Mixed:          zscore(scalar, array), rank(scalar, array)
  Infix:          +  -  *  /  (Python operators also work)

Example expressions:
  div(ts_mean(returns(60)), ts_std(returns(60)))
  zscore(last(close(120)), close(120))
  corr(returns(60), log_arr(volume(60)))
  sub(zscore(last(close(60)), close(60)), zscore(last(close(240)), close(240)))
  ts_skew(returns(120))
  div(sub(last(close(1)), vwap(60)), ts_std(close(60)))
"""

import ast
import re
import textwrap
from typing import Tuple


ALL_OPS = frozenset([
    "close", "open", "high", "low", "volume", "returns", "vwap",
    "diff", "log_arr", "normalize", "ema_arr", "slice_arr",
    "add_arr", "sub_arr", "mul_arr", "div_arr",
    "ts_mean", "ts_std", "ts_max", "ts_min", "ts_sum",
    "ts_skew", "ts_kurt", "last", "first", "corr", "cov",
    "neg", "abs_s", "log_s", "tanh_s", "sign_s",
    "add", "sub", "mul", "div",
    "zscore", "rank",
])


class ExprCompiler:
    def __init__(self, warmup_param: int = 48):
        self.warmup_param = warmup_param
        self._counter = 0
        self._stmts: list = []

    # ------------------------------------------------------------------ public

    def compile(self, expr_str: str) -> str:
        """Compile DSL expression → full GeneratedAlphaFactor class source."""
        expr_str = expr_str.strip()
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"DSL syntax error: {exc}") from exc

        result, is_arr = self._visit(tree.body)
        if is_arr:
            result = f"float({result}[-1])"

        indent = "            "
        body_lines = [indent + s for s in self._stmts]
        body_lines.append(f"{indent}return float({result})")
        method_body = "\n".join(body_lines)
        return self._wrap_class(method_body)

    # ----------------------------------------------------------------- visitor

    def _visit(self, node) -> Tuple[str, bool]:
        if isinstance(node, ast.Call):
            return self._visit_call(node)
        if isinstance(node, ast.Constant):
            return str(node.value), False
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            inner, _ = self._visit(node.operand)
            return f"(-({inner}))", False
        if isinstance(node, ast.BinOp):
            return self._visit_binop(node)
        raise ValueError(f"Unsupported node: {ast.dump(node)}")

    def _visit_binop(self, node) -> Tuple[str, bool]:
        lc, la = self._visit(node.left)
        rc, ra = self._visit(node.right)
        is_arr = la or ra
        if is_arr:
            var = self._new_var()
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
            if type(node.op) in op_map:
                self._emit(f"{var} = {lc} {op_map[type(node.op)]} {rc}")
            elif isinstance(node.op, ast.Div):
                self._emit(f"{var} = {lc} / (np.abs({rc}) + 1e-8)")
            else:
                raise ValueError(f"Unsupported array op: {type(node.op).__name__}")
            return var, True
        else:
            op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
            if type(node.op) in op_map:
                return f"(({lc}) {op_map[type(node.op)]} ({rc}))", False
            if isinstance(node.op, ast.Div):
                return f"(({lc}) / (abs({rc}) + 1e-8))", False
            raise ValueError(f"Unsupported scalar op: {type(node.op).__name__}")

    def _expect_arity(self, fn: str, args, n: int):
        if len(args) != n:
            raise ValueError(f"{fn}() expects {n} argument(s), got {len(args)}")

    def _expect_array(self, expr: str, is_arr: bool, fn: str) -> str:
        if not is_arr:
            raise ValueError(f"{fn} expects an ARRAY argument, got SCALAR")
        return expr

    def _expect_scalar(self, expr: str, is_arr: bool, fn: str) -> str:
        if is_arr:
            raise ValueError(f"{fn} expects a SCALAR argument, got ARRAY. Wrap arrays with last/ts_mean/ts_std/etc.")
        return expr

    def _align_arrays(self, a: str, b: str) -> tuple[str, str]:
        """Emit aligned suffix arrays to avoid shape mismatch for corr/cov/array ops."""
        n = self._new_var()
        aa = self._new_var()
        bb = self._new_var()
        self._emit(f"{n} = min(len({a}), len({b}))")
        self._emit(f"{aa} = {a}[-{n}:]")
        self._emit(f"{bb} = {b}[-{n}:]")
        return aa, bb

    def _visit_call(self, node) -> Tuple[str, bool]:
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"Only simple function names allowed: {ast.dump(node)}")
        fn = node.func.id
        args = node.args

        # ── field accessors ──────────────────────────────────────────
        if fn in ("close", "open", "high", "low", "volume"):
            self._expect_arity(fn, args, 1)
            w = self._get_window(args, fn)
            var = self._new_var()
            self._emit(f"{var} = self.get_{fn}_array({w})")
            return var, True

        if fn == "returns":
            self._expect_arity(fn, args, 1)
            w = self._get_window(args, fn)
            var = self._new_var()
            self._emit(f"{var} = self.get_returns({w})")
            return var, True

        if fn == "vwap":
            self._expect_arity(fn, args, 1)
            w = self._get_window(args, fn)
            vc, vv, var = self._new_var(), self._new_var(), self._new_var()
            self._emit(f"{vc} = self.get_close_array({w})")
            self._emit(f"{vv} = self.get_volume_array({w})")
            self._emit(f"{var} = float(np.sum({vc} * {vv}) / (np.sum({vv}) + 1e-8))")
            return var, False

        # ── array → array ────────────────────────────────────────────
        if fn == "diff":
            self._expect_arity(fn, args, 1)
            a, is_arr = self._visit(args[0])
            self._expect_array(a, is_arr, fn)
            var = self._new_var()
            self._emit(f"{var} = np.diff({a})")
            return var, True

        if fn == "log_arr":
            self._expect_arity(fn, args, 1)
            a, is_arr = self._visit(args[0])
            self._expect_array(a, is_arr, fn)
            var = self._new_var()
            self._emit(f"{var} = np.log(np.abs({a}) + 1e-8)")
            return var, True

        if fn == "normalize":
            self._expect_arity(fn, args, 1)
            a, is_arr = self._visit(args[0])
            self._expect_array(a, is_arr, fn)
            var = self._new_var()
            self._emit(f"{var} = ({a} - np.mean({a})) / (np.std({a}) + 1e-8)")
            return var, True

        if fn == "ema_arr":
            # Historical name kept for compatibility, but returns a SCALAR EMA value.
            self._expect_arity(fn, args, 2)
            a, is_arr = self._visit(args[0])
            self._expect_array(a, is_arr, fn)
            span = self._get_int(args[1], "ema_arr span")
            var = self._new_var()
            aw, ww = f"_alpha_{var}", f"_w_{var}"
            self._emit(f"{aw} = 2.0 / ({span} + 1)")
            self._emit(f"{ww} = (1 - {aw}) ** np.arange(len({a}) - 1, -1, -1)")
            self._emit(f"{ww} /= ({ww}.sum() + 1e-8)")
            self._emit(f"{var} = float(np.dot({ww}, {a}))")
            return var, False

        if fn == "slice_arr":
            self._expect_arity(fn, args, 2)
            a, is_arr = self._visit(args[0])
            self._expect_array(a, is_arr, fn)
            w = self._get_int(args[1], "slice_arr w")
            var = self._new_var()
            self._emit(f"{var} = {a}[-{w}:]")
            return var, True

        if fn in ("add_arr", "sub_arr", "mul_arr", "div_arr"):
            self._expect_arity(fn, args, 2)
            a, ia = self._visit(args[0])
            b, ib = self._visit(args[1])
            self._expect_array(a, ia, fn)
            self._expect_array(b, ib, fn)
            a, b = self._align_arrays(a, b)
            var = self._new_var()
            if fn == "div_arr":
                self._emit(f"{var} = {a} / (np.abs({b}) + 1e-8)")
            else:
                op = {"add_arr": "+", "sub_arr": "-", "mul_arr": "*"}[fn]
                self._emit(f"{var} = {a} {op} {b}")
            return var, True

        # ── array → scalar ───────────────────────────────────────────
        scalar_map = {
            "ts_mean": "float(np.mean({a}))",
            "ts_std":  "float(np.std({a}) + 1e-8)",
            "ts_max":  "float(np.max({a}))",
            "ts_min":  "float(np.min({a}))",
            "ts_sum":  "float(np.sum({a}))",
            "ts_skew": "float(np.skew({a}))",
            "ts_kurt": "float(np.kurtosis({a}))",
            "last":    "float({a}[-1])",
            "first":   "float({a}[0])",
        }
        if fn in scalar_map:
            self._expect_arity(fn, args, 1)
            a, is_arr = self._visit(args[0])
            self._expect_array(a, is_arr, fn)
            return scalar_map[fn].format(a=a), False

        if fn == "corr":
            self._expect_arity(fn, args, 2)
            a, ia = self._visit(args[0])
            b, ib = self._visit(args[1])
            self._expect_array(a, ia, fn)
            self._expect_array(b, ib, fn)
            a, b = self._align_arrays(a, b)
            return f"float(np.nan_to_num(np.corrcoef({a}, {b})[0, 1], nan=0.0, posinf=0.0, neginf=0.0))", False

        if fn == "cov":
            self._expect_arity(fn, args, 2)
            a, ia = self._visit(args[0])
            b, ib = self._visit(args[1])
            self._expect_array(a, ia, fn)
            self._expect_array(b, ib, fn)
            a, b = self._align_arrays(a, b)
            return f"float(np.nan_to_num(np.cov({a}, {b})[0, 1], nan=0.0, posinf=0.0, neginf=0.0))", False

        # ── mixed ────────────────────────────────────────────────────
        if fn == "zscore":
            self._expect_arity(fn, args, 2)
            s, is_s_arr = self._visit(args[0])
            a, is_a_arr = self._visit(args[1])
            self._expect_scalar(s, is_s_arr, fn)
            self._expect_array(a, is_a_arr, fn)
            return f"(({s}) - float(np.mean({a}))) / (float(np.std({a})) + 1e-8)", False

        if fn == "rank":
            self._expect_arity(fn, args, 2)
            s, is_s_arr = self._visit(args[0])
            a, is_a_arr = self._visit(args[1])
            self._expect_scalar(s, is_s_arr, fn)
            self._expect_array(a, is_a_arr, fn)
            return f"float(np.sum({a} <= ({s}))) / float(len({a}) + 1)", False

        # ── scalar unary ─────────────────────────────────────────────
        unary_map = {
            "neg":    "(-({s}))",
            "abs_s":  "abs({s})",
            "log_s":  "float(np.log(abs({s}) + 1e-8))",
            "tanh_s": "float(np.tanh({s}))",
            "sign_s": "float(np.sign({s}))",
        }
        if fn in unary_map:
            self._expect_arity(fn, args, 1)
            s, is_arr = self._visit(args[0])
            self._expect_scalar(s, is_arr, fn)
            return unary_map[fn].format(s=s), False

        # ── scalar binary ─────────────────────────────────────────────
        bin_map = {"add": "+", "sub": "-", "mul": "*"}
        if fn in bin_map:
            self._expect_arity(fn, args, 2)
            a, ia = self._visit(args[0])
            b, ib = self._visit(args[1])
            self._expect_scalar(a, ia, fn)
            self._expect_scalar(b, ib, fn)
            return f"(({a}) {bin_map[fn]} ({b}))", False

        if fn == "div":
            self._expect_arity(fn, args, 2)
            a, ia = self._visit(args[0])
            b, ib = self._visit(args[1])
            self._expect_scalar(a, ia, fn)
            self._expect_scalar(b, ib, fn)
            return f"(({a}) / (abs({b}) + 1e-8))", False

        raise ValueError(f"Unknown DSL operator '{fn}'. Available: {sorted(ALL_OPS)}")

    # ----------------------------------------------------------------- helpers

    def _new_var(self) -> str:
        self._counter += 1
        return f"_v{self._counter}"

    def _emit(self, line: str):
        self._stmts.append(line)

    def _get_int(self, node, ctx: str = "") -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        raise ValueError(
            f"Expected integer literal{' for ' + ctx if ctx else ''}: {ast.dump(node)}"
        )

    def _get_window(self, args, fn: str) -> int:
        if not args:
            raise ValueError(f"{fn}() requires a window argument")
        return self._get_int(args[0], f"{fn} window")

    def _wrap_class(self, method_body: str) -> str:
        wp = self.warmup_param
        prefix = f"""
import backtrader as bt
import numpy as np
from scipy import stats as _scipy_stats
import math

if not hasattr(np, 'skew'):
    np.skew = _scipy_stats.skew
if not hasattr(np, 'kurtosis'):
    np.kurtosis = _scipy_stats.kurtosis
if not hasattr(np, 'entropy'):
    np.entropy = _scipy_stats.entropy

_ORIG_NP_EXP = np.exp
def _safe_exp(x, *a, **kw):
    return _ORIG_NP_EXP(np.clip(np.asarray(x, dtype=np.float64), -60, 60), *a, **kw)
np.exp = _safe_exp


class BaseAlphaFactor(bt.Indicator):
    lines = ('score',)
    params = (('warmup', {wp}),)

    def __init__(self):
        self.addminperiod(self.p.warmup)
        self._error_count = 0
        self._last_error = None

    def next(self):
        try:
            score = self.compute_score()
        except Exception:
            self._error_count += 1
            if self._last_error is None:
                import traceback
                self._last_error = traceback.format_exc()
            self.lines.score[0] = float('nan')
            return
        if score is None or (isinstance(score, float) and (math.isnan(score) or math.isinf(score))):
            self.lines.score[0] = float('nan')
        else:
            self.lines.score[0] = float(score)

    def compute_score(self):
        raise NotImplementedError

    def _safe_n(self, n):
        return min(n, len(self.data))

    def get_close_array(self, n):
        n = self._safe_n(n)
        return np.array([self.data.close[-i] for i in range(n - 1, -1, -1)])

    def get_open_array(self, n):
        n = self._safe_n(n)
        return np.array([self.data.open[-i] for i in range(n - 1, -1, -1)])

    def get_high_array(self, n):
        n = self._safe_n(n)
        return np.array([self.data.high[-i] for i in range(n - 1, -1, -1)])

    def get_low_array(self, n):
        n = self._safe_n(n)
        return np.array([self.data.low[-i] for i in range(n - 1, -1, -1)])

    def get_volume_array(self, n):
        n = self._safe_n(n)
        return np.array([self.data.volume[-i] for i in range(n - 1, -1, -1)])

    def get_returns(self, n):
        closes = self.get_close_array(n + 1)
        return np.diff(closes) / (closes[:-1] + 1e-8)


class GeneratedAlphaFactor(BaseAlphaFactor):
    params = (('warmup', {wp}),)

    def compute_score(self):
"""
        return textwrap.dedent(prefix).rstrip() + "\n" + method_body.rstrip() + "\n"


# ------------------------------------------------------------------ public API

def compile_expr(expr_str: str, warmup_param: int = 48) -> str:
    """Compile DSL expression to GeneratedAlphaFactor class source."""
    return ExprCompiler(warmup_param).compile(expr_str)


def extract_expr(llm_output: str) -> str:
    """Extract content from <expr>...</expr> tags."""
    m = re.search(r"<expr>(.*?)</expr>", llm_output, re.DOTALL)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    cases = [
        ("Sharpe-like",      "div(ts_mean(returns(60)), ts_std(returns(60)))"),
        ("Z-score",          "zscore(last(close(120)), close(120))"),
        ("Vol-price corr",   "corr(returns(60), log_arr(volume(60)))"),
        ("Multi-scale MR",   "sub(zscore(last(close(60)), close(60)), zscore(last(close(240)), close(240)))"),
        ("Infix sugar",      "ts_mean(returns(60)) / (ts_std(returns(60)) + 0.001)"),
        ("Skewness",         "ts_skew(returns(120))"),
        ("VWAP dev",         "div(sub(last(close(1)), vwap(60)), ts_std(close(60)))"),
        ("Norm returns",     "ts_mean(normalize(returns(120)))"),
    ]
    for name, expr in cases:
        try:
            code = compile_expr(expr, warmup_param=48)
            # Extract just the compute_score body for display
            lines = code.split("\n")
            body = [l for l in lines if "_v" in l or "return" in l]
            print(f"✅ {name}: {expr}")
            print(f"   → {body}")
        except Exception as e:
            print(f"❌ {name}: {e}")
