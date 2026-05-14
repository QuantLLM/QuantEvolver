from __future__ import annotations

import math
from typing import Type

import numpy as np


def _patch_numpy() -> None:
    try:
        from scipy import stats as scipy_stats
    except Exception:  # pragma: no cover
        scipy_stats = None
    if scipy_stats is not None:
        for name in ("skew", "kurtosis", "zscore", "entropy"):
            if not hasattr(np, name) and hasattr(scipy_stats, name):
                setattr(np, name, getattr(scipy_stats, name))
    orig_exp = np.exp
    if getattr(orig_exp, "__name__", "") != "_safe_exp":
        def _safe_exp(x, *args, **kwargs):
            return orig_exp(np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0), *args, **kwargs)
        np.exp = _safe_exp


def load_factor_class_from_code(factor_code: str):
    _patch_numpy()
    try:
        import backtrader as bt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("backtrader is required for factor loading/evaluation") from exc
    namespace = {
        "bt": bt,
        "backtrader": bt,
        "np": np,
        "numpy": np,
        "math": math,
    }
    try:
        import scipy
        from scipy import stats
        namespace.update({"scipy": scipy, "stats": stats})
    except Exception:
        pass
    exec(factor_code, namespace)
    if "GeneratedAlphaFactor" not in namespace:
        raise ValueError("Code does not define GeneratedAlphaFactor")
    return namespace["GeneratedAlphaFactor"]
