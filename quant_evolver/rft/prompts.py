from __future__ import annotations

from quant_evolver.dsl import ALL_OPS


def _bar_label(bar_minutes: int) -> str:
    known = {
        5: "5-minute",
        15: "15-minute",
        60: "1-hour",
        240: "4-hour",
        1440: "1-day",
    }
    if bar_minutes in known:
        return known[bar_minutes]
    if bar_minutes % 1440 == 0:
        days = bar_minutes // 1440
        return f"{days}-day"
    if bar_minutes % 60 == 0:
        hours = bar_minutes // 60
        return f"{hours}-hour"
    return f"{bar_minutes}-minute"


def _lookback_hint(bar_minutes: int) -> str:
    windows = [20, 60, 120, 240]
    if bar_minutes >= 1440:
        return ", ".join(f"{w} bars = {w * bar_minutes // 1440}d" for w in windows)
    return ", ".join(f"{w} bars = {w * bar_minutes // 60}h" for w in windows)


def common_market_context(bar_minutes: int = 5, reward_kind: str = "direction", horizon_bars: int = 1) -> str:
    label = _bar_label(bar_minutes)
    horizon_min = int(bar_minutes) * int(horizon_bars)
    reward_kind = str(reward_kind or "direction").lower()
    if reward_kind in {"ic", "rank_ic"}:
        reward_desc = "time-series IC (Pearson correlation between factor values and future returns across time)." if reward_kind == "ic" else "time-series RankIC (Spearman correlation between factor ranks and future-return ranks across time)."
        eval_desc = f"factor_score predicts future_return over the next {horizon_bars} bars (~{horizon_min} minutes)."
    else:
        reward_desc = "direction_acc only; reward_score = (direction_acc - 0.5) * 200."
        eval_desc = "sign(factor_score) predicts the NEXT bar close direction."
    return f"""Context:
- Strategy Type: Single Asset Timing.
- Universe: one configured tradable asset.
- Timeframe: {label} bars.
- Evaluation: {eval_desc}
- Reward metric: {reward_desc}
- Each bar is {bar_minutes} minutes. Prefer lookbacks spanning meaningful history:
  {_lookback_hint(bar_minutes)}.
- Avoid future bias. Use only completed current/past bars.
"""


def common_cross_sectional_context(bar_minutes: int = 5, horizon_bars: int = 6) -> str:
    label = _bar_label(bar_minutes)
    horizon_min = int(bar_minutes) * int(horizon_bars)
    return f"""Context:
- Strategy Type: Multi-Asset Cross-Sectional Factor.
- Universe: multiple tradable assets evaluated at the same timestamp.
- Timeframe: {label} bars.
- Evaluation: factor_score ranks symbols cross-sectionally; higher factor values should predict higher future returns.
- Target: future_return over the next {horizon_bars} bars (~{horizon_min} minutes).
- Reward metric: mean cross-sectional RankIC, i.e. Spearman correlation between factor ranks and future-return ranks across symbols at each timestamp, averaged over time.
- Each bar is {bar_minutes} minutes. Prefer lookbacks spanning meaningful history:
  {_lookback_hint(bar_minutes)}.
- Avoid future bias. Use only each symbol's completed current/past bars. Do not use other symbols directly; the evaluator applies the same formula to every symbol and ranks the outputs.
"""


def dsl_help() -> str:
    ops = ", ".join(sorted(ALL_OPS))
    return f"""DSL output format — CRITICAL:
- Output EXACTLY ONE single-line expression inside <expr></expr> tags.
- No imports, no assignments, no class, no markdown code fence.
- Available operators: {ops}

DSL type rules — CRITICAL:
- ARRAY producers: close(n), open(n), high(n), low(n), volume(n), returns(n), log_arr(array), add_arr/sub_arr/mul_arr/div_arr(...).
- SCALAR producers: last(array), first(array), vwap(n), ts_mean(array), ts_std(array), zscore(value,array), corr(array,array), cov(array,array).
- last(...) and first(...) require an ARRAY. Never write last(vwap(120)), last(ts_mean(...)), or last(zscore(...)).
- ts_mean/ts_std/ts_sum/ts_min/ts_max require an ARRAY. Never write ts_std(vwap(120)).
- Use `last(close(120))`, but use `vwap(120)` directly.

Common examples:
<expr>div(ts_mean(returns(60)), ts_std(returns(60)))</expr>
<expr>neg(zscore(last(close(120)), close(120)))</expr>
<expr>corr(returns(60), log_arr(volume(60)))</expr>
<expr>div(sub(last(close(1)), vwap(60)), ts_std(close(60)))</expr>
<expr>sub(zscore(last(close(60)), close(60)), zscore(last(close(240)), close(240)))</expr>
"""


def free_code_help() -> str:
    return """Python output format — CRITICAL:
- Output EXACTLY ONE complete Python code block inside <code></code> tags.
- Define class GeneratedAlphaFactor(bt.Indicator).
- The indicator must expose lines = ('score',).
- compute_score/next must use only current and past bars: self.data.close[0], self.data.close[-1], etc.
- Use numpy as np; backtrader is available as bt.
- Return/set a finite float score; use defensive checks for nan/inf/division by zero.
- No future leakage, no file/network IO, no external data.
- Prefer compact code with helper methods for close/open/high/low/volume arrays.
"""


def build_rft_messages(
    mode: str,
    seed_expr: str | None = None,
    bar_minutes: int = 5,
    *,
    seed_name: str = "",
    family: str = "",
    time_split: str = "",
    start_date: str = "",
    end_date: str = "",
    scenario: str = "",
    mutation_hints: list[str] | None = None,
    reward_kind: str = "direction",
    horizon_bars: int = 1,
) -> list[dict[str, str]]:
    is_cross_sectional = reward_kind == "cross_sectional_rankic"
    market_context = common_cross_sectional_context(bar_minutes, horizon_bars) if is_cross_sectional else common_market_context(bar_minutes, reward_kind, horizon_bars)
    objective = (
        "multi-asset cross-sectional future-return ranking"
        if is_cross_sectional
        else ("single-asset future-return forecasting" if reward_kind in {"ic", "rank_ic"} else "single-asset direction prediction")
    )
    system = f"""You are an expert quantitative factor researcher.
Design one alpha factor for {objective}.

{market_context}

{dsl_help()}

Good factors are non-trivial, numerically stable, have enough temporal variance, and use multi-timescale or price-volume structure.
"""
    if mode == "seeded_ocd":
        if not seed_expr:
            raise ValueError("seeded_ocd requires seed_expr")
        task_context = ""
        if seed_name or family or time_split or start_date or end_date or scenario or mutation_hints:
            task_context = f"""
Task context:
- Seed name: {seed_name or 'unnamed_seed'}
- Seed family: {family or 'general'}
- Time split: {time_split or 'default'} {f'({start_date} to {end_date})' if start_date or end_date else ''}
- Scenario variant: {scenario or ('default cross-sectional RankIC' if is_cross_sectional else 'default single-asset next-bar direction')}
- Mutation hints: {', '.join(mutation_hints or []) if mutation_hints else 'change structure, timescale, normalization, or interaction'}
"""
        user = f"""Do not think step by step. Do not output <think>. Output only one <expr>...</expr> line.

You are given a seed factor generated by a strong model or previous search.
{task_context}
Seed expression:
<expr>{seed_expr}</expr>

Mutate it structurally to improve {'cross-sectional RankIC against future returns' if is_cross_sectional else ('time-series future-return correlation' if reward_kind in {'ic', 'rank_ic'} else 'next-bar direction accuracy')}. Prefer one of:
- change timescales,
- replace an operator,
- add price-volume or volatility normalization,
- invert the signal if the seed seems contrarian,
- combine short-vs-long variants.

Output only the improved expression in <expr></expr> tags.

/no_think
"""
    elif mode == "ocd":
        task_context = ""
        if family or time_split or start_date or end_date or scenario or mutation_hints:
            task_context = f"""
Task context:
- Task family: {family or 'no_seed'}
- Time split: {time_split or 'default'} {f'({start_date} to {end_date})' if start_date or end_date else ''}
- Scenario variant: {scenario or ('default cross-sectional RankIC' if is_cross_sectional else 'default single-asset next-bar direction')}
- Design hint: {', '.join(mutation_hints or []) if mutation_hints else 'from scratch; diversify operators, timescales, normalization, and price-volume interactions'}
"""
        single_asset_goal = "predict future returns with strong time-series correlation" if reward_kind in {"ic", "rank_ic"} else "predict next-bar direction"
        user = f"""Do not think step by step. Do not output <think>.
{task_context}
Design a novel alpha factor from scratch using the DSL. Prefer multi-timescale, ratio-based, or nonlinear price-volume constructions that can {('rank assets by future returns' if is_cross_sectional else single_asset_goal)}. Output only <expr>...</expr>.

/no_think"""
    elif mode == "seed_only":
        if not seed_expr:
            raise ValueError("seed_only requires seed_expr")
        user = f"Do not think step by step. Do not output <think>. For logging only, echo this seed expression: <expr>{seed_expr}</expr>\n\n/no_think"
    elif mode == "free_code":
        task_context = ""
        if family or time_split or start_date or end_date or scenario or mutation_hints:
            task_context = f"""
Task context:
- Task family: {family or 'free_code'}
- Time split: {time_split or 'default'} {f'({start_date} to {end_date})' if start_date or end_date else ''}
- Scenario variant: {scenario or 'default single-asset next-bar direction'}
- Design hint: {', '.join(mutation_hints or []) if mutation_hints else 'from scratch Python factor; diversify logic, timescales, normalization, and price-volume interactions'}
"""
        system = f"""You are an expert quantitative factor researcher.
Design one Python alpha factor for {objective}.

{market_context}

{free_code_help()}

Good factors are non-trivial, numerically stable, have enough temporal variance, and use multi-timescale or price-volume structure.
"""
        user = f"""Do not think step by step. Do not output <think>.
{task_context}
Design a novel Backtrader Python alpha factor from scratch. Output only <code>...</code>.

/no_think"""
    else:
        raise ValueError(f"pure_verl currently supports free_code/ocd/seeded_ocd/seed_only, got {mode}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
