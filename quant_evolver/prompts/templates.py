from __future__ import annotations

from quant_evolver.scenarios.schema import ScenarioConfig


def build_seed_generation_prompt(scenario: ScenarioConfig, reward_config: dict, dsl_ops: list[str], n: int = 50) -> str:
    return f"""You are a senior quantitative researcher.
Generate {n} diverse candidate alpha factor seeds for this scenario.

Scenario:
{scenario.to_prompt_context()}

Reward configuration:
{reward_config}

DSL operators:
{', '.join(dsl_ops)}

Return YAML with a top-level `seeds` list. Each seed must contain:
- name
- expr: one valid DSL expression only
- intuition
- category

Cover many distinct categories; do not concentrate on one template. Use categories such as:
momentum, mean_reversion, volume_price_divergence, volatility_breakout,
volatility_compression, range_position, intraday_reversal, liquidity_shock,
trend_strength, return_autocorrelation, volume_acceleration, high_low_range,
open_close_pressure, ema_deviation, price_volume_correlation, rank_normalized_signal.
For each category, vary lookbacks and structures; avoid producing near-duplicates
that only change one number.

STRICT DSL SYNTAX RULES:
- Field accessors always require an integer window: close(60), volume(120), returns(60). Never write bare close/volume.
- returns takes an integer window only: returns(60), NOT returns(close).
- slice_arr takes exactly two args: slice_arr(arr, 60), NOT Python slicing and NOT None.
- ema_arr takes exactly two args: ema_arr(close(120), 20).
- corr/cov require arrays with compatible lengths, e.g. corr(returns(60), log_arr(volume(60))).
- Output scalar expressions; if an operator returns an array, wrap it with last/ts_mean/ts_std/corr/etc.
- ARRAY producers: close(n), open(n), high(n), low(n), volume(n), returns(n), log_arr(array), add_arr/sub_arr/mul_arr/div_arr(...).
- SCALAR producers: last(array), first(array), vwap(n), ts_mean(array), ts_std(array), zscore(value,array), corr(array,array), cov(array,array).
- last(...) and first(...) require an ARRAY. Never write last(vwap(120)), last(ts_mean(...)), or last(zscore(...)).
- ts_mean/ts_std/ts_sum/ts_min/ts_max require an ARRAY. Never write ts_std(vwap(120)).
- Use `last(close(120))`, but use `vwap(120)` directly.
- Use only the listed operator names.

Valid examples:
- div(ts_mean(returns(60)), ts_std(returns(60)))
- neg(zscore(last(close(120)), close(120)))
- corr(returns(60), log_arr(volume(60)))
- div(sub(last(close(1)), vwap(60)), ts_std(close(60)))
- sub(zscore(last(close(60)), close(60)), zscore(last(close(240)), close(240)))

Invalid examples to avoid:
- ema_arr(close, 12)          # bare close is invalid
- returns(close)              # returns needs integer window
- slice_arr(close, -20, None) # Python slicing style is invalid
- sign_s(close(60))           # sign_s expects scalar, not array
- last(vwap(120))             # vwap already returns scalar
- last(ts_mean(close(120)))   # ts_mean already returns scalar

Important: candidates are allowed to be imperfect; they will be backtested and filtered. Prioritize diversity across signal families.
"""


def build_mutation_prompt(scenario: ScenarioConfig, seed_expr: str, seed_metrics: dict | None = None) -> str:
    return f"""You are evolving a quantitative factor.

Scenario:
{scenario.to_prompt_context()}

Seed expression:
<expr>{seed_expr}</expr>

Seed metrics:
{seed_metrics or {}}

Mutate the seed structurally to improve the configured reward. Prefer changing timescale, operator, normalization, sign, or price-volume interaction. Output exactly one improved expression in <expr></expr> tags.

Type reminder: close(n)/volume(n)/returns(n) are ARRAY; last(array), vwap(n), ts_mean(array), zscore(value,array) are SCALAR. Never call last() on a SCALAR such as vwap(), ts_mean(), or zscore().
"""
