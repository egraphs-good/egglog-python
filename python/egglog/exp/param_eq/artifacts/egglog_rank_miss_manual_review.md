# Egglog Rank Miss Manual Rule Review

Scope: baseline `param_eq` rank misses in `egglog_rank_misses.csv`, reviewed against the current `analysis_rules`, `basic_rules`, and `fun_rules` in `pipeline.py`. Evidence includes the default longer-run probe and the larger-backoff probe; this is a rule-family review, not a proof of algebraic minimality.

## Summary

- Rows reviewed: 242
- Rows where current rules reduced parameters with more budget: 1
- Rows with a completed larger-budget probe and no parameter reduction: 241
- Rows where the big-backoff rendered form changed but parameter count did not improve: 40

## Classification Counts

- `requires_new_exp_additive_rules`: 113
- `requires_new_log_abs_rules`: 69
- `requires_domain_sensitive_denominator_rules`: 21
- `no_obvious_current_rule_path_or_rank_target_issue`: 20
- `requires_more_log_rules_or_better_log_factoring`: 10
- `requires_polynomial_or_affine_collection`: 8
- `current_rules_can_reduce_with_more_budget`: 1

## Interpretation

- Additional applications of the current rules only produced one parameter-count improvement in the probes: `kotanchek / SBP / raw_index=173 / original`, from 5 to 4 parameters, reaching rank 4.
- Most misses are not budget-only under the current rule set. They cluster around missing rule families: `log(abs(...))` decomposition, exponential additive normalization such as `exp(c + x)`, polynomial/affine coefficient collection, and safe denominator/common-factor rules.
- The disabled common-denominator rewrites matter for some denominator-heavy rows, but re-enabling the old forms would be unsound; those rows need new domain-aware denominator rules rather than more time.
- Rows in `no_obvious_current_rule_path_or_rank_target_issue` had no parameter reduction in completed probes and do not clearly match one of the missing local rule families by text. These are the best candidates for deeper per-row mathematical review or rank/counting investigation.

## Small Examples By Class

### `requires_new_exp_additive_rules`

- `pagie` / `Bingo` / raw `25` / `sympy`: rank 2, baseline params 3, big-backoff params 3.0. `exp(2.377396411352944 * (log(x0 * x1) + 0.3582973557925481) / (log(x0) + 8.605174590777912))`
- `pagie` / `GP-GOMEA` / raw `110` / `original`: rank 3, baseline params 4, big-backoff params 4.0. `1.950389 - 1.108799 * (exp(exp(-6.234) - x0 * x0) + exp(exp(-24.872) - x1 * x1))`
- `pagie` / `GP-GOMEA` / raw `118` / `original`: rank 3, baseline params 4, big-backoff params 4.0. `2.072676 - 0.00052 * (exp(x1 + 8.426 - exp(x1)) + exp(8.611 + x0 - exp(x0)))`

### `requires_new_log_abs_rules`

- `pagie` / `Bingo` / raw `25` / `original`: rank 2, baseline params 4, big-backoff params 4.0. `exp(-2.377396411352944 * ((-10.503285479940024 - log(abs(0.14985143797609368 * x0))) ** (-1.0) * log(abs(1.4308910409656768 * (x0 * x1)))))`
- `kotanchek` / `Bingo` / raw `21` / `original`: rank 3, baseline params 4, big-backoff params 4.0. `2.0 * 0.17850583364543127 * ((x0 + exp(x0)) * (-1.0 * 0.17850583364543127 * (x0 * exp(x0)) + exp(2.0 * 0.17850583364543127 * (x0 + exp(x0))) - log(abs(2.0 * 0.17850583364543127 * (x0 + exp(x0))))) ** (-1.0))`
- `kotanchek` / `Bingo` / raw `14` / `original`: rank 3, baseline params 5, big-backoff params 5.0. `(-0.008870475378536939 + -0.0506336679649604 * x1) * ((x0 + -0.3460574947000204 * (1.0326587094648079 + exp(x0))) ** (-1.0) * log(abs(-9.541496629390634 + 2.0 * x1)))`

### `requires_domain_sensitive_denominator_rules`

- `kotanchek` / `Bingo` / raw `0` / `original`: rank 4, baseline params 5, big-backoff params 5.0. `0.11064466475608078 + -0.010036545250561161 * (2.0 * x0 + 2.0 * x1) + 0.713072197849276 * ((0.022522799045566234 + x0 * x0) * (x0 * x0 + 2.0 ** (-1.0) * (x1 ** (-1.0) * exp(x0 * x0))) ** (-1.0))`
- `kotanchek` / `Bingo` / raw `20` / `original`: rank 4, baseline params 5, big-backoff params 5.0. `-0.03309619870833754 * (-3.819150332778322 + x0 + (0.03907279509253594 + x0 * x0) * ((-15.272972634101004 + -7.53747357012547 * x1) * exp(x0 * x0) ** (-1.0)))`
- `kotanchek` / `PySR` / raw `195` / `sympy`: rank 4, baseline params 5, big-backoff params 5.0. `(0.6193677368705359 * x0 + x1) * (x0 * (x0 + x1 - exp(x1) - 3.302672572491928) + x1 ** 2.0 * (0.4943409476627152 - 2.0 * x1)) / ((x0 + x1 - exp(x1) - 3.302672572491928) * (-1.0 * x1 + exp(x0 ** 2.0) + exp(x1) + 2.853886507925958))`

### `no_obvious_current_rule_path_or_rank_target_issue`

- `kotanchek` / `Bingo` / raw `8` / `sympy`: rank 2, baseline params 5, big-backoff params 5.0. `-0.11720391936222797 * x1 * (x0 * (x0 + 1.2816489216174494 * (0.7212632969011202 * x1 - 1.0) ** 2.0 - 6.4849819024041455) - 0.6931234338478329) * exp(-1.0 * x0)`
- `pagie` / `SBP` / raw `151` / `original`: rank 2, baseline params 5, big-backoff params 5.0. `0.166642 - 0.018385 * (x0 * x1 + ((-5.488 + x1) * (x1 + 0.546) + x1) * x1 + x0 * x0 + exp(-13.828 * x0))`
- `kotanchek` / `Bingo` / raw `26` / `sympy`: rank 5, baseline params 6, big-backoff params 6.0. `(((0.5185123293134022 * x0 + 0.13470217521610287) * (0.0008622481242864235 * exp(x0) - 2.0 * log(x0 + exp(x0))) - 0.11848913269441189 * exp(x0)) * (0.6964203985908678 * exp(x0) - 2.0 * log(x0 + exp(x0))) + 0.2497806421525048 * exp(2.0 * ...`

### `requires_more_log_rules_or_better_log_factoring`

- `pagie` / `Bingo` / raw `24` / `sympy`: rank 3, baseline params 5, big-backoff params 5.0. `0.2866350699692545 * log(x0 * x1) - 0.1177641535011456 * log(-4625426.158330705 / log(0.5182334989833757 / x1)) + 2.988374508357782`
- `kotanchek` / `Bingo` / raw `14` / `sympy`: rank 3, baseline params 5, big-backoff params 5.0. `(0.0506336679649604 * x1 + 0.008870475378536939) * log(2.0 * x1 - 9.541496629390634) / (-1.0 * x0 + 0.3460574947000204 * exp(x0) + 0.35735928587754767)`
- `pagie` / `Bingo` / raw `20` / `sympy`: rank 3, baseline params 5, big-backoff params 5.0. `0.00010507476032088567 * (819.22982975795 * x0 + x1 - 1.9180501896236644) * log(x0 * x1 * (448579.1335037871 * x0 ** 3.0 * x1 ** 3.0 + 569.384534287446)) / x0`

### `requires_polynomial_or_affine_collection`

- `pagie` / `SBP` / raw `167` / `original`: rank 3, baseline params 4, big-backoff params 4.0. `-0.065845 - 0.048266 * (x0 - (x1 + x1) - (x1 + (x1 + (x1 + (3.579 - x0) * ((3.247 - x0) * x0)) - x1 * x1)))`
- `kotanchek` / `Bingo` / raw `27` / `sympy`: rank 4, baseline params 5, big-backoff params 5.0. `(3.678253204741038 * x0 + 1.9305547847352778 * x1 - 0.39686078577236517) / (x0 ** 6.0 - x0 + 2.057897964881403 * x1 + 10.665734721501936)`
- `kotanchek` / `SBP` / raw `166` / `original`: rank 3, baseline params 4, big-backoff params 4.0. `-0.025349 + 0.010788 * ((x0 + x0 - x0 * x0) * (-5.542 + x0) * (-5.542 + x0) + x1 + x1 + 12.501 + x0 + x1)`

### `current_rules_can_reduce_with_more_budget`

- `kotanchek` / `SBP` / raw `173` / `original`: rank 4, baseline params 5, big-backoff params 4.0. `-8.689343 - 0.039334 * (x0 - x1 + (15.347 - (11.471 * (x0 + 19.197) + x0 * x0)) + (-19.039 - (x0 - 7.988) * (x0 * x0)))`

