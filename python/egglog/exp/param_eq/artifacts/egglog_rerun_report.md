# Egglog Baseline Rerun Report

## Overview

- Total rows: `714`
- Rows with rank: `682`
- Before rank misses: `242`
- After rank misses: `183`
- Fixed by extra rules: `59`
- Still missed: `183`
- Newly missed: `0`
- Execution issues: `0`
- Under-rank rows: `48`
- Param regressions: `1`

## Outcomes

| Outcome | Rows |
| --- | --- |
| fixed_by_extra_rules | 58 |
| missing_rank | 32 |
| still_missed | 183 |
| unchanged_ranked | 393 |
| under_rank | 48 |

## Remaining Misses By Cause

| Remaining issue | Rows |
| --- | --- |
| existing_rules_more_iterations | 5 |
| needs_many_params_or_rank_artifact | 33 |
| rule_added_but_not_matching | 53 |
| rule_not_added | 12 |
| unreviewed_remaining | 80 |

### Longer-Run Probe Status

| Probe status | Rows |
| --- | --- |
| ok | 162 |
| timeout | 21 |

### Remaining Misses By Review Family

| Rule family | Rows |
| --- | --- |
| affine_constant_collection | 14 |
| affine_duplicate_collection | 4 |
| coefficient_lattice_factoring | 4 |
| exp_additive_constant_extraction | 14 |
| exp_product_or_quotient_normalization | 11 |
| log_abs_scale_extraction | 7 |
| no_rank_reducing_rule | 23 |
| pending | 80 |
| quotient_coefficient_exposure | 10 |
| shared_parameter_or_rank_artifact | 6 |
| square_quotient_normalization | 4 |
| tolerance_coefficient_snapping | 6 |

## Smallest Fixed Previous Misses

| Key | Before gap | After gap | Family | Expression |
| --- | --- | --- | --- | --- |
| kotanchek/Bingo/raw 0/original | 1.0 | 0.0 | quotient_coefficient_exposure | 0.11064466475608078 + -0.010036545250561161 * (2.0 * x0 + 2.0 * x1) + 0.713072197849276 * ((0.022522799045566234 + x0... |
| kotanchek/GP-GOMEA/raw 100/original | 1.0 | 0.0 | exp_additive_constant_extraction | 0.012093 - 0.012131 * (exp(x0 + x0 - x0 * x0) * ((2.494 - 14.259) * x1 + exp(x1 - 0.338))) |
| pagie/Bingo/raw 6/original | 1.0 | 0.0 | log_abs_scale_extraction | 0.33963694568414915 * log(abs(-35.167844692926785 * (-1.0 * (2.0 ** (-1.0) * 1.1727867299483856 ** (-1.0)) * (x0 * ((... |
| pagie/EPLEX/raw 42/sympy | 1.0 | 0.0 | log_abs_scale_extraction | log(abs(log(abs(x0 ** 2.0 + 0.694)) + 2.03252032520325 * log(abs(log(abs((13.991224004743 * x1 ** 4.0 + 0.867051) / x... |
| kotanchek/SBP/raw 167/original | 1.0 | 0.0 | exp_additive_constant_extraction | 0.008014 - 0.000927 * ((x1 + (4.989 - 9.32)) * ((3.46 + x1 + x1) * (exp(x0 + (x0 - (-1.978 + x0 * x0))) * x1))) |
| pagie/SBP/raw 159/original | 1.0 | 0.0 | exp_product_or_quotient_normalization | 0.154306 - 0.001732 * ((-20.41 + exp(11.767) * (x0 + (x0 + (x0 + x1))) * (-1.883 + x0) * exp(-7.374 - x0)) * x0 - x1) |
| pagie/EPLEX/raw 52/sympy | 1.0 | 0.0 | log_abs_scale_extraction | log(abs(-0.181476 * log(abs(0.433)) - log(abs((x1 * (0.455625 * x0 ** 2.0 + 0.2386768) + 0.0489999999999999) * log(ab... |
| kotanchek/SBP/raw 171/original | 1.0 | 0.0 | exp_additive_constant_extraction | 0.033047 - 0.000135 * ((exp(x1) * x1 + exp(x0) * (-2.136 * exp(x0)) * exp(5.051 - x0 * x0)) * (-2.136 + (x1 + 3.852))) |
| pagie/PySR/raw 196/original | 1.0 | 0.0 | exp_product_or_quotient_normalization | (exp(x0 * (x1 / (x0 / x1) - x0) - -0.5903025615190395) + x1 * x1 * (x0 / (x0 * exp(x1 * x1)) / 0.9318791503632365) - ... |
| pagie/Bingo/raw 21/original | 3.0 | 0.0 | square_quotient_normalization | log(abs(log(abs(5.852257156264611 + exp(log(abs(0.0004240447616209535 * (x0 * x1)))) * exp(log(abs(0.0004240447616209... |
| kotanchek/SBP/raw 173/original | 1.0 | 0.0 | existing_rules_only | -8.689343 - 0.039334 * (x0 - x1 + (15.347 - (11.471 * (x0 + 19.197) + x0 * x0)) + (-19.039 - (x0 - 7.988) * (x0 * x0))) |
| kotanchek/PySR/raw 187/original | 1.0 | 0.0 | exp_additive_constant_extraction | exp(x0 + x0 - x0 * (x0 + 0.040585192647742296) + -1.0149512623308583) / (x1 * (exp(x1) * 0.040585192647742296) + (3.8... |
| kotanchek/PySR/raw 182/sympy | 1.0 | 0.0 | quotient_coefficient_exposure | (-1.8973651416268622 * x0 * x1 * (-1.0 * x1 + 0.0454913207896129 * exp(x1) + 0.3415087039348648) + (0.168137828555371... |
| pagie/EPLEX/raw 55/sympy | 1.0 | 0.0 | exp_additive_constant_extraction | log(abs(log(abs(-0.876)))) / log(abs(exp(2.0 * log(abs(0.607)) - 2.0 * exp(6.0 * log(abs(-0.826)) + 3.0 * exp(log(abs... |
| pagie/EPLEX/raw 59/sympy | 1.0 | 0.0 | log_abs_scale_extraction | exp(log(abs(log(abs(log(abs(-13.1752305665349 * x0 * x1 * (exp(log(abs(log(abs(-0.829187396351575 * x0 * x1)) + 0.016... |
| kotanchek/GP-GOMEA/raw 102/original | 1.0 | 0.0 | exp_additive_constant_extraction | -0.108816 - 6.5e-05 * ((-8.386 * x0 - (x1 + 0.898)) * exp(7.588 - x0) - (x0 - x1) * (13.986 + 18.716) * (exp(x1) - (x... |
| pagie/EPLEX/raw 56/original | 1.0 | 0.0 | square_quotient_normalization | exp(-0.777 / ((x1 * (exp(-0.306 + x1) - x1)) ** 2.0 * 1.032)) - exp(0.351 / (x0 * 0.3) ** 2.0 / -2.161 * exp(-0.942))... |
| kotanchek/PySR/raw 209/sympy | 3.0 | 0.0 | quotient_coefficient_exposure | x1 * (x0 + x1 - 0.01811268853996139 * (-2.487415653599196 * x0 + exp(x0) + 0.7901026138870583) * (-3.0 * x1 + exp(x1)... |
| kotanchek/PySR/raw 207/original | 1.0 | 0.0 | exp_additive_constant_extraction | (x1 * (x0 * x1 + ((x1 * (0.8308993593471311 * x0) - (log(1.0353531371341038) * exp(x1 - -0.35375558989724504) + 0.112... |
| kotanchek/PySR/raw 194/original | 1.0 | 0.0 | quotient_coefficient_exposure | exp((x1 * x1 + x0 * x0 + x0) * -0.17830776423842798) / 1.4092295785725586 * x1 / ((exp(x0) - x0 - x0) / 0.34806215833... |

## Smallest Remaining Misses

| Key | Gap | Cause | Family | Expression |
| --- | --- | --- | --- | --- |
| pagie/Bingo/raw 25/sympy | 1.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | exp(2.377396411352944 * (log(x0 * x1) + 0.3582973557925481) / (log(x0) + 8.605174590777912)) |
| pagie/Bingo/raw 25/original | 1.0 | rule_added_but_not_matching | log_abs_scale_extraction | exp(-2.377396411352944 * ((-10.503285479940024 - log(abs(0.14985143797609368 * x0))) ** (-1.0) * log(abs(1.4308910409... |
| pagie/GP-GOMEA/raw 110/original | 1.0 | rule_added_but_not_matching | exp_additive_constant_extraction | 1.950389 - 1.108799 * (exp(exp(-6.234) - x0 * x0) + exp(exp(-24.872) - x1 * x1)) |
| pagie/GP-GOMEA/raw 118/original | 1.0 | rule_added_but_not_matching | exp_additive_constant_extraction | 2.072676 - 0.00052 * (exp(x1 + 8.426 - exp(x1)) + exp(8.611 + x0 - exp(x0))) |
| kotanchek/Bingo/raw 21/original | 1.0 | rule_added_but_not_matching | quotient_coefficient_exposure | 2.0 * 0.17850583364543127 * ((x0 + exp(x0)) * (-1.0 * 0.17850583364543127 * (x0 * exp(x0)) + exp(2.0 * 0.178505833645... |
| kotanchek/GP-GOMEA/raw 114/original | 1.0 | rule_added_but_not_matching | affine_constant_collection | 0.010559 + 0.013455 * ((4.203 - x1 + exp(-3.785)) * (x1 * x1 + (x1 + x1)) * exp(x0 + x0 - x0 * x0)) |
| pagie/SBP/raw 167/original | 1.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | -0.065845 - 0.048266 * (x0 - (x1 + x1) - (x1 + (x1 + (x1 + (3.579 - x0) * ((3.247 - x0) * x0)) - x1 * x1))) |
| pagie/SBP/raw 171/original | 2.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | 0.028119 + 0.006351 * exp(x0 + x0 - x0 * x0 - (x1 + x0 + (-4.674 - x0)) * (x0 - x1 * -0.703 - x0)) |
| pagie/SBP/raw 171/sympy | 2.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | 0.006351 * exp(-1.0 * x0 ** 2.0 + 2.0 * x0 - 0.703 * x1 ** 2.0 + 3.285822 * x1) + 0.028119 |
| pagie/SBP/raw 175/sympy | 2.0 | needs_many_params_or_rank_artifact | exp_product_or_quotient_normalization | (0.00540440450573385 * x1 ** 2.0 * (x1 - 6.944) * (x1 - 4.3) * exp(x0) + 0.015024 * exp(x0 * (x0 - 1.0))) * exp(x0 * ... |
| pagie/Bingo/raw 24/sympy | 2.0 | rule_added_but_not_matching | affine_constant_collection | 0.2866350699692545 * log(x0 * x1) - 0.1177641535011456 * log(-4625426.158330705 / log(0.5182334989833757 / x1)) + 2.9... |
| pagie/GP-GOMEA/raw 96/original | 1.0 | rule_added_but_not_matching | exp_additive_constant_extraction | 1.918745 - 23802.792114 * (exp(-10.432 - x0 * x0) + exp(2.7 - 12.507 - x1 * x1)) |
| kotanchek/Bingo/raw 14/sympy | 2.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | (0.0506336679649604 * x1 + 0.008870475378536939) * log(2.0 * x1 - 9.541496629390634) / (-1.0 * x0 + 0.346057494700020... |
| kotanchek/Bingo/raw 14/original | 2.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | (-0.008870475378536939 + -0.0506336679649604 * x1) * ((x0 + -0.3460574947000204 * (1.0326587094648079 + exp(x0))) ** ... |
| kotanchek/Bingo/raw 8/sympy | 3.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | -0.11720391936222797 * x1 * (x0 * (x0 + 1.2816489216174494 * (0.7212632969011202 * x1 - 1.0) ** 2.0 - 6.4849819024041... |
| kotanchek/Bingo/raw 27/original | 1.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | -1.9305547847352778 * ((10.665734721501936 + x0 * (x0 * x0) * (x0 * (x0 * x0)) + 2.057897964881403 * x1 - x0) ** (-1.... |
| kotanchek/Bingo/raw 27/sympy | 1.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | (3.678253204741038 * x0 + 1.9305547847352778 * x1 - 0.39686078577236517) / (x0 ** 6.0 - x0 + 2.057897964881403 * x1 +... |
| kotanchek/Bingo/raw 21/sympy | 1.0 | rule_added_but_not_matching | quotient_coefficient_exposure | 0.35701166729086253 * (-1.0 * x0 - exp(x0)) / (0.17850583364543127 * x0 * exp(x0) - exp(0.35701166729086253 * x0 + 0.... |
| pagie/Bingo/raw 20/sympy | 2.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | 0.00010507476032088567 * (819.22982975795 * x0 + x1 - 1.9180501896236644) * log(x0 * x1 * (448579.1335037871 * x0 ** ... |
| pagie/SBP/raw 169/original | 1.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | 0.012782 + 0.160826 * (exp(x0 * x0 - exp(x0)) * (x0 - (-13.585 - x0 + exp(2.825 - x1 * x1)) * x0 + 2.825)) |
| pagie/SBP/raw 175/original | 2.0 | rule_added_but_not_matching | exp_additive_constant_extraction | 0.015024 + 291401.986945 * (exp(-17.803 + x0) * ((x1 - 6.944) * x1) * ((x1 - 5.3) * x1 + x1) * exp(x0 - x0 * x0)) |
| kotanchek/SBP/raw 166/original | 1.0 | rule_added_but_not_matching | square_quotient_normalization | -0.025349 + 0.010788 * ((x0 + x0 - x0 * x0) * (-5.542 + x0) * (-5.542 + x0) + x1 + x1 + 12.501 + x0 + x1) |
| kotanchek/SBP/raw 150/original | 1.0 | rule_added_but_not_matching | exp_additive_constant_extraction | 0.053829 - 0.000194 * ((exp(x0 + 19.367 - x0 * x0) * exp(-13.819 + x0) - x1 * exp(x1)) * (-1.237 - x1) + exp(x0)) |
| pagie/Bingo/raw 23/original | 1.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | -5.64393372152411 + 0.002708862303985314 * x1 ** (-1.0) + -0.06435725235582418 * ((-2311.5964730737646 - x0) * log(ab... |
| pagie/SBP/raw 151/original | 3.0 | needs_many_params_or_rank_artifact | no_rank_reducing_rule | 0.166642 - 0.018385 * (x0 * x1 + ((-5.488 + x1) * (x1 + 0.546) + x1) * x1 + x0 * x0 + exp(-13.828 * x0)) |
| kotanchek/Bingo/raw 20/original | 1.0 | needs_many_params_or_rank_artifact | exp_product_or_quotient_normalization | -0.03309619870833754 * (-3.819150332778322 + x0 + (0.03907279509253594 + x0 * x0) * ((-15.272972634101004 + -7.537473... |
| kotanchek/PySR/raw 190/original | 2.0 | rule_added_but_not_matching | affine_constant_collection | exp(x0 + -0.21546645087388747 - x0 * x0 - (x1 - x0)) * ((x1 + (exp((x1 - x1 * x1 + (x1 + 0.7393127343318426)) * x1) +... |
| kotanchek/PySR/raw 195/sympy | 1.0 | rule_added_but_not_matching | quotient_coefficient_exposure | (0.6193677368705359 * x0 + x1) * (x0 * (x0 + x1 - exp(x1) - 3.302672572491928) + x1 ** 2.0 * (0.4943409476627152 - 2.... |
| pagie/GP-GOMEA/raw 109/original | 1.0 | rule_added_but_not_matching | exp_additive_constant_extraction | 1.949837 - 1e-05 * (exp(9.658 + 1.885 - x1 * x1) + exp(10.589 + 1.026 - x0 * x0)) |
| pagie/GP-GOMEA/raw 111/original | 1.0 | rule_added_but_not_matching | exp_product_or_quotient_normalization | 1.93893 - 5.4461 * (exp(-0.454 - 1.011 - x1 * x1) + exp(-1.344 - 0.479 - x0 * x0)) |

## Cause Interpretation

- `existing_rules_more_iterations`: the longer-run probe improved or reached rank; this is budget/scheduler-sensitive.
- `rule_added_but_not_matching`: the reviewed family is in the new minimal rule set, but the rerun still missed; inspect orientation, guards, or needed composition.
- `rule_not_added`: the reviewed family was intentionally not part of this minimal rule set.
- `needs_many_params_or_rank_artifact`: reviewed as irreducible under tree occurrence counting or likely rank/counting artifact.
- `unreviewed_remaining`: no completed usable review taxonomy exists for the row.
