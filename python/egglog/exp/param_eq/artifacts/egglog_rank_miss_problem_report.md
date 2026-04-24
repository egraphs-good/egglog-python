# Egglog Rank Miss Problem Report

This report groups the currently reviewed rank-miss rows by the obstruction identified during manual/agent review. It only includes rows with `agent_status=done`; pending rows are not summarized as evidence.

## Overview

- Reviewed rows: `132`
- Pending rows not included: `110`
- New rule would reach rank: `83`
- New rule would partially reduce only: `12`
- Cannot reduce further or rank artifact: `36`
- More runs of existing rules likely enough: `1`

### Counts By Problem

| Problem category | Rows | Reaches rank | Partial only | Cannot reduce | More runs |
|---|---:|---:|---:|---:|---:|
| Exp Additive Constant Extraction | 23 | 21 | 2 | 0 | 0 |
| Quotient Coefficient Exposure | 16 | 14 | 2 | 0 | 0 |
| Affine Constant Collection | 14 | 11 | 3 | 0 | 0 |
| Log Abs Scale Extraction | 13 | 10 | 3 | 0 | 0 |
| Exp Product Or Quotient Normalization | 9 | 9 | 0 | 0 | 0 |
| Square Quotient Normalization | 6 | 6 | 0 | 0 | 0 |
| Tolerance Coefficient Snapping | 6 | 5 | 1 | 0 | 0 |
| Coefficient Lattice Factoring | 4 | 4 | 0 | 0 | 0 |
| Affine Duplicate Or Signed Constant Collection | 4 | 3 | 1 | 0 | 0 |
| Cannot Reduce Further Or Rank Artifact | 36 | 0 | 0 | 36 | 0 |
| More Runs Of Existing Rules | 1 | 0 | 0 | 0 | 1 |

### Counts By Original Conclusion

- `new_rule_reduces_to_rank`: 83
- `needs_many_params_or_rank_artifact`: 36
- `new_rule_partial_reduction`: 12
- `existing_rules_more_iterations`: 1

## Exp Additive Constant Extraction

- Rows: `23`
- Rule/action: Add rules that pull numeric additive constants out of exponentials and absorb the resulting exp(constant) into an existing multiplicative coefficient. Common forms: `a / exp(x + b) -> (a / exp(b)) / exp(x)`, `k * exp(t + c) -> (k * exp(c)) * exp(t)`, and quadratic-completion variants for `exp((a - x) * (x - b))` when a free outer scale can absorb the constant.

### Outcomes

- `new_rule_reduces_to_rank`: 21
- `new_rule_partial_reduction`: 2

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 53 | pagie/GP-GOMEA/raw 110/original | 1 | `new_rule_reduces_to_rank` | exp(c - x) -> exp(c) * exp(-x); exp(c + x) -> exp(c) * exp(x) | Existing rules can finish after the new exp rule exposes constants. |
| 54 | pagie/GP-GOMEA/raw 118/original | 1 | `new_rule_reduces_to_rank` | k*(exp((u+c1)-v)+exp((w+c2)-z)) -> (k*exp(c1))*exp(u-v)+(k*exp(c2))*exp(w-z); include symmetric c+u/u+c forms | Plain exp(c+u)->exp(c)*exp(u) exposes constants, but reaching rank needs distribution/scalar absorption across the exp sum. |
| 59 | pagie/GP-GOMEA/raw 96/original | 1 | `new_rule_reduces_to_rank` | a*(k - exp(c+u) - exp(d+v)) -> (a*exp(c))*(k/exp(c) - exp(u) - exp(d-c)*exp(v)); include sign/order variants | Reaches rank if scaled exp additive extraction is available; no current-rule-only path. |
| 60 | kotanchek/GP-GOMEA/raw 100/original | 1 | `new_rule_reduces_to_rank` | for af>0: af*(z*(bf*x+exp(y+cf))) -> z*((af*bf)*x+exp(y+(cf+log(af)))); include subtraction and term-order variants | New rule reduces 4->3; current-rule-only path not found. |
| 63 | pagie/SBP/raw 175/original | 3 | `new_rule_partial_reduction` | Num(a)*exp(Num(b)+x) -> Num(a*b.exp())*exp(x), plus exp(x+Num(b)) and commuted product variants; finite/profitable/non-subsuming guards | Still not rank 2; remaining intercept, scale, and two distinct roots are generically independent. Likely finite-sample/rank artifact after partial reduction. |
| 64 | kotanchek/SBP/raw 150/original | 1 | `new_rule_reduces_to_rank` | exp((u+Num(a))-v) * exp(Num(b)+w) -> exp(Num(a+b)) * exp((u-v)+w), with order variants; apply when ParamCost lowers | Reaches rank; needs variants to expose constants in mixed add/sub shapes. |
| 66 | pagie/GP-GOMEA/raw 92/original | 1 | `new_rule_reduces_to_rank` | Num(k)*(Num(m)+sum s_i*exp(Num(c_i)+u_i)) -> Num(k*m)+sum s_i*Num(k*exp(c_i))*exp(u_i); finite/profitable guards | Reaches rank; plain unary exp split without scalar absorption may only partially help. |
| 67 | pagie/GP-GOMEA/raw 100/original | 1 | `new_rule_reduces_to_rank` | Num(af)*(Num(kf)-exp(Num(cf)+u)-exp(Num(df)+v)) -> Num(af*exp(cf))*(Num(kf/exp(cf))-exp(u)-Num(exp(df-cf))*exp(v)); finite/profitable variants | Reaches rank; needs order/subtraction variants and profitability control. |
| 68 | pagie/GP-GOMEA/raw 109/original | 1 | `new_rule_reduces_to_rank` | s*(K-exp(c+u)-exp(d+v)) -> (s*exp(c))*(K/exp(c)-exp(u)-exp(d-c)*exp(v)); finite/profitable guards | Reaches rank; no current-rule-only path without exp additive-scale edge. |
| 71 | kotanchek/SBP/raw 167/original | 1 | `new_rule_reduces_to_rank` | Num(k)*(v*exp(Num(c)+u)) -> Num(k*c.exp())*(v*exp(u)); include exp(u+Num(c)), exp(u-Num(c)), and commuted/associative product variants. Finite/profitable/non-subsuming guards. | Reaches rank; current rules plus more iterations cannot expose this without exp additive extraction. |
| 73 | kotanchek/SBP/raw 157/original | 1 | `new_rule_reduces_to_rank` | Guarded family: split exp(k+u), exp(u+k), exp(k-u) into exp(k)*exp(u)/exp(-u), then allow finite literal scale distribution over additive children only when generated coefficients fold and ParamCost lowers, followed by affine coefficient collection. | Reaches rank; engineering concern is expansion blowup. |
| 77 | pagie/SBP/raw 161/original | 1 | `new_rule_reduces_to_rank` | For arbitrary Num terms u,v and finite f64 constants a,b, add guarded rewrite exp((u + Num(a)) - v) * Num(b) -> (Num(b) * exp(Num(a))) * exp(u - v), guarded at least by a != 0 and b != 0, with commuted multiplication/addition variants. Constant folding then collapses Num(b)*exp(Num(a)). | No rank artifact needed. The one-parameter gap is local and validated by extraction to rank after adding this guarded family. |
| 79 | kotanchek/SBP/raw 171/original | 1 | `new_rule_reduces_to_rank` | For Num constants c=Num(cf), k=Num(kf), and nonconstant t: rewrite(c*exp(k+t)) -> Num(cf*kf.exp())*exp(t), guarded by cf != 0, finite exp(kf), t != Num(0); include exp(t+k) and exp(k-t) spelling variants. | No rank artifact needed. Related exp-product merge/split may be useful but should be cost-guarded to avoid e-graph growth. |
| 82 | kotanchek/PySR/raw 187/original | 1 | `new_rule_reduces_to_rank` | Num(a) * exp(x + Num(b)) -> Num(a * b.exp()) * exp(x), plus exp(Num(b)+x) and commuted product variants; require finite a,b and keep non-subsuming/profit-guarded. | No denominator or log rule is needed for the one-rank gap; further shape cleanup changes syntax but is not required. |
| 184 | pagie/EPLEX/raw 55/sympy | 1 | `new_rule_partial_reduction` | exp(c + u) -> exp(c) * exp(u) when c is a numeric constant and u contains variables; expose constant additive offset inside exp arguments before parameterization. | Partial only. Additional normalization may be needed to fully match rank 5; this is the smallest visible rank-reducing obstruction. |
| 87 | kotanchek/GP-GOMEA/raw 102/original | 1 | `new_rule_reduces_to_rank` | k*(u*exp(c+v)-w) -> (k*exp(c))*(u*exp(v)) - k*w, guarded to literal non-integer float c and coefficient-like k; extract exp(c) and absorb into branch coefficient. | This accounts for full 7->6 gap. Remaining constants are structurally independent; no affine collection or duplicate relation needed. |
| 88 | kotanchek/PySR/raw 197/sympy | 1 | `new_rule_reduces_to_rank` | For constants A,k,a,b and term E independent of numeric-only collection: A*exp(E + k*(x + a)*(x + b)) -> (A*exp(k*a*b))*exp(E + k*x*(x + a + b)). This is the smallest rank-reducing form for the hidden constant in the shifted quadratic exponent. | This does not require many-parameter sharing or a rank artifact. The rule also performs local affine collection inside the exp argument, but the rank drop comes from extracting the additive exp constant into the outer coefficient. |
| 91 | kotanchek/PySR/raw 207/original | 1 | `new_rule_reduces_to_rank` | For parameter literals c,a,b and param-free term t: c * (exp(t + a) + b) -> (c * exp(a)) * exp(t) + (c * b), with literal products evaluated. | This explains the single rank gap if rank counts non-integer floats only. It assumes the rank artifact does not penalize distributing this local additive factor; if tree-shape costs dominate elsewhere, verify against the rank implementation. |
| 98 | kotanchek/EPLEX/raw 40/sympy | 2 | `new_rule_reduces_to_rank` | For real exp terms, rewrite exp(x*(u + c)) -> exp(x*u)*exp(c*x) when c is an integer-valued literal or otherwise not a fitted parameter. Equivalently allow exp(x*(u+c))/exp(x*u) -> exp(c*x). | Assumes existing rules already multiply/cancel exp factors and simplify exp(5*x0)*exp(-5*x0). If those are absent, the same family should be added with product normalization, but the first missing obstruction is additive constant extraction. |
| 99 | pagie/PySR/raw 203/original | 2 | `new_rule_reduces_to_rank` | exp(A + c) / exp(d) -> exp(A) * exp(c - d), treating exp(c - d) as one learned coefficient; also allow exp(A + c) / k -> exp(A) * exp(c) / k when k is a numeric constant. | This does not simplify the x1 term. The log(x1*(x1*0.5177))*-1.874 form may have separate abs/domain issues, but it is not needed to explain this row's two-rank miss. |
| 100 | kotanchek/PySR/raw 205/sympy | 1 | `new_rule_reduces_to_rank` | exp((a - x) * (x - b)) -> exp(((a - b) * (a - b)) / 4) * exp(-((x - ((a + b) / 2)) * (x - ((a + b) / 2)))) | Need commuted/orientation variants such as exp((x-b)*(a-x)). Rank reduction relies on an existing multiplicative scale so the extracted exp constant replaces that scale rather than adding a new independent parameter. |
| 191 | kotanchek/EPLEX/raw 35/original | 1 | `new_rule_reduces_to_rank` | rewrite(exp(log(abs(exp(b + z) - y * exp(c + z))) + a)).to(abs(exp((a + b) + z) - y * exp((a + c) + z))) for constant a,b,c; also reached after existing x - -a => x + a. | This closes the one-param rank gap for the row. If implemented only as a generic exp(log(abs(u))+a)->abs(exp(a)*u), current rules may still need distribution/exp-product normalization; the targeted shared-tail exponential form is the smallest direct rank-redu… |
| 106 | pagie/PySR/raw 200/original | 1 | `new_rule_reduces_to_rank` | rewrite(a / exp(x + b)).to((a / exp(b)) / exp(x)) for constant a,b; optionally also b+x orientation if commutativity does not reliably expose x+b. | This is the smallest sufficient rank-reducing rule for this row. Existing rules handle the multiplicative coefficient collection already visible in the baseline. No shared-parameter or many-parameter artifact is needed for the one-param gap here. |

## Quotient Coefficient Exposure

- Rows: `16`
- Rule/action: Add quotient-normalization rules that expose removable denominator scales and same-denominator coefficient collection. Common forms: `z / (y / a + b) -> (a * z) / (y + a*b)`, `(r + a*D) / D -> a + r/D`, `(S + c*D) * (N/D) -> c*N + S*N/D`, and `a/y + b/y -> (a+b)/y` with nonzero-denominator guards.

### Outcomes

- `new_rule_reduces_to_rank`: 14
- `new_rule_partial_reduction`: 2

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 166 | kotanchek/Bingo/raw 21/original | 1 | `new_rule_reduces_to_rank` | a*(x*((b*y+e)-l))**-1 -> (x*(y/(a/b)+(e-l)/a))**-1 when a,b are nonzero and a/b is integer-cost/profitable | Needs profitability guard; folding to -0.5*y would still count as a parameter, so the integer ratio orientation matters. |
| 21 | kotanchek/Bingo/raw 0/original | 1 | `new_rule_reduces_to_rank` | c*((u*(x+a*y))**-1) -> (c/a)*((u*((1/a)*x+y))**-1), a != 0 | Needs new reciprocal scalar pull-out; current rules lack scalar extraction through reciprocal/power -1. |
| 61 | kotanchek/Bingo/raw 21/sympy | 1 | `new_rule_reduces_to_rank` | (a*x)/(b*y + z) -> x/(y/(a/b) + z/a); guards: a != 0, b != 0, a/b integer-valued/profitable; add sign/order variants | Reaches rank; profitability guard avoids cases where a/b introduces a counted float. |
| 23 | kotanchek/PySR/raw 195/sympy | 1 | `new_rule_reduces_to_rank` | (a*(b*c+r))/(c*d) -> (a*(b+r/c))/d; guards c != 0 and d != 0; include commuted variants | Reaches rank; current rules cannot distribute cancellation through additive numerator. |
| 45 | pagie/Bingo/raw 23/sympy | 2 | `new_rule_partial_reduction` | (x/z)+(y/z) -> (x+y)/z with z != 0; plus subtraction/grouping variants | Remaining form b+d/x1+(a*x0+c)/(e-log(x0*x1)) has five generically independent parameter directions; n_rank=4 likely artifact. |
| 26 | pagie/PySR/raw 203/sympy | 1 | `new_rule_reduces_to_rank` | a / (b / z + c) -> (a / b) / (1 / z + c / b), with a,b,c as parameters and z any expression. | This accounts for rank_gap=1 with the smallest local quotient normalization. Integer exponent 2.0 is not counted; the redundant scale is denominator coefficient 3.4342474157379796. |
| 27 | kotanchek/Bingo/raw 29/sympy | 1 | `new_rule_reduces_to_rank` | rewrite((u*v + w*z)/(v*z)).to(u/z + w/v), with guards v != 0 and z != 0; include commuted add/mul variants only as needed. | No row-local gap remains. Implementation needs matching commuted product/sum forms without broad denominator distribution blowup. |
| 84 | kotanchek/PySR/raw 182/sympy | 1 | `new_rule_reduces_to_rank` | ((a + b*c)*d)/c -> (a*d)/c + b*d, when c != 0; orient only when one addend contains a factor equal to denominator and the other does not. | No broader distributivity needed. Closest taxonomy is quotient_coefficient_exposure because it exposes denominator-matching numerator component for cancellation. |
| 48 | pagie/Bingo/raw 28/sympy | 6 | `new_rule_partial_reduction` | For shared nonzero denominator z, rewrite (a*x)/z + b/z -> (a*x + b)/z and (a*x)/z - b/z -> (a*x - b)/z, with commuted variants; broader x/z+y/z requires profitability/domain guard. | Partial only: visible reduction 8->7, not rank 2. Remaining target likely needs hidden provenance, non-local sharing, or rank/counting artifact. |
| 30 | kotanchek/PySR/raw 209/sympy | 3 | `new_rule_reduces_to_rank` | Add a targeted exposure/cancellation rule: (S + C*D)*(N/D) -> C*N + (S*N)/D, plus commuted and subtraction variants, only when the same denominator expression D is matched exactly or canonically. | Implementation needs commutative product canonicalization so D in the numerator product matches D in the reciprocal/quotient. If that exact D match is available, no affine, exp, log_abs, square quotient, tolerance snapping, or shared-rank artifact rule is nee… |
| 92 | kotanchek/PySR/raw 194/original | 1 | `new_rule_reduces_to_rank` | z / (y / a + b) -> (a * z) / (y + a * b), with a != 0 | If implemented only as the two-step existing x/a+b collection plus a separate x/(y/a) clearing rule, extraction must still find the enclosing quotient. The smallest direct rule is the affine-denominator quotient exposure above. |
| 93 | pagie/PySR/raw 205/original | 1 | `new_rule_reduces_to_rank` | Rewrite a/exp(E) to a*exp(-E), then collect multiplicative constants in products containing exp terms: c*((P)*Q)/exp(A+B) -> (c*Q)*P/exp(A+B), exposing the single outer coefficient used by rank. | The first term already matches baseline form modulo ordinary multiplication reassociation. The rank gap is from failing to expose/factor the outer quotient coefficient for the second term, not from needing many parameters or a rank artifact. |
| 95 | pagie/PySR/raw 197/original | 1 | `new_rule_reduces_to_rank` | rewrite(a / y + b / y).to((a + b) / y), with y != 0; signed numerators cover subtraction after existing x - a and x + -y rewrites. | No many-parameter artifact is needed here. The only rank-relevant gap is exposing and collecting duplicate quotient coefficients; exp quotient normalization alone would not reduce the two reciprocal coefficients to one. |
| 188 | pagie/EPLEX/raw 55/original | 1 | `new_rule_reduces_to_rank` | For p<0 and positive integer literals m,n: p / log(abs(exp(a - (exp(b - u)**2 * v))**m)**n) -> -1 / (((m*n)/(-p))*a - exp(2*b + log((m*n)/(-p)) - 2*u) * v). Here m=3,n=2. | A standalone log_abs_scale_extraction rule would only expose 6*(a-Q) and likely leave 6 params. The rank reduction needs quotient scale normalization plus absorption of the positive scale into exp(b-u)^2; implement either as this specialized rule or as coordi… |
| 241 | kotanchek/Bingo/raw 13/original | 2 | `new_rule_reduces_to_rank` | For (a0 + a1*x0 + a2*x1)/(b0 + b1*(x0+c)^2), divide numerator and denominator by b1: ((a0/b1) + (a1/b1)*x0 + (a2/b1)*x1)/((b0/b1) + (x0+c)^2). Then optionally absorb one remaining numerator scale by factoring a2/b1 from the numerator. | If current rules already divide denominator scale but not numerator linear scale, the minimal missing piece is the second normalization only. Need exact pipeline rule inventory to distinguish one combined rule from two sequential existing/new rewrites. |
| 105 | kotanchek/PySR/raw 180/sympy | 1 | `new_rule_reduces_to_rank` | For D != 0, add rewrite((r + a * D) / D).to(a + r / D) plus the commuted numerator form ((a * D + r) / D).to(a + r / D). Existing affine collection should expose a * D first. | This assumes the e*D addend is exposed by existing affine collection and AC matching; otherwise a supporting collection/pass-budget issue remains. No many-parameter or rank artifact is needed for this row: the single missing quotient split accounts for the on… |

## Affine Constant Collection

- Rows: `14`
- Rule/action: Add affine collection rules that combine adjacent numeric offsets and scaled constants inside additive/linear contexts. Common forms: `a*(x+c)+d -> a*(x + c + d/a)`, `k*(E + c1 - c2) -> k*(E + (c1-c2))`, and affine shift exposure under `log(abs(k*x + p)) -> log(abs(k*(x + p/k)))` while preserving scale semantics.

### Outcomes

- `new_rule_reduces_to_rank`: 11
- `new_rule_partial_reduction`: 3

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 55 | kotanchek/GP-GOMEA/raw 114/original | 1 | `new_rule_reduces_to_rank` | rewrite((a - x) + b).to((a + b) - x) for a=Num(af), b=Num(bf); optional symmetric b + (a - x) | Current larger probes stay at 4; proposed local rule reaches n_rank=3. |
| 42 | pagie/Bingo/raw 24/sympy | 2 | `new_rule_partial_reduction` | a*log(c/y)+b -> (-a)*log(-y)+(b+a*log(-c)), c < 0, y < 0 | D remains inside log(-log(D/x1)); reaching rank 3 needs more than current rules. |
| 65 | kotanchek/PySR/raw 190/original | 2 | `new_rule_reduces_to_rank` | exp(Num(a)+x)->Num(exp(a))*exp(x), finite a; and x+Num(a)*x -> (Num(1.0)+Num(a))*x, with symmetric variants and finite/profitable guards | Reaches rank; implementation concern is controlling exp-split and implicit-one coefficient collection growth. |
| 24 | kotanchek/Bingo/raw 8/original | 4 | `new_rule_partial_reduction` | Num(k)*((x+Num(s))*(x+Num(s))) -> Num(k*s*s)*((Num(1/s)*x+Num(1))**Num(2)); s nonzero, finite/profitable variants | Still not rank 2; remaining five parameters are generically independent by polynomial coefficient recovery, so likely rank/counting artifact after partial reduction. |
| 172 | pagie/EPLEX/raw 30/sympy | 1 | `new_rule_reduces_to_rank` | ((x+Num(a))*(x+Num(b)))+Num(c) -> x*(x+Num(a+b))+Num(a*b+c); finite/profitable guards | Reaches rank; active obstruction is monic quadratic collection, not log_abs. |
| 237 | kotanchek/SBP/raw 166/sympy | 2 | `new_rule_reduces_to_rank` | Narrow guarded family matching x*(((b*x+c)**2)*(a*(x+k))+d)+e*y+f -> d*((-x*(x+k))*(x+c/b)**2+x+(e/d)*y)+f when d ~= -a*b*b and e/d integer/profitable | Reaches rank; rule is narrow and should be guarded to avoid polynomial blowup. |
| 74 | pagie/SBP/raw 176/original | 2 | `new_rule_partial_reduction` | General one-way rules such as Num(a) + (Num(b) - y) -> Num(a + b) - y, (Num(a) - y) + Num(b) -> Num(a + b) - y, and Num(a) + (y + Num(b)) -> Num(a + b) + y, guarded on finite a, b, and a + b, and kept non-subsuming/profitability-guarded to avoid search blowup. | The proposed rule closes only one of the two rank-gap params, giving a 5-param exact form. Reaching 4 would require an approximate/data-local drop or a different external rank convention; as an analytic expression over an open x0,x1 domain, the remaining five… |
| 6 | kotanchek/SBP/raw 172/original | 1 | `new_rule_reduces_to_rank` | For finite f64 constants af and bf with finite af + bf, where a = Num(af), b = Num(bf): rewrite((a - x) + b).to((a + b) - x). A useful symmetric companion is rewrite(b + (a - x)).to((a + b) - x). The rule is general over any Num x and guarded to avoid NaN/Inf constant-folding edge cases. | No rank artifact needed; the missing degree is localized to one affine constant-collection pattern. Broader coverage may need related variants such as a+(b-x), (x+a)-b, or nested additive reassociation. |
| 81 | pagie/SBP/raw 170/original | 1 | `new_rule_reduces_to_rank` | c1 + (c2 + t) -> (c1 + c2) + t for float constants inside affine sums, e.g. -10.978 + (15.158203042332545 - exp(x1) + 15.599*x1) -> 4.180203042332545 - exp(x1) + 15.599*x1. | No remaining rank gap if affine constant collection runs after the baseline-forming constant fold; it directly explains baseline_after_params=5 to n_rank=4. |
| 28 | kotanchek/PySR/raw 195/original | 1 | `new_rule_reduces_to_rank` | ((a + c1) - b) + c2 -> (a - b) + (c1 + c2), for finite constants c1,c2; collect constants across intervening subtraction in affine additive chain. | Remaining counted floats are 0.6193677368705359, 2.853886507925958, -0.24717047383135762, and 3.302672572491928; no further rule needed. |
| 85 | kotanchek/GP-GOMEA/raw 98/original | 1 | `new_rule_reduces_to_rank` | ((a - x) * (x + b)) + c -> (c + a*b) - x * (x + (b - a)); add commuted plus variants as needed. | No broader exp-additive rule needed; remaining gap is monic quadratic/affine-product constant collection. |
| 31 | kotanchek/PySR/raw 201/original | 1 | `new_rule_reduces_to_rank` | Within an affine product, normalize additive float literals in the argument: k * (E + c1 - c2) => k * (E + (c1 - c2)), preserving integer-valued structural coefficients such as x1 + x1. | This row does not require exp normalization or square quotient rules. It may still leave x0 + x0 - x0 uncollected structurally, but that affects syntax more than parameter rank under the stated counting rule. |
| 32 | kotanchek/PySR/raw 203/original | 1 | `new_rule_reduces_to_rank` | Collect adjacent affine constants across multiplication and addition: (a*(x+c)+d) -> a*(x+(c+d/a)) when a,d are nonzero params and a is safe to divide by. Here 3.3628776435387486*(x0-0.14626012317910758)-0.050504132883325455 becomes 3.3628776435387486*(x0-0.1612782529586318). | No evidence this row needs denominator normalization, exp/log rewrites, quotient coefficient exposure, or shared-parameter/rank-artifact handling. The repeated 3.3628776435387486 remains present in numerator and denominator in both forms, so the observed miss… |
| 196 | kotanchek/EPLEX/raw 33/sympy | 2 | `new_rule_reduces_to_rank` | For a*(b*x + c) inside abs/log, rewrite to (a*b)*(x + c/b) when b is a nonzero integer-valued literal and c is param-like: log(abs(k*x + p)) -> log(abs(k*(x + p/k))). | This accounts for the two-rank miss if current rules can already collect outer product constants and affine additive constants. If current rules lack those too, the same row also needs existing affine constant collection outside log, but the smallest rank-fac… |

## Log Abs Scale Extraction

- Rows: `13`
- Rule/action: Add `log(abs(...))` scale-extraction and square/product rules. Common forms: `log(abs(k*u)) -> log(abs(k)) + log(abs(u))`, `log(abs(t*t)) -> 2*log(abs(t))`, and `exp(log(abs(k*u*exp(v)))) -> abs(k)*abs(u)*exp(v)` where side conditions preserve the original zero set.

### Outcomes

- `new_rule_reduces_to_rank`: 10
- `new_rule_partial_reduction`: 3

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 165 | pagie/Bingo/raw 25/original | 2 | `new_rule_partial_reduction` | log(abs(c*x)) -> log(abs(c)) + log(abs(x)), c != 0; log(abs(x*y)) -> log(abs(x)) + log(abs(y)) | Rank target 2 still unexplained; likely stronger non-local transformation or rank artifact. |
| 169 | pagie/Bingo/raw 6/original | 1 | `new_rule_reduces_to_rank` | log(abs(a * y)) -> log(abs(a)) + log(abs(y)) and symmetric log(abs(y * a)); guards: a != 0.0, y != 0.0 | Reaches rank for this row; implement both product orders with nonzero guards. |
| 170 | pagie/EPLEX/raw 42/sympy | 1 | `new_rule_reduces_to_rank` | log(abs(log(abs(af*u)) + b)) -> log(abs(log(abs(u)) + (b + log(abs(af))))); guards: af != 0 and relevant inner expressions nonzero; include quotient variants | Reaches rank; implement as general scale-shift family rather than row-specific fold. |
| 173 | pagie/EPLEX/raw 42/original | 2 | `new_rule_reduces_to_rank` | Use three guarded families: a/(x/b) -> (a*b)/x for b != 0 and x != 0; (x/b)**n -> (1/(b**n))*(x**n) for b != 0 and integer-valued n; c + log(abs(t)) -> log(abs(exp(c)*t)) for finite c and t != 0, pushing the positive scale into additive coefficients when profitable. | No artifact claim needed. Implementation must avoid unsound domain broadening and avoid duplicate scale occurrences that the occurrence-counting cost model would penalize. |
| 174 | pagie/EPLEX/raw 52/sympy | 1 | `new_rule_reduces_to_rank` | For finite nonzero literal a and arbitrary Num y, add non-subsuming variants log(abs(Num(a)*y)) -> log(abs(Num(a))) + log(abs(y)) and log(abs(y*Num(a))) -> log(abs(y)) + log(abs(Num(a))), guarded by finite a, a != 0, y != 0, and preferably ParamCost decrease. | No remaining rank gap after this rule family. |
| 175 | pagie/EPLEX/raw 45/sympy | 1 | `new_rule_reduces_to_rank` | For af,bf:f64 and y:Num, add guarded symmetric rewrites exp(Num(bf)*log(abs(Num(af)*y))) -> (abs(Num(af))**Num(bf))*exp(Num(bf)*log(abs(y))) and exp(Num(bf)*log(abs(y*Num(af)))) -> (abs(Num(af))**Num(bf))*exp(Num(bf)*log(abs(y))), with guards af != 0 and y != 0. | True missing rule, not iteration artifact. May require directionality/scheduler guard because it introduces alternatives under exp/log/abs. |
| 179 | pagie/Bingo/raw 24/original | 2 | `new_rule_partial_reduction` | log(abs(a * y**-1.0)) -> log(abs(a)) - log(abs(y)), with a != 0 and y != 0. | Single log_abs_scale_extraction closes only one of two gap parameters. Rank 3 needs another relation, shared-parameter recovery, or may be a rank artifact. |
| 181 | pagie/EPLEX/raw 45/original | 2 | `new_rule_reduces_to_rank` | For finite nonzero a,b,c and Num u,v, add guarded log-abs scale extraction in quotient contexts, e.g. exp(log(abs(u/(a/v)))/b) -> abs(a)**(-1/b)*exp(log(abs(u*v))/b), and log(abs((u/a)/c)) -> log(abs(u/(a*c))). | Bare log(abs(t/a)) may be insufficient; rank-closing version should include direct exp-log-abs quotient context or equivalent quotient exposure plus exp additive extraction. |
| 185 | pagie/EPLEX/raw 59/sympy | 1 | `new_rule_reduces_to_rank` | For finite a != 0 and b, rewrite log(abs(a*y)) + b -> log(abs((a*exp(b))*y)); include b + log(abs(a*y)) and left-associated product variants. | No remaining gap. Split form would also expose relation but needs constant/addition collection; direct absorption is smallest. |
| 186 | pagie/Bingo/raw 15/original | 1 | `new_rule_reduces_to_rank` | log(abs(k * u)) => log(abs(u)) + log(abs(k)) for nonzero scalar parameter k, with the extracted log(abs(k)) treated as a scalar constant and then collected into surrounding affine constants. | This assumes the rank target counted the log scale as removable rather than requiring a separate learned constant for log(abs(k)). If current rules already have this identity but lack enough saturation iterations, then classify as existing_rules_more_iteratio… |
| 187 | pagie/Bingo/raw 5/original | 3 | `new_rule_reduces_to_rank` | log(abs(t * t)) => 2 * log(abs(t)) | After the rewrite, coefficient collection may expose 0.1771790010121568 as a single param coefficient. Remaining rank differences, if any, would likely come from current rank accounting for the reciprocal power -1.0 or the unfactored rational-like x0*x0/(2889… |
| 194 | kotanchek/EPLEX/raw 35/sympy | 2 | `new_rule_reduces_to_rank` | For real v and numeric k, rewrite exp(log(abs(k*u*exp(v)))) -> abs(k)*abs(u)*exp(v); analogously log(abs(u*exp(v))) -> log(abs(u))+v. Run before param counting, then existing exp-product and coefficient collection can combine exposed factors. | If current rules do not already collect exposed numeric products or cancel adjacent exp factors after this exposure, add those as follow-up plumbing; algebraically this row does not need a shared-parameter or many-parameter rank-artifact explanation. |
| 195 | kotanchek/EPLEX/raw 33/original | 3 | `new_rule_partial_reduction` | Add guarded symmetric rules: log(abs(a * y)) -> log(a) + log(abs(y)) and log(abs(y * a)) -> log(abs(y)) + log(a), with a=Num(af), af>0.0, y!=Num(0.0). | One parameter remains beyond n_rank after this local rule. The remaining constants are not obviously eliminated by current affine/quotient rules; reaching rank 7 likely needs a separate coefficient-sharing/lattice rule or the n_rank value reflects search/rank… |

## Exp Product Or Quotient Normalization

- Rows: `9`
- Rule/action: Add exponential product/cancellation rules that combine or cancel multiplicative exponentials across products and quotient-like contexts. Common forms: `exp(u)*exp(v) -> exp(u+v)`, `exp(u)*exp(-u) -> 1`, `((u*exp(a)+v)*exp(-a)) -> u + v*exp(-a)`, and guarded normalization of products involving `exp(c)` and `exp(-c)`.

### Outcomes

- `new_rule_reduces_to_rank`: 9

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 69 | pagie/GP-GOMEA/raw 111/original | 1 | `new_rule_reduces_to_rank` | Num(a)*(exp(Num(b)+x)+exp(Num(c)+y)) -> Num(a*exp(b))*exp(x)+Num(a*exp(c))*exp(y); plus single-term and commuted variants; finite guards | Reaches rank; missing guarded exp-additive scalar absorption/distribution, not more iterations. |
| 75 | pagie/SBP/raw 159/original | 1 | `new_rule_reduces_to_rank` | For finite k,c with k > 0 and finite(log(k)+c), add non-subsuming/profitability-guarded variants of ((Num(k)*y)*exp(Num(c)+u)) -> y*exp(Num(log(k)+c)+u) and ((Num(k)*y)*exp(Num(c)-u)) -> y*exp(Num(log(k)+c)-u), plus commuted/associative variants and pre-folded exp(Num(a))*y*exp(Num(c)+/-u) -> y*exp… | No rank gap remains. Remaining five floats occupy distinct roles: additive offset, outer scale, inner additive constant, affine x0 shift, and exp shift. Rule should be guarded and non-subsuming to control product-association growth. |
| 80 | pagie/PySR/raw 196/original | 1 | `new_rule_reduces_to_rank` | Guarded family: x/(x*y)->1/y, x/(y*x)->1/y; x*(y/x-z)->y-x*z; 1/exp(t)->exp(0-t), exp(u)*exp(v)->exp(u+v), exp(u)/exp(v)->exp(u-v); Num(k)*exp(Num(c)+t)->Num(k*c.exp())*exp(t); scalar distribution s*(u+v)->s*u+s*v only when it exposes literal merges and lowers ParamCost. | No rank artifact needed. Risk is e-graph growth from distribution and exp combination; use one-way/profitability guards. |
| 25 | kotanchek/PySR/raw 204/original | 1 | `new_rule_reduces_to_rank` | Normalize exp((u / a) * (v / b)) to exp((u * v) / (a * b)), equivalently exp((1 / (a * b)) * u * v), when a and b are nonzero constants. | No remaining rank gap after this normalization. Other gaps are not needed for this row because the only extra parameter is the duplicate quotient scale inside the exp argument. |
| 97 | kotanchek/EPLEX/raw 38/sympy | 1 | `new_rule_reduces_to_rank` | exp((a - b * exp(c)) * exp(-c)) * exp(-c) => exp(a * exp(-c) - b - c) | This is not solved by more affine collection alone: the blocker is multiplicative exponential cancellation under a product of exp terms. If the rank metric does not share the c subterm after rewriting, this may need canonical DAG/common-subexpression normaliz… |
| 101 | kotanchek/PySR/raw 184/original | 2 | `new_rule_reduces_to_rank` | exp(a) * exp(b) -> exp(a + b), with follow-on normalization of same-denominator sums: u/c + v/c -> (u + v)/c. | This rule addresses the visible rank miss from split exponentials. Further rank reduction may need existing affine duplicate collection inside the combined numerator, but the smallest missing family is exp product normalization. |
| 104 | kotanchek/PySR/raw 185/sympy | 2 | `new_rule_reduces_to_rank` | ((u * exp(a) + v) * exp(-a)) -> (u + v * exp(-a)); commuted/additive variants included. Here u=x1+(-0.2946268534853358), v=1.0, a=1.598800435582104*x1*(x1+0.3553849416107877). | This is the smallest sufficient rule for this row. It should be oriented with a cost guard so it fires only when it exposes exp(a)*exp(-a) cancellation or reduces rank; unrestricted distribution can increase search space. |
| 107 | kotanchek/GP-GOMEA/raw 98/sympy | 3 | `new_rule_reduces_to_rank` | exp(u) * exp(v) -> exp(u + v), followed by exp(0) -> 1 and additive inverse cancellation u + -u -> 0; equivalently exp(u) * exp(-u) -> 1 as the smallest direct rule. | This explains the clearest rank-miss obstruction for the A exponent pair. Further rank recovery may still require distribution or a rule that recognizes exp(u)/exp(u) or exp(u)*exp(-u) before or during affine/multiplicative normalization. |
| 108 | kotanchek/EPLEX/raw 58/sympy | 1 | `new_rule_reduces_to_rank` | With an enclosing free multiplicative coefficient k, rewrite k*exp((a - x)*(x - b))*G -> k2*exp(x*(a + b - x))*G, where k2 = k*exp(-a*b) and G is independent of this normalization. | This does not reduce the inner exp parameters. If applied where no free outer scale exists, the extracted exp(-a*b) would become another coefficient and may not reduce rank. Need rule ordering before final param counting. |

## Square Quotient Normalization

- Rows: `6`
- Rule/action: Add square-aware quotient rules that move constant scale factors out of squared denominators or cancel repeated squared symbolic factors. Common forms: `c / (a*y)^2 -> (c/a^2) / y^2` and guarded cancellation of `t**4 / (v + t**2)**2` patterns when denominator side conditions are preserved.

### Outcomes

- `new_rule_reduces_to_rank`: 6

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 236 | kotanchek/SBP/raw 166/original | 1 | `new_rule_reduces_to_rank` | rewrite(x * x).to(x ** Num(2.0)); only when factors are same e-class; non-subsuming; integer exponent only | Reaches rank; watch search growth and keep rule one-way/non-subsuming. |
| 47 | pagie/Bingo/raw 5/sympy | 1 | `new_rule_reduces_to_rank` | For nonzero constant a and guarded q != 0, m+r/q != 0: k*log(((m*q+r)**2.0)/((a*q)**2.0))+b -> k*log((m+r/q)**2.0)+(b-k*log(a**2.0)), with commuted/subtraction variants and finite guards. | Concrete 5-param expression exists. Implementation should restrict to exponent 2.0 or small even powers, constant scale a, and only fire when factor relation is present to avoid blowup. |
| 180 | pagie/Bingo/raw 21/original | 3 | `new_rule_reduces_to_rank` | rewrite(x * x).to(x ** 2) with exact integer exponent 2; optionally also rewrite((x * x) * (y * y)).to((x * y) ** 2). | No additional rank-reducing rule is needed if extraction keeps square form. Abs-aware exp(log(abs(t))) normalization may simplify but is not required to reach rank 4. |
| 183 | pagie/EPLEX/raw 39/sympy | 1 | `new_rule_reduces_to_rank` | a ** 4.0 / (b * a ** 2.0 + c) ** 2.0 => 1.0 / (b + c / a ** 2.0) ** 2.0. | No remaining gap after square quotient normalization. x0 term has no analogous repeated coefficient to eliminate. |
| 89 | pagie/EPLEX/raw 56/original | 1 | `new_rule_reduces_to_rank` | Normalize squared scaled denominators: c / (a * y)^2 -> (c / a^2) / y^2, with a a nonzero constant. Here this rewrites -0.06332088084610697 / (x0 * 0.3)^2 to -0.703565342734522 / x0^2. | No further rank-reducing rule is needed for this row. The outer division by -0.958 remains one parameter either way, and the x1-side exponent still needs its independent scale and additive exp shift, so the final count after this rule is 4. |
| 96 | pagie/EPLEX/raw 46/sympy | 1 | `new_rule_reduces_to_rank` | rewrite((z * (t ** 4)) / (w * (v + t ** 2) ** 2)).to(z / (w * (1 + v / (t ** 2)) ** 2), t != 0) | Implementation may need the rule to match through surrounding product/quotient association. If the metric shares identical literals globally, the baseline already has 6 unique non-integer floats and the reported gap is only occurrence/rank accounting. |

## Tolerance Coefficient Snapping

- Rows: `6`
- Rule/action: Add a narrowly-scoped numeric snapping pass for known-safe fitted constants. Examples from reviewed rows include near-zero additive constants, exact or near integer-valued float literals like `2.0`, and near-rational coefficients such as `0.3333333333333325 -> 1/3`. This is approximate unless restricted to exactly integer-valued floats.

### Outcomes

- `new_rule_reduces_to_rank`: 5
- `new_rule_partial_reduction`: 1

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 4 | kotanchek/SBP/raw 155/sympy | 1 | `new_rule_reduces_to_rank` | For finite nonzero coefficient a, finite coefficient b, arbitrary Num terms x and y, and integer-valued literal k, if abs(b-k*a)<=CONST_MERGE_TOLERANCE or abs(b/a-k)<=relative_tolerance, rewrite a*x + b*y -> a*(x+k*y) and a*x - b*y -> a*(x-k*y). | Reaches rank via approximate coefficient snapping; not a rank artifact, but not exact rewrite semantics either. |
| 239 | pagie/SBP/raw 167/sympy | 2 | `new_rule_partial_reduction` | For finite nonzero coefficient a, finite coefficient b, terms x,y, and integer-valued k, if abs(b/a-k)<=CONST_MERGE_TOLERANCE or abs(b-k*a)<=CONST_MERGE_TOLERANCE, rewrite a*x + b*y -> a*(x+k*y), keeping k integer-valued/free. | Partial only. After snapping, four non-integer floats remain; n_rank=3 likely needs hidden rounded relation or is a numerical/counting artifact. |
| 12 | pagie/SBP/raw 159/sympy | 1 | `new_rule_reduces_to_rank` | When a Float coefficient in an affine/linear factor is within tolerance of a small rational p/q, canonicalize it to exact p/q, e.g. 0.3333333333333325 * x1 -> x1 / 3 inside x0 + c*x1. | No exp product cancellation or quotient exposure is needed to close this one-rank miss. Other constants may have product relationships, but the smallest sufficient obstruction is the unsnapped 1/3 coefficient. |
| 94 | pagie/Operon/raw 120/sympy | 1 | `new_rule_reduces_to_rank` | Before parameter counting, canonicalize integer-valued float coefficients in multiplicative positions: if Float(c) is within tolerance of integer k, rewrite Mul(Float(c), t) to Mul(Int(k), t), with k=1 removed and k=-1 rendered as Neg(t). This covers exp(-1.0 * z) -> exp(-z). | No remaining rank gap after this rule. The duplicated inner coefficient -1.0047436520881448 is already one shared non-integer parameter; the x ** 2.0 literals are integer-valued powers and should stay outside the parameter count. |
| 193 | pagie/EPLEX/raw 37/sympy | 1 | `new_rule_reduces_to_rank` | For additive literals with abs(a) <= eps, rewrite x + Num(a) -> x and Num(a) + x -> x; eps must be at least 9e-06 for this row. | No further rank gap remains after snapping 9e-06. Log/abs scale extraction and exp(log(.)) rules could reduce nodes or expose alternate forms, but they are not needed to reach rank 8 for this row. |
| 15 | kotanchek/SBP/raw 163/sympy | 1 | `new_rule_reduces_to_rank` | After coefficient exposure/collection, rewrite real literals within tolerance of an integer to the integer literal, e.g. 2.0 -> 2, preserving exact integer semantics for downstream param counting. | No algebraic gap remains for this row if integer-valued floats are excluded from params. If the rank pipeline intentionally counts 2.0 as a parameter, then this is a rank/counting convention artifact rather than a rewrite miss. |

## Coefficient Lattice Factoring

- Rows: `4`
- Rule/action: Add guarded coefficient-path factoring for expressions where several literals only appear through multiplicative coefficient paths. These rows generally need a more global scalar-normalization pass rather than a single local binary rewrite; orient with cost/rank guards to avoid distributive blow-up.

### Outcomes

- `new_rule_reduces_to_rank`: 4

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 8 | pagie/SBP/raw 163/sympy | 1 | `new_rule_reduces_to_rank` | On a flattened additive sum, rewrite c*t + a*p + b*q to a*(p + m*t) + b*(q + n*t) when a,b != 0, m,n are small integer-valued literals, and c is within tolerance of m*a+n*b; also allow c*t+b*q -> b*(q+k*t+d*t) when d ~= c/b-k. Guard by ParamCost decrease and bounded integer offsets. | Likely needs a flattened affine/polynomial collector rather than purely binary local rewrite. Tolerance and bounded integer offsets are critical to avoid search explosion. |
| 182 | pagie/EPLEX/raw 51/sympy | 1 | `new_rule_reduces_to_rank` | For U=x0**4, if an inner log_abs quotient has (U*(V - a) - b) * (s/(U + r)) and b ~= a*r, rewrite it as s*((U*V)/(U + r) - a); equivalently (U*V - a*(U+r))/(U+r) -> U*V/(U+r) - a. | No remaining gap. Equality b=a*r is approximate at displayed precision, so use tolerance/snapping guard. |
| 83 | kotanchek/PySR/raw 208/sympy | 1 | `new_rule_reduces_to_rank` | For finite constants a,b and terms x,y, if a+b is an integer-valued literal k, rewrite a*x - b*y -> a*(x+y) - k*y, with sign/order variants and ParamCost-decrease guard. | No gap remains. Equivalent post-normalization relation exists, but original-form a+b=k rule is smaller. |
| 90 | kotanchek/PySR/raw 193/original | 2 | `new_rule_reduces_to_rank` | For scalar literals c,d,n,m,r: exp(c+F)*(U/n - exp(d+G)*(V/m - r)) -> exp(F)*(p*U - exp(G)*(q*V - h)), where p=exp(c)/n, q=exp(c+d)/m, h=exp(c+d)*r; fold p,q,h as single coefficients. | A broad distributivity rule may blow up search; implement as a guarded coefficient-path normalizer for exp constants and quotient denominators inside additive terms. No many-parameter or numerical rank artifact is needed for this row. |

## Affine Duplicate Or Signed Constant Collection

- Rows: `4`
- Rule/action: Add duplicate affine-constant sharing and sign-normalized literal collection. Examples include shared shifts like `(x0+c)` and `(x1+c)`, repeated signed magnitudes `c` and `-c`, or duplicate fitted roots that should count as one parameter when the expression uses the same learned constant in two affine sites.

### Outcomes

- `new_rule_reduces_to_rank`: 3
- `new_rule_partial_reduction`: 1

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 238 | kotanchek/SBP/raw 165/original | 1 | `new_rule_reduces_to_rank` | For any subexpression u, term x, and finite f64 constants a,b, rewrite (((u + Num(a)) - x) - x) + Num(b) -> u - Num(2.0)*x + Num(a+b), plus negative-spelling variants such as ((u - (x + Num(a))) - x) - Num(b) -> u - Num(2.0)*x - Num(a+b). Guard with finite constants and profitability/no ParamCost i… | No gap remains. Keep the rule one-way/non-subsuming or profitability-gated and include operand-order variants needed after commutativity. |
| 7 | pagie/SBP/raw 172/original | 2 | `new_rule_partial_reduction` | Add guarded affine-duplicate collection such as ((u - Num(a))*v + Num(a)) -> u*v + Num(a)*(1.0 - v), with commuted addition/surrounding-context variants. Guard on finite a and profitability/no ParamCost increase; one-way/non-subsuming to avoid expansion churn. | Partial only. The remaining five directions appear independent; n_rank=4 is likely finite-sample/numerical rank artifact or external counting convention after this partial reduction. |
| 240 | pagie/SBP/raw 154/original | 1 | `new_rule_reduces_to_rank` | Canonicalize identical affine shift constants across different symbolic terms: occurrences like `(x_i + c)` and `(x_j + c)` should expose and reuse one shared parameter for the same non-integer constant `c`, including subtraction-rendered forms `(x_i - p)`. | No evidence here for quotient, reciprocal, square, exp, log(abs), or coefficient-lattice rules. A tolerance rule could notice `5.12 + 4.881 ~= 10.001`, but exact duplicate affine collection is the smaller sufficient rank-closing rule. |
| 103 | kotanchek/PySR/raw 206/original | 1 | `new_rule_reduces_to_rank` | Canonicalize signed non-integer literals before param counting: replace literal -c with unary negation of literal c, or otherwise treat c and -c as one shared parameter with sign outside the parameter. | After sign-normalized duplicate collection, the visible non-integer parameter magnitudes are 2.6270807618049434, 0.7307483550809931, 0.5695620761153319, 0.048636102379870004, 3.0878544411384157, and 0.9859666285998858, matching n_rank=6. |

## Cannot Reduce Further Or Rank Artifact

- Rows: `36`
- Rule/action: Do not add a local rewrite based on these rows alone. The reviewed obstruction was either genuinely many-parameter under tree/literal counting, a shared-parameter accounting mismatch, finite-data rank artifact, or a case where the proposed algebra would not reduce the parameter count to the archived rank.

### Subtypes

- `no_rank_reducing_rule`: 23
- `shared_parameter_or_rank_artifact`: 8
- `exp_product_or_quotient_normalization`: 4
- `log_abs_scale_extraction`: 1

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 52 | pagie/Bingo/raw 25/sympy | 1 | `needs_many_params_or_rank_artifact` | none | Likely rank/counting or finite-data artifact rather than rewrite-budget miss. |
| 234 | pagie/SBP/raw 167/original | 1 | `needs_many_params_or_rank_artifact` | none reducing params; polynomial expansion/collection only simplifies shape | Current rules flatten syntax only; rank 3 likely numerical/local rank or counting convention, not globally valid rewrite. |
| 56 | pagie/SBP/raw 171/original | 2 | `needs_many_params_or_rank_artifact` | No param-reducing rewrite proposed; optional node-only affine cancellation: (x + y) - x -> y and (x + y) + (a - y) -> x + a. | n_rank=2 does not look reachable by sound structural rewrites under current ParamCost; likely rank/counting artifact or genuinely four-parameter form. |
| 57 | pagie/SBP/raw 171/sympy | 2 | `needs_many_params_or_rank_artifact` | none | n_rank=2 appears to be rank/metric artifact rather than reachable by sound rewrite. |
| 58 | pagie/SBP/raw 175/sympy | 2 | `needs_many_params_or_rank_artifact` | none reducing params; exp(u)*exp(v)->exp(u+v) with distribution only gives shape simplification | Rank target 2 is not explained by globally sound rewrite; likely finite-data/numerical rank or counting artifact. |
| 43 | kotanchek/Bingo/raw 14/sympy | 2 | `needs_many_params_or_rank_artifact` | none | n_rank=3 not explained by sound algebraic rewrites; likely rank/metadata or data-local artifact. |
| 167 | kotanchek/Bingo/raw 14/original | 2 | `needs_many_params_or_rank_artifact` | none | n_rank=3 looks like benchmark/data rank artifact, not missing current-rule application. |
| 1 | kotanchek/Bingo/raw 8/sympy | 3 | `needs_many_params_or_rank_artifact` | none | Square recentering or expansion changes syntax but not count; n_rank=2 likely finite-data/numerical rank or counting artifact. |
| 168 | kotanchek/Bingo/raw 27/original | 1 | `needs_many_params_or_rank_artifact` | none for param count; optional node-only exp(log(y))->y when y>0 or exp(log(abs(x)))->abs(x) when x!=0 | n_rank=4 likely finite-data/numerical-rank or external counting; rational collection may clean shape but not count. |
| 235 | kotanchek/Bingo/raw 27/sympy | 1 | `needs_many_params_or_rank_artifact` | none for parameter count; optional rational affine collection only normalizes shape | n_rank=4 not explained by current rules or sound global rewrite; likely external rank/data artifact. |
| 44 | pagie/Bingo/raw 20/sympy | 2 | `needs_many_params_or_rank_artifact` | none | A guarded log-polynomial factoring rule may normalize shape but does not reach rank 3. |
| 62 | pagie/SBP/raw 169/original | 1 | `needs_many_params_or_rank_artifact` | none | Closing this likely requires shared-parameter/let-binding accounting, not a local rewrite. |
| 171 | pagie/Bingo/raw 23/original | 1 | `needs_many_params_or_rank_artifact` | none for param count; optional guarded u**-1 -> 1/u and log(abs(k*u))->log(abs(k))+log(abs(u)) normalizers | n_rank=4 likely external numerical/rank artifact; no visible sound param-reducing rewrite. |
| 2 | pagie/SBP/raw 151/original | 3 | `needs_many_params_or_rank_artifact` | none reducing params; polynomial expansion/normalization only improves shape | n_rank=2 needs external data/domain constraints; rewrite-only reduction to two params would be unsound generically. |
| 22 | kotanchek/Bingo/raw 20/original | 1 | `needs_many_params_or_rank_artifact` | cleanup only: exp(x)**Num(k)->exp(Num(k)*x) for finite k; exp(x)*exp(y)->exp(x+y); exp(x)/exp(y)->exp(x-y) | n_rank=4 likely numerical/rank artifact; cleanup may help shape but should not reduce count. |
| 3 | kotanchek/Bingo/raw 26/sympy | 1 | `needs_many_params_or_rank_artifact` | none reducing params; optional exp product/cancellation cleanup | n_rank=5 likely rank artifact or different convention; no sound five-param tree target found. |
| 70 | kotanchek/SBP/raw 155/original | 1 | `needs_many_params_or_rank_artifact` | no exact rank-reducing rule; optional exp(Num(c)+y)->Num(exp(c))*exp(y), finite c; do not drop near-zero constants without approximate semantics | n_rank=4 likely numerical rank artifact from tiny D; matching it requires deliberate approximate-rule policy, not more exact rewrites. |
| 5 | pagie/SBP/raw 165/original | 1 | `needs_many_params_or_rank_artifact` | No exact rank-reducing rewrite. Optional guarded polynomial collection only normalizes shape. | n_rank=4 likely finite-data/numerical rank, external convention, or approximate/tolerance issue rather than sound exact rewrite. |
| 72 | pagie/SBP/raw 176/sympy | 1 | `needs_many_params_or_rank_artifact` | No rank-reducing rule. Optional cleanup: exp(Num(c)+y)->Num(exp(c))*exp(y) and Num(a)*exp(u)->exp(log(Num(a))+u) for a>0, finite constants. | n_rank=4 likely rank artifact or different parameter convention. More current-rule iterations not main issue. |
| 76 | pagie/SBP/raw 179/original | 2 | `needs_many_params_or_rank_artifact` | No guarded general rank-reducing rule recommended. Non-param-lowering canonicalizations such as u - u*v -> u*(1 - v) or u*v + v -> v*(u + 1) could improve shape but would not close the rank gap. | Reported n_rank=4 is likely numerical/sampling rank artifact. Validate by rerunning rank with more sample points/higher precision/tighter SVD diagnostics if needed. |
| 78 | pagie/SBP/raw 169/sympy | 1 | `needs_many_params_or_rank_artifact` | Optional non-rank-closing exp-context family: for positive finite literal a, rewrite a*exp(t) and exp(t)*a to exp(log(a)+t); add exp(u)*exp(v)->exp(u+v), exp(u)/exp(v)->exp(u-v), and guarded exp-factor-out only when there is a matching exp inverse/cancellation opportunity. | Closing the rank gap appears to require shared-parameter/rank-aware accounting or a let/DAG representation, not more iterations or a rank-closing local tree rewrite. |
| 9 | kotanchek/Bingo/raw 20/sympy | 1 | `needs_many_params_or_rank_artifact` | No param-reducing rule for this row. Optional cleanup: ((u*exp(t)+v)*exp(s)) -> u + v*exp(s) when t+s is known/equivalent to 0, or exp(t)*exp(s)->1 under inverse-exponent guard. Non-subsuming/profitability guarded because it reduces shape but not params here. | Rank target likely numerical/local rank or restricted-sample artifact. Longer current-rule probes stayed at five; missing exp cancellation improves nodes, not param count. |
| 176 | kotanchek/Bingo/raw 26/original | 1 | `needs_many_params_or_rank_artifact` | No parameter-count-reducing rule. Optional cleanup: rewrite (x**Num(-1.0))*y or y*(x**Num(-1.0)) to y/x, guarded by x != 0 and ParamCost nonincrease; this only normalizes exp(x0)**-1.0 and does not reduce params. | No sound local rewrite found to reach n_rank=5; likely numerical/rank convention artifact. Validate by recomputing rank with diagnostics for the six basis directions. |
| 46 | pagie/Bingo/raw 0/sympy | 2 | `needs_many_params_or_rank_artifact` | No exact rank-reducing Num rewrite. Optional guarded log-scale normalization log(Num(a)*y) -> Num(log(a))+log(y), with associated/symmetric product variants, guarded by finite a > 0 and log-domain preservation, only when ParamCost decreases. | Current tree has six counted float occurrences; sharing-aware representation might avoid duplicated A-derived scale, but exact open-domain model still has five degrees. n_rank=4 likely external numerical/counting artifact. |
| 177 | pagie/Bingo/raw 0/original | 2 | `needs_many_params_or_rank_artifact` | No rank-closing rewrite. Optional shape/diagnostic rule log(abs(Num(a)*y)) -> log(abs(Num(abs(a)))) + log(abs(y)) and symmetric variant, guarded by finite a != 0, y != 0, and preferably ParamCost/profitability. | n_rank=4 likely numerical/sampling or shared-parameter artifact, not missing local rewrite to tree language. |
| 178 | pagie/Bingo/raw 28/original | 5 | `needs_many_params_or_rank_artifact` | No rank-closing local rewrite. A guarded log_abs_scale_extraction rule log(abs(a*y))->log(abs(a))+log(abs(y)) would expose scales but would not reduce this row to rank 2. | n_rank=2 is not explained by visible algebra. It likely requires hidden shared provenance, a non-obvious coefficient-lattice relation, or is a rank artifact. |
| 10 | pagie/SBP/raw 179/sympy | 2 | `needs_many_params_or_rank_artifact` | No exact single local rank-reducing rule. The smallest exact normalization is (A*(exp(u)+c)+B*exp(u))*exp(-u) -> A*(1+c*exp(-u))+B, but it leaves the same six non-integer floats. | Rank 4 appears to come from shared-parameter/coefficient-rank structure rather than a small tree rewrite reducing literal floats to four. |
| 11 | pagie/SBP/raw 165/sympy | 1 | `needs_many_params_or_rank_artifact` | none; optional common-scale collection -a*u - b*v -> -a*(u + (b/a)*v) only normalizes shape here. | n_rank=4 likely reflects numerical rank, external convention, or model-family reconstruction not justified by exact local rewrite. |
| 86 | kotanchek/Bingo/raw 24/original | 3 | `needs_many_params_or_rank_artifact` | none exact rank-reducing; optional shape-only rules are exp(Num(a)+x) -> Num(exp(a))*exp(x) and (-x)**Num(-1.0) -> Num(-1.0)*(x**Num(-1.0)), guarded on finite constants. | n_rank=5 likely external rank/counting artifact or needs let-style sharing plus another hidden dependency; no smallest sound local tree rewrite reaches five. |
| 29 | kotanchek/Bingo/raw 24/sympy | 3 | `needs_many_params_or_rank_artifact` | No single local rewrite from the listed rule-gap families exposes a 5-parameter form. The only visible saving is recognizing the repeated coefficient 0.32506758388128154 in the denominator and in exp(-0.32506758388128154*exp(x0**2)) as one shared parameter, which is a rank/accounting issue rather t… | Rank 5 likely comes from the downstream rank model allowing shared parameters or fitting artifacts not represented by current rewrite rules. A genuine 5-param proof would need two additional numeric dependencies among a,b,c,e,f,g,d, but no small symbolic-rule… |
| 13 | pagie/SBP/raw 151/sympy | 3 | `needs_many_params_or_rank_artifact` | No new rank-reducing rewrite. The smallest sound form is the existing common-factor collection: a*(x1*(r + x0) + exp(k*x0) + x1^3 + x0^2 + s) + b*x1^2, with a,b,r,k,s all non-integer params. | The rank gap appears to come from the rank artifact or shared latent-parameter accounting outside single-expression rewrite algebra. No affine, quotient, exp, log, or square normalization rule reduces this row to 2 params without hiding non-integer ratios or… |
| 14 | pagie/SBP/raw 172/sympy | 1 | `needs_many_params_or_rank_artifact` | No sound rank-reducing rewrite. Mark as shared-parameter/rank artifact unless external parameter ties justify equating one of {15.767, 6.2586666666666675, 4.990666666666667, 0.00963, 0.179028}. | The rank gap of 1 does not appear closable by current algebraic rewrite families without tolerance snapping or an external model constraint. If n_rank=4 is trusted, the missing equality is semantic/estimation metadata rather than a local expression rewrite. |
| 189 | pagie/EPLEX/raw 44/sympy | 1 | `needs_many_params_or_rank_artifact` | No algebraic rewrite is the smallest fix. Parameterization should intern identical non-integer float literals, or CSE the repeated term U = p*log(abs(x0/x1))**4 + 1 so both U occurrences share the same p. | If the pipeline already interns exact equal float literals before ranking, then this row is inconclusive without inspecting that implementation. From the rendered baseline alone, the apparent one-rank gap is a shared-parameter/rank artifact, not a new rank-re… |
| 102 | kotanchek/PySR/raw 188/original | 1 | `needs_many_params_or_rank_artifact` | No local algebraic rule. If the rank metric is meant to share equal learned constants, model this as let p = 2.5453490974043493 and reuse p in both exponentials; otherwise keep baseline rank 8. | The row looks like a parameter-sharing/rank-accounting artifact rather than a missing algebraic simplification. A true rank-7 form needs global sharing of the repeated 2.5453490974043493 or a rank metric that counts equal literals once; current tree rewrites… |
| 190 | pagie/Bingo/raw 20/original | 2 | `needs_many_params_or_rank_artifact` | No rank-reducing rewrite proposed. Small candidates like log(abs(c*u))->log(abs(u))+log(abs(c)) are valid identities but do not remove a parameter here because the added constant is multiplied by a nonconstant affine factor. | The reported rank 3 is most plausibly from finite-dataset collinearity, shared parameters outside this rendered form, or a rank-estimation artifact. Confirming which would require the actual pagie row data/Jacobian or the rank artifact generation code. |
| 192 | kotanchek/EPLEX/raw 44/sympy | 1 | `needs_many_params_or_rank_artifact` | Canonicalize finite integer-valued Float exponents before param counting, e.g. Pow(u, 3.0) -> Pow(u, 3); do not add a rank-reducing algebraic rewrite for this row. | A common-denominator rewrite could combine x1**3/d**3 + exp(x0**3)/d**3 into (x1**3 + exp(x0**3))/d**3, but that does not remove a non-integer parameter and is not the smallest explanation for the one-rank gap. |

## More Runs Of Existing Rules

- Rows: `1`
- Rule/action: These rows were classified as likely reducible by the existing rule set with more saturation time, pass budget, or less aggressive backoff. No new rule family is required before trying a longer run.

### Outcomes

- `existing_rules_more_iterations`: 1

### Rows

| Row | Key | Gap | Outcome | Proposed rule / diagnosis | Remaining notes |
|---:|---|---:|---|---|---|
| 0 | kotanchek/SBP/raw 173/original | 1 | `existing_rules_more_iterations` | none; current basic_rules already contains a*x + b -> a*(x + b/a), which applies with a=-11.471, x=(-17.85910443727661 - x0), and b=-201.87275573295364. | No algebraic rule gap remains. The issue is default search budget/scheduling, not missing rules or rank artifact. |
