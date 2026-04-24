# Egglog Rank Miss Agent Review

Rows reviewed by one row-specific agent: 132 / 242
Pending rows: 110

## Conclusion Counts

- `new_rule_reduces_to_rank`: 83
- `needs_many_params_or_rank_artifact`: 36
- `new_rule_partial_reduction`: 12
- `existing_rules_more_iterations`: 1

## New Rule Families Found

- `no_rank_reducing_rule`: 23
- `exp_additive_constant_extraction`: 23
- `quotient_coefficient_exposure`: 16
- `affine_constant_collection`: 14
- `log_abs_scale_extraction`: 14
- `exp_product_or_quotient_normalization`: 13
- `shared_parameter_or_rank_artifact`: 8
- `tolerance_coefficient_snapping`: 6
- `square_quotient_normalization`: 6
- `affine_duplicate_collection`: 4
- `coefficient_lattice_factoring`: 4
- `existing_rules_only`: 1

## Reviewed Rows

### pagie/Bingo/raw 25/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: Current rules do not reduce 3->2. Even splitting log(x0*x1) or exp/additive forms leaves A, B, C generically independent in exp(A*(u+v+B)/(u+C)).
- domain: Requires x0>0, x1>0, log(x0)+C != 0 for the independence argument; no globally sound 2-param rewrite identified.
- remaining gap: Likely rank/counting or finite-data artifact rather than rewrite-budget miss.

### pagie/Bingo/raw 25/original

- conclusion: `new_rule_partial_reduction`
- rule family: `log_abs_scale_extraction`
- proposed rule: `log(abs(c*x)) -> log(abs(c)) + log(abs(x)), c != 0; log(abs(x*y)) -> log(abs(x)) + log(abs(y))`
- proof: Splitting log(abs(C*x0)) folds B and C into P=B-log(abs(C)); splitting log(abs(D*(x0*x1))) folds D into Q=log(abs(D)); 4->3. No sound 3->2 path found.
- domain: For c != 0, both sides are defined exactly when x != 0; product split requires nonzero factors.
- remaining gap: Rank target 2 still unexplained; likely stronger non-local transformation or rank artifact.

### pagie/GP-GOMEA/raw 110/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `exp(c - x) -> exp(c) * exp(-x); exp(c + x) -> exp(c) * exp(x)`
- proof: Fold exp(-6.234), exp(-24.872) to C,D; rewrite exp(C-X), exp(D-Y); factor exp(C) from the sum and combine B*exp(C), leaving A, B*exp(C), exp(D)/exp(C): 4->3.
- domain: Safe over reals because exp is total and exp(c-x)=exp(c)*exp(-x); exp(C)>0 for division.
- remaining gap: Existing rules can finish after the new exp rule exposes constants.

### pagie/GP-GOMEA/raw 118/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `k*(exp((u+c1)-v)+exp((w+c2)-z)) -> (k*exp(c1))*exp(u-v)+(k*exp(c2))*exp(w-z); include symmetric c+u/u+c forms`
- proof: Current rules stop at 4 params. Exp additive extraction gives 0.00052*exp(8.426)=2.37338748595467 and 0.00052*exp(8.611)=2.85570358866897, yielding 2.072676 - 2.37338748595467*exp(x1-exp(x1)) - 2.85570358866897*exp(x0-exp(x0)) with 3 params.
- domain: exp(a+b)=exp(a)*exp(b) is total over reals; only f64 overflow caveat, finite here.
- remaining gap: Plain exp(c+u)->exp(c)*exp(u) exposes constants, but reaching rank needs distribution/scalar absorption across the exp sum.

### kotanchek/Bingo/raw 21/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `a*(x*((b*y+e)-l))**-1 -> (x*(y/(a/b)+(e-l)/a))**-1 when a,b are nonzero and a/b is integer-cost/profitable`
- proof: Current rules fold 2*c to A and -c separately, leaving 4 counted floats. Scaling the reciprocal denominator by A preserves the integer ratio A/c=2, yielding a form with A reused and -2.0 as integer-cost, reducing to 3 counted params.
- domain: Requires A,b nonzero and reciprocal base nonzero; scaling by nonzero A preserves zeros. log(abs(A*u)) domain unchanged because A>0.
- remaining gap: Needs profitability guard; folding to -0.5*y would still count as a parameter, so the integer ratio orientation matters.

### kotanchek/GP-GOMEA/raw 114/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `rewrite((a - x) + b).to((a + b) - x) for a=Num(af), b=Num(bf); optional symmetric b + (a - x)`
- proof: Existing rules fold exp(-3.785) to 0.022708862776731332 but leave (4.203 - x1) + 0.022708862776731332. Collecting constants gives 4.225708862776731 - x1, reducing to constants 0.010559, 0.013455, 4.225708862776731: 4->3.
- domain: Pure add/sub reassociation over reals; no domain side conditions.
- remaining gap: Current larger probes stay at 4; proposed local rule reaches n_rank=3.

### pagie/SBP/raw 167/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none reducing params; polynomial expansion/collection only simplifies shape`
- proof: Expanding (p-x0)*((q-x0)*x0) gives constants c,k,p+q,p*q. These remain four independent degrees; derivative directions are generically independent.
- domain: Polynomial identities are domain-safe over reals.
- remaining gap: Current rules flatten syntax only; rank 3 likely numerical/local rank or counting convention, not globally valid rewrite.

### pagie/SBP/raw 171/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No param-reducing rewrite proposed; optional node-only affine cancellation: (x + y) - x -> y and (x + y) + (a - y) -> x + a.`
- proof: Affine cancellation simplifies exponent to 2*x0 - x0^2 - 0.703*x1^2 + 3.285822*x1, but still has outer 0.028119 and 0.006351 plus two independent x1-shape constants (0.703 and 4.674, or 0.703 and 3.285822).
- domain: Additive cancellations are unconditional over reals; exp is total. There is no pure additive exponent constant to absorb into 0.006351 without preserving an equivalent shape constant.
- remaining gap: n_rank=2 does not look reachable by sound structural rewrites under current ParamCost; likely rank/counting artifact or genuinely four-parameter form.

### pagie/SBP/raw 171/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: Completing the square gives B*exp(-(x0-1)^2 - 0.703*(x1-2.337)^2) + 0.028119 with B=0.006351*exp(4.839483007), still four non-integer constants B, 0.703, 2.337, 0.028119. Comparing quadratic log coefficients shows these are generically identifiable.
- domain: Real x0,x1; exp total; A=0.006351 nonzero and 0.703>0. Exp absorption/completing-square are safe but do not lower count.
- remaining gap: n_rank=2 appears to be rank/metric artifact rather than reachable by sound rewrite.

### kotanchek/Bingo/raw 0/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `c*((u*(x+a*y))**-1) -> (c/a)*((u*((1/a)*x+y))**-1), a != 0`
- proof: The row contains X + 0.5*T inside a reciprocal. Factor 0.5 out: c/0.5 times reciprocal of u*(2*X+T). 0.5 disappears as a parameter and 2 is integer-valued: 5->4.
- domain: a != 0 preserves zero-denominator domain: u*(x+a*y)=0 iff u*((1/a)*x+y)=0 up to nonzero scale.
- remaining gap: Needs new reciprocal scalar pull-out; current rules lack scalar extraction through reciprocal/power -1.

### pagie/SBP/raw 175/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `none reducing params; exp(u)*exp(v)->exp(u+v) with distribution only gives shape simplification`
- proof: Let A=0.00540440450573385, B=0.015024, c=6.944, d=4.3, q=x0*(x0-1), r=x0*(1-x0)=-q. Then E=(A*x1^2*(x1-c)*(x1-d)*exp(x0)+B*exp(q))*exp(r)=B+A*x1^2*(x1-c)*(x1-d)*exp(2*x0-x0^2), still A,B,c,d = 4 params.
- domain: exp(q)*exp(-q)=1 is safe over reals; no log/division/sqrt restrictions. Polynomial factor has genuine 3-param freedom when A!=0 and c!=d; B is independent offset.
- remaining gap: Rank target 2 is not explained by globally sound rewrite; likely finite-data/numerical rank or counting artifact.

### pagie/Bingo/raw 24/sympy

- conclusion: `new_rule_partial_reduction`
- rule family: `affine_constant_collection`
- proposed rule: `a*log(c/y)+b -> (-a)*log(-y)+(b+a*log(-c)), c < 0, y < 0`
- proof: For C<0 and y=log(D/x1)<0, -B*log(C/y)+K becomes B*log(-y)+(K-B*log(-C)), folding C and K: 5->4. No sound 4->3 path found.
- domain: Domain condition y<0 is exactly needed for C/y>0 when C<0.
- remaining gap: D remains inside log(-log(D/x1)); reaching rank 3 needs more than current rules.

### pagie/GP-GOMEA/raw 96/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `a*(k - exp(c+u) - exp(d+v)) -> (a*exp(c))*(k/exp(c) - exp(u) - exp(d-c)*exp(v)); include sign/order variants`
- proof: Current rules fold 2.7-12.507 and factor to 23802.792114*(8.061008098589655e-05 - exp(-10.432-x0*x0) - exp(-9.807-x1*x1)). Extracting exp constants and outer scale gives 0.7015649253789731*(2.7349500104548805 - exp(-x0*x0) - 1.8682459574322257*exp(-x1*x1)), with 3 floats.
- domain: Sound over reals because exp(c)>0 and exp(c+u)=exp(c)*exp(u); no log/abs restrictions.
- remaining gap: Reaches rank if scaled exp additive extraction is available; no current-rule-only path.

### kotanchek/Bingo/raw 14/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: Existing factoring gives 0.0506336679649604*(x1+0.1751892709940647) and log(2.0*(x1-4.770748314695317)). The two x1 shifts differ. Splitting log(2*u) adds a non-droppable A*log(2) term. Denominator has two independent constants after fixed -x0 coefficient.
- domain: Requires x1>4.770748314695317 and denominator != 0. Log product split is domain-equivalent but deleting log(2) would be unsound.
- remaining gap: n_rank=3 not explained by sound algebraic rewrites; likely rank/metadata or data-local artifact.

### kotanchek/Bingo/raw 14/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: Numerator a+b*x1 can factor to b*(x1+a/b) but remains 2 params; denominator x0+c*(d+exp(x0)) has c and c*d; log(abs(e+2*x1)) has shift e. Inverse or abs/log factoring does not reduce below 5 under current cost model.
- domain: Requires x0+c*(d+exp(x0)) != 0 and e+2*x1 != 0. log(abs(2*u)) split is valid for u!=0 but materializes log(2).
- remaining gap: n_rank=3 looks like benchmark/data rank artifact, not missing current-rule application.

### kotanchek/Bingo/raw 8/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: Writing E=-a*x1*exp(-x0)*(x0^2+x0*b*(g*x1-1)^2-d*x0-e), expansion yields five independent coefficient directions. Since a,b,g are nonzero, the coefficients generically recover a,g,b,d,e; no sound 2-param rewrite is evident.
- domain: Real x0,x1; exp(-x0) total/nonzero. No log/sqrt/division side conditions.
- remaining gap: Square recentering or expansion changes syntax but not count; n_rank=2 likely finite-data/numerical rank or counting artifact.

### kotanchek/Bingo/raw 27/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none for param count; optional node-only exp(log(y))->y when y>0 or exp(log(abs(x)))->abs(x) when x!=0`
- proof: Current rules only refactor affine numerator. On x0 sign components, exp(log(abs(x0)))=abs(x0)=+/-x0, giving rational form (alpha+beta*x0+gamma*x1)/(B+x0^6+C*x1-x0); cross-multiplying generically fixes five degrees.
- domain: Original log(abs(x0)) requires x0!=0. exp(log(abs(x0)))->abs(x0) needs guard to avoid domain broadening at x0=0; x0>0 assumption is extra-domain and still leaves five params.
- remaining gap: n_rank=4 likely finite-data/numerical-rank or external counting; rational collection may clean shape but not count.

### kotanchek/GP-GOMEA/raw 100/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `for af>0: af*(z*(bf*x+exp(y+cf))) -> z*((af*bf)*x+exp(y+(cf+log(af)))); include subtraction and term-order variants`
- proof: Existing rules fold 2.494-14.259=-11.765 but do not absorb a positive scale into exp. With B=0.012131, C=-11.765, cf=-0.338: A-B*E*(C*x1+exp(x1+cf)) = A-E*((B*C)*x1+exp(x1+cf+log(B))), leaving 3 params.
- domain: Requires positive absorbed scale; row has B>0. Negative scale cannot be absorbed into real exp; zero scale is separate simplification.
- remaining gap: New rule reduces 4->3; current-rule-only path not found.

### pagie/Bingo/raw 6/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `log(abs(a * y)) -> log(abs(a)) + log(abs(y)) and symmetric log(abs(y * a)); guards: a != 0.0, y != 0.0`
- proof: Current rules only split direct log(a*y) with positive a; here the negative scale is hidden under abs. Extracting log(abs(-35.167844692926785)) combines with outer -1.1727867299483856, leaving params A, shifted constant, 0.42633497398286985, 58.20319882282552: 5->4.
- domain: For a != 0 and y != 0, abs(a*y)=abs(a)*abs(y)>0 and zeros are preserved.
- remaining gap: Reaches rank for this row; implement both product orders with nonzero guards.

### kotanchek/Bingo/raw 27/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none for parameter count; optional rational affine collection only normalizes shape`
- proof: Current rules do not collect the whole rational as (a*x+b*y+c)/(p(x)+d*y+e). Cross-multiplying two such forms forces a,b,c from x*p,y*p,p coefficients, then forces d,e; five non-integer floats are generically identifiable.
- domain: Over reals where denominator is nonzero; 6.0 is integer-valued and not a param.
- remaining gap: n_rank=4 not explained by current rules or sound global rewrite; likely external rank/data artifact.

### kotanchek/Bingo/raw 21/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `(a*x)/(b*y + z) -> x/(y/(a/b) + z/a); guards: a != 0, b != 0, a/b integer-valued/profitable; add sign/order variants`
- proof: Current rules stop at (-A*u)/(B*v - exp(A*u)+log(u)-C), counted as A,B,A,C. Scaling the denominator by A uses A/B=2, producing u/(v/-2 + tail/-A), counted as A,A,C because -2 is integer-valued: 4->3.
- domain: Nonzero scaling preserves denominator zeros and adds no log/exp domain restrictions.
- remaining gap: Reaches rank; profitability guard avoids cases where a/b introduces a counted float.

### pagie/Bingo/raw 20/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: The large log coefficient is buried inside c*T+d, so current log(a*y) cannot expose it. But the five counted params k,a,b,c,d are locally independent: k,a,b derivatives involve Q*L,L,L/x0 while c,d derivatives are rational in T over c*T+d and cannot be spanned by the L directions.
- domain: 3.0 is integer-valued. Any log factor rule would need positivity guards but would preserve two genuine inner-log params.
- remaining gap: A guarded log-polynomial factoring rule may normalize shape but does not reach rank 3.

### pagie/EPLEX/raw 42/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `log(abs(log(abs(af*u)) + b)) -> log(abs(log(abs(u)) + (b + log(abs(af))))); guards: af != 0 and relevant inner expressions nonzero; include quotient variants`
- proof: Current rules cannot split scale hidden under nested log(abs(...)). Extracting log(abs(13.991224004743)) folds with +0.313 to one shift, dropping from five floats to four: 0.694, 2.03252032520325, 0.061971061267125115, shifted inner-log constant.
- domain: Uses log(abs(af*u)) = log(abs(u)) + log(abs(af)) on af*u != 0; x1 != 0 and shifted-log nonzero guards needed.
- remaining gap: Reaches rank; implement as general scale-shift family rather than row-specific fold.

### pagie/SBP/raw 169/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none`
- proof: Current rules can reorder/factor but cannot eliminate the duplicated 2.825 across an additive term and exp(2.825 - x1^2). Under occurrence-count cost there are five non-integer float occurrences: 0.012782, 0.160826, -13.585, 2.825, 2.825.
- domain: Integer literals in x0*x0 do not count. Treating the two 2.825 occurrences as a shared symbol is not represented by the current Num tree cost.
- remaining gap: Closing this likely requires shared-parameter/let-binding accounting, not a local rewrite.

### pagie/SBP/raw 175/original

- conclusion: `new_rule_partial_reduction`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `Num(a)*exp(Num(b)+x) -> Num(a*b.exp())*exp(x), plus exp(x+Num(b)) and commuted product variants; finite/profitable/non-subsuming guards`
- proof: Current rules cannot split exp(-17.803+x0) or absorb it into adjacent coefficient. Rewriting 291401.986945*exp(-17.803+x0) to 0.005404404505733848*exp(x0) reduces 5->4, giving shape 0.015024 + 0.005404404505733848*exp(2*x0-x0*x0)*x1^2*(x1-6.944)*(x1-4.3).
- domain: exp(b+x)=exp(b)*exp(x) is real-total; finite f64 guards avoid inf/NaN. Use non-subsuming rewrite so extraction can ignore unprofitable splits.
- remaining gap: Still not rank 2; remaining intercept, scale, and two distinct roots are generically independent. Likely finite-sample/rank artifact after partial reduction.

### kotanchek/SBP/raw 150/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `exp((u+Num(a))-v) * exp(Num(b)+w) -> exp(Num(a+b)) * exp((u-v)+w), with order variants; apply when ParamCost lowers`
- proof: Current rules do not combine exp products or extract additive constants. exp(x0+19.367-x0*x0)*exp(-13.819+x0) = exp(5.548)*exp(2*x0-x0*x0). Result has non-integer params 0.053829, 0.000194, exp(5.548), -1.237: 5->4.
- domain: exp product identity is total over reals; guard is for finite f64/profitability, not domain.
- remaining gap: Reaches rank; needs variants to expose constants in mixed add/sub shapes.

### kotanchek/SBP/raw 166/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `square_quotient_normalization`
- proposed rule: `rewrite(x * x).to(x ** Num(2.0)); only when factors are same e-class; non-subsuming; integer exponent only`
- proof: Current rules leave duplicated (-5.542+x0)*(-5.542+x0), counting -5.542 twice. Rewriting to (-5.542+x0)**2 reduces non-integer params to 0.010788, 10.151259547645532, -5.542: 4->3. Agent reports temporary in-memory ruleset reached ParamCost(3,22).
- domain: x*x = x**2 is safe over reals for integer exponent 2.0; exponent is integer-valued and usually not a param.
- remaining gap: Reaches rank; watch search growth and keep rule one-way/non-subsuming.

### pagie/Bingo/raw 23/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none for param count; optional guarded u**-1 -> 1/u and log(abs(k*u))->log(abs(k))+log(abs(u)) normalizers`
- proof: Current rules do not normalize u**-1 into 1/u, so inverse-product rules cannot combine x0^-1*x1^-1; log(abs(k*u)) also not split. But after safe normalization the family a+b/x1+c/((d-x0)*log(abs(e/(x0*x1)))) has five generically independent parameter directions.
- domain: Requires x0,x1, affine denominator, and log term nonzero. log-abs split and reciprocal normalization need corresponding nonzero guards.
- remaining gap: n_rank=4 likely external numerical/rank artifact; no visible sound param-reducing rewrite.

### pagie/SBP/raw 151/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none reducing params; polynomial expansion/normalization only improves shape`
- proof: Current rules factor x1 but do not expand/canonicalize polynomial products. Expanding gives A - B*(x0*x1+x0^2+x1^3+(C+D+1)*x1^2+C*D*x1+exp(E*x0)). Effective invariants A,-B,-B*(C+D+1),-B*C*D,E are locally independent when B!=0,C!=D.
- domain: Pure polynomial/exp over reals; integer literals introduced by expansion do not count.
- remaining gap: n_rank=2 needs external data/domain constraints; rewrite-only reduction to two params would be unsound generically.

### kotanchek/Bingo/raw 20/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `cleanup only: exp(x)**Num(k)->exp(Num(k)*x) for finite k; exp(x)*exp(y)->exp(x+y); exp(x)/exp(y)->exp(x-y)`
- proof: Current rules factor affine constants but do not normalize exp(x0*x0)**-1 or combine exp terms. However expression has five independent non-integer params A,B,C,D,E in A*(B+x0+D*(x0^2+C)*(x1+E)*exp(-x0^2)); tangent directions are generically independent.
- domain: Exp-specific cleanup is safe because exp is positive; arbitrary reciprocal/power distribution would need guards.
- remaining gap: n_rank=4 likely numerical/rank artifact; cleanup may help shape but should not reduce count.

### kotanchek/PySR/raw 190/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `exp(Num(a)+x)->Num(exp(a))*exp(x), finite a; and x+Num(a)*x -> (Num(1.0)+Num(a))*x, with symmetric variants and finite/profitable guards`
- proof: Current rules cannot expose -0.21546645087388747 inside exp or collect x1 + 0.05221153846822916*x1. Rewriting gives K*exp(G)*x1*(x1 + c2*T), where K=exp(c0)*c1*(1+c3), leaving only K, c2, and c1 as non-integer float params: 5->3.
- domain: Both identities are total over reals; use finite literal guards and avoid no-op matches. No log/sqrt/division side conditions.
- remaining gap: Reaches rank; implementation concern is controlling exp-split and implicit-one coefficient collection growth.

### kotanchek/PySR/raw 195/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `(a*(b*c+r))/(c*d) -> (a*(b+r/c))/d; guards c != 0 and d != 0; include commuted variants`
- proof: Let C=x0+x1-exp(x1)-3.302672572491928, A=0.6193677368705359*x0+x1, R=x1^2*(0.4943409476627152-2*x1), D=... Expression is A*(x0*C+R)/(C*D). Splitting denominator factor gives A*(x0+R/C)/D, removing duplicate C constant and reducing 5->4.
- domain: Valid only where canceled factor C and remaining denominator D are nonzero; needs real partial-domain guard, not just syntactic c != Num(0).
- remaining gap: Reaches rank; current rules cannot distribute cancellation through additive numerator.

### pagie/GP-GOMEA/raw 92/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `Num(k)*(Num(m)+sum s_i*exp(Num(c_i)+u_i)) -> Num(k*m)+sum s_i*Num(k*exp(c_i))*exp(u_i); finite/profitable guards`
- proof: Current rules factor to 42823.103036*(4.5545e-05 - exp(-10.557-x0^2)-exp(-10.564-x1^2)) but cannot extract exp constants or distribute scale. Proposed rule yields 1.950381 - 1.1138617444599723*exp(-x0^2) - 1.1060919382970058*exp(-x1^2): 4->3.
- domain: exp identities are real-total; guards are f64 finite/profitability.
- remaining gap: Reaches rank; plain unary exp split without scalar absorption may only partially help.

### pagie/GP-GOMEA/raw 100/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `Num(af)*(Num(kf)-exp(Num(cf)+u)-exp(Num(df)+v)) -> Num(af*exp(cf))*(Num(kf/exp(cf))-exp(u)-Num(exp(df-cf))*exp(v)); finite/profitable variants`
- proof: Current rules factor to 2.1e-05*(92875.7619-exp(10.888-x0^2)-exp(10.889-x1^2)). Proposed rule gives 1.1241327840379391*(1.7350183427567174 - exp(-x0^2) - 1.0010005001667077*exp(-x1^2)), exactly 3 params.
- domain: exp(c+u)=exp(c)*exp(u) is real-total; keep signs outside exp and guard finite f64 folds.
- remaining gap: Reaches rank; needs order/subtraction variants and profitability control.

### pagie/GP-GOMEA/raw 109/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `s*(K-exp(c+u)-exp(d+v)) -> (s*exp(c))*(K/exp(c)-exp(u)-exp(d-c)*exp(v)); finite/profitable guards`
- proof: Current rules factor to 1e-05*(194983.7-exp(11.543-x1^2)-exp(11.615-x0^2)). Proposed rule yields 1.0305313417752837*(1.8920695770795672 - exp(-x1^2) - 1.0746553440638147*exp(-x0^2)), with 3 params.
- domain: Real-total exp identities; finite f64 and profitability guards only.
- remaining gap: Reaches rank; no current-rule-only path without exp additive-scale edge.

### pagie/GP-GOMEA/raw 111/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `Num(a)*(exp(Num(b)+x)+exp(Num(c)+y)) -> Num(a*exp(b))*exp(x)+Num(a*exp(c))*exp(y); plus single-term and commuted variants; finite guards`
- proof: Current rules produce 5.4461*(0.3560217403279411 - exp(-1.465-x1^2) - exp(-1.823-x0^2)). Absorbing exp shifts gives 1.93893 - 1.2584738493846883*exp(-x1^2) - 0.8797651837046532*exp(-x0^2), exactly 3 params.
- domain: exp total over reals; f64 finite guards avoid inf/NaN.
- remaining gap: Reaches rank; missing guarded exp-additive scalar absorption/distribution, not more iterations.

### kotanchek/Bingo/raw 8/original

- conclusion: `new_rule_partial_reduction`
- rule family: `affine_constant_collection`
- proposed rule: `Num(k)*((x+Num(s))*(x+Num(s))) -> Num(k*s*s)*((Num(1/s)*x+Num(1))**Num(2)); s nonzero, finite/profitable variants`
- proof: Current rules do not recognize coefficient times identical affine factors. Rewriting -0.6667403548542734*((-1.3864562418418644+x1)^2) to -1.2816489216174494*(0.7212632969011202*x1-1)^2 reduces 6->5.
- domain: Identity over reals when s != 0; finite guards avoid inf/NaN. exp(x0)**-1 -> exp(-x0) is shape-only here.
- remaining gap: Still not rank 2; remaining five parameters are generically independent by polynomial coefficient recovery, so likely rank/counting artifact after partial reduction.

### kotanchek/Bingo/raw 26/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none reducing params; optional exp product/cancellation cleanup`
- proof: Current rules cannot combine exp factors or distribute exp(-2*x0), but normalized z=exp(-x0)*log(x0+exp(x0)) form is f + A*(z-q)*((x0+s)*(z-t)+u), with six locally independent floats A,s,t,q,u,f.
- domain: Requires x0+exp(x0)>0 for log; exp normalization is domain-safe but not count-reducing. Integer 2/-2 not counted.
- remaining gap: n_rank=5 likely rank artifact or different convention; no sound five-param tree target found.

### pagie/Bingo/raw 23/sympy

- conclusion: `new_rule_partial_reduction`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `(x/z)+(y/z) -> (x+y)/z with z != 0; plus subtraction/grouping variants`
- proof: Current rules split log scale but cannot combine a*x0/L + c/L. Common denominator factoring gives d/x1 + b + (a*x0+c)/L, reducing duplicate L constant and 6->5.
- domain: Requires shared denominator nonzero; here also x1 != 0 and log quotient/domain constraints.
- remaining gap: Remaining form b+d/x1+(a*x0+c)/(e-log(x0*x1)) has five generically independent parameter directions; n_rank=4 likely artifact.

### pagie/EPLEX/raw 30/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `((x+Num(a))*(x+Num(b)))+Num(c) -> x*(x+Num(a+b))+Num(a*b+c); finite/profitable guards`
- proof: Current rules already absorb 1.077 scale but do not collect (x1+0.178084)*(x1+0.2515946853343705)+0.46147174676957475. Proposed rule folds to x1*(x1+0.4296786853343705)+0.5062767347126608, reducing 5->4. Agent reports transient extraction reached ParamCost(4,15).
- domain: Polynomial identity; surrounding log(abs(log(abs(...)))) domain unchanged by algebraic denominator equality.
- remaining gap: Reaches rank; active obstruction is monic quadratic collection, not log_abs.

### kotanchek/SBP/raw 166/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `Narrow guarded family matching x*(((b*x+c)**2)*(a*(x+k))+d)+e*y+f -> d*((-x*(x+k))*(x+c/b)**2+x+(e/d)*y)+f when d ~= -a*b*b and e/d integer/profitable`
- proof: Current rules reduce 0.66268-0.33134*x to -0.33134*(x-2) but cannot absorb square slope and +0.010788. Guards hold: d ~= -a*b*b, e/d=3, c/b ~= -5.542. Target has 0.010788, -5.542, 0.109511788: 5->3.
- domain: Polynomial identity under guarded numeric equalities; integer 0,2,3 and exponent 2 not counted.
- remaining gap: Reaches rank; rule is narrow and should be guarded to avoid polynomial blowup.

### kotanchek/SBP/raw 155/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `no exact rank-reducing rule; optional exp(Num(c)+y)->Num(exp(c))*exp(y), finite c; do not drop near-zero constants without approximate semantics`
- proof: Current rules fold exp(-20.972) and affine constants but cannot split exp additive constant. Exact normal form a*(B+G*x1-x1^2-x0^2-exp(D+C*x1)) has five independent floats; tiny D ~= 7.8e-10 is exact and independent.
- domain: Exp additive split is exact but only partial; dropping D would be approximate/tolerance-based, not exact.
- remaining gap: n_rank=4 likely numerical rank artifact from tiny D; matching it requires deliberate approximate-rule policy, not more exact rewrites.

### kotanchek/SBP/raw 155/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `tolerance_coefficient_snapping`
- proposed rule: `For finite nonzero coefficient a, finite coefficient b, arbitrary Num terms x and y, and integer-valued literal k, if abs(b-k*a)<=CONST_MERGE_TOLERANCE or abs(b/a-k)<=relative_tolerance, rewrite a*x + b*y -> a*(x+k*y) and a*x - b*y -> a*(x-k*y).`
- proof: Current exact factoring cannot equate -0.0251390000196031 with -0.025139; factoring would introduce ratio 1.0000000007797882, counted as a float. Difference is 1.96e-11 below CONST_MERGE_TOLERANCE. Snapping the exp coefficient to -0.025139 lets existing rules extract -0.025139*(exp(0.586*x1)+x0**2+x1**2-7.42*x1)+0.188863171, reducing 5->4.
- domain: Approximate, not exact algebraic identity. Needs explicit tolerance policy, finite/nonzero guards, representative-choice rules, and regression tests. Integer-valued literals like 1.0/2.0 not counted.
- remaining gap: Reaches rank via approximate coefficient snapping; not a rank artifact, but not exact rewrite semantics either.

### kotanchek/SBP/raw 167/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `Num(k)*(v*exp(Num(c)+u)) -> Num(k*c.exp())*(v*exp(u)); include exp(u+Num(c)), exp(u-Num(c)), and commuted/associative product variants. Finite/profitable/non-subsuming guards.`
- proof: Current rules fold 4.989-9.32 to -4.331 but cannot extract -1.978 from exp(x0+(x0-(-1.978+x0*x0))). Extracting exp(1.978) and absorbing into 0.000927 gives 0.006700608120847708. Remaining params are 0.008014, 0.006700608120847708, -4.331, 3.46: 5->4.
- domain: exp(c+u)=exp(c)*exp(u) is total over reals; guards only for finite f64/profitability. No log/sqrt/division.
- remaining gap: Reaches rank; current rules plus more iterations cannot expose this without exp additive extraction.

### pagie/SBP/raw 165/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No exact rank-reducing rewrite. Optional guarded polynomial collection only normalizes shape.`
- proof: Current rules fold constants and factor outer scale to 0.01085*(91.40605069124423 + ...). Expanding inside scale yields a*(g + r2*x0^2 + r1*x0 + r5*x1 + r0), with independent functions g=x0^3+exp(x0)-exp(x1), x0^2, x0, x1, and 1. The five counted floats 0.01085, 91.40605069124423, 6.659, -12.488, 14.814 are independent; no second exact dependency justifies n_rank=4.
- domain: Polynomial arithmetic and exp only; exact over all real x0,x1 and finite constants. If dividing by outer scale during normalization, guard scale != 0.
- remaining gap: n_rank=4 likely finite-data/numerical rank, external convention, or approximate/tolerance issue rather than sound exact rewrite.

### pagie/SBP/raw 176/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No rank-reducing rule. Optional cleanup: exp(Num(c)+y)->Num(exp(c))*exp(y) and Num(a)*exp(u)->exp(log(Num(a))+u) for a>0, finite constants.`
- proof: Current rules factor shared 0.001527 scale to baseline with params {0.001527, -16.248, 58.0528865237319, 298.56868294999475, 44.89194499017682}. Family s*(q*x0*exp(x0-x0^2) - x1*exp(x1) + x1*exp(k*x0) + r*x1 - m) has independent slots s,q,k,r,m. Reintroducing provenance exp(5.699), exp(3.89), 0.06855/0.001527 does not reduce count.
- domain: Domain requires x0,x1 real; exp total. Integer-valued 1/2 signs not counted.
- remaining gap: n_rank=4 likely rank artifact or different parameter convention. More current-rule iterations not main issue.

### kotanchek/SBP/raw 157/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `Guarded family: split exp(k+u), exp(u+k), exp(k-u) into exp(k)*exp(u)/exp(-u), then allow finite literal scale distribution over additive children only when generated coefficients fold and ParamCost lowers, followed by affine coefficient collection.`
- proof: Current rules fold 11.267*7.23 to 81.46041 but cannot split exp(7.23-x0^2) or distribute outer 0.000155*x0 scale. With exp split and guarded distribution, A + B*x0*(C*(x1-PQ)+x1+exp(Q-x0^2)*R) becomes A + B*(C+1)*x0*x1 - B*C*P*Q*x0 + B*exp(Q)*x0*exp(-x0^2)*R. Remaining params: 0.144574, 0.00066774, 0.041768010623400006, 0.21393448813349325: 5->4.
- domain: exp additive split total over reals; distribution/affine collection polynomial identities. Use finite/profitability guards to control expansion.
- remaining gap: Reaches rank; engineering concern is expansion blowup.

### pagie/SBP/raw 176/original

- conclusion: `new_rule_partial_reduction`
- rule family: `affine_constant_collection`
- proposed rule: `General one-way rules such as Num(a) + (Num(b) - y) -> Num(a + b) - y, (Num(a) - y) + Num(b) -> Num(a + b) - y, and Num(a) + (y + Num(b)) -> Num(a + b) + y, guarded on finite a, b, and a + b, and kept non-subsuming/profitability-guarded to avoid search blowup.`
- proof: Current-rule obstruction first: analysis_rules folds exp(3.89) to 48.91088652373189, but basic_rules has no constant-collection rule for 9.142 + (48.91088652373189 - y). Longer current-rule probes with larger backoff and no-backoff still extract 6 params, so this is not just an iteration budget miss. The proposed affine constant collection rewrites the x1 bracket to 58.052886523731885 - (exp(x1) - exp(x0 * -16.248)), reducing the row from 6 counted non-integer floats to 5. It does not soundly reach n_rank=4: after this merge the exact parameter directions can be represented by A + B*(T_C(x0) + x1*(H - exp(x1) + exp(F*x0))). The d/e redundancy is gone into H, but A, B, C, H, and F remain generically independent: A is the only constant offset, C changes only the x0-only Gaussian branch T_C, H controls the x1 basis, F changes the x0 sensitivity of x1*exp(F*x0), and B is forced by the distinct -x1*exp(x1) component shared with no other parameter direction.
- domain: All original literals here are non-integer floats and count as params under ParamCost; integer-valued literals would usually not. The affine collection rule is exact over real arithmetic and has no division-domain side condition, but should guard f64 finiteness of the folded sum. Exp additive splitting, e.g. exp(c + u) -> exp(c)*exp(u), is exact with finite-exp guards, but in this full expression it does not by itself reduce the parameter count because the outer scale B must remain independent for the x1 terms.
- remaining gap: The proposed rule closes only one of the two rank-gap params, giving a 5-param exact form. Reaching 4 would require an approximate/data-local drop or a different external rank convention; as an analytic expression over an open x0,x1 domain, the remaining five parameters are independent, so the n_rank=4 target is likely a finite-sample/numerical rank artifact after the partial reduction.

### kotanchek/SBP/raw 172/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `For finite f64 constants af and bf with finite af + bf, where a = Num(af), b = Num(bf): rewrite((a - x) + b).to((a + b) - x). A useful symmetric companion is rewrite(b + (a - x)).to((a + b) - x). The rule is general over any Num x and guarded to avoid NaN/Inf constant-folding edge cases.`
- proof: Current rules fold adjacent Num(a)+Num(b) and rewrite x-a to x+-a, but do not collect constants split across symbolic subtraction in (a - x) + b. The baseline keeps (8.072 - x0 + 7.155)*x1, so 8.072 and 7.155 remain separate params. Applying affine collection gives (15.227 - x0)*x1 and reaches 0.009565*(4695.615595638752 + (x1+x1) - exp(x1) + (15.227-x0)*x1 - (x0+x0)*x0), with three non-integer floats, matching n_rank=3. Larger current-rule budgets did not change the 4-param extraction.
- domain: Counted parameters are non-integer floats. Integer-valued literals and duplicated variables such as x1+x1 or x0+x0 are not parameter-counting terms. exp(18.719) is already handled by current constant folding and participates in the folded 4695.615595638752 constant.
- remaining gap: No rank artifact needed; the missing degree is localized to one affine constant-collection pattern. Broader coverage may need related variants such as a+(b-x), (x+a)-b, or nested additive reassociation.

### pagie/SBP/raw 159/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `For finite k,c with k > 0 and finite(log(k)+c), add non-subsuming/profitability-guarded variants of ((Num(k)*y)*exp(Num(c)+u)) -> y*exp(Num(log(k)+c)+u) and ((Num(k)*y)*exp(Num(c)-u)) -> y*exp(Num(log(k)+c)-u), plus commuted/associative variants and pre-folded exp(Num(a))*y*exp(Num(c)+/-u) -> y*exp(Num(a+c)+/-u).`
- proof: analysis_rules folds exp(11.767) to 128926.78963076824, but current rules cannot absorb a positive scalar multiplier into an additive exp shift. Rewriting 128926.78963076824*y*exp(-7.374 - x0) to y*exp(4.393 - x0), since log(128926.78963076824)-7.374 = 4.393, leaves params 0.154306, 0.001732, -20.41, -1.883, and 4.393, reducing 6->5 and matching n_rank=5.
- domain: The identity uses total positive exp, but scalar absorption requires k > 0 and finite f64 guards for log/folded shift. Non-integer floats count as params; integer-valued literals and repeated variables in x0+x0+x0+x1 do not.
- remaining gap: No rank gap remains. Remaining five floats occupy distinct roles: additive offset, outer scale, inner additive constant, affine x0 shift, and exp shift. Rule should be guarded and non-subsuming to control product-association growth.

### pagie/SBP/raw 179/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No guarded general rank-reducing rule recommended. Non-param-lowering canonicalizations such as u - u*v -> u*(1 - v) or u*v + v -> v*(u + 1) could improve shape but would not close the rank gap.`
- proof: Current rules can fold constants and factor explicit common multiplicative terms, but have no param-lowering rewrite for this expression. Visible normalizations like x0 - x0*T -> x0*(1-T) and (x1-c)*x1 + x1 -> x1*(x1-(c-1)) do not reduce the six non-integer floats. Let A=0.105238, B=0.004516, C=5.432, D=3.749, E=2.638, F=1.033, P=x1^2+(1-C)*x1, Q=x0-x1-D, R=exp(E-x0)-F. The expression is A - B*x0 + B*x0*P*Q*R. Parameter derivatives span independent functions generically, so symbolic rank is six rather than four.
- domain: Real-valued exp expression; independence assumes generic x0,x1 variation and B != 0. Integer-valued literals are structural, non-integer floats are parameters. A low numerical rank would need dataset/tolerance evidence rather than a semantics-preserving rewrite.
- remaining gap: Reported n_rank=4 is likely numerical/sampling rank artifact. Validate by rerunning rank with more sample points/higher precision/tighter SVD diagnostics if needed.

### kotanchek/SBP/raw 165/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_duplicate_collection`
- proposed rule: `For any subexpression u, term x, and finite f64 constants a,b, rewrite (((u + Num(a)) - x) - x) + Num(b) -> u - Num(2.0)*x + Num(a+b), plus negative-spelling variants such as ((u - (x + Num(a))) - x) - Num(b) -> u - Num(2.0)*x - Num(a+b). Guard with finite constants and profitability/no ParamCost increase.`
- proof: Current rules expose x0*x0 + 1.015 - x0 - x0 + -3.101 but stop there; they do not collect two identical linear terms while merging constants separated by those terms. Applying the rule gives x0*x0 - 2.0*x0 - 2.086. The whole expression then has non-integer floats 0.009361, 0.001754, 4.923, 16.384, and 2.086, exactly n_rank=5.
- domain: Polynomial identity over all real x; no division/log/sqrt domain restrictions. Introduced 2.0 is integer-valued and not a parameter; 2.086 replaces 1.015 and -3.101.
- remaining gap: No gap remains. Keep the rule one-way/non-subsuming or profitability-gated and include operand-order variants needed after commutativity.

### pagie/EPLEX/raw 42/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `Use three guarded families: a/(x/b) -> (a*b)/x for b != 0 and x != 0; (x/b)**n -> (1/(b**n))*(x**n) for b != 0 and integer-valued n; c + log(abs(t)) -> log(abs(exp(c)*t)) for finite c and t != 0, pushing the positive scale into additive coefficients when profitable.`
- proof: Current rules do not match 1.889/(x1/0.459), do not distribute integer power over (x1/0.415)**3.0, and do not absorb abs-wrapped log offsets. Exposing coefficients gives 1.889*0.459/x1 + (1/0.415**3.0)*x1**3.0. Then 0.313 + log(abs(inner)) becomes log(abs(exp(0.313)*inner)), absorbing scale into coefficients P=exp(0.313)*1.889*0.459 and Q=exp(0.313)/0.415**3.0. Full expression has four non-integer params: P, Q, 0.492, 0.694, matching n_rank=4.
- domain: Reciprocal and power rules require nonzero denominators. log-offset absorption is valid where t != 0 because exp(c)>0 and abs(exp(c)*t)=exp(c)*abs(t). Integer-valued powers 2.0 and 3.0 are structural, not params.
- remaining gap: No artifact claim needed. Implementation must avoid unsound domain broadening and avoid duplicate scale occurrences that the occurrence-counting cost model would penalize.

### pagie/EPLEX/raw 52/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `For finite nonzero literal a and arbitrary Num y, add non-subsuming variants log(abs(Num(a)*y)) -> log(abs(Num(a))) + log(abs(y)) and log(abs(y*Num(a))) -> log(abs(y)) + log(abs(Num(a))), guarded by finite a, a != 0, y != 0, and preferably ParamCost decrease.`
- proof: Current rules have log(a*y), log(y*a), log(a/y), log(y/a), and log(a**y), but only outside abs and for positive constants. Baseline remains log(abs(C - log(abs(K*T)))) with params C, K, B, D. Applying log(abs(K*T)) -> log(abs(K)) + log(abs(T)) and existing x-(y+z) plus constant folding gives log(abs(-2.383569211003397 - log(abs(x1*(x0**2.0+0.5238448285322359)+0.10754458161865547)))), leaving three non-integer floats and matching n_rank=3.
- domain: Exact on original domain because |a*y|=|a|*|y| for finite a != 0 and y != 0. Negative a is allowed due to abs. Integer-valued 2.0 is structural; all non-integer floats count.
- remaining gap: No remaining rank gap after this rule family.

### pagie/SBP/raw 161/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `For arbitrary Num terms u,v and finite f64 constants a,b, add guarded rewrite exp((u + Num(a)) - v) * Num(b) -> (Num(b) * exp(Num(a))) * exp(u - v), guarded at least by a != 0 and b != 0, with commuted multiplication/addition variants. Constant folding then collapses Num(b)*exp(Num(a)).`
- proof: Current rules reassociate/fold constants but do not move additive constants across exp. Here 0.069 is trapped inside exp(exp(x1)+0.069-x1), while -0.069 is a scalar multiplying that exp result. The proposed rule lets constant folding combine -0.069*exp(0.069) into -0.07392909843123589. A scratch run with only this proposed rule extracted a 5-param form, reducing ParamCost from 6 to 5 and matching n_rank=5.
- domain: Identity exp((u+a)-v)*b = b*exp(a)*exp(u-v) is valid over real arithmetic; exp has no positivity precondition. Non-integer floats count as params; integer-valued literals usually do not.
- remaining gap: No rank artifact needed. The one-parameter gap is local and validated by extraction to rank after adding this guarded family.

### pagie/SBP/raw 169/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `Optional non-rank-closing exp-context family: for positive finite literal a, rewrite a*exp(t) and exp(t)*a to exp(log(a)+t); add exp(u)*exp(v)->exp(u+v), exp(u)/exp(v)->exp(u-v), and guarded exp-factor-out only when there is a matching exp inverse/cancellation opportunity.`
- proof: Current rules lack exp product combination/cancellation and cannot turn 16.860944986089*exp(-x1**2.0) into exp(2.825-x1**2.0), even though log(16.860944986089)=2.825. These rules expose the relation but under tree occurrence cost still leave five non-integer literal occurrences: 0.160826, 0.012782, 13.585, and 2.825 twice.
- domain: Apparent four-degree family is A=0.160826, B=0.012782, C=13.585, D=2.825 with 16.860944986089=exp(D). The same D is used in two separate tree contexts: additive offset and exponent coefficient.
- remaining gap: Closing the rank gap appears to require shared-parameter/rank-aware accounting or a let/DAG representation, not more iterations or a rank-closing local tree rewrite.

### pagie/SBP/raw 172/original

- conclusion: `new_rule_partial_reduction`
- rule family: `affine_duplicate_collection`
- proposed rule: `Add guarded affine-duplicate collection such as ((u - Num(a))*v + Num(a)) -> u*v + Num(a)*(1.0 - v), with commuted addition/surrounding-context variants. Guard on finite a and profitability/no ParamCost increase; one-way/non-subsuming to avoid expansion churn.`
- proof: Current top-level factoring gives 0.00963*(18.590654205607475 - S), but binary rules do not globally collect affine/Horner duplicates like ((u-a)*v+a). Applying the rule with u=x0+(x0+x0), a=18.776, v=x0 reduces the two 18.776 occurrences to one. Result still has five non-integer floats: 0.00963, 18.590654205607475, -15.767, 18.776, and -14.972.
- domain: Exact polynomial identity; no division/log/sqrt domain conditions. Non-integer floats count; 1.0/2.0/3.0 are structural. Independence assumes nonzero scale and open x0,x1 domain.
- remaining gap: Partial only. The remaining five directions appear independent; n_rank=4 is likely finite-sample/numerical rank artifact or external counting convention after this partial reduction.

### kotanchek/SBP/raw 171/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `For Num constants c=Num(cf), k=Num(kf), and nonconstant t: rewrite(c*exp(k+t)) -> Num(cf*kf.exp())*exp(t), guarded by cf != 0, finite exp(kf), t != Num(0); include exp(t+k) and exp(k-t) spelling variants.`
- proof: Current rules fold -2.136+3.852 to 1.7159999999999997, but cannot combine remaining -2.136 coefficient with the 5.051 inside exp(5.051 - x0*x0). Rewriting -2.136*exp(5.051 + -1.0*(x0*x0)) to -333.597414814633*exp(-1.0*(x0*x0)) leaves params 0.033047, 0.000135, 1.7159999999999997, and -333.597414814633, reaching n_rank=4.
- domain: Non-integer floats are params; integer-valued -1.0/1.0/2.0 are structural. Rule applies to adjacent numeric coefficient times exp(constant+tail) with finite/overflow guards.
- remaining gap: No rank artifact needed. Related exp-product merge/split may be useful but should be cost-guarded to avoid e-graph growth.

### pagie/EPLEX/raw 45/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `For af,bf:f64 and y:Num, add guarded symmetric rewrites exp(Num(bf)*log(abs(Num(af)*y))) -> (abs(Num(af))**Num(bf))*exp(Num(bf)*log(abs(y))) and exp(Num(bf)*log(abs(y*Num(af)))) -> (abs(Num(af))**Num(bf))*exp(Num(bf)*log(abs(y))), with guards af != 0 and y != 0.`
- proof: Current rules split bare log(a*y) etc. but not the factor inside exp(1.8450184501845*log(abs(2.23713646532438*x1*exp(log(abs(x0)))))). The proposed rule combines 0.607 with 2.23713646532438**1.8450184501845, producing a 5-param expression and matching n_rank=5. Long/no-backoff probes stayed at six under current rules.
- domain: Sound on domain af != 0, y != 0 because exp(b*log(abs(a*y))) = |a|**b * exp(b*log(abs(y))). Non-integer floats count; 2.0 is structural.
- remaining gap: True missing rule, not iteration artifact. May require directionality/scheduler guard because it introduces alternatives under exp/log/abs.

### pagie/SBP/raw 163/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `coefficient_lattice_factoring`
- proposed rule: `On a flattened additive sum, rewrite c*t + a*p + b*q to a*(p + m*t) + b*(q + n*t) when a,b != 0, m,n are small integer-valued literals, and c is within tolerance of m*a+n*b; also allow c*t+b*q -> b*(q+k*t+d*t) when d ~= c/b-k. Guard by ParamCost decrease and bounded integer offsets.`
- proof: Current rules already factor the equal-and-opposite exp coefficients into -0.01063*(exp(x1)-exp(x0)); the remaining relation is coefficient-lattice, not local cancellation. Identities: 0.08391322=0.01063*7.894, 0.15719644=0.01063*(2*7.894-1), 0.11393234=0.01063*7.894*(1+0.3577400557385356). Equivalent target c0 + a*((exp(x0)-exp(x1)-x1) + r*((x0+x1-x0*x0)+(x1+d*x0))) has four non-integer params: 0.103875, 0.01063, 7.894, 0.3577400557385356.
- domain: No domain issue beyond normal arithmetic. Integer-valued offsets/exponent 2.0 are structural; non-integer floats count. Basis terms are independent, so equivalence is checked by coefficient matching.
- remaining gap: Likely needs a flattened affine/polynomial collector rather than purely binary local rewrite. Tolerance and bounded integer offsets are critical to avoid search explosion.

### kotanchek/Bingo/raw 20/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `No param-reducing rule for this row. Optional cleanup: ((u*exp(t)+v)*exp(s)) -> u + v*exp(s) when t+s is known/equivalent to 0, or exp(t)*exp(s)->1 under inverse-exponent guard. Non-subsuming/profitability guarded because it reduces shape but not params here.`
- proof: Current rules lack reverse distribution of outer exp(-x0**2) through a sum and exp(t)*exp(-t) cancellation. Applying that cleanup to ((A-B*x0)*exp(t)+H*(t+C)*(x1+G))*exp(-t) gives A-B*x0+H*(t+C)*(x1+G)*exp(-t), but the non-integer floats remain A,B,H,C,G: five parameters. Independence: x1 coefficient H*(t+C)*exp(-t) determines H,C; then x1-free terms determine A,B,G over an open x0 interval.
- domain: exp is total/nonzero; cancellation needs inverse-exponent guard. Exponent literals 2.0 and -1.0 are structural; five non-integer floats remain parameters.
- remaining gap: Rank target likely numerical/local rank or restricted-sample artifact. Longer current-rule probes stayed at five; missing exp cancellation improves nodes, not param count.

### kotanchek/Bingo/raw 26/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No parameter-count-reducing rule. Optional cleanup: rewrite (x**Num(-1.0))*y or y*(x**Num(-1.0)) to y/x, guarded by x != 0 and ParamCost nonincrease; this only normalizes exp(x0)**-1.0 and does not reduce params.`
- proof: Let z=exp(x0)**-1.0*log(abs(x0+exp(x0))). Baseline is f + A*(z-q)*((x0+s)*(z-t)+u), with six non-integer floats f,A,q,t,s,u. Expanding over basis x0*z**2, z**2, x0*z, z, x0, 1 recovers A,s,q+t,q*t,u,f up to a discrete q/t swap, so there is no continuous one-parameter invariance. Current larger probes also retained six params.
- domain: Argument holds where x0+exp(x0)>0 and log abs is defined. Non-integer floats count; integer-valued 2.0/-1.0 are structural.
- remaining gap: No sound local rewrite found to reach n_rank=5; likely numerical/rank convention artifact. Validate by recomputing rank with diagnostics for the six basis directions.

### pagie/Bingo/raw 0/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No exact rank-reducing Num rewrite. Optional guarded log-scale normalization log(Num(a)*y) -> Num(log(a))+log(y), with associated/symmetric product variants, guarded by finite a > 0 and log-domain preservation, only when ParamCost decreases.`
- proof: Current fun_rules split one positive scalar out of a direct log product, folding log(66438756992.645454)-54.93119319072758 into -30.011646777931055. The reciprocal occurrence remains scaled because splitting it introduces denominator shifts and duplicate scale terms that do not lower tree ParamCost. Analytic family C + D*log(u_A + E + B/u_A), u_A=log(-x0*(A*x1+2/x1)), has generically independent C,D,A,E,B directions, so even sharing A gives five degrees, not four.
- domain: Requires x1 != 0, inner log positive, u_A != 0, outer log positive. Non-integer floats count; integer-valued -1/2/-2 are structural.
- remaining gap: Current tree has six counted float occurrences; sharing-aware representation might avoid duplicated A-derived scale, but exact open-domain model still has five degrees. n_rank=4 likely external numerical/counting artifact.

### pagie/Bingo/raw 5/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `square_quotient_normalization`
- proposed rule: `For nonzero constant a and guarded q != 0, m+r/q != 0: k*log(((m*q+r)**2.0)/((a*q)**2.0))+b -> k*log((m+r/q)**2.0)+(b-k*log(a**2.0)), with commuted/subtraction variants and finite guards.`
- proof: Current rules lack u**2/v**2 -> (u/v)**2, guarded cancellation of (m*q+r)/q -> m+r/q, and offset absorption. Since 28890.10432966032*3.461392830531733e-05 ~= 1, denominator is (a*q)**2 with q=x0+28890.10432966032. Algebra gives ((m*q+r)**2)/((a*q)**2)=((m+r/q)**2)/(a**2), so the offset folds to 2.5354794172640975. Result has five non-integer floats, matching n_rank=5.
- domain: a != 0; q and m+r/q nonzero preserve log/square domain. Integer 2.0/-1.0 are structural; new offset replaces old intercept rather than adding a param.
- remaining gap: Concrete 5-param expression exists. Implementation should restrict to exponent 2.0 or small even powers, constant scale a, and only fire when factor relation is present to avoid blowup.

### pagie/PySR/raw 196/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `Guarded family: x/(x*y)->1/y, x/(y*x)->1/y; x*(y/x-z)->y-x*z; 1/exp(t)->exp(0-t), exp(u)*exp(v)->exp(u+v), exp(u)/exp(v)->exp(u-v); Num(k)*exp(Num(c)+t)->Num(k*c.exp())*exp(t); scalar distribution s*(u+v)->s*u+s*v only when it exposes literal merges and lowers ParamCost.`
- proof: Current rules cannot cancel through product denominators, x*(y/x-z), exp quotient/product, or distribute scalar over a sum. With proposed rules q=x0/(x0*exp(x1*x1)) becomes exp(-x1*x1), and x0*(x1/(x0/x1)-x0) becomes x1*x1-x0*x0. The whole expression normalizes to merged constants -1.113084302832721, -0.6619166689994231, -1.0360719341398616, and 1.9503833114546634: four non-integer params matching rank.
- domain: Original requires x1 != 0 and x0 != 0 due to divisions; exp denominator nonzero; 0.931879... nonzero. Rules must preserve nonzero-domain guards. Integer 0/1/-1/2 are structural.
- remaining gap: No rank artifact needed. Risk is e-graph growth from distribution and exp combination; use one-way/profitability guards.

### pagie/Bingo/raw 0/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `log_abs_scale_extraction`
- proposed rule: `No rank-closing rewrite. Optional shape/diagnostic rule log(abs(Num(a)*y)) -> log(abs(Num(abs(a)))) + log(abs(y)) and symmetric variant, guarded by finite a != 0, y != 0, and preferably ParamCost/profitability.`
- proof: Current rules have bare log(a*y) splits but not log(abs(a*y)); baseline factors T to -2*x0*(x1**-1 + 33219378496.322727*x1). Since L=log(abs(T)) occurs both as L and D/L, tree cost counts the inner scale twice. Even with ideal sharing, expression A+B*log(abs(C+D/L+L)) with L=log(abs(-2*x0*(x1**-1+k*x1))) has at least five independent parameters A,B,C,D,k. C/D square-completion does not apply: D-C^2/4 differs by ~1.013, outside tolerance.
- domain: Requires nonzero log/abs arguments. Optional log-abs scale rule is sound for finite nonzero a and y. Integer -2/-1 structural; extracting log(abs(-2)) must fold into another param to avoid adding one.
- remaining gap: n_rank=4 likely numerical/sampling or shared-parameter artifact, not missing local rewrite to tree language.

### kotanchek/PySR/raw 204/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `Normalize exp((u / a) * (v / b)) to exp((u * v) / (a * b)), equivalently exp((1 / (a * b)) * u * v), when a and b are nonzero constants.`
- proof: The obstruction is that baseline keeps two independent constants in exp(x0 / 0.9061053191483835 * (x0 / 2.1124692156696177)). Algebraically those constants only occur through their product: exp((x0/a)*(x0/b)) = exp(x0^2/(a*b)). Replacing a and b by one coefficient or denominator leaves outer scale 0.2236500177835562, exp quadratic coefficient, numerator offset 0.09167619346933147, and affine offset 0.21837017173690837: 4 params, matching n_rank=4.
- domain: Requires a != 0 and b != 0. The observed constants are positive, so the quotient/product rewrite is safe over the real domain; exp adds no sign constraints.
- remaining gap: No remaining rank gap after this normalization. Other gaps are not needed for this row because the only extra parameter is the duplicate quotient scale inside the exp argument.

### pagie/Bingo/raw 28/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `No rank-closing local rewrite. A guarded log_abs_scale_extraction rule log(abs(a*y))->log(abs(a))+log(abs(y)) would expose scales but would not reduce this row to rank 2.`
- proof: Let L=log(abs(900226.4764680645*(x0*x1))). The row is c0+c1*log(abs(c5*x0^-1*x1^-1*L))/log(abs(c2+c3*L^-1)). Current rules are blocked by abs and **(-1.0) not normalized to division. Counting non-integer floats gives at least c0,c1,c2,c3,900226.4764680645,c5 = six params; log-abs scale extraction transforms constants to logs/ratios but does not eliminate enough to reach rank 2.
- domain: Safe log_abs_scale_extraction needs a != 0 and y != 0; sign is handled by abs. Quotient exposure also needs nonzero numerator/denominator. Integer -1.0 is structural.
- remaining gap: n_rank=2 is not explained by visible algebra. It likely requires hidden shared provenance, a non-obvious coefficient-lattice relation, or is a rank artifact.

### pagie/Bingo/raw 24/original

- conclusion: `new_rule_partial_reduction`
- rule family: `log_abs_scale_extraction`
- proposed rule: `log(abs(a * y**-1.0)) -> log(abs(a)) - log(abs(y)), with a != 0 and y != 0.`
- proof: Obstruction: current fun_rules have log(a*y), log(a/y), and log(y/a) scale extraction only outside abs and for positive constants, while this row needs log(abs(-4625426.158330705 * L**-1.0)) and log(abs(0.5182334989833758 * x1**-1.0)). The rule changes the outer term to log(abs(-4625426.158330705)) - log(abs(L)); folding plus affine collection replaces 10.425711371181222 and -4625426.158330705 by one offset, reducing 5 params to 4, not n_rank=3.
- domain: The abs form permits negative constants. Sound where a != 0, y != 0, and original log/reciprocal subterms are defined. Exponent -1.0 is structural.
- remaining gap: Single log_abs_scale_extraction closes only one of two gap parameters. Rank 3 needs another relation, shared-parameter recovery, or may be a rank artifact.

### pagie/PySR/raw 203/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `a / (b / z + c) -> (a / b) / (1 / z + c / b), with a,b,c as parameters and z any expression.`
- proof: The obstruction is that the denominator hides one affine scale inside b / z + c, so parameter equivalence spends rank on both numerator a and denominator scale b. Applying the rule to 2.792238560484319 / (3.4342474157379796 / z + 2.844338639118127), z=(x1 ** 2.0) ** 1.8741636407391655, gives 0.813054... / (1 / z + 0.828225...), reducing the quotient float count from 4 to 3 and matching 7 -> 6.
- domain: Valid where b != 0 and b / z + c != 0; z must be defined and nonzero. The reciprocal form has the same x1=0 singularity as the original quotient term.
- remaining gap: This accounts for rank_gap=1 with the smallest local quotient normalization. Integer exponent 2.0 is not counted; the redundant scale is denominator coefficient 3.4342474157379796.

### pagie/Bingo/raw 21/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `square_quotient_normalization`
- proposed rule: `rewrite(x * x).to(x ** 2) with exact integer exponent 2; optionally also rewrite((x * x) * (y * y)).to((x * y) ** 2).`
- proof: Obstruction: the miss is duplicate product squares. Current rules have commutativity/associativity and log rules, but no product-to-square collection, so occurrences of 0.0004240447616209535 and 7480.0610581772835 remain duplicated in A*A and B*B. Let A=exp(log(abs(0.0004240447616209535*(x0*x1)))) and B=7480.0610581772835+log(abs(0.0004240447616209535*(x0*x1))). Rewriting A*A and B*B to squares leaves 4 non-integer floats, matching n_rank=4.
- domain: x*x -> x**2 is domain-preserving for real square semantics with integer exponent 2 and preserves existing log/abs obligations. The exponent 2 is structural, not a fitted parameter.
- remaining gap: No additional rank-reducing rule is needed if extraction keeps square form. Abs-aware exp(log(abs(t))) normalization may simplify but is not required to reach rank 4.

### pagie/SBP/raw 170/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `c1 + (c2 + t) -> (c1 + c2) + t for float constants inside affine sums, e.g. -10.978 + (15.158203042332545 - exp(x1) + 15.599*x1) -> 4.180203042332545 - exp(x1) + 15.599*x1.`
- proof: Obstruction: baseline still has two separate additive float constants in the same affine expression. The earlier fold 15.694 - (exp(x1) + exp(-0.624)) -> 15.158203042332545 - exp(x1) only explains how baseline is reached; the rank-reducing step is collecting -10.978 and 15.158203042332545 into 4.180203042332545, reducing non-integer float count from 5 to 4 and preserving algebra.
- domain: Valid for real affine addition; with floating semantics use the same constant-evaluation policy/tolerance as existing folding. Counted floats 5.3e-05, -0.00822052, -10.978, 15.158203042332545, 15.599 become 5.3e-05, -0.00822052, 4.180203042332545, 15.599.
- remaining gap: No remaining rank gap if affine constant collection runs after the baseline-forming constant fold; it directly explains baseline_after_params=5 to n_rank=4.

### pagie/SBP/raw 179/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No exact single local rank-reducing rule. The smallest exact normalization is (A*(exp(u)+c)+B*exp(u))*exp(-u) -> A*(1+c*exp(-u))+B, but it leaves the same six non-integer floats.`
- proof: Affine collection has already exposed 0.004665028=0.004516*1.033 and 23.303365810451727=0.105238/0.004516, while outer exp(-x0) blocks exp(x0)*exp(-x0) cancellation across the sum. Applying cancellation gives 0.004665028*x0*x1*(x1-4.432)*(x1-x0+3.749)*(1-13.538436794046856*exp(-x0))+0.004516*(23.303365810451727-x0), still six non-integer floats.
- domain: Candidate rewrites preserve six scalar values or trade one derived scalar for another, e.g. 0.004665028 <-> 0.004516*1.033. Integer -1.0 is structural.
- remaining gap: Rank 4 appears to come from shared-parameter/coefficient-rank structure rather than a small tree rewrite reducing literal floats to four.

### kotanchek/SBP/raw 173/original

- conclusion: `existing_rules_more_iterations`
- rule family: `existing_rules_only`
- proposed rule: `none; current basic_rules already contains a*x + b -> a*(x + b/a), which applies with a=-11.471, x=(-17.85910443727661 - x0), and b=-201.87275573295364.`
- proof: Default extraction stops with an affine constant outside the scaled term. Existing rule rewrites -201.87275573295364 + -11.471*(-17.85910443727661 - x0) to -11.471*(-0.2605728591270484 - x0), giving four counted floats: 0.039334, 7.988, -11.471, -0.2605728591270484. Longer current-rules probes already reach after_params=4, matching n_rank=4.
- domain: Pure real polynomial/affine arithmetic; no log, sqrt, exp, or denominator condition introduced. Division by a is guarded by a=-11.471 nonzero finite.
- remaining gap: No algebraic rule gap remains. The issue is default search budget/scheduling, not missing rules or rank artifact.

### pagie/SBP/raw 167/sympy

- conclusion: `new_rule_partial_reduction`
- rule family: `tolerance_coefficient_snapping`
- proposed rule: `For finite nonzero coefficient a, finite coefficient b, terms x,y, and integer-valued k, if abs(b/a-k)<=CONST_MERGE_TOLERANCE or abs(b-k*a)<=CONST_MERGE_TOLERANCE, rewrite a*x + b*y -> a*(x+k*y), keeping k integer-valued/free.`
- proof: Current affine factoring sees 0.24133/0.048266 as 4.999999999999999, counted as a non-integer float though intended ratio is 5. Snapping gives 0.048266*((x0-3.247)*(x0-3.579)*x0 - x1**2 - x0 + 5*x1 - 5*0.2728421663282642), reducing only the spurious near-integer parameter. Expanded coefficients still recover a, p+q, p*q, and c.
- domain: Pure polynomial arithmetic; no denominator/log/sqrt/exp side conditions. Snapping is approximate and needs finite/nonzero guards. Integer 2.0 and snapped 5.0 are structural.
- remaining gap: Partial only. After snapping, four non-integer floats remain; n_rank=3 likely needs hidden rounded relation or is a numerical/counting artifact.

### pagie/SBP/raw 165/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none; optional common-scale collection -a*u - b*v -> -a*(u + (b/a)*v) only normalizes shape here.`
- proof: The apparent missed normalization is absorbing -0.1463448*x0**2 into the existing -0.01085 factor, but 0.1463448/0.01085 = 13.488, replacing one non-integer float with another. Baseline counts five floats: 0.01085, 14.814, 18.147000000000002, 0.1463448, 0.0894957768; optional factoring still counts five: 0.01085, 14.814, 18.147000000000002, 13.488, 0.0894957768.
- domain: Polynomial arithmetic and exp over real x0,x1 with finite constants. If optional scale collection divides by scale, guard scale != 0.
- remaining gap: n_rank=4 likely reflects numerical rank, external convention, or model-family reconstruction not justified by exact local rewrite.

### pagie/EPLEX/raw 45/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `For finite nonzero a,b,c and Num u,v, add guarded log-abs scale extraction in quotient contexts, e.g. exp(log(abs(u/(a/v)))/b) -> abs(a)**(-1/b)*exp(log(abs(u*v))/b), and log(abs((u/a)/c)) -> log(abs(u/(a*c))).`
- proof: Current rules split log(a*y), log(y/a), and log(exp(x)), but not scale factors hidden under log(abs(...)) or exp(log(abs(...))/b). Baseline keeps seven floats after folding log(abs(-0.021)). The inner rule combines 0.607 and 0.447 while retaining -0.542; the outer quotient rule combines -0.927 and -1.297. Remaining floats are folded log(abs(-0.021)), combined inner coefficient, -0.542, -0.302, and combined outer quotient scale, matching n_rank=5.
- domain: Requires finite nonzero scales and original log/abs definedness: a,b,c != 0, u*v != 0, u != 0. abs removes sign restrictions.
- remaining gap: Bare log(abs(t/a)) may be insufficient; rank-closing version should include direct exp-log-abs quotient context or equivalent quotient exposure plus exp additive extraction.

### kotanchek/PySR/raw 187/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `Num(a) * exp(x + Num(b)) -> Num(a * b.exp()) * exp(x), plus exp(Num(b)+x) and commuted product variants; require finite a,b and keep non-subsuming/profit-guarded.`
- proof: Current rules fold exp(Num) and log(Num), but do not extract additive constants from exp to merge with adjacent scalar. Baseline has seven counted floats. Applying rule with x=x0+x0-x0*(x0+0.040585192647742296), b=-1.0149512623308583 combines 0.6992234510287301*exp(b)=0.25341262808852694. Count then becomes six, matching n_rank=6.
- domain: exp(b+x)=exp(b)*exp(x) is sound over reals and adds no log-domain restriction. Existing denominator nonzero conditions remain. Use finite f64 guards to avoid NaN/inf.
- remaining gap: No denominator or log rule is needed for the one-rank gap; further shape cleanup changes syntax but is not required.

### pagie/EPLEX/raw 51/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `coefficient_lattice_factoring`
- proposed rule: `For U=x0**4, if an inner log_abs quotient has (U*(V - a) - b) * (s/(U + r)) and b ~= a*r, rewrite it as s*((U*V)/(U + r) - a); equivalently (U*V - a*(U+r))/(U+r) -> U*V/(U+r) - a.`
- proof: Current rules normalize the denominator to s/(U+r) but leave b independent, missing b=a*r. Here a=0.0624330452543088, r=4.784355555555555, and a*r ~= 0.298701886912704 = b. Thus x0**4*(x1-a)-b = x0**4*x1 - a*(x0**4+r), so division by x0**4+r gives x0**4*x1/(x0**4+r) - a. Baseline has c,d,a,b,s,r,g = 7 floats; rewritten has c,d,a,s,r,g = 6, matching n_rank.
- domain: Algebraic where x0**4+r is nonzero; surrounding abs/log still excludes zeros. r is positive here, but general rule should keep denominator-zero exclusions.
- remaining gap: No remaining gap. Equality b=a*r is approximate at displayed precision, so use tolerance/snapping guard.

### kotanchek/PySR/raw 208/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `coefficient_lattice_factoring`
- proposed rule: `For finite constants a,b and terms x,y, if a+b is an integer-valued literal k, rewrite a*x - b*y -> a*(x+y) - k*y, with sign/order variants and ParamCost-decrease guard.`
- proof: Current rules factor multiplicative scales in the affine bracket but do not exploit integer-complement coefficient relation between x0 and exp(x1). Here 1.8394445968539719 + 0.16055540314602817 = 2.0, so 1.8394445968539719*x0 - 0.16055540314602817*exp(x1) rewrites to 1.8394445968539719*(x0+exp(x1)) - 2.0*exp(x1). Full expression then uses exactly six non-integer floats, matching n_rank=6.
- domain: Affine arithmetic identity over real terms; exp(x1) is opaque and total. No division/log/sqrt side conditions. Guard finite f64 and integer-valued computed sum.
- remaining gap: No gap remains. Equivalent post-normalization relation exists, but original-form a+b=k rule is smaller.

### kotanchek/Bingo/raw 29/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `rewrite((u*v + w*z)/(v*z)).to(u/z + w/v), with guards v != 0 and z != 0; include commuted add/mul variants only as needed.`
- proof: Current rules do not cancel separate numerator summands against separate factors of product denominator, so x1 - 6.784280947796324 remains duplicated. Let X=x0-exp(x0), Y=x1-6.784280947796324, Z=exp(x0)-52.61627831646421, C=35.56866639996601*x1, A=-6.656270926980652e-05*x1-1.550707023441505e-05. Rendered form is A*(Y*Z*exp(x0)+C*X)*Z/(X*Y). Rule gives A*(C/Y + Z*exp(x0)/X)*Z, with six float leaves, matching n_rank.
- domain: Valid where v,z nonzero; here guards are original denominator factors X != 0 and Y != 0. exp is total; integer -1/1 are structural.
- remaining gap: No row-local gap remains. Implementation needs matching commuted product/sum forms without broad denominator distribution blowup.

### kotanchek/PySR/raw 195/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `((a + c1) - b) + c2 -> (a - b) + (c1 + c2), for finite constants c1,c2; collect constants across intervening subtraction in affine additive chain.`
- proof: Baseline has two separate floats in the inner denominator, 1.6282363698705866 and 1.6744362026213413, because exp(0.5154765131661209) was folded but not collected with the earlier additive constant across -(x0+x1). Combining them gives 3.302672572491928, changing exp(x1)+1.6282363698705866-(x0+x1)+1.6744362026213413 into exp(x1)-(x0+x1)+3.302672572491928. This removes one float, matching 5->4.
- domain: Real affine reassociation over finite constants; denominator-zero behavior unchanged under exact real semantics. Guard/approximate if preserving IEEE evaluation order.
- remaining gap: Remaining counted floats are 0.6193677368705359, 2.853886507925958, -0.24717047383135762, and 3.302672572491928; no further rule needed.

### pagie/EPLEX/raw 39/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `square_quotient_normalization`
- proposed rule: `a ** 4.0 / (b * a ** 2.0 + c) ** 2.0 => 1.0 / (b + c / a ** 2.0) ** 2.0.`
- proof: Baseline duplicates 0.843881856540084 because (0.843881856540084*x1**2+1) appears as a**4 numerator and as a**2 inside squared denominator. Let a=0.843881856540084*x1**2+1, b=x1**2, c=0.938527657604728. Then a**4/(b*a**2+c)**2 = 1/(b+c/a**2)**2. After folding log(abs(-0.414))**2, result has six non-integer float occurrences instead of seven, matching n_rank=6.
- domain: Introduces division by a**2, so require a != 0 and denominators nonzero. In this row a is strictly positive over real x1. Integer -1/1/2/4 excluded.
- remaining gap: No remaining gap after square quotient normalization. x0 term has no analogous repeated coefficient to eliminate.

### kotanchek/PySR/raw 182/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `((a + b*c)*d)/c -> (a*d)/c + b*d, when c != 0; orient only when one addend contains a factor equal to denominator and the other does not.`
- proof: Denominator C=exp(x1-x0)-x1+2.2629335889059945 is a factor in exactly one numerator addend, but the whole numerator is multiplied by D=exp(-x0**2), so ordinary cancellation cannot fire. Baseline is (A+B*C)*D/C. Rule rewrites to A*D/C + B*D, then exp(x0**2)*exp(-x0**2)->1 reduces B*D to 0.16813782855537174 - 0.05041963936946082*x0, matching the one-rank improvement.
- domain: Valid where C != 0. Uses exp(t)*exp(-t)=1, valid for real exp. Non-integer floats count; -1.0 and 2.0 structural.
- remaining gap: No broader distributivity needed. Closest taxonomy is quotient_coefficient_exposure because it exposes denominator-matching numerator component for cancellation.

### pagie/EPLEX/raw 55/sympy

- conclusion: `new_rule_partial_reduction`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `exp(c + u) -> exp(c) * exp(u) when c is a numeric constant and u contains variables; expose constant additive offset inside exp arguments before parameterization.`
- proof: Baseline has variable-dependent exp nodes whose arguments mix numeric offset with variable term. In the innermost x0 branch, -1.0*exp(-0.18496503345507379 - 2.0*log(abs(x0**2))) contains a non-integer float inside additive exp argument. Pulling that constant out gives a parameterized scale factor multiplying variable-only exp term, reducing the exposed mixed affine-in-exp structure by one, but larger denominator still has nested exp terms and outer constants.
- domain: exp(c+u)=exp(c)*exp(u) is domain-safe for finite real c,u. Surrounding log(abs(exp(...))) remains positive. May change overflow/underflow numerically, so use symbolic canonicalization.
- remaining gap: Partial only. Additional normalization may be needed to fully match rank 5; this is the smallest visible rank-reducing obstruction.

### pagie/EPLEX/raw 59/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `For finite a != 0 and b, rewrite log(abs(a*y)) + b -> log(abs((a*exp(b))*y)); include b + log(abs(a*y)) and left-associated product variants.`
- proof: Current rules split log(a*y) only outside abs and for positive a, so log(abs(-0.829187396351575*x0*x1)) + 0.01646 remains two floats. The rule rewrites it to log(abs(-0.8429487662722076*x0*x1)), since -0.8429487662722076 = -0.829187396351575*exp(0.01646). This drops counted floats from seven to six, matching n_rank=6.
- domain: Valid when a != 0, b finite, exp(b) finite, and y != 0. abs handles negative a and preserves zero/outer-log singularities.
- remaining gap: No remaining gap. Split form would also expose relation but needs constant/addition collection; direct absorption is smallest.

### pagie/Bingo/raw 28/sympy

- conclusion: `new_rule_partial_reduction`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `For shared nonzero denominator z, rewrite (a*x)/z + b/z -> (a*x + b)/z and (a*x)/z - b/z -> (a*x - b)/z, with commuted variants; broader x/z+y/z requires profitability/domain guard.`
- proof: Current rules cannot collect two terms over the same symbolic denominator, so two occurrences of log(x0*x1)+13.71040165117035 remain counted. Let L=log(x0*x1), T=x0*x1, C=13.71040165117035. Shared-denominator collection rewrites -3.3162197817782353*L/(L+C)+30.73693812541886/(L+C) to (-3.3162197817782353*L+30.73693812541886)/(L+C), dropping one float. Similar numerator exposure does not reduce further.
- domain: Sound only where shared denominator z is defined and nonzero; surrounding log args must stay positive. Negative log scale factors require stronger sign/domain guards.
- remaining gap: Partial only: visible reduction 8->7, not rank 2. Remaining target likely needs hidden provenance, non-local sharing, or rank/counting artifact.

### kotanchek/GP-GOMEA/raw 98/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `((a - x) * (x + b)) + c -> (c + a*b) - x * (x + (b - a)); add commuted plus variants as needed.`
- proof: Current rules fold -0.289+14.81 to 14.521 but do not collect constants across an affine product plus constant offset. With a=-7.097,b=4.18,c=14.521, (a-x1)*(x1+b)+c becomes (14.521 + -7.097*4.18) - x1*(x1+11.277) = -15.14446 - x1*(x1+11.277). Full expression then has six non-integer floats, matching n_rank=6.
- domain: Pure polynomial identity over reals; exp total. Guard finite f64 folds and orient by ParamCost/profitability. Integer coefficients are structural.
- remaining gap: No broader exp-additive rule needed; remaining gap is monic quadratic/affine-product constant collection.

### kotanchek/Bingo/raw 24/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `no_rank_reducing_rule`
- proposed rule: `none exact rank-reducing; optional shape-only rules are exp(Num(a)+x) -> Num(exp(a))*exp(x) and (-x)**Num(-1.0) -> Num(-1.0)*(x**Num(-1.0)), guarded on finite constants.`
- proof: Current rules fold exp(Num) but do not split exp(c+u) or normalize negative reciprocal denominators. In positive-denominator form: f + (b*(x0+x0^2)*(x1+d)-c)/(x0^2+p*exp(x0^2)+a) + exp(-a)*exp(-p*exp(x0^2)). Exp additive extraction only replaces second a by exp(-a) and leaves p repeated, so tree still has eight float occurrences. Even with ideal sharing, x1 coefficient fixes b,a,p; x1-free rational fixes d,c; f is offset: six independent params, not five.
- domain: exp total/positive; reciprocal preserves original denominator-nonzero condition. Integer -1.0 structural; fitted constants are non-integer floats.
- remaining gap: n_rank=5 likely external rank/counting artifact or needs let-style sharing plus another hidden dependency; no smallest sound local tree rewrite reaches five.

### kotanchek/GP-GOMEA/raw 102/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `k*(u*exp(c+v)-w) -> (k*exp(c))*(u*exp(v)) - k*w, guarded to literal non-integer float c and coefficient-like k; extract exp(c) and absorb into branch coefficient.`
- proof: From baseline, final miss is additive constant 7.588 inside exp(7.588 - x0). It appears only as multiplicative scale exp(7.588) on the first exponential branch. Rewriting exp(7.588-x0) to exp(7.588)*exp(-x0) and absorbing exp(7.588) into surrounding numeric coefficient removes one counted float, taking baseline constants (-0.108816, 6.5e-05, -8.386, 0.898, 7.588, 32.702, 13.305) to six independent params.
- domain: Valid for positive real exp semantics. Apply as coefficient exposure/absorption around products/subtraction, not approximate eager materialization unless pipeline intentionally materializes fitted coefficients.
- remaining gap: This accounts for full 7->6 gap. Remaining constants are structurally independent; no affine collection or duplicate relation needed.

### pagie/SBP/raw 154/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_duplicate_collection`
- proposed rule: `Canonicalize identical affine shift constants across different symbolic terms: occurrences like `(x_i + c)` and `(x_j + c)` should expose and reuse one shared parameter for the same non-integer constant `c`, including subtraction-rendered forms `(x_i - p)`.`
- proof: Current-rule obstruction: the two `-4.881` literals are buried as separate affine shifts, `x0 + -4.881` inside the cubic x0 factor and `x1 + -4.881` inside the x1 factor, so the rendered baseline counts both occurrences. All other non-integer literals appear only once with no exact integer-coefficient relation needed. Sharing the duplicate affine offset changes the parameter count from 8 literal occurrences to 7: `0.018176`, `0.001712`, `-8.935`, `18.954`, `-5.12`, shared `-4.881`, and `18.328`. This matches n_rank=7.
- domain: Pure polynomial/affine normalization only; no exp/log/domain side conditions. Integer-valued structural coefficients such as implicit `1` and `-1` are not counted as parameters.
- remaining gap: No evidence here for quotient, reciprocal, square, exp, log(abs), or coefficient-lattice rules. A tolerance rule could notice `5.12 + 4.881 ~= 10.001`, but exact duplicate affine collection is the smaller sufficient rank-closing rule.

### kotanchek/PySR/raw 197/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `For constants A,k,a,b and term E independent of numeric-only collection: A*exp(E + k*(x + a)*(x + b)) -> (A*exp(k*a*b))*exp(E + k*x*(x + a + b)). This is the smallest rank-reducing form for the hidden constant in the shifted quadratic exponent.`
- proof: Obstruction: current rules factored the outer affine coefficient, giving the baseline 7-param form, but they leave the constant term inside exp(k*(x0+a)*(x0+b)) hidden. With a=-0.03948916953537872, b=-1.2172800892692854, k=-0.8631793118974627, rewrite k*(x0+a)*(x0+b)=k*x0*(x0+a+b)+k*a*b and absorb exp(k*a*b) into the outer scale. Params become adjusted scale, affine root 8.986149794812901, affine shift 0.9808871412991315, exp x1 coeff, exp quadratic coeff k, and exp affine sum a+b: final parameter count 6. x1**4.0 is integer-valued and not counted.
- domain: Valid for real finite expressions because exp(u+v)=exp(u)*exp(v). The absorbed exp(k*a*b) is positive and only changes the multiplicative scale. No new division, log, abs, or sign-sensitive branch is introduced.
- remaining gap: This does not require many-parameter sharing or a rank artifact. The rule also performs local affine collection inside the exp argument, but the rank drop comes from extracting the additive exp constant into the outer coefficient.

### pagie/EPLEX/raw 56/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `square_quotient_normalization`
- proposed rule: `Normalize squared scaled denominators: c / (a * y)^2 -> (c / a^2) / y^2, with a a nonzero constant. Here this rewrites -0.06332088084610697 / (x0 * 0.3)^2 to -0.703565342734522 / x0^2.`
- proof: The baseline has 5 non-integer parameters: -0.752906976744186, -0.306, -0.06332088084610697, 0.3, and -0.958. Existing rules have already collected the outer products/divisions and the exp(-0.942) constant into -0.06332088084610697. The remaining obstruction is the coefficient 0.3 hidden under a square in the denominator. Since (x0 * 0.3)^2 = 0.09 * x0^2, the exponent coefficient becomes -0.06332088084610697 / 0.09 = -0.703565342734522. This removes the separate 0.3 parameter, leaving 4 parameters, matching n_rank=4.
- domain: Requires a != 0. The rewrite preserves the same singular set for y=0, since both forms divide by y^2. For real square semantics, a^2 is positive, so no branch or sign issue is introduced by moving the scale into the numerator coefficient.
- remaining gap: No further rank-reducing rule is needed for this row. The outer division by -0.958 remains one parameter either way, and the x1-side exponent still needs its independent scale and additive exp shift, so the final count after this rule is 4.

### kotanchek/Bingo/raw 24/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No single local rewrite from the listed rule-gap families exposes a 5-parameter form. The only visible saving is recognizing the repeated coefficient 0.32506758388128154 in the denominator and in exp(-0.32506758388128154*exp(x0**2)) as one shared parameter, which is a rank/accounting issue rather than a simplifying rewrite.`
- proof: Current-rule obstruction first: the expression is already in a normalized sum of a quotient plus an affine constant plus an exponential term. The quotient numerator has independent non-integer constants a, b, c; the denominator has d, e; the tail has f, g and reuses d. None of affine constant collection, duplicate affine constants, exp additive extraction, product/quotient exp normalization, log(abs(k*y)), reciprocal exposure, square quotient normalization, or lattice factoring creates an algebraic equality eliminating three constants. Counting non-integer coefficients with shared d gives final parameter count 7, not rank 5.
- domain: Assumes x0,x1 real and exp is ordinary real exponential. Integer-valued exponent 2.0 is treated as structural, not a learned parameter. The repeated d is safe to share syntactically, but it does not change the function class algebraically.
- remaining gap: Rank 5 likely comes from the downstream rank model allowing shared parameters or fitting artifacts not represented by current rewrite rules. A genuine 5-param proof would need two additional numeric dependencies among a,b,c,e,f,g,d, but no small symbolic-rule family listed exposes them.

### kotanchek/PySR/raw 209/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `Add a targeted exposure/cancellation rule: (S + C*D)*(N/D) -> C*N + (S*N)/D, plus commuted and subtraction variants, only when the same denominator expression D is matched exactly or canonically.`
- proof: Let D=(-2.487415653599196*x0+exp(x0)+0.7901026138870583)*(-3.0*x1+exp(x1)+7.787500628085413), S=x0+x1+0.4553546802016498, a=0.01811268853996139. The row is (S-a*D)*(x1/D). Current obstruction: the additive numerator hides the exact denominator factor, so ordinary product cancellation cannot remove the duplicate D occurrence. The proposed rule yields -a*x1 + x1*S/D. Non-integer params are a, 0.4553546802016498, 2.487415653599196, 0.7901026138870583, 7.787500628085413: final parameter count 5, matching n_rank; -3.0 is integer-valued and not counted.
- domain: Requires D != 0, which is already required by the original quotient. No additional exp, log, abs, or sign assumptions are introduced. The transformation preserves the same poles at zeros of either denominator factor.
- remaining gap: Implementation needs commutative product canonicalization so D in the numerator product matches D in the reciprocal/quotient. If that exact D match is available, no affine, exp, log_abs, square quotient, tolerance snapping, or shared-rank artifact rule is needed for this row.

### pagie/SBP/raw 159/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `tolerance_coefficient_snapping`
- proposed rule: `When a Float coefficient in an affine/linear factor is within tolerance of a small rational p/q, canonicalize it to exact p/q, e.g. 0.3333333333333325 * x1 -> x1 / 3 inside x0 + c*x1.`
- proof: Current rules already expose the inner product as (x0 + -1.883) * (x0 + 0.3333333333333325*x1), but leave the near-rational coefficient as a free float parameter. Baseline params are -0.42026652729485225, -1.883, 0.3333333333333325, -0.08411357484865532, 0.001732, 89.09122401847574 = 6, excluding -1.0. Snapping 0.3333333333333325 to exact 1/3 makes it rank-free, leaving 5 float params and matching n_rank=5.
- domain: Safe only under the same numeric tolerance used for param equality; require small numerator/denominator and preserve sign. This is algebraic over real-valued x0,x1 and does not change exp-domain assumptions.
- remaining gap: No exp product cancellation or quotient exposure is needed to close this one-rank miss. Other constants may have product relationships, but the smallest sufficient obstruction is the unsnapped 1/3 coefficient.

### pagie/SBP/raw 151/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No new rank-reducing rewrite. The smallest sound form is the existing common-factor collection: a*(x1*(r + x0) + exp(k*x0) + x1^3 + x0^2 + s) + b*x1^2, with a,b,r,k,s all non-integer params.`
- proof: Obstruction: current rules already expose the common coefficient -0.018385, but doing so replaces the x1 and constant coefficients by non-integer ratios -2.996448 and -9.064019581180311, which still count as params. Expansion proves equality: -0.018385*x1*(-2.996448+x0) gives -0.018385*x0*x1 + 0.05508969648*x1; -0.018385*(-9.064019581180311) gives +0.166642. The exp scale -13.828 and x1^2 coefficient 0.07247367 are independent non-integer params. Minimum visible count remains 5, not rank 2.
- domain: Integer exponents 2.0 and 3.0 can be structural. Non-integer floats -0.018385, 0.07247367, -13.828, -2.996448, and -9.064019581180311 count as params after factoring. Snapping -2.996448 to -3 or -9.064019581180311 to -9 would change the expression.
- remaining gap: The rank gap appears to come from the rank artifact or shared latent-parameter accounting outside single-expression rewrite algebra. No affine, quotient, exp, log, or square normalization rule reduces this row to 2 params without hiding non-integer ratios or using tolerance-based approximation.

### pagie/SBP/raw 172/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No sound rank-reducing rewrite. Mark as shared-parameter/rank artifact unless external parameter ties justify equating one of {15.767, 6.2586666666666675, 4.990666666666667, 0.00963, 0.179028}.`
- proof: Obstruction: the baseline has already collected the duplicated x0 and -x0^2 coefficient and factored the shared -0.00963 exp/polynomial scale, leaving -0.00963*(exp(-15.767*x0)+exp(x1)-3*(6.2586666666666675*(x0-x0^2)+x0^3+4.990666666666667*x1))+0.179028. With integer -3.0 free, the remaining non-integer parameters are scale, exp slope, two independent affine polynomial ratios, and offset. The two ratios multiply independent basis functions, so collecting or reassociating terms cannot remove either one exactly.
- domain: Non-integer floats count as params; integer-valued -3.0 does not. exp(-15.767*x0) gives an independent slope parameter. The x1 linear term and x0-x0^2 polynomial block are algebraically independent, so their decimal ratios cannot be merged by generic rewrites.
- remaining gap: The rank gap of 1 does not appear closable by current algebraic rewrite families without tolerance snapping or an external model constraint. If n_rank=4 is trusted, the missing equality is semantic/estimation metadata rather than a local expression rewrite.

### kotanchek/PySR/raw 193/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `coefficient_lattice_factoring`
- proposed rule: `For scalar literals c,d,n,m,r: exp(c+F)*(U/n - exp(d+G)*(V/m - r)) -> exp(F)*(p*U - exp(G)*(q*V - h)), where p=exp(c)/n, q=exp(c+d)/m, h=exp(c+d)*r; fold p,q,h as single coefficients.`
- proof: Obstruction first: current rules fold literal arithmetic and have limited quotient-in-sum factoring, but they do not extract additive constants from exp or distribute/collect scalar coefficient paths through subtraction, so c,n,d,m,r remain separate. Let F=-x0*x0+2*x0-x1, G=-x0+(x0+e-x1)*x1, U=x1*(x1+b), V=x1*x1. The row is exp(c+F)*(U/n - exp(d+G)*(V/m - r)). It equals exp(F)*(p*U - exp(G)*(q*V - h)) with params b,e,p,q,h. Thus the seven noninteger literals c,b,n,d,e,m,r collapse to five independent coefficient parameters, matching n_rank=5.
- domain: Valid over real arithmetic with n=0.9598793817164541 and m=0.9039712460608937 nonzero. exp is total, so extracting exp(c) and exp(d) adds no domain restriction. Integer literals such as 1 and 2 are structural, not params.
- remaining gap: A broad distributivity rule may blow up search; implement as a guarded coefficient-path normalizer for exp constants and quotient denominators inside additive terms. No many-parameter or numerical rank artifact is needed for this row.

### kotanchek/PySR/raw 207/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `For parameter literals c,a,b and param-free term t: c * (exp(t + a) + b) -> (c * exp(a)) * exp(t) + (c * b), with literal products evaluated.`
- proof: Current-rule obstruction: the baseline has already collected -(log(1.0353531371341038)*exp(x1+0.35375558989724504)+0.11255368129091607) into -0.034742563822104125*(exp(x1+0.35375558989724504)+3.2396481119596157), but no current rule extracts the additive constant from exp through the outer coefficient. Applying the proposed rule to that subterm gives A*exp(x1)+B where A=-0.034742563822104125*exp(0.35375558989724504) and B=-0.034742563822104125*3.2396481119596157. This replaces params c,a,b with A,B, reducing 8 params to 7.
- domain: Identity is exact over real arithmetic for exp and multiplication. The log literal was already evaluated from a positive constant, so the proposed rule adds no new log-domain condition. Denominator and all x0-dependent structure are untouched.
- remaining gap: This explains the single rank gap if rank counts non-integer floats only. It assumes the rank artifact does not penalize distributing this local additive factor; if tree-shape costs dominate elsewhere, verify against the rank implementation.

### kotanchek/PySR/raw 194/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `z / (y / a + b) -> (a * z) / (y + a * b), with a != 0`
- proof: Current-rule obstruction: rules can collect constants inside an affine denominator via x / a + b -> (x + b*a) / a, but they do not clear that exposed denominator scale through an enclosing quotient. Let y=exp(x0)-x0-x0, a=0.34806215833743864, b=1.7208022848793882, and z be the remaining numerator. The subterm z / (y/a + b) becomes (a*z)/(y+a*b). Existing constant folding then combines a with the existing factor 1/1.4092295785725586, replacing three params {a,b,1.409...} by two params {a*b,a/1.409...}. Other params remain the exp coefficient, 0.9729207943380669, and -0.1759177414425294, matching rank 5.
- domain: Requires a != 0 and preserves the original undefined set where y/a+b = 0, equivalently y+a*b = 0. No log/sqrt domain changes. Non-integer floats count as params; integer literals like 1 and 2 do not.
- remaining gap: If implemented only as the two-step existing x/a+b collection plus a separate x/(y/a) clearing rule, extraction must still find the enclosing quotient. The smallest direct rule is the affine-denominator quotient exposure above.

### pagie/PySR/raw 205/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `Rewrite a/exp(E) to a*exp(-E), then collect multiplicative constants in products containing exp terms: c*((P)*Q)/exp(A+B) -> (c*Q)*P/exp(A+B), exposing the single outer coefficient used by rank.`
- proof: Obstruction first: current rules can simplify x1/x1 -> 1 and collect (0.8979952519492354*x1-x1)*x1 -> -0.1020047480507646*x1^2, and exp(0.5335679036188624) is a param constant. Algebra gives (-0.1020047480507646*x1^2+log(2.7971484481986244))*(0.13355299915112226*x1^2-exp(0.5335679036188624))/exp(x0^2+1.294289683924389). Expanding the numerator factors out -0.013623040029834211, yielding -0.013623040029834211*((x1^2-10.083849100616565)*(x1^2-12.766502981662837))/exp(x0^2+1.294289683924389).
- domain: x1/x1 -> 1 requires x1 != 0 if using real algebra. exp denominators are nonzero. log(2.7971484481986244) is defined and positive. Non-integer floats count as params; integer-valued literals like powers of 2 do not drive the gap.
- remaining gap: The first term already matches baseline form modulo ordinary multiplication reassociation. The rank gap is from failing to expose/factor the outer quotient coefficient for the second term, not from needing many parameters or a rank artifact.

### pagie/Operon/raw 120/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `tolerance_coefficient_snapping`
- proposed rule: `Before parameter counting, canonicalize integer-valued float coefficients in multiplicative positions: if Float(c) is within tolerance of integer k, rewrite Mul(Float(c), t) to Mul(Int(k), t), with k=1 removed and k=-1 rendered as Neg(t). This covers exp(-1.0 * z) -> exp(-z).`
- proof: Current-rule obstruction: the rendered baseline still contains -1.0 as the coefficient in exp(-1.0 * exp(-1.0047436520881448 * x1 ** 2.0)), so param counting charges it as an extra float parameter. The genuinely non-integer constants are 0.22864008270232225, -9.69348884031831, -1.0047436520881448, 0.3440097684222109, 1.635892432732362, -1.1765190489038249, and -0.8640792395159441: seven total. Snapping only -1.0 to the integer literal -1 changes exp(-1.0 * z) to exp(-z), reducing baseline_after_params from 8 to n_rank 7.
- domain: No special real-domain assumption is needed for this row: exp is total on real inputs and the rewrite only canonicalizes an exactly integer-valued scalar coefficient. Do not snap arbitrary near-integers unless the tolerance is deliberately part of the param_eq abstraction.
- remaining gap: No remaining rank gap after this rule. The duplicated inner coefficient -1.0047436520881448 is already one shared non-integer parameter; the x ** 2.0 literals are integer-valued powers and should stay outside the parameter count.

### pagie/Bingo/raw 15/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `log(abs(k * u)) => log(abs(u)) + log(abs(k)) for nonzero scalar parameter k, with the extracted log(abs(k)) treated as a scalar constant and then collected into surrounding affine constants.`
- proof: Let k=-314.9892428270721 and L=log(abs(k*(x0*x1))). Current rules expose (-1)*k^-1 as 0.0031747115902272137, but they leave k embedded inside each log(abs(k*...)). Expanding log(abs(k*u)) gives log(abs(u))+log(abs(k)). In the first L occurrence this adds only a scalar multiple of x0^-1*x1^-1*(39.54832653926522+x1), and in the nested quotient log it converts log(abs(k*x0*(x1*L)^-1)) to log(abs(x0*(x1*L)^-1))+log(abs(k)). The final extracted scalar is multiplied by -240.49898207944213 and absorbed into the affine constant 292.42415431458545, eliminating one independent float parameter from the rendered form.
- domain: Valid where k!=0 and log arguments are nonzero. abs makes the scale split real-valued: abs(k*u)=abs(k)*abs(u). Integer literals -1 and powers -1 are structural, not params; k and non-integer coefficients/constants are params.
- remaining gap: This assumes the rank target counted the log scale as removable rather than requiring a separate learned constant for log(abs(k)). If current rules already have this identity but lack enough saturation iterations, then classify as existing_rules_more_iterations; the visible obstruction is the unextracted scalar inside log(abs(k*...)).

### pagie/PySR/raw 197/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `rewrite(a / y + b / y).to((a + b) / y), with y != 0; signed numerators cover subtraction after existing x - a and x + -y rewrites.`
- proof: Current-rule obstruction: the pipeline can fold exp(0.8366837949675797) to 2.308698151681927 and can collect affine constants, but it has no same-denominator collection for a/y + b/y. The miss is exactly the pair -2.308698151681927/exp(x1*x1) + -1.8924009776034219/exp(x1*x1). A quotient coefficient rule rewrites this to (-4.201099129285349)/exp(x1*x1), replacing two non-integer coefficient params with one. All other shown transforms are already reachable by constant folding, sign normalization, and affine constant collection, so this closes the rank_gap 1.
- domain: For this row y = exp(x1*x1), so y is strictly positive and the denominator side condition is safe. The log term log(x1*x1 + 0.9893514112972329) and the x0 exponential argument are unchanged. Generic rule should require y != 0 or be limited to known-nonzero denominators.
- remaining gap: No many-parameter artifact is needed here. The only rank-relevant gap is exposing and collecting duplicate quotient coefficients; exp quotient normalization alone would not reduce the two reciprocal coefficients to one.

### pagie/Bingo/raw 5/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `log(abs(t * t)) => 2 * log(abs(t))`
- proof: Current rules factor the duplicated affine-like inner term enough to render log(abs(t * t)), but stop before using the square inside abs/log. Let t = x0*x0*(520439.72713022487*(28890.10432966032+x0)^(-1.0)+-165.43872587343168*(x1*x1))-x1. Since abs(t*t)=abs(t)^2 and log(abs(t)^2)=2*log(abs(t)) where t != 0, the expression becomes 0.2033733319566922 + (2*0.0885895005060784)*log(abs(t)). That removes one duplicated structural occurrence and folds the outside coefficient, matching the observed 3-rank gap source more directly than broader factoring rules.
- domain: Valid on the original expression domain where t != 0 and log(abs(t*t)) is defined. The rewrite preserves real-valued semantics there; at t=0 both sides are undefined or non-finite, so no finite-domain behavior is lost.
- remaining gap: After the rewrite, coefficient collection may expose 0.1771790010121568 as a single param coefficient. Remaining rank differences, if any, would likely come from current rank accounting for the reciprocal power -1.0 or the unfactored rational-like x0*x0/(28890.10432966032+x0), not from the duplicated log argument.

### pagie/EPLEX/raw 55/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `For p<0 and positive integer literals m,n: p / log(abs(exp(a - (exp(b - u)**2 * v))**m)**n) -> -1 / (((m*n)/(-p))*a - exp(2*b + log((m*n)/(-p)) - 2*u) * v). Here m=3,n=2.`
- proof: Current-rule obstruction: log(exp(x))->x cannot fire because exp(a-Q) is wrapped by **3, abs, **2, and the quotient numerator p is not exposed as a scalable denominator coefficient. Algebra gives log(abs(exp(a-Q)^3)^2)=6*(a-Q), with Q=exp(b-u)^2*v=exp(2*b-2*u)*v. Since p=-2.0220093000996027<0, p/(6*(a-Q)) = -1/(((6)/(-p))*a - exp(2*b + log(6/(-p)) - 2*u)*v). This replaces params p,a,b by two non-integer combinations; c=-0.896, e=-0.09248251672753682, f=-0.5816058058270378 remain, giving rank 5.
- domain: Requires p nonzero; the displayed sign form uses p<0. m,n are integer-valued positive literals and should not count as params. exp(...) is strictly positive, so the outer abs does not change value. Inner log(abs(x0**2)) and log(abs(x1**2)) keep the original exclusions at x0=0 or x1=0.
- remaining gap: A standalone log_abs_scale_extraction rule would only expose 6*(a-Q) and likely leave 6 params. The rank reduction needs quotient scale normalization plus absorption of the positive scale into exp(b-u)^2; implement either as this specialized rule or as coordinated quotient exposure and exp-additive extraction.

### pagie/EPLEX/raw 46/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `square_quotient_normalization`
- proposed rule: `rewrite((z * (t ** 4)) / (w * (v + t ** 2) ** 2)).to(z / (w * (1 + v / (t ** 2)) ** 2), t != 0)`
- proof: Current rules collect affine constants, giving the rendered baseline, but they do not divide a numerator/denominator by a repeated symbolic square. In the second exp let t=(0.108837568328275*x0**6+1), v=0.00954923344395646*x0**12, z=0.10536832436291177*(x0**2+-6.850597671733092), w=x0**2. Then z*t**4/(w*(v+t**2)**2)=z/(w*(1+v/t**2)**2). This removes the second occurrence of the c=0.108837568328275 literal, leaving 6 non-integer float params overall, matching n_rank.
- domain: Rule needs t != 0 and usual denominator defined. For this row t=1+c*x0**6 with c>0, so t>0 over reals; x0=0 was already singular from /x0**2. Integer-valued exponents 2,4,6,12 should not count as params.
- remaining gap: Implementation may need the rule to match through surrounding product/quotient association. If the metric shares identical literals globally, the baseline already has 6 unique non-integer floats and the reported gap is only occurrence/rank accounting.

### pagie/EPLEX/raw 44/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No algebraic rewrite is the smallest fix. Parameterization should intern identical non-integer float literals, or CSE the repeated term U = p*log(abs(x0/x1))**4 + 1 so both U occurrences share the same p.`
- proof: Current-rule obstruction: the rank miss is not from an unnormalized quotient/log/square shape, but from counting the repeated coefficient 0.875656742556918 twice. Let L=log(abs(x0/x1)) and U=p*L**4+1. The expression contains U**4 in the numerator and (U**2 - 0.851410399769728)**2 in the denominator, with the same p. Unique non-integer parameters are -0.786769, 1.26138411854031, 0.39714058776807, 0.168375426124201, p, 0.851410399769728, and 2.463: seven total. Baseline_after_params=8 is explained by unshared duplicate p occurrences.
- domain: Integer-valued literals 1.0, 2.0, and 4.0 should not be counted as free parameters here. The repeated p occurs under even powers and inside log(abs(...)), but sharing p does not require domain-changing algebra.
- remaining gap: If the pipeline already interns exact equal float literals before ranking, then this row is inconclusive without inspecting that implementation. From the rendered baseline alone, the apparent one-rank gap is a shared-parameter/rank artifact, not a new rank-reducing rewrite.

### kotanchek/EPLEX/raw 38/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `exp((a - b * exp(c)) * exp(-c)) * exp(-c) => exp(a * exp(-c) - b - c)`
- proof: Current-rule obstruction: rules normalized the inner negative product but did not combine the outer product of exponentials across the shared exp(c) and exp(-c) scale. Let c = x0 + exp(x1*(1.39933902481093 - x1)), a = 3.45907080661025*x0, and b = x1 + 0.219847859107203*exp(x0 - 3.05276088841039*exp((-1.21652690533432 - x0*exp(x1))*exp(-x1))). The remaining hard factor is exp((a - b*exp(c))*exp(-c))*exp(-c). Algebra gives (a - b*exp(c))*exp(-c) - c = a*exp(-c) - b - c, removing the duplicate c-shaped exponent site and matching the one-rank gap.
- domain: Valid over real expressions because exp(c)>0 and exp(u)*exp(v)=exp(u+v) is unconditional. No log-domain side condition is introduced. Non-integer floats counted as params; integer-valued -1.0 is structural.
- remaining gap: This is not solved by more affine collection alone: the blocker is multiplicative exponential cancellation under a product of exp terms. If the rank metric does not share the c subterm after rewriting, this may need canonical DAG/common-subexpression normalization, but algebraically the smallest sufficient rewrite is the exp product normalization above.

### kotanchek/EPLEX/raw 40/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `For real exp terms, rewrite exp(x*(u + c)) -> exp(x*u)*exp(c*x) when c is an integer-valued literal or otherwise not a fitted parameter. Equivalently allow exp(x*(u+c))/exp(x*u) -> exp(c*x).`
- proof: Current obstruction: the shared exponent a*x0*x1*exp(-x0) is present once as exp(a*x0*x1*exp(-x0)) but hidden twice inside exp(x0*(a*x1*exp(-x0)+k)). With a=4.85453828157037, extraction gives exp(x0*(u-5))=E*exp(-5*x0) and exp(x0*(u+1))=E*exp(x0), where E=exp(a*x0*x1*exp(-x0)). Then the numerator term exp(5*x0)*exp(-5*x0) collapses, and denominator terms expose the same E structure, removing the two rank-miss exponent offsets.
- domain: Valid for real exponentials; no positivity side condition beyond preserving the original denominator nonzero domain. Treat a and non-integer coefficients as params. Integer-valued literals 6.0, 5.0, -5.0, 2.0, 1.0, -1.0 should not add fitted-param rank.
- remaining gap: Assumes existing rules already multiply/cancel exp factors and simplify exp(5*x0)*exp(-5*x0). If those are absent, the same family should be added with product normalization, but the first missing obstruction is additive constant extraction.

### kotanchek/Bingo/raw 13/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `For (a0 + a1*x0 + a2*x1)/(b0 + b1*(x0+c)^2), divide numerator and denominator by b1: ((a0/b1) + (a1/b1)*x0 + (a2/b1)*x1)/((b0/b1) + (x0+c)^2). Then optionally absorb one remaining numerator scale by factoring a2/b1 from the numerator.`
- proof: Current-rule obstruction: existing affine/square collection exposes the shifted square, but leaves an arbitrary denominator scale b1 and three raw numerator coefficients, so the same rational form is still over-parametrized. Since the whole expression is a quotient, multiplying numerator and denominator by 1/b1 is semantics-preserving where b1 != 0 and removes one free denominator coefficient. A second scale redundancy remains in the numerator linear form: k*(u0+u1*x0+x1)/(v+(x0+c)^2) uses k,u0,u1,v,c, matching n_rank=5. This accounts for the rank gap 7->5.
- domain: Valid on denominator-scale parameter b1 != 0. The expression already has exponent -1, so this is quotient normalization rather than an exp/log rule. Non-integer floats are parameters; literal -1 exponent is integer-valued and should not count.
- remaining gap: If current rules already divide denominator scale but not numerator linear scale, the minimal missing piece is the second normalization only. Need exact pipeline rule inventory to distinguish one combined rule from two sequential existing/new rewrites.

### pagie/PySR/raw 203/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `exp(A + c) / exp(d) -> exp(A) * exp(c - d), treating exp(c - d) as one learned coefficient; also allow exp(A + c) / k -> exp(A) * exp(c) / k when k is a numeric constant.`
- proof: Current-rule obstruction: the denominator exp(0.07429085443356889 * 0.020641932342605206) is folded to 1.0015346832137049, but the matching additive constant 0.07429085443356889 inside exp(x0 * -0.7734843695524414 * x0 + 0.07429085443356889) is not pulled out, so the same literal is counted again and the quotient keeps an extra coefficient site. Applying exp(A+c)/exp(d)->exp(A)*exp(c-d) gives A=-0.7734843695524414*x0*x0, c=0.07429085443356889, d=0.001533499..., replacing two non-integer constants c and d/k by one coefficient exp(c-d). This accounts for the rank gap of 2 when combined with current constant folding.
- domain: Valid over real exp for all real A,c,d. Integer-valued structural literals are not counted as params; all listed decimal coefficients are params. No log-domain side condition is needed for this rule.
- remaining gap: This does not simplify the x1 term. The log(x1*(x1*0.5177))*-1.874 form may have separate abs/domain issues, but it is not needed to explain this row's two-rank miss.

### kotanchek/PySR/raw 205/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `exp((a - x) * (x - b)) -> exp(((a - b) * (a - b)) / 4) * exp(-((x - ((a + b) / 2)) * (x - ((a + b) / 2))))`
- proof: Current rules collect affine constants from the two polynomial factors, giving the rendered baseline, but they leave exp((1.1883794736981237 - x0) * (x0 - 0.8765025611709572)) with two root parameters. Algebra: (a-x)(x-b)=-(x-(a+b)/2)^2+(a-b)^2/4. The proposed rule extracts exp((a-b)^2/4) as a multiplicative constant and leaves one exponent parameter, mu=(a+b)/2. Here that constant folds into the existing outer scale 0.01220774123327107, so params become scale, 4.4045, 6.3674, 0.2433, 0.2978, 2.5460, mu: 7 total.
- domain: Valid for real x,a,b; exp has no positivity restriction. Integer literals 2 and 4 are structural, not params. Non-integer derived values exp((a-b)^2/4) and (a+b)/2 count as params, but the extracted factor merges with the existing outer coefficient.
- remaining gap: Need commuted/orientation variants such as exp((x-b)*(a-x)). Rank reduction relies on an existing multiplicative scale so the extracted exp constant replaces that scale rather than adding a new independent parameter.

### kotanchek/PySR/raw 184/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `exp(a) * exp(b) -> exp(a + b), with follow-on normalization of same-denominator sums: u/c + v/c -> (u + v)/c.`
- proof: Current-rule obstruction is that the expression keeps two multiplicative exp factors with the same outer denominator 2.725807745015068, so their affine/quadratic numerators are not exposed as one exponent. Let c=2.725807745015068. The two exponent arguments are (-1.5650412257786985 - x1 - x1)/c and ((x0 / -0.6595888099113719 * x0 - x1 / 0.8882302129082629))/c. Combining exp factors gives exp(((-1.5650412257786985 - 2*x1) + (x0 / -0.6595888099113719 * x0 - x1 / 0.8882302129082629))/c), eliminating one exp/product layer and exposing a single exponent, plausibly accounting for the rank gap without adding new non-integer params.
- domain: Valid over real-valued arguments because exp(a)*exp(b)=exp(a+b) for all real a,b. The denominator c is a nonzero learned float parameter. Integer literal 2 from x1+x1 is structural, not a new parameter.
- remaining gap: This rule addresses the visible rank miss from split exponentials. Further rank reduction may need existing affine duplicate collection inside the combined numerator, but the smallest missing family is exp product normalization.

### kotanchek/PySR/raw 201/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `Within an affine product, normalize additive float literals in the argument: k * (E + c1 - c2) => k * (E + (c1 - c2)), preserving integer-valued structural coefficients such as x1 + x1.`
- proof: Current-rule obstruction: the rendered baseline has already exposed quotient/product coefficients, giving -0.12734936583369594 * (exp(x1) + x0 + x0 - exp(x0) + 0.028361584423514188 - x0 - 2.4083162757635677), but it does not collect the two non-integer additive constants inside that affine argument. Algebraically those are one parameter: 0.028361584423514188 - 2.4083162757635677 = -2.3799546913400535. Rewriting to -0.12734936583369594 * (exp(x1) + x0 + x0 - exp(x0) - x0 - 2.3799546913400535) removes one float parameter, reducing baseline_after_params 6 to n_rank 5.
- domain: Pure real algebra over addition, multiplication, division by nonzero learned constants, and exp. No log/abs domain side conditions. Integer-valued duplicate coefficients from x1 + x1 or x0 + x0 - x0 should not be counted as learned params.
- remaining gap: This row does not require exp normalization or square quotient rules. It may still leave x0 + x0 - x0 uncollected structurally, but that affects syntax more than parameter rank under the stated counting rule.

### kotanchek/PySR/raw 188/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No local algebraic rule. If the rank metric is meant to share equal learned constants, model this as let p = 2.5453490974043493 and reuse p in both exponentials; otherwise keep baseline rank 8.`
- proof: Current-rule obstruction: existing rules have already normalized the only clear reducible coefficient pattern, rewriting exp(0.8325766331394892/0.6756390603608541 - x1*2.5453490974043493*x1*x1) to exp(2.5453490974043493*(0.4841301322429487 - x1*x1*x1)). The remaining rank gap is that 2.5453490974043493 appears once in the outer exponent and once in the inner exponent, in additive/multiplicative contexts that cannot be merged by local algebra. Collecting x1 - p*x1 to (1-p)*x1 or expanding p*(k-x1^3) preserves one non-integer coefficient occurrence per site, so it does not reduce the tree to 7 params.
- domain: All displayed non-integer floats are parameters by occurrence; integer literals like powers and multiplication by x1 are structural. The double negative around -0.40562409423561496 is still one additive constant. No log/abs or positivity-domain rule is relevant.
- remaining gap: The row looks like a parameter-sharing/rank-accounting artifact rather than a missing algebraic simplification. A true rank-7 form needs global sharing of the repeated 2.5453490974043493 or a rank metric that counts equal literals once; current tree rewrites cannot expose that as one local parameter.

### pagie/Bingo/raw 20/original

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `No rank-reducing rewrite proposed. Small candidates like log(abs(c*u))->log(abs(u))+log(abs(c)) are valid identities but do not remove a parameter here because the added constant is multiplied by a nonconstant affine factor.`
- proof: Obstruction: current collection exposes the log argument as k*x0*x1*((x0*x1)^3+e), but k and e are not algebraically disposable under the surrounding multiplier A=b+c/x0-x1/x0. Extracting k gives A*(log|x0*x1*((x0*x1)^3+e)|+log|k|); dropping log|k| would lose the nonzero term log|k|*A. The e parameter also changes log((x0*x1)^3+e), not just a scale. Together with the outer coefficient and two affine constants, this remains five generic non-integer parameters, not rank 3.
- domain: Requires x0 != 0 from x0**-1. Log terms require nonzero arguments; abs allows negative constant scales, and scale extraction would use log(abs(k)) for k != 0. Integer-valued literals such as -1.0 are not counted as params.
- remaining gap: The reported rank 3 is most plausibly from finite-dataset collinearity, shared parameters outside this rendered form, or a rank-estimation artifact. Confirming which would require the actual pagie row data/Jacobian or the rank artifact generation code.

### kotanchek/PySR/raw 203/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `Collect adjacent affine constants across multiplication and addition: (a*(x+c)+d) -> a*(x+(c+d/a)) when a,d are nonzero params and a is safe to divide by. Here 3.3628776435387486*(x0-0.14626012317910758)-0.050504132883325455 becomes 3.3628776435387486*(x0-0.1612782529586318).`
- proof: Current-rule obstruction: the raw numerator exposes two independent float literals in an affine x0 term, a*(x0+c)+d, while the baseline already folds them into one shifted affine term. Algebra: c+d/a = -0.14626012317910758 + (-0.050504132883325455)/3.3628776435387486 = -0.1612782529586318 approximately. Replacing the pair c,d with one derived constant removes one counted non-integer float without changing variables, exp terms, denominator, or the outer scale/offset. That exactly explains the 1-rank gap from baseline_after_params 8 to n_rank 7.
- domain: Requires a != 0 and ordinary real arithmetic. This is an approximate float-param identity, so matching should use the same numeric tolerance/canonicalization policy as existing param_eq constant folds. Integer literals are structural here; counted params are non-integer floats.
- remaining gap: No evidence this row needs denominator normalization, exp/log rewrites, quotient coefficient exposure, or shared-parameter/rank-artifact handling. The repeated 3.3628776435387486 remains present in numerator and denominator in both forms, so the observed miss is localized to affine constant collection.

### kotanchek/PySR/raw 206/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_duplicate_collection`
- proposed rule: `Canonicalize signed non-integer literals before param counting: replace literal -c with unary negation of literal c, or otherwise treat c and -c as one shared parameter with sign outside the parameter.`
- proof: Current-rule obstruction: the rendered baseline has already collected the local constants: 0.5590230865341662 - -0.42694354206571955 -> 0.9859666285998858, and x1 - 0.21002997412236857 + -0.048636102379870004*(exp(x1)-1.2305420553970035) -> x1 -0.048636102379870004*(3.0878544411384157+exp(x1)). The remaining rank gap is the same magnitude 2.6270807618049434 appearing once as an additive -2.6270807618049434 in the exp exponent and once as +2.6270807618049434 multiplying x1*(x1+x0). If signed literals are counted separately, baseline has 7 params; sign-normalized sharing gives 6.
- domain: This is exact algebra over real-valued expressions. Integer literals such as implicit 1 and -1 are not the relevant params. No tolerance snapping is needed because the repeated magnitude is textually identical.
- remaining gap: After sign-normalized duplicate collection, the visible non-integer parameter magnitudes are 2.6270807618049434, 0.7307483550809931, 0.5695620761153319, 0.048636102379870004, 3.0878544411384157, and 0.9859666285998858, matching n_rank=6.

### kotanchek/PySR/raw 185/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `((u * exp(a) + v) * exp(-a)) -> (u + v * exp(-a)); commuted/additive variants included. Here u=x1+(-0.2946268534853358), v=1.0, a=1.598800435582104*x1*(x1+0.3553849416107877).`
- proof: Current-rule obstruction: the cancelling exp(a)*exp(-a) factors are separated by an additive wrapper, so product/quotient exp normalization cannot see them without distributing the outer exp(-a) across u*exp(a)+v. Algebra: (u*exp(a)+v)*exp(-a)=u*exp(a)*exp(-a)+v*exp(-a)=u+v*exp(-a). Applying this yields (0.7265240564533295*x1+0.48872997975581867)*(x1-0.2946268534853358+exp(-a))/denom, reducing the artifact by the missed two ranks.
- domain: Valid over real-valued expressions for all finite a because exp(a) is nonzero. No denominator-domain change is introduced; the original denominator factors remain unchanged. Treat non-integer floats as params; integer-valued 1.0, -1.0, -3.0, 2.0 are structural literals.
- remaining gap: This is the smallest sufficient rule for this row. It should be oriented with a cost guard so it fires only when it exposes exp(a)*exp(-a) cancellation or reduces rank; unrestricted distribution can increase search space.

### kotanchek/EPLEX/raw 35/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `rewrite(exp(log(abs(exp(b + z) - y * exp(c + z))) + a)).to(abs(exp((a + b) + z) - y * exp((a + c) + z))) for constant a,b,c; also reached after existing x - -a => x + a.`
- proof: Current obstruction: rules have log(exp(x))->x and constant folding, but no exp(log(abs(u))+a) inverse/extraction, so the rendered +0.9732422709999999 remains an independent param. In this row u=exp(0.5117085777865424-x0)-x1*exp(-0.373-x0). Since exp(a)>0, exp(log(abs(u))+a)=abs(exp(a)*u). Pushing that positive scale into the two exponential terms gives abs(exp(1.4849508487865423-x0)-x1*exp(0.6002422709999999-x0)), replacing params {0.5117085777865424,-0.373,0.9732422709999999} with two combined params. Total params drop 8 to 7.
- domain: Valid over reals where the LHS is defined: u != 0 because log(abs(u)) requires positive argument. exp(a) is always positive, so abs(exp(a)*u)=exp(a)*abs(u). Integer-valued exponents/sign literals are not counted as params.
- remaining gap: This closes the one-param rank gap for the row. If implemented only as a generic exp(log(abs(u))+a)->abs(exp(a)*u), current rules may still need distribution/exp-product normalization; the targeted shared-tail exponential form is the smallest direct rank-reducing rule here.

### kotanchek/EPLEX/raw 44/sympy

- conclusion: `needs_many_params_or_rank_artifact`
- rule family: `shared_parameter_or_rank_artifact`
- proposed rule: `Canonicalize finite integer-valued Float exponents before param counting, e.g. Pow(u, 3.0) -> Pow(u, 3); do not add a rank-reducing algebraic rewrite for this row.`
- proof: Obstruction: the current rules appear to have already done the relevant constant/affine work. log(abs(-0.146))**3.0 - 0.083 is rendered as -7.206868039343769, log(abs(1.303369)) + 1.116 as 1.3809524506836304, and the outer x0 terms are collected as 1.3809524506836304 + x0*(-0.071 + 0.057*x1/L). Counting only non-integer numeric literals gives exactly five params: 1.3809524506836304, -0.071, 0.057, -7.206868039343769, -1.194. The apparent sixth item is the structural exponent 3.0, so the rank miss reduces to rank accounting/canonicalization, not a missing algebraic rule.
- domain: Safe only for finite Float values exactly equal to an integer and used as exponents. Here 3.0 is an odd integer exponent, so replacing it with 3 preserves real-valued behavior even when the base is negative.
- remaining gap: A common-denominator rewrite could combine x1**3/d**3 + exp(x0**3)/d**3 into (x1**3 + exp(x0**3))/d**3, but that does not remove a non-integer parameter and is not the smallest explanation for the one-rank gap.

### kotanchek/PySR/raw 180/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `quotient_coefficient_exposure`
- proposed rule: `For D != 0, add rewrite((r + a * D) / D).to(a + r / D) plus the commuted numerator form ((a * D + r) / D).to(a + r / D). Existing affine collection should expose a * D first.`
- proof: Obstruction: current rules can build common denominators and normalize quotient coefficients, but do not split a numerator addend that is a scalar multiple of the whole denominator. Let D=exp(x0 ** 2.0)+exp(x1)+3.5047393470316366 and e=0.011424247853121624. The numerator has e*exp(x0 ** 2.0)+e*exp(x1)+0.04003901096107706 plus R=x1*(...). Since 0.04003901096107706 = e*3.5047393470316366 within tolerance, existing constant collection can expose e*D. The proposed rule rewrites (R+e*D)/D to e+R/D, eliminating the dependent constant f and reducing 7 params to rank 6.
- domain: The row denominator is positive because both exp terms and 3.5047393470316366 are positive, so D != 0 holds. The general rule should require D != 0 and preserves the original quotient domain. The log domain x0 + 0.7116175810554622 > 0 is unchanged. The exponent literal 2.0 is integer-valued and should not count as a parameter.
- remaining gap: This assumes the e*D addend is exposed by existing affine collection and AC matching; otherwise a supporting collection/pass-budget issue remains. No many-parameter or rank artifact is needed for this row: the single missing quotient split accounts for the one-parameter gap.

### pagie/EPLEX/raw 37/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `tolerance_coefficient_snapping`
- proposed rule: `For additive literals with abs(a) <= eps, rewrite x + Num(a) -> x and Num(a) + x -> x; eps must be at least 9e-06 for this row.`
- proof: Current-rule obstruction: rules remove exact 0 via 0 + x -> x and fold exact constants, but they do not snap near-zero literals. The baseline has exactly 9 non-integer float params, with the final top-level + 9e-06 contributing one. Rewriting that additive near-zero constant to 0 lets existing commutativity/identity remove it, leaving the other 8 non-integer params unchanged: 3.017599791376, 0.13360110193664, 0.703249523422628, 0.137, 0.758725341426404, 0.014641, -0.557375869540689, -0.335544732704243.
- domain: This is approximate, not exact algebra. It is appropriate only if param_eq accepts tolerance semantics for fitted constants. Integer-valued 1.0 and 2.0 literals are not counted as params here.
- remaining gap: No further rank gap remains after snapping 9e-06. Log/abs scale extraction and exp(log(.)) rules could reduce nodes or expose alternate forms, but they are not needed to reach rank 8 for this row.

### kotanchek/EPLEX/raw 35/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `log_abs_scale_extraction`
- proposed rule: `For real v and numeric k, rewrite exp(log(abs(k*u*exp(v)))) -> abs(k)*abs(u)*exp(v); analogously log(abs(u*exp(v))) -> log(abs(u))+v. Run before param counting, then existing exp-product and coefficient collection can combine exposed factors.`
- proof: Obstruction: current rules affine-factor 1.66813890656534 - 0.688665232843956*x1 into 0.688665232843956*(2.422278382889299 - x1), but the factor is still hidden under exp(log(abs(...*exp(-x0)))), so the outer exp(x0) cancellation and coefficient collection cannot fire. The rule rewrites that block times exp(x0) to 0.688665232843956*abs(2.422278382889299 - x1), letting 2.64651127063146 and 0.688665232843956 collapse to one coefficient. The first log also loses a hidden exp scale without adding noninteger params. Remaining params: K,b,A*e,r,0.452,c,C = 7.
- domain: Assumes real x0,x1. exp(v)>0, so abs(u*exp(v))=abs(u)*exp(v); the log form is equivalent wherever u != 0 and preserves the same zero singularities. Positive numeric k may leave abs. Integer-valued literals -2,-1,3 are not counted as params.
- remaining gap: If current rules do not already collect exposed numeric products or cancel adjacent exp factors after this exposure, add those as follow-up plumbing; algebraically this row does not need a shared-parameter or many-parameter rank-artifact explanation.

### kotanchek/EPLEX/raw 33/original

- conclusion: `new_rule_partial_reduction`
- rule family: `log_abs_scale_extraction`
- proposed rule: `Add guarded symmetric rules: log(abs(a * y)) -> log(a) + log(abs(y)) and log(abs(y * a)) -> log(abs(y)) + log(a), with a=Num(af), af>0.0, y!=Num(0.0).`
- proof: Obstruction first: current fun_rules extract scale only for log(a*y), log(y*a), log(a/y), log(y/a); after existing affine factoring this row has log(abs(0.268*(x0+c0))) and log(abs(0.384*(x1+c1))), so abs blocks scale extraction. The proposed rule gives log(0.268)+log(abs(x0+c0)) and log(0.384)+log(abs(x1+c1)); analysis then folds log constants into -1.412 and -0.008. That removes the two exposed scale literals, reducing baseline 10 params to about 8, not all the way to n_rank 7.
- domain: Valid for positive scale a and y nonzero: log(abs(a*y)) = log(a) + log(abs(y)). This matches existing positive-scale log rules. Negative scales could be handled separately with log(abs(a)), but positive-only is the smallest rule needed here because the exposed scales are 0.268 and 0.384.
- remaining gap: One parameter remains beyond n_rank after this local rule. The remaining constants are not obviously eliminated by current affine/quotient rules; reaching rank 7 likely needs a separate coefficient-sharing/lattice rule or the n_rank value reflects search/rank accounting rather than literal-count algebra.

### kotanchek/EPLEX/raw 33/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `affine_constant_collection`
- proposed rule: `For a*(b*x + c) inside abs/log, rewrite to (a*b)*(x + c/b) when b is a nonzero integer-valued literal and c is param-like: log(abs(k*x + p)) -> log(abs(k*(x + p/k))).`
- proof: Current obstruction is affine shift exposure under log(abs(.)): the rendered baseline has log(abs(0.384*(x1 + -7.0717871423147916))) and log(abs(0.268*(x0 + -0.7770646701367612))), while the input keeps 0.384*x1 - 2.71556626264888 and 0.268*x0 - 0.208253331596652. The remaining outside constants already factor cleanly: 0.4*(...)+0.096 = 0.4*((...)*x1+0.24), and 0.073497378816*(q)^2+0.002 = 0.073497378816*((q)^2+0.02721185479290331). Pulling 0.4*0.073497378816 gives 0.029398951526400003, matching baseline.
- domain: Valid for real x because abs(k*(x+s)) = abs(k)*abs(x+s), but log(abs(k*(x+s))) differs from log(abs(x+s)) by log(abs(k)); this rule keeps k inside abs, so no additive log constant is introduced. Requires k nonzero.
- remaining gap: This accounts for the two-rank miss if current rules can already collect outer product constants and affine additive constants. If current rules lack those too, the same row also needs existing affine constant collection outside log, but the smallest rank-facing obstruction is the affine form under log(abs).

### pagie/PySR/raw 200/original

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_additive_constant_extraction`
- proposed rule: `rewrite(a / exp(x + b)).to((a / exp(b)) / exp(x)) for constant a,b; optionally also b+x orientation if commutativity does not reliably expose x+b.`
- proof: Current obstruction: rules fold constants and collect products/quotients, but exp arguments are opaque except exp(Num), log(exp), and log rules. The baseline already reduced x1*1.0127629362668584*(x1*0.7545716786341436) to 0.7642022288773276*(x1*x1) and the x0 quotient chain to x0*0.15728893873963976. The remaining gap is -0.5802718580838973/exp(0.7642022288773276*(x1*x1)+-0.6072178507085199): extract the additive exp constant to combine -0.5802718580838973 and -0.6072178507085199 into one coefficient, reducing 6 params to rank 5.
- domain: For real inputs, exp(b)>0, so the rewrite preserves denominator safety and semantics. b=-0.6072178507085199 is a non-integer float param; integer-valued literals such as the implicit -1 in x0*(0.15728893873963976*x0-x0) should not count as params.
- remaining gap: This is the smallest sufficient rank-reducing rule for this row. Existing rules handle the multiplicative coefficient collection already visible in the baseline. No shared-parameter or many-parameter artifact is needed for the one-param gap here.

### kotanchek/GP-GOMEA/raw 98/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `exp(u) * exp(v) -> exp(u + v), followed by exp(0) -> 1 and additive inverse cancellation u + -u -> 0; equivalently exp(u) * exp(-u) -> 1 as the smallest direct rule.`
- proof: Current-rule obstruction: the baseline keeps -0.047292 * exp(0.555*x0*(x0-1.761)) inside a sum that is multiplied by exp(-0.555*x0*(x0-1.761)); existing rules expose the shared 0.010811 factor in x1 terms but do not normalize products of opposite exponentials across multiplication. Distributing the outer exp(-A) over the sum creates -0.047292*exp(A)*exp(-A), where A=0.555*x0*(x0-1.761), which reduces to -0.047292. This removes the non-integer coefficient 0.047292 as a param-like term and collapses one exponential product, matching the rank gap direction.
- domain: Valid over real-valued A because exp(A) is nonzero and exp(A)*exp(-A)=1. Integer-valued literals are structural; non-integer coefficients 0.555, 1.761, 0.010811, 0.047292, 0.156986531 count as params.
- remaining gap: This explains the clearest rank-miss obstruction for the A exponent pair. Further rank recovery may still require distribution or a rule that recognizes exp(u)/exp(u) or exp(u)*exp(-u) before or during affine/multiplicative normalization.

### kotanchek/EPLEX/raw 58/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `exp_product_or_quotient_normalization`
- proposed rule: `With an enclosing free multiplicative coefficient k, rewrite k*exp((a - x)*(x - b))*G -> k2*exp(x*(a + b - x))*G, where k2 = k*exp(-a*b) and G is independent of this normalization.`
- proof: Current-rule obstruction: baseline leaves exp((0.996 - x0)*(x0 + -0.896393)), so both root literals 0.996 and 0.896393 are counted. Algebraically, (a - x)*(x - b) = x*(a + b - x) - a*b. Thus exp((a - x)*(x - b)) = exp(-a*b)*exp(x*(a + b - x)). The positive constant exp(-a*b) can be absorbed into the existing outer scale -0.761854261089838. This replaces two non-integer root params by one sum param a+b=1.892393, dropping baseline params from 9 to rank 8.
- domain: Exact over real-valued x0 with ordinary exp. Requires a and b to be variable-independent and an enclosing free multiplicative scale to absorb exp(-a*b). Integer-valued structural coefficients like -1 are not counted as params.
- remaining gap: This does not reduce the inner exp parameters. If applied where no free outer scale exists, the extracted exp(-a*b) would become another coefficient and may not reduce rank. Need rule ordering before final param counting.

### kotanchek/SBP/raw 163/sympy

- conclusion: `new_rule_reduces_to_rank`
- rule family: `tolerance_coefficient_snapping`
- proposed rule: `After coefficient exposure/collection, rewrite real literals within tolerance of an integer to the integer literal, e.g. 2.0 -> 2, preserving exact integer semantics for downstream param counting.`
- proof: Current-rule obstruction: the existing rules already factor 0.009239 and expose 0.018478/0.009239 as 2.0, producing baseline 0.009239*(exp(17.386*x0)*(22.5997402316268-(exp(x1)-exp(x0)*(x0-4.627))-(2.0*((x0-9.621)*x1)-x0))-1.0)*exp(-17.386*x0). The only apparent extra counted literal versus n_rank is integer-valued float 2.0. Snapping 2.0 to integer 2 leaves non-integer params {0.009239,17.386,22.5997402316268,4.627,9.621}, matching rank 5.
- domain: Safe when the literal is exactly integer-valued or within a strict numeric tolerance used elsewhere for coefficient recovery. Do not snap non-integer constants such as 4.627, 9.621, 17.386, 0.009239, or 22.5997402316268.
- remaining gap: No algebraic gap remains for this row if integer-valued floats are excluded from params. If the rank pipeline intentionally counts 2.0 as a parameter, then this is a rank/counting convention artifact rather than a rewrite miss.
