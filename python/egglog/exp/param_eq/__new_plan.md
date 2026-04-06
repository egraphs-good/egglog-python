pre:

use a local folder path for experimental and egglog
so we can modify them easier.

use `uv sync --reinstall-package egglog --all-extras` to update based on egglog experimental

rename egg_like to fresh_rematch or something

try removing haskell backoff and compensating by decreasing backoff size?

use sum of all_function_sizes instead of serialized sum of counts for performance?

perm scheduler add to egglog experimental, return new ExperminetalEgraph struct so we can store extra data like perminent schedulers. Make all of them perm by default.
expose in Python by default, by adding a top level scope somehow,
like `.global()` or something...? that way we dont need a custom scheduler in egglog
python.

Remove `stop-when-no-updates` in experimental that we added... since dont use it anymore


replace `is_const(a, ca)` with `is_const(a)` where `ca`
   is generated as a new var like `_ca` inside the function since it isn't referenced elsewhere.

once all that is done if things still work, we are ready to package the repo up
to a stable place. So remove all extra code for checking different versions, review
the code as a whole and make any simplifications/remove dead code. We are done trying to fix the schedules and rules to match baseline, we now want a clean part to build off of for multisets, with the replication notebook just showing that we hit
the baseline basically.

---

Overall goal: See how replacing binary ops with containers
can get us closer to saturation or at least decrease the number
of nodes and reduce need for backoff which is not complete.



We want to remove +, -, /, *, and replace with `polynomial(Map[Map[Expr, i64], i64])`. Input into the e-graph with that and after extract out in that form.


First, we want to make a plan by going through and leaving a comment next to each rewrite to see how it would change under the new plan. We can imagine we have any functional operations over
maps that we want at this time, and then once we settle on our design, we can go and actually add those.

Many rewrites will be simply deleted, like `a*b -> b*a`, and many others can be collapsed into one, like

```
    yield rewrite((x / a) + (y / b)).to(
        (x + (y / (b * a))) / a,
        *is_const(a, ca),
        *is_const(b, cb),
        *is_not_const(x),
        *is_not_const(y),
    )
```

is just a directed rewrite so that if we have a constant term in one of the monomials, we try factoring it out.

Stop after adding comments before each rule so I can manually inspect and see what makes sense before trying to implement it.
