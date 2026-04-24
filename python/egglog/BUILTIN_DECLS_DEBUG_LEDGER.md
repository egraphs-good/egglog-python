# Builtin Decls Debug Ledger

## Question

Why did `_convert_function()` need a manual builtin seed list instead of starting from `Declarations()`?

## Status

Resolved with one shared helper and two narrow call-site fixes.

## Smallest Repro

With the manual builtin seed removed from `python/egglog/builtins.py`:

```python
from egglog import *
from egglog.builtins import map_fold_kv

map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty())
```

Observed:

- raises `KeyError: Ident(name='f64', module='egglog.builtins')`

Nearby controls:

- `map_fold_kv(lambda acc, k, v: acc + v, i64(0), Map[i64, i64].empty())` passes
- `map_filter_defined_kv(lambda k, v: Maybe[f64].some(v), Map[i64, f64].empty())` passes

## Exact Probes

### Probe 1: broad witness

Command:

```bash
uv run python -m egglog.exp.param_eq --expr 'x*y + x*z + w' --variant container
```

Observed:

- fails during `_analysis_rules`
- first meaningful exception is the same `KeyError` on builtin `f64`

### Probe 2: direct type-resolution comparison

Command:

```bash
uv run python - <<'PY'
from egglog.builtins import f64, i64, String, Map, Maybe
from egglog.declarations import Declarations, Ident
from egglog.runtime import resolve_type_annotation_mutate

for label, tp in [('f64', f64), ('Map[i64,String]', Map[i64, String]), ('Maybe[f64]', Maybe[f64])]:
    decls = Declarations()
    resolve_type_annotation_mutate(decls, tp)
    print(label, sorted(str(k) for k in decls._classes))
PY
```

Observed:

- direct `f64` resolution includes builtin `f64`
- `Map[i64, String]` resolution includes `Map`, `MultiSet`, `i64`, etc.
- `Maybe[f64]` resolution includes `Maybe` and `f64`

Interpretation:

- plain builtin runtime classes are not globally broken

### Probe 3: instrument `_convert_function`

Observed inside the failing `map_fold_kv(... Map[i64, f64] ...)` callback conversion:

- `get_type_args()` is `(f64, f64, i64, f64)`
- after `resolve_type_annotation_mutate(decls, tp)` for all four args, `decls._classes` contains:
  - `Map`, `MultiSet`, `i64`, `Unit`, `String`, `Bool`, `Vec`, `UnstableFn`
  - but **not** `f64`
- the callback arg `RuntimeExpr`s therefore carry decls without `f64`
- the first `acc + v` lookup fails when `RuntimeExpr.__egg_class_decl__` asks for builtin `f64`

This is the first concrete failure mechanism.

### Probe 4: synthetic runtime-class repro

Command:

```bash
uv run python - <<'PY'
from egglog import *
from egglog.declarations import Declarations, Ident, TypeRefWithVars
from egglog.runtime import RuntimeClass, resolve_type_annotation_mutate
from egglog.thunk import Thunk

base = Declarations.create(Map, i64, String)
synthetic_f64 = RuntimeClass(Thunk.value(base), TypeRefWithVars(Ident.builtin('f64')))
probe = Declarations()
resolve_type_annotation_mutate(probe, synthetic_f64)
print(sorted(str(k) for k in probe._classes))
PY
```

Observed:

- the resolved declarations still do **not** contain builtin `f64`

Control:

- `resolve_type_annotation_mutate(Declarations(), f64)` does contain builtin `f64`

Interpretation:

- the problem is specifically with borrowed `RuntimeClass` wrappers whose decl thunk does not already contain their own builtin class

### Probe 5: confirming probe

Probe-only monkeypatch:

- keep `_convert_function` otherwise unchanged
- after resolving each callback type arg, also union the canonical builtin runtime class object for that type arg

Observed:

- the tiny repro `map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty())` flips to passing

### Probe 6: compare the two candidate fix sites on the original canaries

Candidate A:

- monkeypatch `resolve_type_annotation_mutate(...)` so borrowed builtin `RuntimeClass` values materialize their canonical builtin class

Candidate B:

- monkeypatch `with_type_args(...)` so builtin type args are represented by canonical builtin `RuntimeClass` values instead of borrowed outer-decls wrappers

Observed:

- both candidate patches flip the original smallest repro:
  - `map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty())`
- both candidate patches also flip the real `param_eq` canary:
  - `run_paper_pipeline_container("x*y + x*z + w")`

Interpretation:

- either location can repair the original regression
- the remaining question is which one is the narrower semantic fix

### Probe 7: broader parameterized-builtin canary

Command:

```bash
uv run python - <<'PY'
from egglog import *
from egglog.builtins import map_map_values

map_map_values(lambda k, v: v.unwrap(), Map[i64, Maybe[f64]].empty())
PY
```

Observed:

- baseline fails with:
  - `KeyError: Ident(name='Maybe', module='egglog.builtins')`
- neither of the quick probe patches above fixes this case

Full traceback localization:

- the failure happens in `TypeConstraintSolver.substitute_typevars_try_function(...)`
- specifically while probing the lambda with dummy `RuntimeExpr`s before `with_type_args(...)` is entered

Interpretation:

- this is a separate latent bug in the dummy-arg type-inference path
- it is **not** the mechanism that forced the manual builtin seed in `_convert_function`
- but it fails for the same deeper reason: the callback is operating on a type ref whose declarations were never materialized from the type itself

### Probe 8: scope comparison for the two candidate fix sites

Command:

```bash
rg -n "resolve_type_annotation_mutate\\(|with_type_args\\(" python/egglog
```

Observed:

- `with_type_args(...)` is used only in the conversion/type-args machinery
- `resolve_type_annotation_mutate(...)` is used broadly across `runtime.py`, `egraph.py`, and builtin conversion paths

Interpretation:

- if both sites repair the original canaries, `with_type_args(...)` is the narrower place to fix the original regression

### Probe 9: compare an older passing higher-order builtin against the new failing one

Commands run under the old pre-fix behavior:

```python
Vec(f64(1.0)).map(lambda x: x + 1.0)
map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty())
```

Observed inside `_convert_function()`:

- `Vec(...).map(...)`
  - raw type args: `(f64, f64)`
  - callback `decls._classes` already contained builtin `f64`
- `map_fold_kv(...)`
  - raw type args: `(f64, f64, i64, f64)`
  - callback `decls._classes` contained `Map`, `MultiSet`, `i64`, etc.
  - but did **not** contain builtin `f64`
  - the synthetic callback args therefore had type `f64` with declarations that did not contain `f64`
  - the first `acc + v` lookup failed immediately

Interpretation:

- the original bug was latent, but not universal
- older method-based higher-order builtins already carried the needed builtin types in the outer decl context
- the new top-level `Map` callback path created borrowed callback type args from an outer decl context that did not contain the callback builtin type

### Probe 10: compare an older passing dummy-arg inference case against the new failing one

Commands run under the old pre-fix behavior:

```python
MultiSet(Rational(1, 2)).map(lambda x: x.to_f64() + 1.0)
map_map_values(lambda k, v: v.unwrap(), Map[i64, Maybe[f64]].empty())
```

Observed inside `substitute_typevars_try_function(...)`:

- `MultiSet(...).map(...)`
  - `probe_decls` already contained `Rational` and `f64`
  - dummy inference succeeded
- `map_map_values(...)`
  - `probe_decls` contained `Map`, `MultiSet`, `i64`, etc.
  - but did **not** contain `Maybe`
  - the dummy arg with type `Maybe[f64]` failed before the lambda body could infer its return type

Interpretation:

- this second bug was also latent, but only surfaced once the new top-level `Map` builtin callbacks introduced dummy arg types whose class declarations were not already present in the outer decl context

### Probe 11: can a no-import technique rely only on `retrieve_conversion_decls()`?

Command:

```bash
uv run python - <<'PY'
from egglog import *
from egglog.conversion import retrieve_conversion_decls
from egglog.declarations import Ident

decls = retrieve_conversion_decls()
for name in ['f64', 'Maybe', 'Pair', 'Rational', 'Map', 'Vec', 'MultiSet']:
    print(name, Ident.builtin(name) in decls._classes)
PY
```

Observed:

- `f64`, `Maybe`, `Pair`, `Map`, `Vec`, `MultiSet` are present
- `Rational` is **not** present

Interpretation:

- a no-import fix cannot simply say "use `retrieve_conversion_decls()` as the canonical source of all runtime classes"
- the conversion declarations universe is incomplete for this purpose

### Probe 12: does a registry-based no-import technique work?

Probe-only monkeypatch:

- replace `runtime_class_from_type_ref(...)` with a registry lookup keyed by `Ident`
- reconstruct parameterized types from registered zero-arg runtime classes
- no `importlib` lookup

Observed:

- `map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty())` passes
- `map_map_values(lambda k, v: v.unwrap(), Map[i64, Maybe[f64]].empty())` passes

Interpretation:

- importing by module/name is not fundamentally required
- a runtime-class registry populated at class creation time should be sufficient

## Current Hypothesis

For the original regression only:

`with_type_args(...)` creates synthetic `RuntimeClass` objects for callback type arguments using the *outer* declaration thunk.

For `map_fold_kv(..., Map[i64, f64].empty())`, those borrowed runtime classes for `f64` inherit the outer `Map`-oriented declaration context, which contains `Map`, `MultiSet`, `i64`, etc., but not builtin `f64`.

That makes `_convert_function()` see a borrowed `RuntimeClass` for `f64` whose captured declarations are type-incomplete. The manual seed list masked this by preloading builtin classes into `_convert_function`.

## Working Root Cause

The shared missing mechanism is:

- "when a bare type ref is turned back into a runtime object for callback conversion or dummy-arg inference, the declarations required by that type are not materialized from the type itself"

That shows up in two different places:

1. `with_type_args(...)`
   - creates borrowed `RuntimeClass` values from the outer decl thunk
   - those borrowed builtin runtime classes can omit their own builtin declarations
2. `TypeConstraintSolver.substitute_typevars_try_function(...)`
   - creates dummy `RuntimeExpr`s from substituted `JustTypeRef`s
   - but never materializes the declarations for those dummy arg types first

## Accepted Fix

Shared helper:

- add `runtime_class_from_type_ref(...)`
- add `materialize_type_ref_mutate(...)`

Integration points:

1. `with_type_args(...)`
   - use `runtime_class_from_type_ref(...)` when possible instead of always creating `RuntimeClass(decls, a.to_var())`
2. `TypeConstraintSolver.substitute_typevars_try_function(...)`
   - materialize each substituted dummy arg type into `probe_decls` before constructing the dummy `RuntimeExpr`

Why this shape won:

- it fixes both target failures
- it keeps the behavior local to the two boundaries where bare type refs were being rehydrated without declarations
- it avoids broad lazy behavior in `RuntimeExpr`
- it avoids widening `resolve_type_annotation_mutate(...)`, which is a more central helper

## Narrowed Root Cause

What was broken before:

- bare type refs were being rehydrated in two places from an *ambient declarations context* instead of from a canonical runtime class source

Why older code still looked fine:

- older higher-order builtins like `Vec.map(...)` and `MultiSet.map(...)` happened to start from outer decl contexts that already contained the needed callback builtin types

Why the new changes triggered it:

- the newly added top-level `Map` callback builtins (`map_fold_kv`, `map_map_values`, etc.) created callbacks whose outer decl contexts contained `Map`-oriented declarations but not the callback builtin types like `f64`, `Maybe`, or `Rational`
- that exposed the preexisting assumption that ambient decls were "complete enough" to rehydrate callback type refs and dummy arg types

Why `importlib` was used in the accepted patch:

- it was the shortest way to recover a canonical runtime class object from a bare type ref once this latent bug was exposed
- but Probe 12 shows it is not the only viable mechanism

## Better Follow-Up

If we want to remove the import-based lookup, the best candidate is:

- a runtime-class registry keyed by `Ident`, populated when canonical zero-arg `RuntimeClass` objects are created
- `runtime_class_from_type_ref(...)` can then reconstruct parameterized types from that registry without `importlib`

This looks better than relying on `retrieve_conversion_decls()`, because Probe 11 showed that conversion decls are not a complete universe of runtime classes (for example, `Rational` is missing there).

## Observed Result

Target repros:

- `map_fold_kv(lambda acc, k, v: acc + v, f64(0.0), Map[i64, f64].empty())` -> passes
- `map_map_values(lambda k, v: v.unwrap(), Map[i64, Maybe[f64]].empty())` -> passes

Regression canaries:

- `python/tests/test_high_level.py -q` -> `182 passed, 2 xfailed`
- `python/tests/test_unstable_fn.py -q` -> `22 passed`
- `python -m egglog.exp.param_eq --expr 'x*y + x*z + w' --variant container` -> passes

## Next Probe

No active probe for this bug family.

If a future failure appears on non-importable user-defined runtime classes, test whether
`runtime_class_from_type_ref(...)` needs a second lookup path beyond module import.
