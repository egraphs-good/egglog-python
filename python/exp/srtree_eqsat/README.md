# Srtree-EqSat Replication Helpers

This folder contains the offline helper commands used to compare the Egglog reproduction against the original `srtree-eqsat` source implementation.

The intended flow is:

1. Run the Haskell helper from the source repo to gather reference numbers for specific rows.
2. Run the Egglog helper from `egg-smol-python` to compare baseline and multiset pipelines.
3. Copy the Haskell numbers into the doc-script notebook so it stays rerunnable without the source repo.

## Haskell comparison

From `/Users/saul/p/srtree-eqsat`:

```bash
stack exec -- runghc /Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/haskell_compare.hs 1 50
```

This prints, for each requested row:

- runtime
- stop reason from the exported API path, which is currently `unavailable_from_exported_api`
- parameter counts before and after `floatConstsToParam`
- input and output node counts
- final simplified expression rendered with `showOutput PYTHON`

The source repo's public `simplifyEqSat` export does not expose intermediate e-graph sizes or stop reasons, so `final_memo_size` and `final_eclass_count` are printed as `na`.

## Egglog comparison

From `/Users/saul/p/egg-smol-python`:

```bash
uv run --project /Users/saul/p/egg-smol-python python /Users/saul/p/egg-smol-python/python/exp/srtree_eqsat/run_compare.py
```

That runs:

- the baseline Egglog reproduction with Haskell-matched backoff settings
- the multiset hypothesis path with explicit stage reports

and prints compact tables for rows `1` and `50`.
