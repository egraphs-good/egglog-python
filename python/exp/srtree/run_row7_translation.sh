#!/usr/bin/env bash
set -euo pipefail

uv run --project /Users/saul/p/egg-smol-python python -u \
  /Users/saul/p/srtree/examples/egglog_ac_multiset_from_srtree.py \
  --corpus /Users/saul/p/srtree/results/ablation/1199_pareto_random \
  --row 7 \
  --iters 50
