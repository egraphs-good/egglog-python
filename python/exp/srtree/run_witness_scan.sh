#!/usr/bin/env bash
set -euo pipefail

SRTREE_ROOT=/Users/saul/p/srtree

cd "$SRTREE_ROOT"
. /Users/saul/.ghcup/env
cabal exec -- runghc -i"$SRTREE_ROOT/src" -i"$SRTREE_ROOT/examples" \
  "$SRTREE_ROOT/examples/find_eqsat_cap_witness.hs" --max-iters 50 "$@"
