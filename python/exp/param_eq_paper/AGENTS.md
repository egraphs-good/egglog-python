# Param-Eq Paper Notes

This directory contains the paper-replication artifacts and runners for the
older `param-eq-haskell` experiment harness.

## Haskell source of truth

The archived Haskell repo used for paper comparisons lives at:

- `/Users/saul/p/param-eq-haskell`

If live Haskell runs are blocked, use the archived outputs under:

- `/Users/saul/p/param-eq-haskell/results`

Those archived result files are the fallback source of truth for row-level
Haskell outputs.

## LLVM requirement

The pinned Haskell toolchain uses GHC `9.0.2` on Apple Silicon. On this
machine, that GHC falls back to LLVM, so Stack commands fail unless an LLVM
`opt` binary is on `PATH`.

The local setup currently in place is:

- `llvm@12` keg: `/opt/homebrew/opt/llvm@12`
- stable symlink: `/Users/saul/installs/llvm-12 -> /opt/homebrew/opt/llvm@12`
- local Homebrew formula copy:
  `/opt/homebrew/Library/Taps/saul/homebrew-local-old-llvm/Formula/llvm@12.rb`

For any shell running the Haskell build or probes:

```bash
export PATH="$HOME/installs/llvm-12/bin:$PATH"
opt --version
```

Without that, commands like:

```bash
stack run pagie original trees
```

fail with:

```text
ghc-9.0.2: could not execute: opt
Warning: Couldn't figure out LLVM version!
Make sure you have installed LLVM between [9 and 13)
```

## Local Homebrew workflow for `llvm@12`

Homebrew disabled `llvm@12` upstream, so the least-invasive install path here
is:

1. keep a local tap with a copied `llvm@12.rb`
2. remove the local `disable!` gate there
3. remove any stale max-macOS restriction that blocks the current machine
4. install from that local formula with API resolution disabled

Install shape:

```bash
HOMEBREW_NO_INSTALL_FROM_API=1 brew install saul/local-old-llvm/llvm@12
mkdir -p ~/installs
ln -sfn /opt/homebrew/opt/llvm@12 ~/installs/llvm-12
```

## Old GHC libffi header fix

After LLVM was fixed locally, the next Stack blocker was a missing arm64 libffi
target header during dependency builds.

The current machine has this RTS header symlink in place already:

- `/Users/saul/.stack/programs/aarch64-osx/ghc-9.0.2/lib/ghc-9.0.2/lib/aarch64-osx-ghc-9.0.2/rts-1.0.2/include/ffitarget_arm64.h`

It points to the Command Line Tools SDK header:

- `/Library/Developer/CommandLineTools/SDKs/MacOSX26.2.sdk/usr/include/ffi/ffitarget_arm64.h`

If that symlink ever disappears, restore it with:

```bash
sdk_path="$(xcrun --show-sdk-path)"
ghc_rts_include="$HOME/.stack/programs/aarch64-osx/ghc-9.0.2/lib/ghc-9.0.2/lib/aarch64-osx-ghc-9.0.2/rts-1.0.2/include"
ln -sfn "$sdk_path/usr/include/ffi/ffitarget_arm64.h" "$ghc_rts_include/ffitarget_arm64.h"
```

## One-time Haskell build

After LLVM and the header fix are in place:

```bash
cd /Users/saul/p/param-eq-haskell
stack build --only-dependencies
```

That is the expensive one-time setup step. After it succeeds, Stack runs and
small Haskell probes are much cheaper to rerun.

## Useful live Haskell commands

Run from `/Users/saul/p/param-eq-haskell` with LLVM 12 on `PATH`:

```bash
stack run pagie original trees
stack run pagie original counts
stack run kotanchek original trees
stack run kotanchek original counts
```

## Analysis probe

When checking whether Egglog still matches `FixTree.hs` analysis semantics, use
the local probe file:

- `/Users/saul/p/param-eq-haskell/tmp_probe_analysis.hs`

It is meant for mixed-class checks such as:

- `x - x`
- `2 - 2`
- `x / x`
- `2 / 2`

and reports:

- class analysis before and after rewrites
- node sets before and after rewrites
- extracted best term with `cost2`
