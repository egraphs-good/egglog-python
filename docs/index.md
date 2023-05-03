# [`egglog`](https://github.com/egraphs-good/egglog/) Python

`egglog` is a Python package that provides bindings to the Rust library of the same name,
allowing you to use e-graphs in Python for optimization, symbolic computation, and analysis.

```shell
pip install egglog
```

## Status of this project

This package is in development and is not ready for production use. `egglog` itself
is also subject to changes and is less stable than [`egg`](https://github.com/egraphs-good/egg).
If you are looking for a more stable e-graphs library in Python, you can try [`snake-egg`](https://github.com/egraphs-good/snake-egg), which wraps `egg`.

`egglog` is a rewrite of the `egg` library to use [relational e-matching](https://arxiv.org/abs/2108.02290) and to add datalog features.
See the ["Better Together: Unifying Datalog and Equality Saturation"](https://arxiv.org/abs/2304.04332) paper for more details

> We present egglog, a fixpoint reasoning system that unifies Datalog and equality saturation (EqSat). Like Datalog, it supports efficient incremental execution, cooperating analyses, and lattice-based reasoning. Like EqSat, it supports term rewriting, efficient congruence closure, and extraction of optimized terms.

## How documentation is organized

We use the [Diátaxis framework](https://diataxis.fr/) to organize our documentation. This helps with figuring out where different content should live and how it should be organized.

```{toctree}
auto_examples/index
tutorials
how-to-guides
explanation
reference
```
