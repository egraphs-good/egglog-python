---
file_format: mystnb
---

# Reference

This package serves two purposes:

1. Exposing the machinery of the Rust implementation to Python.
2. Providing a Pythonic interface to build e-graphs.

Instead of skipping directly to the second step, we first expose the primitives
from the rust library as close as we can to the original API. Then we build a
higher level API on top of that, that is more friendly to use.

```{toctree}
:maxdepth: 1
changelog
reference/usage
reference/contributing
reference/high-level
reference/egglog-translation
reference/python-integration
reference/bindings
```
