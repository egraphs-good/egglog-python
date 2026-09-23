---
file_format: mystnb
---

# Low Level: `egglog.bindings`

This modules contains bindings for the rust `egglog` library which are meant to be as close to the source as possible. This might result in not the most ergonomic API, if so, we can build higher level abstractions on top of it.

Example:

```{code-cell} python
from egglog.bindings import *

eqsat_basic = """(datatype Math
  (Num i64)
  (Var String)
  (Add Math Math)
  (Mul Math Math))

;; expr1 = 2 * (x + 3)
(let expr1 (Mul (Num 2) (Add (Var "x") (Num 3))))
;; expr2 = 6 + 2 * x
(let expr2 (Add (Num 6) (Mul (Num 2) (Var "x"))))


(rewrite (Add a b)
         (Add b a))
(rewrite (Mul a (Add b c))
         (Add (Mul a b) (Mul a c)))
(rewrite (Add (Num a) (Num b))
         (Num (+ a b)))
(rewrite (Mul (Num a) (Num b))
         (Num (* a b)))

(run 10)
(check (= expr1 expr2))"""

egraph = EGraph()
commands = egraph.parse_program(eqsat_basic)
egraph.run_program(*commands)
```

`EggSmolError.replayable_by_fail` conservatively reports whether the failing
command can be reproduced by wrapping it in Egglog's `(fail ...)` command.

Experimental commands may return typed data through
`UserDefinedCommandOutput`. Command execution first returns a
`UserDefinedOutput` wrapper; call `output.output.as_multi_extract()` on its
payload. This returns a `MultiExtractOutput` for `multi-extract` results and
`None` for other custom outputs. Its `termdag` is shared by all results, while
`terms` contains one ordered list of variant term IDs per input root.

The commands are a representation which is close the AST of the egglog text language. We
can see this by printing the commands:

```{code-cell} python
for command in commands:
    print(command)
```

## Shared programs

`Program` owns the same versioned command records used by Rust. It can be built
from binding commands, parsed from Egglog source, or imported from JSON:

```{code-cell} python
from egglog.bindings import EGraph, Program

program = Program.parse("(function answer () i64 :no-merge)\n(set (answer) 42)\n(check (= (answer) 42))")
payload = program.to_json()
restored = Program.from_json(payload)
EGraph().run_shared_program(restored)
assert restored.to_json() == payload
print(restored.to_replayable_egglog())
```

`Program.json_schema()` returns JSON Schema generated from the Rust definitions.
JSON preserves native fields that the legacy Python command classes do not
expose. Imported programs retain their native records throughout execution and
export; they are not converted back through those classes. JSON import/export
enforces the core wire format's size and nesting limits, raising `ValueError`
when exceeded. Native execution does not require the program to fit those limits.

`to_egglog()` renders diagnostic source. `to_replayable_egglog()` additionally
parses and compares the result, ignoring source locations and redundant
single-element schedule sequences, and raises
`ValueError` when the text syntax cannot preserve the records. This check does
not guarantee execution will succeed. The ordinary `run_program(*commands)`
API also executes through shared records. Programs remain surface commands:
the engine still expands, resolves, and typechecks them.

Enable `EGraph(record_program=True)` to record submitted commands in order.
`recorded_program()` exports the commands; `stop_recording()` returns a
`CommandRecord` whose `to_json()` includes each command's outcome and whose
`program()` exports its attempted command stream. `start_recording()` resets
and enables recording. Unlike legacy `record=True`/`commands()`, this record
retains failed attempts and successful prefixes of failing batches.

A command record is not a state snapshot or a rollback mechanism. Failed
commands may have partial effects, and replay stops at errors. It does not
record outputs, direct value queries, extraction callbacks, runtime settings,
or external file contents. It retains push/pop history rather than deleting
commands when a scope is popped. A Python callback exception can be reported
after later commands in its native batch have executed; the record attributes
that exception to its originating command without rolling back the suffix.

Structurally valid commands may require extensions or external resources.
Python's bindings install experimental commands and the `PyObject` sort; a
plain Rust engine must install equivalent capabilities to execute programs
that use them. JSON contains Python pickle payloads when `PyObject` is used,
and executing these payloads requires a trusted Python environment. Live
custom extraction-cost callbacks are outside the portable program format.

## Thread safety

Independent `EGraph` and `Extractor` instances can run concurrently. Serialize
access to shared mutable objects, including `EGraph` and `TermDag`; overlapping
accesses involving mutation raise an error instead of waiting. Python callbacks
must synchronize shared mutable state. The
[high-level API](python-integration.md#thread-safety) also requires initialization
before concurrent use.

## API

```{eval-rst}
.. automodule:: egglog.bindings
   :members:
   :undoc-members:
```
