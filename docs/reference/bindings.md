# Encoded transport: `egglog.bindings`

The public Python frontend authors semantic protobuf records and sends encoded
requests to the shared Experimental adapter. Python names, call forms, keyword
arguments, defaults and mutation conventions belong to the frontend, alongside
owned references into those semantic records. Use {doc}`python-integration` for
the normal Python API. The generated protocol types live in
`egglog_proto.egglog.v1.egglog_pb`.

`egglog.bindings` is the private native transport. Its `_ProtoEngine` accepts
bytes for creation, execution, cloning, destruction, history and backend-wide
builtin retrieval. Execution returns an ordered list of encoded success/error
frames, plus an original Python callback exception when one occurred. Other
operations return encoded messages. `PrimitiveFailure`, re-exported
from `egglog`, marks an expected host primitive non-match.

The native Python-object provider installs its real custom sort and primitives
inside each transport. `_export_pyobject_catalog()` exports their configured
`BuiltinCatalog` for generation; builtin signatures come from their native owners.
The Python generator joins that semantic catalog to its local presentation map;
the map does not repeat native signatures.
Programs, snapshots and history contain user definitions and refer to configured
builtins by canonical keys. They do not contain builtin descriptor snapshots.
`Program.rulesets` carries complete named collection definitions as setup before
commands. Each supplied binding replaces its previous collection; omitted names
retain their bindings. `Run.rulesets` selects an ordered union of literal
collections and persistent names. Unknown names select no rules. Collections
carry rule records directly, and shared rule content retains its matching
progress across binding replacement. Function invocation uses `Apply` over
the function's structural type; Python retains its `UnstableFn` callable facade
and partial-value introspection. The engine checks native primitive applicability
when the program is submitted.

The extension has no public native EGraph, AST/command classes, raw values,
extractor, snapshot, plan wrappers or tracing configuration functions. Extraction,
compaction and snapshots use the generated protocol records. Ordinary execution
returns `RunResult` with `updated` and `can_stop`; explicit diagnostics use one
opaque `Diagnostic` document. `_ProtoEngine.run_stream` delivers encoded owned
chunks synchronously to a callback, with the same original exception channel.
Each frame contains either an output with its own arenas or an error; earlier
outputs remain valid after a later error. Diagnostics carry a format and opaque
bytes. Completion is acknowledged by an empty success frame or a final error.
`EGraph.freeze()` returns an owned `EGraphSnapshot`.

## Thread safety

Independent byte engines may run concurrently. Serialize access to a shared
mutable engine; overlapping mutation raises an error instead of waiting.
Python callbacks must synchronize shared mutable state. The
[high-level API](python-integration.md#thread-safety) also requires initialization
before concurrent use.

## Current migration limits

Notebook text needs the shared source-to-protobuf producer exposed to Python.
CSV input and replayable source recording remain unfinished. Visualization reads
the exported snapshot without a native serializer. Python custom extraction
callbacks need shared host-model registration; they do not use a second native
engine or extractor API.
