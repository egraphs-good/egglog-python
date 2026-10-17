# Encoded transport: `egglog.bindings`

The public Python frontend authors generated protobuf records and sends encoded
requests to the shared Experimental adapter. Use {doc}`python-integration` for
the normal Python API. The generated protocol types live in
`egglog_proto.egglog.v1.egglog_pb`.

`egglog.bindings` is the private native transport. Its `_ProtoEngine` accepts
bytes for creation, execution, cloning, destruction, resource configuration and
scope return. Results are bytes, except that execution also returns an original
Python callback exception when one occurred. `PrimitiveFailure`, re-exported
from `egglog`, marks an expected host primitive non-match.

The native Python-object provider installs its real custom sort and primitives
inside each transport. `_export_pyobject_catalog()` exports their configured
declarations for generation; builtin signatures come from their native owners.

The extension has no public native EGraph, AST/command classes, raw values,
extractor, snapshot, plan wrappers or tracing configuration functions. Extraction,
compaction, snapshots and reports use the generated protocol records. Report
detail is selected with the generated `egglog.ReportLevel` enum, and
`EGraph.freeze()` returns an owned `EGraphSnapshot`.

## Thread safety

Independent byte engines may run concurrently. Serialize access to a shared
mutable engine; overlapping mutation raises an error instead of waiting.
Python callbacks must synchronize shared mutable state. The
[high-level API](python-integration.md#thread-safety) also requires initialization
before concurrent use.

## Current migration limits

Notebook text needs the shared source-to-protobuf producer exposed to Python.
CSV input, replayable source recording and visualization from exported snapshots
remain unfinished. Python custom extraction callbacks need shared host-model
registration; they do not use a second native engine or extractor API.
