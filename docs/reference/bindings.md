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
Callable definitions are setup records in `Program.constructors`,
`Program.functions`, `Program.relations` and `Program.user_defined_primitives`.
All four collections share one callable namespace and link before commands;
collection order does not determine installation order. Each concrete definition
carries its optional `doc`; table definitions also carry the
`public_alias`/`default_name` table-name choice. Compatible definitions may be
repeated. `RuleBody.query` shares one conjunctive binder, and `RuleBody.actions`
executes actions in order for each match.
Equality-class nodes use `Node.eq` with `Eq.members`. Python keeps
`eq(...).to(...)` for equality facts and `union(...).with_(...)` for merge
actions. Directional function-row writes remain `Set`.
Every equality entry in an input `Program.sorts` establishes that nominal type,
including unused entries. Its optional `doc` carries definition metadata.
Every typed `Node.binding` establishes a global slot in a separate name
namespace; `Node.var` remains lexical. `Command.let` assigns a value to a binding
node, supports same-sort rebinding and loops, and is excluded from rule actions.
Snapshots retain all installed nominal sorts and typed slots, with one Let per
populated slot. Saved code retains binding references while stored data is closed.
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

The protocol contains no source files or byte spans. Documentation, readable names
and diagnostic rule labels remain. Source-produced `Node.var` records keep the
authored binder names; private backend hygiene does not alter these protocol names.
The source parser retains its own input locations for frontend diagnostics.

An attributable error carries a compact, independently owned `Error.context`
Program containing the offending authored command, rule or merge declaration and
its referenced user definitions. `Error.site.path` selects that owner or action
inside the returned context using protobuf field numbers and repeated-element
indices; its optional `node_id` selects the exact failing expression in the
context's node arena. `Error.location` independently identifies the active
command in the submitted request. Malformed or internal failures may have no
known context. Python retains the generated error record on `EggSmolError` and
adds a generated protobuf JSON note for the authored context and site, preserving the
engine message and exception arguments. Presentation never runs the Program or
loads opaque custom values. Original Python callback exceptions retain their
identity and traceback through the separate host exception channel.

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
