"""
Encoded EGraph transport and response ownership.

No expression, declaration, or command is reconstructed from Python semantic
dataclasses here. Language lookup indexes retain canonical declaration Refs;
installation compatibility and idempotence belong to the shared adapter.
Only original Python callback exceptions use a request-owned non-wire channel.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import contextmanager, suppress
from operator import index
from typing import Literal, SupportsIndex, cast
from uuid import uuid4

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from . import bindings
from ._catalog import builtin_catalog
from ._program import Builder, Owner, Packed, Ref, StructuralView, _copy_record, _definition_key, pack, rule_equal
from .declarations import Ident

__all__ = ["EGraphState", "EggSmolError"]

_QUERY_RESOURCES = object()


class EggSmolError(Exception):
    """An engine error returned by the encoded protocol."""


def _thread_count(value: object) -> int:
    """Keep Python's integer-index acceptance within the protocol's uint32 domain."""
    count = index(cast("SupportsIndex", value))
    if count < 0:
        msg = "can't convert negative int to unsigned"
        raise OverflowError(msg)
    if count > (1 << 32) - 1:
        msg = "thread count exceeds the protocol's uint32 range"
        raise OverflowError(msg)
    return count


class EGraphState:
    """Own one native handle and the declaration refs needed to read responses."""

    _recording = False

    def __init__(
        self,
        *,
        num_threads: int = 1,
        save_egglog_string: bool = False,
        seminaive: object = True,
        no_decomp: object = False,
    ) -> None:
        num_threads = _thread_count(num_threads)
        if not isinstance(seminaive, bool) or not isinstance(no_decomp, bool):
            msg = "Compilation flags must be booleans"
            raise TypeError(msg)
        if save_egglog_string and not seminaive:
            msg = "Recording cannot replay initial seminaive=False without a Source creation context"
            raise NotImplementedError(msg)
        self.transport = bindings._ProtoEngine()
        # The public default cost sort is an ambient engine builtin. Creation
        # references it directly; authored Programs still send full declarations.
        request = pb.CreateEGraphRequest(
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            options=pb.EGraphOptions(cost_sort=0, seminaive=seminaive),
            threads=num_threads,
            no_decomp=no_decomp,
        )
        response = pb.CreateEGraphResponse.from_binary(self.transport.create(request.to_binary()))
        self.handle: int | None = response.egraph_id
        self._recording = save_egglog_string
        if save_egglog_string:
            self.transport.enable_recording()
        self.definitions = dict(builtin_catalog().definitions)
        # Language bindings index canonical records, never copies of rule bodies.
        self.rulesets: dict[Ident, Ref] = {}
        self._combined_sources: dict[Ident, Ref] = {}
        self._ruleset_identifiers: dict[Ref, Ident] = {}
        self._direct_rules: set[Ref] = set()
        self._pending_rules: dict[tuple[Ident, Ref, Ref | None], tuple[Ref, Ref]] = {}
        self._pending_roots: dict[tuple[Ident, str, tuple[Ref, ...]], Ref] = {}
        # Exact acknowledged configs, or None after an unacknowledged attempt.
        # Scope ownership follows this handle, never the shared transport.
        self._scheduler_bindings: dict[str, pb.BindScheduler | None] = {}
        self._cost_model_active = 0
        self._native_running = False
        self._active_definitions: tuple[Ref, ...] = ()

    @contextmanager
    def cost_model(self, factory: Callable[[bytes], object], *, additive: bool) -> Generator[str, None, None]:
        """Retain an explicit C/D/codec registration for exactly one operation."""
        if self.handle is None or self._cost_model_active or self._native_running:
            msg = "Cannot register a model on a destroyed or actively extracting EGraph"
            raise RuntimeError(msg)
        sorts = pb.Program(
            ir_version=1,
            sorts=[
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))),
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="PyObject"))),
            ],
        )
        name = self.transport.register_cost_model(self.handle, sorts.to_binary(), 0, 1, additive, factory)
        try:
            yield name
        finally:
            self.transport.release_cost_model(self.handle, name)

    def configure_resources(
        self, num_threads: object = _QUERY_RESOURCES, *, no_decomp: object = _QUERY_RESOURCES
    ) -> pb.ConfigureEGraphResourcesResponse:
        """Query/update this handle over bytes and return its actual native resources."""
        if self._cost_model_active or self._native_running:
            msg = "Cannot execute engine operations during a cost-model callback"
            raise RuntimeError(msg)
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        operation: Oneof[Literal["query"], pb.Unit] | Oneof[Literal["threads"], int] | Oneof[Literal["no_decomp"], bool]
        if no_decomp is not _QUERY_RESOURCES:
            if num_threads is not _QUERY_RESOURCES:
                msg = "Resource updates require exactly one operation"
                raise ValueError(msg)
            if not isinstance(no_decomp, bool):
                msg = "Decomposition setting must be a boolean"
                raise TypeError(msg)
            operation = Oneof[Literal["no_decomp"], bool]("no_decomp", no_decomp)
        else:
            operation = (
                Oneof[Literal["query"], pb.Unit]("query", pb.Unit())
                if num_threads is _QUERY_RESOURCES
                else Oneof[Literal["threads"], int]("threads", _thread_count(num_threads))
            )
        request = pb.ConfigureEGraphResourcesRequest(egraph_id=self.handle, operation=operation)
        response = pb.ConfigureEGraphResourcesResponse.from_binary(
            self.transport.configure_resources(request.to_binary())
        )
        if response.threads < 1:
            msg = "Resource response must contain a positive actual thread count"
            raise ValueError(msg)
        if no_decomp is not _QUERY_RESOURCES and response.no_decomp is not no_decomp:
            msg = "Resource response did not acknowledge the requested decomposition setting"
            raise ValueError(msg)
        return response

    # Receipt/arena adoption and host-error precedence share one error boundary.
    def run_program(  # noqa: C901, PLR0912
        self,
        commands: Iterable[Ref] | Packed,
        *,
        roots: Iterable[Ref] = (),
        include_run_reports: bool = False,
        allow_failed_check: bool = False,
    ) -> tuple[pb.RunProgramResponse, Owner]:
        """Pack canonical command closure once and adopt the returned records."""
        if self._cost_model_active or self._native_running:
            msg = "Cannot execute engine operations during a cost-model callback"
            raise RuntimeError(msg)
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        packed = (
            commands
            if isinstance(commands, Packed)
            else pack(roots, commands=commands, ambient=self.definitions.keys())
        )
        if allow_failed_check and (
            len(packed.program.commands) != 1
            or packed.program.commands[0].kind is None
            or packed.program.commands[0].kind.field != "check"
        ):
            msg = "Failed-check policy requires exactly one top-level Check"
            raise ValueError(msg)
        request = pb.RunProgramRequest(
            egraph_id=self.handle, program=packed.program, profile=True, include_run_reports=include_run_reports
        )
        encoded_request = request.to_binary()
        scheduler_bindings = [
            _copy_record(command.kind.value)
            for command in packed.program.commands
            if command.kind is not None and command.kind.field == "bind_scheduler"
        ]
        if self._recording:
            program = packed.program
            # Snapshot introspection has no source command and cannot install
            # definitions or evaluate authored roots. It remains usable after close.
            snapshot_only = (
                len(program.commands) == 1
                and not program._unknown_fields
                and not program.commands[0]._unknown_fields
                and program.commands[0].kind is not None
                and program.commands[0].kind.field == "freeze"
                and not program.commands[0].kind.value._unknown_fields
                and not any((
                    program.nodes,
                    program.sorts,
                    program.declarations,
                    program.rules,
                    program.rulesets,
                    program.files,
                ))
            )
            if not snapshot_only:
                self.transport.prepare_recording(program.to_binary())
        # Rendering/packing failures above provably did not submit a binding.
        # After transport starts, never retry an unacknowledged deployment.
        for binding in scheduler_bindings:
            self._scheduler_bindings.setdefault(binding.name, None)
        self._native_running = True
        self._active_definitions = packed.definitions
        try:
            encoded, host_error = self.transport.run(encoded_request)
        except BaseException:
            if self._recording:
                self.transport.finish_recording(False)
            raise
        finally:
            self._active_definitions = ()
            self._native_running = False
        # Keep receipt validation inside this boundary to preserve the host cause.
        try:
            response = pb.RunProgramResponse.from_binary(encoded)
            submitted = {
                cast("pb.Ruleset", reference.read()).name: reference
                for reference in packed.definitions
                if reference.role == "rulesets"
            }
            receipt = response.ruleset_publication
            if receipt is None:
                msg = "Response omitted the required ruleset publication receipt"
                raise ValueError(msg)  # noqa: TRY301
            names = set(receipt.names)
            if len(names) != len(receipt.names) or not names <= submitted.keys():
                msg = "Ruleset publication receipt contains duplicate or unsubmitted names"
                raise ValueError(msg)  # noqa: TRY301
            if (response.error is None or response.error.location is not None) and names != submitted.keys():
                msg = "Completed installation omitted submitted ruleset confirmations"
                raise ValueError(msg)  # noqa: TRY301
            for name in receipt.names:
                reference = submitted[name]
                self.definitions["ruleset", name] = reference
                # A completed root confirms its declaration dependencies even when
                # a later installation failed before authored commands began.
                for dependency in pack(roots=[reference], ambient=self.definitions.keys()).definitions:
                    key = _definition_key(dependency.read())
                    if key is not None and dependency.role == "declarations":
                        self.definitions[key] = dependency
            # A command location confirms that declaration installation finished,
            # even when execution failed. An absent location proves no such thing.
            if response.error is None or response.error.location is not None:
                for reference in packed.definitions:
                    key = _definition_key(reference.read())
                    if key is not None and reference.role == "declarations":
                        self.definitions[key] = reference
            owner = self._adopt_value_records(
                pb.Program(
                    ir_version=1,
                    nodes=response.nodes,
                    sorts=response.sorts,
                    files=response.files,
                )
            )
        except BaseException as error:
            if self._recording:
                self.transport.finish_recording(False)
            if host_error is not None and error is not host_error:
                raise error from host_error
            raise
        # The actual response/arena and publication have been consumed. Preserve
        # first-host precedence without rewriting its chain or the native Error.
        if self._recording:
            self.transport.finish_recording(host_error is None and response.error is None)
        if host_error is None and response.error is None:
            for binding in scheduler_bindings:
                self._scheduler_bindings[binding.name] = binding
        if host_error is not None:
            raise host_error
        if (
            allow_failed_check
            and response.error is not None
            and response.error.code == pb.ErrorCode.CHECK_FAILED
            and response.error.location is not None
            and response.error.location.path == [0]
            and not response.error.location.iterations
        ):
            return response, owner
        if response.error is not None:
            error = EggSmolError(response.error.message)
            # Keep the actual generated record with this Python exception.
            error.__dict__["_egglog_proto_error"] = response.error
            raise error
        return response, owner

    def bind_scheduler(self, reference: Ref) -> None:
        """Deploy once per acknowledged config; preserve uncertain live state."""
        if self.handle is None or self._cost_model_active or self._native_running:
            msg = "Cannot deploy a scheduler on a destroyed or actively executing EGraph"
            raise RuntimeError(msg)
        command = reference.read()
        if reference.role != "commands" or command.kind is None or command.kind.field != "bind_scheduler":
            msg = "Scheduler deployment requires a canonical BindScheduler command"
            raise TypeError(msg)
        binding = command.kind.value
        if binding.name in self._scheduler_bindings:
            installed = self._scheduler_bindings[binding.name]
            if installed is None:
                msg = f"Scheduler {binding.name} has an unacknowledged deployment; restore a saved graph before reuse"
                raise RuntimeError(msg)
            if installed != binding:
                msg = f"Scheduler {binding.name} already has a different configuration"
                raise ValueError(msg)
            return
        self.run_program([reference])

    def _adopt_value_records(self, program: pb.Program, *, pending: Iterable[Ref] = ()) -> Owner:
        """Own value arenas while retaining canonical declaration/class identity."""
        if program._unknown_fields:
            msg = "Cannot adopt unknown Program fields"
            raise NotImplementedError(msg)
        definitions = self.definitions.copy()
        observed = {_definition_key(declaration) for declaration in program.declarations}
        # Native observation starts after installation but before its response
        # receipt reaches Python. Retain first-use source contexts only for
        # definitions actually present in the owned native observation.
        for reference in pending:
            if reference.role == "declarations" and (key := _definition_key(reference.read())) in observed:
                assert key is not None
                definitions[key] = reference
        builder = Builder.from_program(
            pb.Program(
                ir_version=program.ir_version,
                nodes=program.nodes,
                sorts=program.sorts,
                files=program.files,
            )
        )
        for declaration in program.declarations:
            if _definition_key(declaration) not in definitions:
                # Native-only records have no original Python class context;
                # their arena references already address the owned values.
                builder.add("declarations", declaration)
        for reference in definitions.values():
            builder.import_ref(reference)
        return builder.publish()

    def _ruleset_members(self, root: Ref) -> list[Ref]:
        """Flatten ordered canonical selections, deduplicating occurrences only."""
        members: dict[Ref, None] = {}
        seen: set[Ref] = set()
        active: set[Ref] = set()
        pending = [(root, False)]
        while pending:
            reference, leaving = pending.pop()
            if leaving:
                active.remove(reference)
                continue
            if reference in active:
                msg = "Cyclic ruleset selection"
                raise ValueError(msg)
            if reference in seen:
                continue
            seen.add(reference)
            active.add(reference)
            pending.append((reference, True))
            record: pb.Ruleset = reference.read()
            if record.kind is None:
                msg = "Ruleset has no body"
                raise ValueError(msg)
            if record.kind.field == "rules":
                for index in record.kind.value.rules:
                    members[reference.owner.ref("rules", index)] = None
            else:
                children = []
                for child in record.kind.value.rulesets:
                    if child.kind is None:
                        msg = "Ruleset reference has no target"
                        raise ValueError(msg)
                    children.append(
                        reference.owner.ref("rulesets", child.kind.value)
                        if child.kind.field == "index"
                        else self.definitions["ruleset", child.kind.value]
                    )
                pending.extend((child, False) for child in reversed(children))
        return list(members)

    def publish_ruleset(self, root: Ref, identifiers: Mapping[Ref, Ident], *, direct: bool = False) -> Ref:  # noqa: C901
        """
        Publish live authored membership without changing any retained body.

        Rightmost same-Ident source selection precedes installation. Once a
        combined source is installed, its children select completed membership
        only; running it must not install later authored child appends.
        """
        selected = {ident: reference for reference, ident in identifiers.items()}
        active: set[Ident] = set()
        rule_keys: dict[Ref, tuple[object, ...]] = {}

        def confirm(reference: Ref, ident: Ident) -> None:
            record: pb.Ruleset = reference.read()
            key = ("ruleset", record.name)
            try:
                if self.definitions.get(key) != reference:
                    self.run_program([], roots=[reference])
            finally:
                if self.definitions.get(key) == reference:
                    self._ruleset_identifiers[reference] = ident
            if self.definitions.get(key) != reference:
                msg = "Ruleset publication did not confirm the requested root"
                raise ValueError(msg)

        def publish(ident: Ident, *, installed_only: bool = False) -> Ref:  # noqa: C901, PLR0912
            if ident in active:
                msg = "Cyclic logical ruleset bindings"
                raise ValueError(msg)
            active.add(ident)
            try:
                current = self.rulesets.get(ident)
                if current is not None:
                    confirm(current, ident)
                if installed_only and ident not in self._combined_sources:
                    if current is None:
                        msg = "Combined ruleset child has no completed membership"
                        raise ValueError(msg)
                    return current
                source = self._combined_sources.get(ident, selected.get(ident))
                if source is None:
                    raise ValueError(f"Missing authored ruleset: {ident}")
                record: pb.Ruleset = source.read()
                if record.kind is None:
                    msg = "Authored ruleset has no body"
                    raise ValueError(msg)
                if record.kind.field == "combined":
                    frozen = ident in self._combined_sources
                    child_bindings = self._ruleset_identifiers if frozen else identifiers
                    children = []
                    for child in record.kind.value.rulesets:
                        if child.kind is None or child.kind.field != "index":
                            msg = "Authored composition must reference owned ruleset snapshots"
                            raise ValueError(msg)
                        child_ident = child_bindings[source.owner.ref("rulesets", child.kind.value)]
                        children.append(publish(child_ident, installed_only=frozen))
                    combined = pb.CombinedRuleset(
                        rulesets=[
                            pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", child.read().name))
                            for child in children
                        ]
                    )
                    if current is not None and current.read().kind.value == combined:
                        return current
                    root_key = (ident, "combined", tuple(children))
                    if root_key not in self._pending_roots:
                        builder = Builder()
                        index = builder.add(
                            "rulesets",
                            pb.Ruleset(
                                name=f"__egglog_python_ruleset_{uuid4().hex}",
                                kind=Oneof[Literal["combined"], pb.CombinedRuleset]("combined", combined),
                            ),
                        )
                        self._pending_roots[root_key] = builder.publish().ref("rulesets", index)
                    candidate = self._pending_roots[root_key]
                    try:
                        confirm(candidate, ident)
                    finally:
                        if self.definitions.get(("ruleset", candidate.read().name)) == candidate:
                            self.rulesets[ident] = candidate
                            self._pending_roots.pop(root_key)
                            if not frozen:
                                self._combined_sources[ident] = source
                                self._ruleset_identifiers.update(identifiers)
                    return candidate
                for index in record.kind.value.rules:
                    rule = source.owner.ref("rules", index)
                    members = [] if current is None else self._ruleset_members(current)
                    if not direct and any(
                        old not in self._direct_rules and rule_equal(rule, old, keys=rule_keys) for old in members
                    ):
                        continue
                    label = rule.read().name
                    if label and any(old.read().name == label for old in members):
                        # Python's public names were unique within the logical
                        # ruleset. Shared IR labels are deliberately not identity:
                        # this is frontend compatibility validation, not a wire error.
                        raise EggSmolError(f"At 0:0 of \nRule already exists: {label}")
                    key = (ident, rule, current)
                    if key not in self._pending_rules:
                        builder = Builder()
                        for dependency in pack(roots=[rule]).definitions:
                            builder.import_ref(dependency)
                        occurrence = builder.add(
                            "rules",
                            _copy_record(
                                rule.read(),
                                # The visitor is synchronous and retains no callback.
                                relocate=lambda role, index: builder.import_ref(rule.owner.ref(role, index)),  # noqa: B023
                            ),
                        )
                        leaf_index = builder.add(
                            "rulesets",
                            pb.Ruleset(
                                name=f"__egglog_python_ruleset_{uuid4().hex}",
                                kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList(rules=[occurrence])),
                            ),
                        )
                        leaf = builder.publish().ref("rulesets", leaf_index)
                        candidate = leaf
                        if current is not None:
                            builder = Builder()
                            version = builder.add(
                                "rulesets",
                                pb.Ruleset(
                                    name=f"__egglog_python_ruleset_{uuid4().hex}",
                                    kind=Oneof[Literal["combined"], pb.CombinedRuleset](
                                        "combined",
                                        pb.CombinedRuleset(
                                            rulesets=[
                                                pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", ref.read().name))
                                                for ref in (current, leaf)
                                            ]
                                        ),
                                    ),
                                ),
                            )
                            candidate = builder.publish().ref("rulesets", version)
                        self._pending_rules[key] = leaf, candidate
                    leaf, candidate = self._pending_rules[key]
                    try:
                        confirm(leaf, ident)
                    finally:
                        if self.definitions.get(("ruleset", leaf.read().name)) == leaf:
                            self.rulesets[ident] = current = candidate
                            if direct:
                                self._direct_rules.update(self._ruleset_members(leaf))
                            self._pending_rules.pop(key)
                    confirm(candidate, ident)
                if current is None:
                    root_key = (ident, "rules", ())
                    if root_key not in self._pending_roots:
                        builder = Builder()
                        index = builder.add(
                            "rulesets",
                            pb.Ruleset(
                                name=f"__egglog_python_ruleset_{uuid4().hex}",
                                kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList()),
                            ),
                        )
                        self._pending_roots[root_key] = builder.publish().ref("rulesets", index)
                    current = self._pending_roots[root_key]
                    try:
                        confirm(current, ident)
                    finally:
                        if self.definitions.get(("ruleset", current.read().name)) == current:
                            self.rulesets[ident] = current
                            self._pending_roots.pop(root_key)
                return current
            finally:
                active.remove(ident)

        return publish(identifiers[root])

    def copy(self) -> EGraphState:
        """Clone the native handle while sharing immutable authored definitions."""
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
        if self._recording:
            self.transport.prepare_recording_scope(True)
        try:
            response = pb.CloneEGraphResponse.from_binary(
                self.transport.clone_graph(pb.CloneEGraphRequest(egraph_id=self.handle).to_binary())
            )
        except BaseException:
            if self._recording:
                self.transport.finish_recording(False)
            raise
        if self._recording:
            self.transport.finish_recording(True)
        result = object.__new__(EGraphState)
        result.transport = self.transport
        result.handle = response.egraph_id
        result._recording = self._recording
        result.definitions = self.definitions.copy()
        result.rulesets = self.rulesets.copy()
        result._combined_sources = self._combined_sources.copy()
        result._ruleset_identifiers = self._ruleset_identifiers.copy()
        result._direct_rules = self._direct_rules.copy()
        result._pending_rules = self._pending_rules.copy()
        result._pending_roots = self._pending_roots.copy()
        result._scheduler_bindings = self._scheduler_bindings.copy()
        result._cost_model_active = 0
        result._native_running = False
        result._active_definitions = ()
        return result

    def return_to(self, saved: EGraphState) -> None:
        """Restore a direct saved owner; successful Return consumes this handle."""
        if self.handle is None or saved.handle is None:
            msg = "Return requires two live EGraph handles"
            raise RuntimeError(msg)
        if self.transport is not saved.transport or self.handle == saved.handle:
            msg = "Return requires distinct handles on the same transport"
            raise ValueError(msg)
        if self._recording:
            self.transport.prepare_recording_scope(False)
        try:
            encoded = self.transport.return_to(
                pb.ReturnToEGraphRequest(active_id=self.handle, saved_id=saved.handle).to_binary()
            )
            # Successful transport consumes the active handle, even if decoding
            # fails. Public pop selects the saved state only after acknowledgement.
            self.handle = None
            pb.ReturnToEGraphResponse.from_binary(encoded)
        except BaseException:
            if self._recording:
                self.transport.finish_recording(False)
            raise
        if self._recording:
            self.transport.finish_recording(True)

    def destroy(self) -> None:
        """Release this handle once without affecting cloned graph handles."""
        if self.handle is not None:
            pb.DestroyEGraphResponse.from_binary(
                self.transport.destroy(pb.DestroyEGraphRequest(egraph_id=self.handle).to_binary())
            )
            self.handle = None

    def __del__(self) -> None:
        # A failed constructor may have no handle. Never let native cleanup
        # replace an exception already propagating during Python disposal.
        if getattr(self, "handle", None) is not None:
            with suppress(Exception):
                self.destroy()


class _CostModelObservation:
    """Lookup-only indexes over one owned normative observation Program."""

    def __init__(self, state: EGraphState, program: pb.Program) -> None:
        self.program = program
        self.owner = state._adopt_value_records(program, pending=state._active_definitions)
        self.registered = frozenset(
            key for declaration in program.declarations if (key := _definition_key(declaration)) is not None
        )
        self.table_backed = frozenset(
            declaration.kind.value.name
            for declaration in program.declarations
            if declaration.kind is not None and declaration.kind.field in {"constructor", "function", "relation"}
        )
        self.rows: dict[tuple[str, tuple[int, ...]], int] = {}
        self.assigned: dict[tuple[str, tuple[int, ...]], int] = {}
        for node_index, node in enumerate(program.nodes):
            if node.kind is not None and node.kind.field == "union":
                for member in node.kind.value.members:
                    alternative = program.nodes[member]
                    if alternative.kind is not None and alternative.kind.field == "call":
                        call = alternative.kind.value
                        self.rows[call.func, tuple(call.args)] = node_index
        for command in program.commands:
            if command.kind is None or command.kind.field != "action" or command.kind.value.kind is None:
                continue
            action = command.kind.value.kind
            if action.field in {"set", "set_cost"} and action.value.target is not None:
                target = action.value.target
                destination = self.rows if action.field == "set" else self.assigned
                destination[target.func, tuple(target.args)] = (
                    action.value.value if action.field == "set" else action.value.cost
                )

    def supplied_values(self, node: int) -> frozenset[Ref]:
        """Follow semantic callback arguments, stopping at opaque class identity."""
        record = self.program.nodes[node]
        pending = list(record.kind.value.args) if record.kind is not None and record.kind.field == "call" else [node]
        result: set[Ref] = set()
        while pending:
            index = pending.pop()
            reference = self.owner.ref("nodes", index)
            if reference in result:
                continue
            result.add(reference)
            record = reference.read()
            if record.kind is not None and record.kind.field == "union":
                continue

            def visit(role: str, child: int) -> int:
                if role == "nodes":
                    pending.append(child)
                return child

            _copy_record(record, relocate=visit)
        return frozenset(result)

    def lookup(self, reference: Ref, *, supplied: frozenset[Ref] | None) -> Ref | None:
        """Resolve existing rows without evaluating arguments or publishing declarations."""
        node: pb.Node = reference.read()
        if node.kind is None or node.kind.field not in {"call", "get_cost"}:
            msg = "Can only look up a table call or assigned cost"
            raise TypeError(msg)
        call = node.kind.value
        if ("callable", call.func) not in self.registered:
            msg = "Lookup tables must be registered before extraction starts"
            raise ValueError(msg)
        allowed = (
            supplied
            if supplied is not None
            else frozenset(self.owner.ref("nodes", i) for i in range(len(self.program.nodes)))
        )

        # A missing row must not hide a later invalid observation.
        resolved_args = [
            self._resolve(reference.owner.ref("nodes", arg), allowed, callback=supplied is not None)
            for arg in call.args
        ]
        arguments = []
        for value in resolved_args:
            if value is None:
                return None
            arguments.append(value)
        key = call.func, tuple(arguments)
        result = (self.assigned if node.kind.field == "get_cost" else self.rows).get(key)
        return None if result is None else self.owner.ref("nodes", result)

    def _resolve(self, value: Ref, allowed: frozenset[Ref], *, callback: bool) -> int | None:
        value = value.owner.ref(value.role, value.index)
        if value in allowed and value.owner is self.owner:
            return value.index
        record: pb.Node = value.read()
        if (structural := self._portable_shape(value)) is not None:
            for candidate in allowed:
                if StructuralView(candidate) == structural:
                    return candidate.index
            if not callback:
                return None
        if not callback and record.kind is not None and record.kind.field == "call":
            target = record.kind.value
            if ("callable", target.func) in self.registered and target.func not in self.table_backed:
                msg = "Lookup cannot evaluate primitive keys; register or bind the expression explicitly"
                raise ValueError(msg)
            resolved_args = [
                self._resolve(value.owner.ref("nodes", arg), allowed, callback=False) for arg in target.args
            ]
            arguments = []
            for resolved in resolved_args:
                if resolved is None:
                    return None
                arguments.append(resolved)
            return self.rows.get((target.func, tuple(arguments)))
        msg = (
            "Cost models can only look up tables using values supplied to the callback"
            if callback
            else (
                "Lookup value belongs to a different EGraph or observation scope; "
                "use EGraph.let(name, original_call) to retain a live value"
            )
        )
        raise ValueError(msg)

    @staticmethod
    def _portable_shape(reference: Ref) -> StructuralView | None:
        """Reject opaque classes and unevaluated calls before structural matching."""
        pending = [reference]
        seen: set[Ref] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            record: pb.Node = current.read()
            if record.kind is None or record.kind.field != "primitive_value":
                return None

            def visit(role: str, child: int, owner: Owner = current.owner) -> int:
                if role == "nodes":
                    pending.append(owner.ref("nodes", child))
                return child

            _copy_record(record, relocate=visit)
        return StructuralView(reference)
