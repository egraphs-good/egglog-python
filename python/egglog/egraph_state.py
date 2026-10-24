"""
Encoded EGraph transport and response ownership.

No expression, declaration, or command is reconstructed from Python semantic
dataclasses here. Language lookup indexes retain canonical declaration Refs;
installation compatibility and idempotence belong to the shared adapter.
Only original Python callback exceptions use a request-owned non-wire channel.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager, suppress
from operator import index
from typing import Literal, SupportsIndex, cast

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from . import bindings
from ._catalog import builtin_catalog
from ._program import Builder, Owner, Packed, Ref, StructuralView, _copy_record, _definition_key, pack

__all__ = ["EGraphState", "EggSmolError"]


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
        num_threads: SupportsIndex = 1,
        save_egglog_string: bool = False,
        record_history: object = False,
        seminaive: object = True,
        no_decomp: object = False,
        report_level: pb.ReportLevel | None = None,
    ) -> None:
        num_threads = _thread_count(num_threads)
        if not isinstance(seminaive, bool) or not isinstance(no_decomp, bool):
            msg = "Compilation flags must be booleans"
            raise TypeError(msg)
        if not isinstance(record_history, bool):
            msg = "History recording must be a boolean"
            raise TypeError(msg)
        if save_egglog_string and not seminaive:
            msg = "Recording cannot replay initial seminaive=False without a Source creation context"
            raise NotImplementedError(msg)
        if report_level is not None and (
            type(report_level) is not pb.ReportLevel or report_level == pb.ReportLevel.UNSPECIFIED
        ):
            msg = "Expected a defined protobuf report level"
            raise TypeError(msg)
        self.transport = bindings._ProtoEngine()
        # The public default cost sort is an ambient engine builtin. Creation
        # references it directly; authored Programs still send full declarations.
        request = pb.CreateEGraphRequest(
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            options=pb.EGraphOptions(cost_sort=0, seminaive=seminaive),
            threads=num_threads,
            no_decomp=no_decomp,
            record_history=record_history,
            report_level=report_level,
        )
        response = pb.CreateEGraphResponse.from_binary(self.transport.create(request.to_binary()))
        self.handle: int | None = response.egraph_id
        if response.threads < 1:
            msg = "Creation response must contain a positive actual thread count"
            raise ValueError(msg)
        self.threads = response.threads
        self.no_decomp = no_decomp
        self._recording = save_egglog_string
        if save_egglog_string:
            self.transport.enable_recording()
        self.definitions = dict(builtin_catalog().definitions)
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

    # Arena adoption and host-error precedence share one error boundary.
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
        # Keep response validation inside this boundary to preserve the host cause.
        try:
            response = pb.RunProgramResponse.from_binary(encoded)
            # Successful preparation or a command location confirms that the
            # complete declaration installation preceded authored execution.
            # Failed installation can be retried with the original definitions.
            if response.error is None or response.error.location is not None:
                for reference in packed.definitions:
                    key = _definition_key(reference.read())
                    if key is not None:
                        self.definitions[key] = reference
            owner = self._adopt_value_records(
                pb.Program(
                    nodes=response.nodes,
                    sorts=response.sorts,
                    files=response.files,
                    rules=response.rules,
                )
            )
        except BaseException as error:
            if self._recording:
                self.transport.finish_recording(False)
            if host_error is not None and error is not host_error:
                raise error from host_error
            raise
        # The actual response and arenas have been consumed. Preserve
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
        # response reaches Python. Retain first-use source contexts only for
        # definitions actually present in the owned native observation.
        for reference in pending:
            if reference.role == "declarations" and (key := _definition_key(reference.read())) in observed:
                assert key is not None
                definitions[key] = reference
        builder = Builder.from_program(
            pb.Program(
                nodes=program.nodes,
                sorts=program.sorts,
                files=program.files,
                rules=program.rules,
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
        result.threads = self.threads
        result.no_decomp = self.no_decomp
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
