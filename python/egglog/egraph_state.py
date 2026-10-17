"""
Encoded EGraph transport and response ownership.

No expression, declaration, or command is reconstructed from Python semantic
dataclasses here. Language lookup indexes retain canonical declaration Refs;
installation compatibility and idempotence belong to the shared adapter.
Only original Python callback exceptions use a request-owned non-wire channel.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import suppress
from operator import index
from typing import Literal, SupportsIndex, cast
from uuid import uuid4

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from . import bindings
from ._catalog import builtin_catalog
from ._program import Builder, Owner, Packed, Ref, _copy_record, _definition_key, pack, rule_equal
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

    def __init__(self, *, num_threads: int = 1) -> None:
        num_threads = _thread_count(num_threads)
        self.transport = bindings._ProtoEngine()
        # The public default cost sort is an ambient engine builtin. Creation
        # references it directly; authored Programs still send full declarations.
        request = pb.CreateEGraphRequest(
            sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
            options=pb.EGraphOptions(cost_sort=0),
            threads=num_threads,
        )
        response = pb.CreateEGraphResponse.from_binary(self.transport.create(request.to_binary()))
        self.handle: int | None = response.egraph_id
        self.definitions = dict(builtin_catalog().definitions)
        # Language bindings index canonical records, never copies of rule bodies.
        self.rulesets: dict[Ident, Ref] = {}
        self._combined_sources: dict[Ident, Ref] = {}
        self._ruleset_identifiers: dict[Ref, Ident] = {}
        self._direct_rules: set[Ref] = set()
        self._pending_rules: dict[tuple[Ident, Ref, Ref | None], tuple[Ref, Ref]] = {}
        self._pending_roots: dict[tuple[Ident, str, tuple[Ref, ...]], Ref] = {}

    def configure_resources(self, num_threads: object = _QUERY_RESOURCES) -> int:
        """Query/update this handle over bytes and return the receiver's actual count."""
        if self.handle is None:
            msg = "EGraph handle has been destroyed"
            raise RuntimeError(msg)
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
        return response.threads

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
        encoded, host_error = self.transport.run(request.to_binary())
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
            builder = Builder.from_program(
                pb.Program(
                    ir_version=1,
                    nodes=response.nodes,
                    sorts=response.sorts,
                    files=response.files,
                )
            )
            for reference in self.definitions.values():
                builder.import_ref(reference)
            owner = builder.publish()
        except BaseException as error:
            if host_error is not None and error is not host_error:
                raise error from host_error
            raise
        # The actual response/arena and publication have been consumed. Preserve
        # first-host precedence without rewriting its chain or the native Error.
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
                    if not direct and any(old not in self._direct_rules and rule_equal(rule, old) for old in members):
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
        response = pb.CloneEGraphResponse.from_binary(
            self.transport.clone_graph(pb.CloneEGraphRequest(egraph_id=self.handle).to_binary())
        )
        result = object.__new__(EGraphState)
        result.transport = self.transport
        result.handle = response.egraph_id
        result.definitions = self.definitions.copy()
        result.rulesets = self.rulesets.copy()
        result._combined_sources = self._combined_sources.copy()
        result._ruleset_identifiers = self._ruleset_identifiers.copy()
        result._direct_rules = self._direct_rules.copy()
        result._pending_rules = self._pending_rules.copy()
        result._pending_roots = self._pending_roots.copy()
        return result

    def return_to(self, saved: EGraphState) -> None:
        """Restore a direct saved owner; successful Return consumes this handle."""
        if self.handle is None or saved.handle is None:
            msg = "Return requires two live EGraph handles"
            raise RuntimeError(msg)
        if self.transport is not saved.transport or self.handle == saved.handle:
            msg = "Return requires distinct handles on the same transport"
            raise ValueError(msg)
        encoded = self.transport.return_to(
            pb.ReturnToEGraphRequest(active_id=self.handle, saved_id=saved.handle).to_binary()
        )
        # Successful transport has consumed the active handle. Even a corrupt
        # acknowledgement must not leave ownership that could Destroy it twice.
        # Public pop selects the saved state only after decoding succeeds.
        self.handle = None
        pb.ReturnToEGraphResponse.from_binary(encoded)

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
