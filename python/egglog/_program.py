"""
Private ownership and relocation for directly authored protobuf records.

Fragments have local address slots, some importing records from other owners.
Only generated message fields select or order edges. Published records are never
mutated; inspection and packing return detached copies. This module does not
typecheck, evaluate, expand defaults, rename binders, or install providers.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from math import isnan
from struct import unpack
from threading import RLock
from types import MappingProxyType
from typing import Any, cast
from weakref import WeakKeyDictionary

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import DescFieldValueList, DescFieldValueMessage, DescMessage, Message, ScalarType

# Every Program-reachable field is deliberately listed. ':' means an arena
# address; '@' means a named definition dependency. Other fields are copied,
# recursively for inline messages. Binder positions are deliberately NOT indices.
_FIELDS = {
    "Program": "ir_version nodes sorts declarations commands files rulesets rules",
    "Node": "sort_id:sorts var primitive_value call union get_cost span",
    "Sort": "eq@sort family func var span",
    "HostSort": "name@sort args:sorts",
    "FuncSort": "params:sorts result:sorts",
    "Span": "file:files start end",
    "SourceFile": "name contents",
    "Union": "members:nodes",
    "Call": "func@callable args:nodes",
    "Lambda": "captures:nodes body:nodes",
    "PrimitiveValue": "i64 f64_bits string bool unit big_int big_rat rational vec set multiset map pair maybe lambda partial_call custom",
    "BigRat": "numerator denominator",
    "Rational": "numerator denominator",
    "ValueList": "items:nodes",
    "MapValue": "entries",
    "MapEntry": "key:nodes value:nodes",
    "PairValue": "first:nodes second:nodes",
    "MaybeValue": "value:nodes",
    "CustomValue": "payload args:nodes",
    "Unit": "",
    "Action": "term:nodes set delete subsume panic set_cost span",
    "Set": "target value:nodes",
    "SetCost": "target cost:nodes",
    "Declaration": "constructor function relation primitive host_sort_family host_primitive eq_sort span bindings doc",
    "Arg": "sort:sorts name",
    "EqSort": "name bindings",
    "Constructor": "name inputs output:sorts cost:nodes unextractable",
    "Function": "name inputs output:sorts merge:nodes",
    "Relation": "name inputs",
    "Primitive": "name inputs output:sorts body:nodes",
    "HostSortFamily": "name arity bindings",
    "HostPrimitive": "name signature application",
    "GenericSignature": "type_params inputs output:sorts varargs",
    "FunctionApplication": "",
    "SortBindings": "python rust egglog",
    "TypeBinding": "path type_params",
    "EgglogTypeBinding": "symbol",
    "CallableBindings": "python rust egglog",
    "BindingOwner": "sort:sorts function",
    "FunctionTypeOwner": "",
    "PythonBindings": "views",
    "PythonCallable": "kind path owner receiver params mutates",
    "PythonParameter": "core_input name default_expr:nodes",
    "RustBindings": "views",
    "RustCallable": "path owner borrowed_self receiver params trait_impl",
    "RustReceiver": "core_input borrowed",
    "RustParameter": "core_input name borrowed",
    "RustTrait": "path args output_associated_type",
    "RustType": "sort:sorts borrowed",
    "EgglogBindings": "views",
    "EgglogCallable": "symbol datatype_member",
    "Ruleset": "name rules combined span doc",
    "RuleList": "rules:rules",
    "RulesetRef": "index:rulesets name@ruleset",
    "RuleDecl": "rule rewrite birewrite name eval_mode no_decomp include_subsumed span doc",
    "Rule": "query:nodes head",
    "Rewrite": "lhs:nodes rhs:nodes conditions:nodes subsume",
    "BiRewrite": "lhs:nodes rhs:nodes conditions:nodes",
    "CombinedRuleset": "rulesets",
    "Command": "action check prove prove_exists run bind_scheduler keep_best extract freeze print_size print_function print_table_stats repeat saturate span",
    "Check": "facts:nodes",
    "Prove": "facts:nodes",
    "ProveExists": "constructor@callable",
    "Run": "ruleset scheduler",
    "BindScheduler": "name scheduler",
    "Scheduler": "back_off",
    "BackOff": "match_limit ban_length node_limit",
    "KeepBest": "tables@callable extractor cost_model",
    "Extract": "roots:nodes variants extractor cost_model",
    "Freeze": "",
    "PrintSize": "tables@callable",
    "PrintFunction": "table@callable max_rows",
    "PrintTableStats": "tables@callable",
    "Repeat": "body until:nodes times",
    "Saturate": "body until:nodes max_iterations",
}
_ROLES: dict[str, type[Message]] = {
    "nodes": pb.Node,
    "sorts": pb.Sort,
    "declarations": pb.Declaration,
    "rules": pb.RuleDecl,
    "rulesets": pb.Ruleset,
    "files": pb.SourceFile,
    "commands": pb.Command,
}
_FIELD_ROLES = {
    name: dict(token.replace("@", ":@").partition(":")[::2] for token in fields.split())
    for name, fields in _FIELDS.items()
}


def _check_inventory() -> None:
    """Fail on schema drift instead of guessing which new integers are addresses."""
    pending = [pb.Program.desc()]
    seen: set[str] = set()
    while pending:
        descriptor = pending.pop()
        if descriptor.name in seen:
            continue
        seen.add(descriptor.name)
        expected = _FIELD_ROLES.get(descriptor.name)
        if expected is None or set(expected) != {field.name for field in descriptor.fields}:
            raise NotImplementedError(f"Unclassified protobuf fields in {descriptor.type_name}")
        for field in descriptor.fields:
            role = expected[field.name]
            if role and not role.startswith("@"):
                value = field.value
                scalar = value.element if isinstance(value, DescFieldValueList) else getattr(value, "scalar", None)
                if role not in _ROLES or scalar != ScalarType.UINT32:
                    raise TypeError(f"Invalid arena field classification: {descriptor.name}.{field.name}")
            if isinstance(field.value, DescFieldValueMessage):
                pending.append(field.value.message)
            elif isinstance(field.value, DescFieldValueList) and isinstance(field.value.element, DescMessage):
                pending.append(field.value.element)
    if seen != set(_FIELDS):
        raise NotImplementedError(f"Stale protobuf inventory entries: {set(_FIELDS) - seen}")


def _index_fields() -> list[tuple[type[Message], str, str]]:
    """Expose the audited address inventory for exhaustive relocation tests."""
    return [
        (getattr(pb, name), field, role)
        for name, fields in _FIELD_ROLES.items()
        for field, role in fields.items()
        if role and not role.startswith("@")
    ]


def _copy_record(  # noqa: C901, PLR0912
    record: Message,
    *,
    relocate: Callable[[str, int], int] | None = None,
    dependency: Callable[[str, str], None] | None = None,
) -> Any:
    """
    Detach inline messages iteratively, optionally relocating arena references.

    Inline message cycles cannot be serialized. Arena cycles remain ordinary
    integers and are handled by pack's allocation-before-traversal invariant.
    """
    result = type(record)()
    pending = [(record, result, False)]
    active: set[int] = set()
    while pending:
        source, target, leaving = pending.pop()
        if leaving:
            active.remove(id(source))
            continue
        if id(source) in active:
            msg = "protobuf inline message cycle"
            raise ValueError(msg)
        active.add(id(source))
        pending.append((source, target, True))
        descriptor = source.desc()
        fields = _FIELD_ROLES.get(descriptor.name)
        if descriptor.type_name != f"egglog.v1.{descriptor.name}" or fields is None:
            raise NotImplementedError(f"Unsupported protobuf record {descriptor.type_name}")
        # Unknown fields might contain indices. Preserving them without knowing
        # their role would silently emit dangling addresses after relocation.
        if source._unknown_fields:
            msg = "Cannot relocate unknown protobuf fields"
            raise NotImplementedError(msg)
        children = []
        for field in descriptor.fields:
            if field.name not in fields:
                raise NotImplementedError(f"Unclassified protobuf field {descriptor.name}.{field.name}")
            if field.presence.name != "IMPLICIT" and field not in source:
                continue
            value = source[field]
            if value is None:
                continue
            repeated = isinstance(field.value, DescFieldValueList)
            values = value if repeated else [value]
            transformed = []
            role = fields[field.name]
            for item in values:
                if not isinstance(item, Message | str | int | float | bool | bytes):
                    raise TypeError(f"Unsupported mutable/non-scalar field {descriptor.name}.{field.name}")
                replacement = item
                if role.startswith("@"):
                    if not isinstance(item, str):
                        raise TypeError(f"Invalid definition name {descriptor.name}.{field.name}")
                    if dependency is not None:
                        dependency(role[1:], item)
                elif role:
                    if isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 2**32:
                        raise TypeError(f"Invalid arena index {descriptor.name}.{field.name}")
                    if relocate is not None:
                        replacement = relocate(role, item)
                elif isinstance(item, Message):
                    detached = type(item)()
                    children.append((item, detached, False))
                    replacement = detached
                transformed.append(replacement)
            target[field] = transformed if repeated else transformed[0]
        pending.extend(reversed(children))
    return result


@dataclass(frozen=True, slots=True)
class Ref:
    """An address, with no independent opcode, type, child list or binder."""

    owner: Owner
    role: str
    index: int

    def read(self) -> Any:
        """Return a detached message so inspection cannot mutate published data."""
        canonical = self.owner.ref(self.role, self.index)
        return _copy_record(cast("Message", canonical.owner._slots[canonical.role][canonical.index]))


_STRUCTURAL_HASHES: WeakKeyDictionary[Owner, dict[tuple[str, int], int]] = WeakKeyDictionary()
_STRUCTURAL_LOCK = RLock()


def _structural_parts(reference: Ref) -> tuple[tuple[object, ...], list[Ref]]:  # noqa: C901
    """Read comparison data directly; only arena addresses become child refs."""
    ref = reference.owner.ref(reference.role, reference.index)
    record = cast("Message", ref.owner._slots[ref.role][ref.index])
    if ref.role in {"rules", "rulesets"} or (
        isinstance(record, pb.Node) and record.kind is not None and record.kind.field == "union"
    ):
        return (ref.role, ref), []
    shape: list[object] = [ref.role]
    children = []
    pending: list[tuple[Message, str]] = [(record, "")]
    while pending:
        message, path = pending.pop()
        shape.extend((path, message.desc().type_name))
        inline = []
        for field in message.desc().fields:
            if field.name == "span" or (field.presence.name != "IMPLICIT" and field not in message):
                continue
            value = message[field]
            if value is None:
                continue
            repeated = isinstance(field.value, DescFieldValueList)
            values = value if repeated else [value]
            shape.extend((field.name, len(values)))
            role = _FIELD_ROLES[message.desc().name][field.name]
            for index, item in enumerate(values):
                if role and not role.startswith("@"):
                    children.append(ref.owner.ref(role, item))
                    shape.append(("ref", role))
                elif isinstance(item, Message):
                    inline.append((item, f"{path}/{field.name}/{index}"))
                elif isinstance(message, pb.PrimitiveValue) and field.name == "f64_bits":
                    number = unpack("!d", item.to_bytes(8, "big"))[0]
                    if isnan(number):
                        msg = "NaN structural comparison requires the pending Python identity decision"
                        raise NotImplementedError(msg)
                    shape.append(number)
                else:
                    shape.append(item)
        pending.extend(reversed(inline))
    return tuple(shape), children


@dataclass(frozen=True, slots=True, eq=False)
class StructuralView:
    """Read-only structural inspection; Ref itself retains address identity."""

    ref: Ref

    def __hash__(self) -> int:
        root = self.ref.owner.ref(self.ref.role, self.ref.index)
        with _STRUCTURAL_LOCK:
            pending = [(root, False)]
            active: set[Ref] = set()
            while pending:
                ref, leaving = pending.pop()
                cache = _STRUCTURAL_HASHES.setdefault(ref.owner, {})
                key = (ref.role, ref.index)
                if key in cache:
                    continue
                shape, children = _structural_parts(ref)
                if leaving:
                    active.remove(ref)
                    cache[key] = hash((shape, tuple(_STRUCTURAL_HASHES[c.owner][c.role, c.index] for c in children)))
                    continue
                if ref in active:
                    msg = "Cyclic protobuf structure without an identity-bearing node"
                    raise ValueError(msg)
                active.add(ref)
                pending.append((ref, True))
                pending.extend((child, False) for child in reversed(children))
            return _STRUCTURAL_HASHES[root.owner][root.role, root.index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructuralView):
            return NotImplemented
        pending = [(self.ref, other.ref)]
        seen = set()
        while pending:
            left, right = pending.pop()
            left = left.owner.ref(left.role, left.index)
            right = right.owner.ref(right.role, right.index)
            if left == right or (left, right) in seen:
                continue
            seen.add((left, right))
            left_shape, left_children = _structural_parts(left)
            right_shape, right_children = _structural_parts(right)
            if left_shape != right_shape or len(left_children) != len(right_children):
                return False
            pending.extend(zip(left_children, right_children, strict=True))
        return True


class Owner:
    """Published protobuf records; imported slots resolve directly to their owner."""

    __slots__ = ("__weakref__", "_keepalive", "_slots")

    def __init__(self, slots: Mapping[str, tuple[Message | Ref, ...]], keepalive: tuple[object, ...] = ()) -> None:
        # Internal transfer from Builder.publish; caller-owned messages must
        # enter through Builder.add/fill/from_program so they are detached.
        self._slots = MappingProxyType(dict(slots))
        # Language-local callbacks/lifetimes are never semantic edges or wire data.
        self._keepalive = tuple(keepalive)

    def ref(self, role: str, index: int) -> Ref:
        if (
            role not in self._slots
            or isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(self._slots[role])
        ):
            raise IndexError(f"Invalid {role} slot {index!r}")
        record = self._slots[role][index]
        return record if isinstance(record, Ref) else Ref(self, role, index)


class Builder:
    """Thread-confined construction; publication transfers its private records once."""

    def __init__(self) -> None:
        self._slots: dict[str, list[Message | Ref | None]] | None = {role: [] for role in _ROLES}
        self._imports: dict[Ref, int] = {}

    def add(self, role: str, record: Message) -> int:
        if self._slots is None:
            msg = "Builder already published"
            raise RuntimeError(msg)
        if role not in _ROLES or not isinstance(record, _ROLES[role]):
            raise TypeError(f"Wrong protobuf record for {role}")
        detached = _copy_record(record)
        index = len(self._slots[role])
        self._slots[role].append(detached)
        return index

    def reserve(self, role: str) -> int:
        if self._slots is None:
            msg = "Builder already published"
            raise RuntimeError(msg)
        if role not in _ROLES:
            raise TypeError(f"Unknown arena role {role}")
        index = len(self._slots[role])
        self._slots[role].append(None)
        return index

    def fill(self, role: str, index: int, record: Message) -> None:
        if self._slots is None:
            msg = "Builder already published"
            raise RuntimeError(msg)
        if role not in _ROLES or not isinstance(record, _ROLES[role]):
            raise TypeError(f"Wrong protobuf record for {role}")
        if not 0 <= index < len(self._slots[role]) or self._slots[role][index] is not None:
            msg = "Only an unfilled reserved slot can be filled"
            raise ValueError(msg)
        self._slots[role][index] = _copy_record(record)

    def import_ref(self, reference: Ref) -> int:
        if self._slots is None:
            msg = "Builder already published"
            raise RuntimeError(msg)
        canonical = reference.owner.ref(reference.role, reference.index)
        if canonical in self._imports:
            return self._imports[canonical]
        index = len(self._slots[canonical.role])
        self._slots[canonical.role].append(canonical)
        self._imports[canonical] = index
        return index

    def publish(self, *, keepalive: tuple[object, ...] = ()) -> Owner:
        if self._slots is None:
            msg = "Builder already published"
            raise RuntimeError(msg)
        if any(record is None for records in self._slots.values() for record in records):
            msg = "Cannot publish unfilled reserved slots"
            raise ValueError(msg)
        slots = {role: tuple(cast("list[Message | Ref]", records)) for role, records in self._slots.items()}
        self._slots = None
        self._imports.clear()
        return Owner(slots, keepalive)

    @classmethod
    def from_program(cls, program: pb.Program) -> Builder:
        """Adopt a decoded Program as a fresh owner, preserving its local indices."""
        if program.ir_version != 1:
            raise NotImplementedError(f"Unsupported IR version {program.ir_version}")
        if program._unknown_fields:
            msg = "Cannot adopt unknown Program fields"
            raise NotImplementedError(msg)
        builder = cls()
        for role in _ROLES:
            for record in getattr(program, role):
                builder.add(role, record)
        return builder


@dataclass(frozen=True, slots=True)
class Packed:
    program: pb.Program
    indices: tuple[int, ...]
    # Canonical source declarations let response owners retain the same
    # presentations without manufacturing another semantic declaration copy.
    definitions: tuple[Ref, ...]


def clone_nodes(roots: Iterable[Ref]) -> tuple[Ref, ...]:  # noqa: C901
    """
    Expand templates with fresh node identities and shared canonical definitions.

    All roots share one relocation map. Sorts, files and declarations are
    immutable imports; only reachable nodes are copied. This does not evaluate
    defaults or merge the independent binders of lambda bodies and captures.
    """
    builder = Builder()
    indices: dict[Ref, int] = {}
    pending: list[Ref] = []
    definitions: dict[Owner, dict[tuple[str, str], set[Ref]]] = {}

    def allocate(source: Ref) -> int:
        source = source.owner.ref(source.role, source.index)
        if source.role != "nodes":
            return builder.import_ref(source)
        if source not in indices:
            indices[source] = builder.reserve("nodes")
            pending.append(source)
        return indices[source]

    roots = tuple(roots)
    if any(root.role != "nodes" for root in roots):
        msg = "Template roots must be node references"
        raise TypeError(msg)
    root_indices = tuple(allocate(root) for root in roots)
    while pending:
        source = pending.pop()

        def dependency(namespace: str, name: str, origin: Owner = source.owner) -> None:
            if origin not in definitions:
                table: dict[tuple[str, str], set[Ref]] = {}
                for index in range(len(origin._slots["declarations"])):
                    ref = origin.ref("declarations", index)
                    key = _definition_key(ref.read())
                    if key is not None:
                        table.setdefault(key, set()).add(ref)
                definitions[origin] = table
            matches = definitions[origin].get((namespace, name), set())
            if len(matches) != 1:
                raise ValueError(f"Template requires one canonical definition for {namespace} {name!r}")
            builder.import_ref(next(iter(matches)))

        def relocate(role: str, index: int, origin: Owner = source.owner) -> int:
            return allocate(origin.ref(role, index))

        record = _copy_record(source.read(), relocate=relocate, dependency=dependency)
        builder.fill("nodes", indices[source], record)
    owner = builder.publish()
    return tuple(owner.ref("nodes", index) for index in root_indices)


def _definition_key(record: Message) -> tuple[str, str] | None:
    """Classify definition namespaces without conflating absent and empty names."""
    if isinstance(record, pb.Declaration):
        if record.kind is None:
            msg = "Declaration has no kind"
            raise ValueError(msg)
        namespace = "sort" if record.kind.field in {"eq_sort", "host_sort_family"} else "callable"
        return namespace, record.kind.value.name
    if isinstance(record, pb.Ruleset) and record.has_field("name"):
        return "ruleset", record.name
    return None


def pack(  # noqa: C901, PLR0912
    roots: Iterable[Ref] = (),
    *,
    commands: Iterable[Ref] = (),
    definitions: Iterable[Ref] = (),
    ambient: Collection[tuple[str, str]] = (),
) -> Packed:
    """
    Pack reachable records once; command roots retain order and multiplicity.

    Definitions supplied here are available by name, not automatic emission
    roots. Missing names require an explicit ambient declaration/provider set.
    Distinct same-name definitions are rejected rather than guessed compatible;
    semantic definition reconciliation belongs to a later adapter-aware layer.
    """
    program = pb.Program(ir_version=1)
    mapped: dict[Ref, int] = {}
    pending: list[Ref] = []
    indexed: set[Owner] = set()
    candidates: dict[tuple[str, str], set[Ref]] = {}
    demands: set[tuple[str, str]] = set()
    emitted: dict[tuple[str, str], Ref] = {}

    def index_owner(owner: Owner) -> None:
        if owner in indexed:
            return
        indexed.add(owner)
        for role in ("declarations", "rulesets"):
            for index in range(len(owner._slots[role])):
                ref = owner.ref(role, index)
                record = cast("Message", ref.owner._slots[ref.role][ref.index])
                key = _definition_key(record)
                if key is None:
                    continue
                candidates.setdefault(key, set()).add(ref)
                if key in emitted and len(candidates[key]) > 1:
                    raise ValueError(f"Ambiguous same-name definitions require reconciliation: {key}")

    def allocate(reference: Ref) -> int:
        ref = reference.owner.ref(reference.role, reference.index)
        if ref.role == "commands":
            msg = "Commands must be passed as ordered command roots"
            raise ValueError(msg)
        if ref not in mapped:
            index_owner(ref.owner)
            records = getattr(program, ref.role)
            mapped[ref] = len(records)
            records.append(_ROLES[ref.role]())
            pending.append(ref)
        return mapped[ref]

    def rewrite(ref: Ref) -> Message:
        source = cast("Message", ref.owner._slots[ref.role][ref.index])
        return _copy_record(
            source,
            relocate=lambda role, index: allocate(ref.owner.ref(role, index)),
            dependency=lambda namespace, name: demands.add((namespace, name)),
        )

    for reference in definitions:
        ref = reference.owner.ref(reference.role, reference.index)
        if ref.role not in {"declarations", "rulesets"}:
            msg = "Definitions must refer to declarations or rulesets"
            raise TypeError(msg)
        index_owner(ref.owner)
    indices = tuple(allocate(ref) for ref in roots)
    for reference in commands:
        ref = reference.owner.ref(reference.role, reference.index)
        if ref.role != "commands":
            msg = "Expected a command reference"
            raise TypeError(msg)
        index_owner(ref.owner)
        program.commands.append(cast("pb.Command", rewrite(ref)))

    while True:
        while pending:
            ref = pending.pop()
            rewritten = rewrite(ref)
            getattr(program, ref.role)[mapped[ref]] = rewritten
            key = _definition_key(rewritten)
            if key is not None:
                if len(candidates.get(key, ())) > 1 or (key in emitted and emitted[key] != ref):
                    raise ValueError(f"Distinct same-name definitions require reconciliation: {key}")
                emitted[key] = ref
        for key in sorted(demands):
            matches = candidates.get(key, set())
            if len(matches) > 1:
                raise ValueError(f"Ambiguous same-name definitions require reconciliation: {key}")
            if matches:
                allocate(next(iter(matches)))
                demands.remove(key)
        if not pending:
            missing = demands - set(ambient)
            if missing:
                raise ValueError(f"missing definitions: {sorted(missing)}")
            break
    return Packed(program, indices, tuple(emitted.values()))


_check_inventory()
