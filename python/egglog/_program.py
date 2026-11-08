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
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakKeyDictionary

from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import DescFieldValueList, DescFieldValueMessage, DescMessage, Message, ScalarType

if TYPE_CHECKING:
    from ._catalog import Catalog
    from ._presentation import Presentation

# Every Program-reachable field is deliberately listed. ':' means an arena
# address; '@' means a named definition dependency. Other fields are copied,
# recursively for inline messages. Binder positions are deliberately NOT indices.
_FIELDS = {
    "Program": "nodes sorts constructors functions relations user_defined_primitives commands rulesets",
    "Node": "sort_id:sorts var binding primitive_value call apply eq get_cost",
    "BuiltinCatalog": "sorts primitive_sorts primitives",
    "Sort": "eq primitive func var doc",
    "PrimitiveSortRef": "name@primitive_sort args:sorts",
    "FuncSort": "params:sorts result:sorts",
    "Eq": "members:nodes",
    "Call": "func@callable args:nodes",
    "Apply": "function:nodes args:nodes",
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
    "Action": "term:nodes set delete subsume panic set_cost",
    "Set": "target value:nodes",
    "SetCost": "target cost:nodes",
    "Constructor": "name inputs:sorts output:sorts cost:nodes unextractable hidden doc public_alias default_name",
    "Function": "name inputs:sorts output:sorts merge:nodes hidden doc public_alias default_name",
    "Relation": "name inputs:sorts doc public_alias default_name",
    "UserDefinedPrimitive": "name inputs:sorts output:sorts body:nodes doc",
    "PrimitiveSort": "name arity doc",
    "Primitive": "name signature doc",
    "GenericSignature": "type_param_count inputs:sorts output:sorts varargs:sorts",
    "Rule": "rule rewrite birewrite name eval_mode no_decomp include_subsumed doc",
    "RuleBody": "query:nodes actions",
    "Rewrite": "lhs:nodes rhs:nodes conditions:nodes subsume",
    "BiRewrite": "lhs:nodes rhs:nodes conditions:nodes",
    "Command": "action let check prove prove_exists run bind_scheduler keep_best extract freeze table_sizes table_rows table_stats loop fail stats explain",
    "Let": "binding:nodes value:nodes",
    "Check": "facts:nodes",
    "Prove": "facts:nodes",
    "ProveExists": "constructor@callable",
    "Run": "rulesets scheduler",
    "Ruleset": "rules",
    "RulesetBinding": "name value",
    "RulesetSelection": "ruleset name",
    "Explain": "groups",
    "BindScheduler": "name back_off",
    "BackOff": "match_limit ban_length node_limit",
    "KeepBest": "tables@callable extractor cost_model",
    "Extract": "roots:nodes variants extractor cost_model",
    "Freeze": "",
    "GetTableSizes": "tables@callable",
    "GetTableRows": "table@callable max_rows",
    "GetTableStats": "tables@callable",
    "Loop": "body until:nodes max_iterations",
}
_ROLES: dict[str, type[Message]] = {
    "nodes": pb.Node,
    "sorts": pb.Sort,
    "constructors": pb.Constructor,
    "functions": pb.Function,
    "relations": pb.Relation,
    "user_defined_primitives": pb.UserDefinedPrimitive,
    "rules": pb.Rule,
    "rulesets": pb.RulesetBinding,
    "commands": pb.Command,
    "primitive_sorts": pb.PrimitiveSort,
    "primitives": pb.Primitive,
}
_PROGRAM_ROLES = (
    "nodes",
    "sorts",
    "constructors",
    "functions",
    "relations",
    "user_defined_primitives",
    "rulesets",
    "commands",
)
_BUILTIN_ROLES = ("sorts", "primitive_sorts", "primitives")
_FIELD_ROLES = {
    name: dict(token.replace("@", ":@").partition(":")[::2] for token in fields.split())
    for name, fields in _FIELDS.items()
}


def _check_inventory() -> None:
    """Fail on schema drift instead of guessing which new integers are addresses."""
    pending = [pb.Program.desc(), pb.BuiltinCatalog.desc()]
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


def _structural_parts(
    reference: Ref, *, identity: bool = True, exact_literals: bool = False
) -> tuple[tuple[object, ...], list[tuple[Ref, str]]]:
    """Read comparison data directly; only arena addresses become child refs."""
    ref = reference.owner.ref(reference.role, reference.index)
    record = cast("Message", ref.owner._slots[ref.role][ref.index])
    if identity and (
        ref.role == "rules" or (isinstance(record, pb.Node) and record.kind is not None and record.kind.field == "eq")
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
            if (isinstance(message, pb.Sort) and field.name == "doc") or (
                field.presence.name != "IMPLICIT" and field not in message
            ):
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
                    children.append((ref.owner.ref(role, item), f"{path}/{field.name}/{index}"))
                    shape.append(("ref", role))
                elif isinstance(item, Message):
                    inline.append((item, f"{path}/{field.name}/{index}"))
                elif isinstance(message, pb.PrimitiveValue) and field.name == "f64_bits" and not exact_literals:
                    number = unpack("!d", item.to_bytes(8, "big"))[0]
                    shape.append(("nan", item) if isnan(number) else number)
                else:
                    shape.append(item)
        pending.extend(reversed(inline))
    return tuple(shape), children


def _structural_equal(
    left: Ref, right: Ref, *, exact_literals: bool = False, owners: list[Owner] | None = None
) -> bool:
    """Compare relocated structure without conflating identity-bearing nodes."""
    pending = [(left, right)]
    seen = set()
    while pending:
        left, right = pending.pop()
        left = left.owner.ref(left.role, left.index)
        right = right.owner.ref(right.role, right.index)
        if owners is not None:
            owners.extend((left.owner, right.owner))
        if left == right or (left, right) in seen:
            continue
        seen.add((left, right))
        if exact_literals:
            left_shape, left_children = _structural_parts(left, exact_literals=True)
            right_shape, right_children = _structural_parts(right, exact_literals=True)
        else:
            left_shape, left_children = _structural_parts(left)
            right_shape, right_children = _structural_parts(right)
        if left_shape != right_shape or len(left_children) != len(right_children):
            return False
        pending.extend((left[0], right[0]) for left, right in zip(left_children, right_children, strict=True))
    return True


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
                    cache[key] = hash((shape, tuple(_STRUCTURAL_HASHES[c.owner][c.role, c.index] for c, _ in children)))
                    continue
                if ref in active:
                    msg = "Cyclic protobuf structure without an identity-bearing node"
                    raise ValueError(msg)
                active.add(ref)
                pending.append((ref, True))
                pending.extend((child, False) for child, _ in reversed(children))
            return _STRUCTURAL_HASHES[root.owner][root.role, root.index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StructuralView):
            return NotImplemented
        return _structural_equal(self.ref, other.ref)


def action_equal(left: Ref, right: Ref) -> bool:  # noqa: C901
    """
    Prove ordered Action syntax equality without identifying execution nodes.

    Exact refs are axioms; empty Unions and rule identities have no
    structural proof. A repeated active distinct pair has no finite proof.
    Only completed proofs are reusable, including in the named closure.
    """
    if left.role != "commands" or right.role != "commands":
        msg = "Action comparison requires command references"
        raise TypeError(msg)
    active: set[tuple[Ref, Ref]] = set()
    completed: set[tuple[Ref, Ref]] = set()
    for closure in (False, True):
        if closure:
            definitions = [
                {_definition_key(ref.read()): ref for ref in pack(commands=[root]).definitions}
                for root in (left, right)
            ]
            if definitions[0].keys() != definitions[1].keys():
                return False
            pending = [(definitions[0][key], definitions[1][key], False) for key in definitions[0]]
        else:
            pending = [(left, right, False)]
        while pending:
            first, second, leaving = pending.pop()
            first = first.owner.ref(first.role, first.index)
            second = second.owner.ref(second.role, second.index)
            pair = first, second
            if leaving:
                active.remove(pair)
                completed.add(pair)
                continue
            if first == second or pair in completed:
                continue
            if pair in active:
                return False
            for ref in pair:
                record = ref.owner._slots[ref.role][ref.index]
                if ref.role == "rules" or (
                    isinstance(record, pb.Node)
                    and record.kind is not None
                    and record.kind.field == "eq"
                    and not record.kind.value.members
                ):
                    return False
            first_shape, first_children = _structural_parts(first, identity=False)
            second_shape, second_children = _structural_parts(second, identity=False)
            if first_shape != second_shape or len(first_children) != len(second_children):
                return False
            active.add(pair)
            pending.append((first, second, True))
            pending.extend(
                (a[0], b[0], False) for a, b in zip(reversed(first_children), reversed(second_children), strict=True)
            )
    return True


def _rule_key(root: Ref) -> tuple[object, ...]:  # noqa: C901
    """
    Derive comparison/hash tokens, preserving allocation topology per use unit.

    Rule addresses remain occurrence identities everywhere else. Query Eq
    syntax is nonallocating; action/deferred-code Eq nodes need a bijection, not
    absolute owner indices. Named closure is exactly packing's dependency closure.
    """
    if root.role != "rules":
        msg = "Rule comparison requires rule references"
        raise TypeError(msg)
    builder = Builder()
    binding = builder.add("rulesets", pb.RulesetBinding(value=pb.Ruleset(rules=[builder.import_rule(root)])))
    closure = {
        _definition_key(ref.read()): ref for ref in pack([builder.publish().ref("rulesets", binding)]).definitions
    }
    ordered = sorted(closure, key=repr)
    roots: list[tuple[Ref, object, bool]] = [(closure[key], key, False) for key in ordered] + [(root, "rule", False)]
    roots = [(ref.owner.ref(ref.role, ref.index), domain, query) for ref, domain, query in roots]
    pending: list[tuple[Ref, object, bool, bool]] = [(*visit, False) for visit in reversed(roots)]
    active: dict[tuple[Ref, object, bool], tuple[tuple[object, ...], list[tuple[Ref, object, bool]]]] = {}
    completed: dict[tuple[Ref, object, bool], int] = {}
    tokens: dict[tuple[object, ...], int] = {}
    unions: dict[object, dict[Ref, int]] = {}
    while pending:
        reference, domain, query, leaving = pending.pop()
        reference = reference.owner.ref(reference.role, reference.index)
        visit = reference, domain, query
        if leaving:
            shape, children_visits = active.pop(visit)
            token = shape, tuple(completed[child] for child in children_visits)
            completed[visit] = tokens.setdefault(token, len(tokens))
            continue
        if visit in completed:
            continue
        if visit in active:
            msg = "Cyclic rule template syntax"
            raise ValueError(msg)
        shape, children = _structural_parts(reference, identity=False)
        record = reference.read()
        if isinstance(record, pb.Node) and record.kind is not None and record.kind.field == "eq" and not query:
            allocations = unions.setdefault(domain, {})
            ordinal = allocations.setdefault(reference, len(allocations))
            shape = ("allocation", ordinal, shape)
        children_visits = []
        active[visit] = shape, children_visits
        pending.append((reference, domain, query, True))
        for child, path in children:
            child_domain, child_query = domain, query
            if reference.role == "rules":
                if path.startswith("/birewrite/"):
                    # Each direction has its own action allocation domain;
                    # neither its query nor the other direction seeds it.
                    for direction, lhs in (("forward", "lhs"), ("reverse", "rhs")):
                        child_query = path.startswith((f"/birewrite/0/{lhs}/", "/birewrite/0/conditions/"))
                        children_visits.append((
                            child.owner.ref(child.role, child.index),
                            (direction, "query" if child_query else "actions"),
                            child_query,
                        ))
                    continue
                child_query = path.startswith(("/rule/0/query/", "/rewrite/0/lhs/", "/rewrite/0/conditions/"))
                child_domain = "query" if child_query else "actions"
            elif reference.role in {"constructors", "functions", "relations", "user_defined_primitives"}:
                # User primitive, merge and cost roots have independent binders.
                child_domain = (reference, path)
                child_query = False
            elif "/lambda/0/body/" in path:
                child_domain, child_query = object(), False
            children_visits.append((child.owner.ref(child.role, child.index), child_domain, child_query))
        pending.extend((*child, False) for child in reversed(children_visits))
    # Interned child IDs erase pure DAG sharing without expanding its paths.
    # Insertion order is canonical postorder, not any owner's arena numbering.
    return tuple(ordered), tuple(completed[visit] for visit in roots), tuple(tokens)


def rule_equal(left: Ref, right: Ref, *, keys: dict[Ref, tuple[object, ...]] | None = None) -> bool:
    """Compare templates, optionally retaining completed keys in a caller-owned memo."""
    if keys is None:
        return _rule_key(left) == _rule_key(right)
    left = left.owner.ref(left.role, left.index)
    right = right.owner.ref(right.role, right.index)
    for reference in (left, right):
        if reference not in keys:
            keys[reference] = _rule_key(reference)
    return keys[left] == keys[right]


class Owner:
    """Published protobuf records; imported slots resolve directly to their owner."""

    __slots__ = (
        "__weakref__",
        "_builtin_indices",
        "_definition_indices",
        "_environment",
        "_keepalive",
        "_presentation_catalog",
        "_presentations",
        "_slots",
    )

    def __init__(
        self,
        slots: Mapping[str, tuple[Message | Ref, ...]],
        keepalive: tuple[object, ...] = (),
        presentations: Mapping[tuple[str, str], Presentation] | None = None,
        environment: tuple[Owner, ...] = (),
    ) -> None:
        # Internal transfer from Builder.publish; caller-owned messages must
        # enter through Builder.add/fill/from_program so they are detached.
        self._slots = MappingProxyType(dict(slots))
        # Language-local callbacks/lifetimes are never semantic edges or wire data.
        self._keepalive = tuple(keepalive)
        self._presentations = MappingProxyType(dict(presentations or {}))
        # The frontend index is derived once from this immutable owned context.
        # Keeping it here avoids both global retention and repeated catalog scans.
        self._presentation_catalog: Catalog | None = None
        self._definition_indices: Mapping[tuple[str, str], tuple[tuple[str, int], ...]] | None = None
        self._environment = environment
        self._builtin_indices = MappingProxyType({
            _definition_key(record): (role, index)
            for role in ("primitive_sorts", "primitives")
            for index, record in enumerate(self._slots[role])
            if isinstance(record, Message)
        })

    @property
    def definition_indices(self) -> Mapping[tuple[str, str], tuple[tuple[str, int], ...]]:
        """Index immediate slots once, without copying transitive contexts or retaining self refs."""
        if self._definition_indices is None:
            indices: dict[tuple[str, str], list[tuple[str, int]]] = {}
            for role in ("sorts", "nodes", "constructors", "functions", "relations", "user_defined_primitives"):
                for index in range(len(self._slots[role])):
                    ref = self.ref(role, index)
                    key = _definition_key(cast("Message", ref.owner._slots[ref.role][ref.index]))
                    if key is not None:
                        indices.setdefault(key, []).append((role, index))
            self._definition_indices = MappingProxyType({key: tuple(slots) for key, slots in indices.items()})
        return self._definition_indices

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

    def __init__(self, *, environment: Iterable[Owner] = ()) -> None:
        self._slots: dict[str, list[Message | Ref | None]] | None = {role: [] for role in _ROLES}
        self._imports: dict[Ref, int] = {}
        self._contexts: set[Owner] = set()
        self.presentations: dict[tuple[str, str], Presentation] = {}
        self._environment = dict.fromkeys(
            provider for owner in environment for provider in (*owner._environment, owner)
        )

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
        self._environment.update(dict.fromkeys(canonical.owner._environment))
        if canonical.owner._builtin_indices:
            self._environment[canonical.owner] = None
        if canonical in self._imports:
            return self._imports[canonical]
        index = len(self._slots[canonical.role])
        self._slots[canonical.role].append(canonical)
        self._imports[canonical] = index
        return index

    def import_context(self, owner: Owner) -> None:
        """Retain named references when embedding code without an arena-owner edge."""
        if owner in self._contexts:
            return
        self._contexts.add(owner)
        self._environment.update(dict.fromkeys(owner._environment))
        if owner._builtin_indices:
            self._environment[owner] = None
        for slots in owner.definition_indices.values():
            for role, index in slots:
                self.import_ref(owner.ref(role, index))

    def import_rule(self, reference: Ref) -> pb.Rule:
        """Embed an owned reusable rule, relocating its closure into this fragment."""
        canonical = reference.owner.ref(reference.role, reference.index)
        if canonical.role != "rules":
            msg = "Inline rules require an owned Rule reference"
            raise TypeError(msg)
        self.import_context(canonical.owner)
        record = canonical.owner._slots[canonical.role][canonical.index]
        assert isinstance(record, pb.Rule)
        return _copy_record(record, relocate=lambda role, index: self.import_ref(canonical.owner.ref(role, index)))

    def publish(
        self,
        *,
        keepalive: tuple[object, ...] = (),
        presentation_factory: Callable[[Owner], Mapping[tuple[str, str], Presentation]] | None = None,
    ) -> Owner:
        if self._slots is None:
            msg = "Builder already published"
            raise RuntimeError(msg)
        if any(record is None for records in self._slots.values() for record in records):
            msg = "Cannot publish unfilled reserved slots"
            raise ValueError(msg)
        slots = {role: tuple(cast("list[Message | Ref]", records)) for role, records in self._slots.items()}
        self._slots = None
        self._imports.clear()
        self._contexts.clear()
        owner = Owner(slots, keepalive, self.presentations, tuple(self._environment))
        # Combining provider contexts must not substitute a different signature
        # for the same canonical key. Keep the original generated records owned.
        definitions: dict[tuple[str, str], Ref] = {}
        for provider in owner._environment:
            for key, (role, index) in provider._builtin_indices.items():
                assert key is not None
                reference = provider.ref(role, index)
                if key in definitions and not _structural_equal(definitions[key], reference, exact_literals=True):
                    raise ValueError(f"Conflicting builtin environments: {key}")
                definitions[key] = reference
        if presentation_factory is not None:
            # Resolve circular local default/context references before the new
            # owner is exposed. Semantic records are already immutable.
            owner._presentations = MappingProxyType(dict(presentation_factory(owner)))
        return owner

    @classmethod
    def from_program(cls, program: pb.Program, *, environment: Iterable[Owner] = ()) -> Builder:
        """Adopt a decoded Program as a fresh owner, preserving its local indices."""
        if program._unknown_fields:
            msg = "Cannot adopt unknown Program fields"
            raise NotImplementedError(msg)
        builder = cls(environment=environment)
        for role in _PROGRAM_ROLES:
            for record in getattr(program, role):
                builder.add(role, record)
        return builder

    @classmethod
    def from_builtins(cls, catalog: pb.BuiltinCatalog) -> Builder:
        """Own backend descriptors separately from executable Program records."""
        if catalog._unknown_fields:
            msg = "Cannot adopt unknown BuiltinCatalog fields"
            raise NotImplementedError(msg)
        builder = cls()
        for role in _BUILTIN_ROLES:
            for record in getattr(catalog, role):
                builder.add(role, record)
        return builder


@dataclass(frozen=True, slots=True)
class Packed:
    program: pb.Program
    indices: tuple[int, ...]
    # Canonical source declarations let response owners retain the same
    # presentations without manufacturing another semantic declaration copy.
    definitions: tuple[Ref, ...]
    rulesets: tuple[Ref, ...] = ()


def clone_nodes(roots: Iterable[Ref]) -> tuple[Ref, ...]:  # noqa: C901
    """
    Expand templates with fresh node identities and shared canonical definitions.

    All roots share one relocation map. Sorts and declarations are
    immutable imports; only reachable nodes are copied. This does not evaluate
    defaults or merge the independent binders of lambda bodies and captures.
    """
    builder = Builder()
    indices: dict[Ref, int] = {}
    pending: list[Ref] = []

    def allocate(source: Ref) -> int:
        source = source.owner.ref(source.role, source.index)
        if source.role != "nodes" or _definition_key(source.read()) is not None:
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
            matches = dict.fromkeys(
                origin.ref(role, index) for role, index in origin.definition_indices.get((namespace, name), ())
            )
            role = "primitives" if namespace == "callable" else "primitive_sorts"
            for index in range(len(origin._slots[role])):
                reference = origin.ref(role, index)
                if _definition_key(cast("Message", reference.owner._slots[role][reference.index])) == (namespace, name):
                    matches[reference] = None
            for environment in (*origin._environment, origin):
                if (slot := environment._builtin_indices.get((namespace, name))) is not None:
                    matches[environment.ref(*slot)] = None
            first = next(iter(matches), None)
            if first is None or any(
                not _structural_equal(first, other, exact_literals=True) for other in matches if other != first
            ):
                raise ValueError(f"Template requires one canonical definition for {namespace} {name!r}")
            builder.import_ref(first)

        def relocate(role: str, index: int, origin: Owner = source.owner) -> int:
            return allocate(origin.ref(role, index))

        record = _copy_record(source.read(), relocate=relocate, dependency=dependency)
        builder.fill("nodes", indices[source], record)
    owner = builder.publish()
    return tuple(owner.ref("nodes", index) for index in root_indices)


def _definition_key(record: Message) -> tuple[str, str] | None:
    """Classify named setup records without conflating absent and empty names."""
    if isinstance(record, pb.PrimitiveSort):
        return "primitive_sort", record.name
    if isinstance(record, pb.Primitive):
        return "callable", record.name
    if isinstance(record, pb.Sort) and record.kind is not None and record.kind.field == "eq":
        return "eq_sort", record.kind.value
    if isinstance(record, pb.Node) and record.kind is not None and record.kind.field == "binding":
        return "binding", record.kind.value
    if isinstance(record, pb.Constructor | pb.Function | pb.Relation | pb.UserDefinedPrimitive):
        return "callable", record.name
    return None


def pack(  # noqa: C901, PLR0912
    roots: Iterable[Ref] = (),
    *,
    commands: Iterable[Ref] = (),
    definitions: Iterable[Ref] = (),
    rulesets: Iterable[Ref] = (),
    ambient: Collection[tuple[str, str]] = (),
) -> Packed:
    """
    Pack reachable records once; command roots retain order and multiplicity.

    Definitions supplied here are available by name, not automatic emission
    roots. Missing names require an explicit ambient declaration/provider set.
    Exact relocated definitions share one emission; differing semantic code
    still requires adapter-aware reconciliation. Literal bits and identity-bearing
    Eq/rule references are never guessed equivalent.
    """
    program = pb.Program()
    mapped: dict[Ref, int] = {}
    pending: list[Ref] = []
    indexed: set[Owner] = set()
    candidates: dict[tuple[str, str], set[Ref]] = {}
    demands: set[tuple[str, str]] = set()
    emitted: dict[tuple[str, str], Ref] = {}
    builtin_keys: set[tuple[str, str]] = set()
    ruleset_names: dict[str, Ref] = {}

    def index_owner(owner: Owner) -> None:
        if owner in indexed:
            return
        indexed.add(owner)
        for provider in (*owner._environment, owner):
            builtin_keys.update(key for key in provider._builtin_indices if key is not None)
            for key, slots in provider.definition_indices.items():
                for role, index in slots:
                    ref = provider.ref(role, index)
                    matches = candidates.setdefault(key, set())
                    if ref not in matches:
                        matches.add(ref)
                        if key in emitted:
                            demands.add(key)

    def reconcile(key: tuple[str, str], preferred: Ref) -> Ref:
        # Keep original candidates until their reachable owner closures have
        # been inspected; a discarded body can import conflicting definitions.
        matches = candidates[key]
        reference = emitted.get(key) or next(
            (old for old in matches if old in mapped),
            next((old for old in matches if key in old.owner._presentations), preferred),
        )
        while True:
            count = len(matches)
            owners: list[Owner] = []
            for old in tuple(matches):
                if not _structural_equal(reference, old, exact_literals=True, owners=owners):
                    raise ValueError(f"Ambiguous same-name definitions require reconciliation: {key}")
            for owner in owners:
                index_owner(owner)
            if len(matches) == count:
                return reference

    def allocate(reference: Ref) -> int:
        ref = reference.owner.ref(reference.role, reference.index)
        if ref.role == "rules":
            msg = "Rules must be embedded in Ruleset collections"
            raise TypeError(msg)
        if ref.role in {"primitive_sorts", "primitives"}:
            msg = "Builtin descriptors are ambient references, not Program roots"
            raise TypeError(msg)
        if ref.role == "commands":
            msg = "Commands must be passed as ordered command roots"
            raise ValueError(msg)
        if ref not in mapped:
            # Collections are setup records; member addresses relocate through
            # the same owned arenas as command syntax.
            if ref.role == "rulesets":
                record = cast("pb.RulesetBinding", ref.read())
                previous = ruleset_names.get(record.name)
                if previous is not None:
                    if not _structural_equal(previous, ref, exact_literals=True):
                        raise ValueError(f"Conflicting ruleset bindings: {record.name!r}")
                    mapped[ref] = mapped[previous]
                    return mapped[ref]
                ruleset_names[record.name] = ref
                mapped[ref] = len(program.rulesets)
                program.rulesets.append(pb.RulesetBinding())
                pending.append(ref)
                index_owner(ref.owner)
                return mapped[ref]
            index_owner(ref.owner)
            key = _definition_key(cast("Message", ref.owner._slots[ref.role][ref.index]))
            if key is not None:
                ref = reconcile(key, ref)
                if ref in mapped:
                    return mapped[ref]
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
        if ref.role in {"primitive_sorts", "primitives"}:
            index_owner(ref.owner)
            continue
        if _definition_key(ref.read()) is None:
            msg = "Definitions must refer to named setup records"
            raise TypeError(msg)
        index_owner(ref.owner)
    for reference in rulesets:
        if reference.role != "rulesets":
            msg = "Expected a ruleset binding reference"
            raise TypeError(msg)
        allocate(reference)
    indices = tuple(allocate(ref) for ref in roots)
    for reference in commands:
        ref = reference.owner.ref(reference.role, reference.index)
        if ref.role != "commands":
            msg = "Expected a command reference"
            raise TypeError(msg)
        index_owner(ref.owner)
        for index in range(len(ref.owner._slots["rulesets"])):
            allocate(ref.owner.ref("rulesets", index))
        program.commands.append(cast("pb.Command", rewrite(ref)))

    while True:
        while pending:
            ref = pending.pop()
            rewritten = rewrite(ref)
            getattr(program, ref.role)[mapped[ref]] = rewritten
            key = _definition_key(rewritten)
            if key is not None:
                reconcile(key, ref)
                emitted[key] = ref
        for key in sorted(demands):
            matches = candidates.get(key, set())
            if matches:
                allocate(reconcile(key, next(iter(matches))))
                demands.remove(key)
        if not pending:
            if any(candidates.get(key) for key in demands):
                continue
            missing = demands - set(ambient) - builtin_keys
            if missing:
                raise ValueError(f"missing definitions: {sorted(missing)}")
            break
    return Packed(program, indices, tuple(emitted.values()), tuple(ruleset_names.values()))


_check_inventory()
