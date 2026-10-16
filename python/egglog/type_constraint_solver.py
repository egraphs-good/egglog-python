"""Solve generic signatures directly over canonical sort records."""

from __future__ import annotations

from collections.abc import Sequence

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Builder, Ref, StructuralView, _copy_record

__all__ = ["TypeConstraintError", "check_type_restrictions", "infer_sort", "substitute_sort"]


class TypeConstraintError(RuntimeError):
    """Typing error when trying to infer the return type."""


def check_type_restrictions(signature: pb.GenericSignature, bindings: dict[int, Ref]) -> None:
    """Validate binder-index metadata and its exact, fully inferred Eq substitution."""
    indices = signature.equality_type_params
    if list(indices) != sorted(set(indices)) or any(i >= len(signature.type_params) for i in indices):
        msg = "Invalid equality type parameter indices"
        raise TypeConstraintError(msg)
    for index in indices:
        sort: pb.Sort | None = bindings[index].read() if index in bindings else None
        if sort is None or sort.kind is None or sort.kind.field != "eq":
            raise TypeConstraintError(f"Type parameter {index} requires an equality sort")


def infer_call_bindings(
    declaration: Ref, arguments: Sequence[Ref], result: Ref
) -> tuple[frozenset[int], tuple[tuple[int, Ref], ...]]:
    """
    Strictly type stored call sorts, distinguishing input- from result-inferred binders.

    No values or converters participate. Ordinary deconstruction retains only
    result-inferred bindings; a concrete named Func also retains its input sorts.
    """
    record: pb.Declaration = declaration.read()
    kind = record.kind.value if record.kind is not None else None
    if isinstance(kind, pb.HostPrimitive) and kind.typing is not None and kind.typing.field == "signature":
        signature = kind.typing.value
        if not signature.has_field("output"):
            msg = "Stored call signature has no output"
            raise TypeConstraintError(msg)
        inputs, tail, output, size = signature.inputs, signature.varargs, signature.output, len(signature.type_params)
    elif isinstance(kind, pb.Function | pb.Constructor | pb.Primitive):
        signature = None
        inputs, tail, output, size = kind.inputs, [], kind.output, 0
    else:
        msg = "Stored call requires a canonical signature"
        raise TypeConstraintError(msg)
    count = len(arguments) - len(inputs)
    if count < 0 or (count % len(tail) if tail else count != 0):
        msg = "Stored call has invalid canonical argument count"
        raise TypeConstraintError(msg)
    bindings: dict[int, Ref] = {}
    for index, actual in enumerate(arguments):
        pattern = inputs[index] if index < len(inputs) else tail[(index - len(inputs)) % len(tail)]
        infer_sort(declaration.owner.ref("sorts", pattern.sort), actual, size, bindings)
    determined = frozenset(bindings)
    infer_sort(declaration.owner.ref("sorts", output), result, size, bindings)
    if len(bindings) != size:
        msg = "Not every generic parameter was determined by the stored call"
        raise TypeConstraintError(msg)
    if signature is not None:
        check_type_restrictions(signature, bindings)
    return determined, tuple(sorted(bindings.items()))


def infer_sort(pattern: Ref, actual: Ref, binder_size: int, bindings: dict[int, Ref]) -> None:  # noqa: C901, PLR0912
    """Unify canonical sort records transactionally within one signature binder."""
    pending = [(pattern, actual)]
    visited: set[tuple[Ref, Ref]] = set()
    staged = bindings.copy()
    while pending:
        expected, received = pending.pop()
        if expected.role != "sorts" or received.role != "sorts":
            msg = "Type inference requires sort references"
            raise TypeConstraintError(msg)
        expected = expected.owner.ref("sorts", expected.index)
        received = received.owner.ref("sorts", received.index)
        if (expected, received) in visited:
            continue
        visited.add((expected, received))
        lhs: pb.Sort = expected.read()
        rhs: pb.Sort = received.read()
        if lhs.kind is None or rhs.kind is None:
            msg = "Cannot infer an absent sort kind"
            raise TypeConstraintError(msg)
        if rhs.kind.field == "var":
            msg = "Actual argument sorts must be closed"
            raise TypeConstraintError(msg)
        if lhs.kind.field == "var":
            variable = lhs.kind.value
            if variable >= binder_size:
                raise TypeConstraintError(f"Sort variable {variable} is outside the signature binder")
            if variable in staged:
                if StructuralView(staged[variable]) != StructuralView(received):
                    raise TypeConstraintError(f"Inconsistent sort for type parameter {variable}")
            else:
                # Check the entire candidate, including nested function/family args.
                concrete = [received]
                seen: set[Ref] = set()
                while concrete:
                    child = concrete.pop()
                    if child in seen:
                        continue
                    seen.add(child)
                    record: pb.Sort = child.read()
                    if record.kind is None or record.kind.field == "var":
                        msg = "Actual argument sorts must be closed"
                        raise TypeConstraintError(msg)
                    match record.kind.field:
                        case "family":
                            indices = record.kind.value.args
                        case "func":
                            indices = [*record.kind.value.params, record.kind.value.result]
                        case "eq":
                            indices = []
                    concrete.extend(child.owner.ref("sorts", index) for index in indices)
                staged[variable] = received
            continue
        if lhs.kind.field != rhs.kind.field:
            raise TypeConstraintError(f"Expected {lhs.kind.field} sort, got {rhs.kind.field}")
        match lhs.kind.field:
            case "eq":
                if lhs.kind.value != rhs.kind.value:
                    raise TypeConstraintError(f"Expected {lhs.kind.value}, got {rhs.kind.value}")
                continue
            case "family":
                assert rhs.kind.field == "family"
                if lhs.kind.value.name != rhs.kind.value.name:
                    raise TypeConstraintError(f"Expected {lhs.kind.value.name}, got {rhs.kind.value.name}")
                expected_children = lhs.kind.value.args
                received_children = rhs.kind.value.args
            case "func":
                assert rhs.kind.field == "func"
                expected_children = [*lhs.kind.value.params, lhs.kind.value.result]
                received_children = [*rhs.kind.value.params, rhs.kind.value.result]
        if len(expected_children) != len(received_children):
            msg = "Sort arity mismatch"
            raise TypeConstraintError(msg)
        pending.extend(
            (expected.owner.ref("sorts", left), received.owner.ref("sorts", right))
            for left, right in zip(expected_children, received_children, strict=True)
        )
    bindings.clear()
    bindings.update(staged)


def substitute_sort(pattern: Ref, bindings: dict[int, Ref]) -> Ref:  # noqa: C901
    """Instantiate sort records iteratively, preserving closed imported records."""
    builder = Builder()
    resolved: dict[Ref, int] = {}
    active: set[Ref] = set()
    pending = [(pattern, False)]
    while pending:
        current, exiting = pending.pop()
        current = current.owner.ref(current.role, current.index)
        if current.role != "sorts":
            msg = "Type substitution requires sort references"
            raise TypeConstraintError(msg)
        if current in resolved:
            continue
        record: pb.Sort = current.read()
        if record.kind is None:
            msg = "Cannot substitute an absent sort kind"
            raise TypeConstraintError(msg)
        if record.kind.field == "var":
            try:
                replacement = bindings[record.kind.value]
            except KeyError as exc:
                raise TypeConstraintError(f"Unresolved type variable: {record.kind.value}") from exc
            resolved[current] = builder.import_ref(replacement)
            continue
        match record.kind.field:
            case "family":
                children = record.kind.value.args
            case "func":
                children = [*record.kind.value.params, record.kind.value.result]
            case "eq":
                children = []
        if not children:
            resolved[current] = builder.import_ref(current)
            continue
        if exiting:

            def relocate(role: str, index: int, origin: Ref = current) -> int:
                reference = origin.owner.ref(role, index)
                return resolved[reference] if role == "sorts" else builder.import_ref(reference)

            resolved[current] = builder.add("sorts", _copy_record(record, relocate=relocate))
            active.remove(current)
            continue
        if current in active:
            msg = "Sort patterns cannot contain cycles"
            raise TypeConstraintError(msg)
        active.add(current)
        pending.append((current, True))
        pending.extend((current.owner.ref("sorts", index), False) for index in children)
    return builder.publish().ref("sorts", resolved[pattern.owner.ref(pattern.role, pattern.index)])
