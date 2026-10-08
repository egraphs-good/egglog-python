"""Solve generic signatures directly over canonical sort records."""

from __future__ import annotations

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Builder, Ref, StructuralView, _copy_record

__all__ = ["TypeConstraintError", "infer_sort", "substitute_sort"]


class TypeConstraintError(RuntimeError):
    """Typing error when trying to infer the return type."""


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
