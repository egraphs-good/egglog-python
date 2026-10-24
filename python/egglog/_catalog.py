"""Derived Python presentation indexes over canonical declaration records."""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Sequence
from functools import cache
from importlib.resources import files

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._presentation import CallKind, builtin_owner, presentation, validate_view
from ._program import Builder, Owner, Ref, _definition_key


@cache
def builtin_catalog() -> Catalog:
    """Decode the native-generated package snapshot, without creating an engine."""
    path = files("egglog").joinpath("_builtin_catalog.pb")
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as exc:
        msg = "The native-generated builtin catalog is missing; regenerate the package resource"
        raise RuntimeError(msg) from exc
    return Catalog(builtin_owner(pb.BuiltinCatalog.from_binary(encoded)))


def _python_path(path: Sequence[str]) -> tuple[str, ...]:
    if not path or any(not part.isidentifier() or keyword.iskeyword(part) for part in path):
        raise ValueError(f"Invalid Python path: {path!r}")
    return tuple(path)


class Catalog:
    """
    Own one catalog snapshot and rebuildable indexes, never copied signatures.

    These indexes contain explicit supplied bindings. Absent Python bindings
    require conservative wrapper derivation at generation, without installing
    synthetic bindings; explicit/derived collisions are generation errors.
    Engine installation and compatibility remain the shared adapter's job.
    """

    def __init__(self, program: pb.Program | Owner) -> None:
        self.owner = program if isinstance(program, Owner) else Builder.from_program(program).publish()
        self.roots = tuple(
            self.owner.ref(role, index)
            for role in ("primitive_sorts", "primitives", "declarations")
            for index in range(len(self.owner._slots[role]))
        )
        self.reindex()

    @classmethod
    def from_refs(cls, declarations: Iterable[Ref]) -> Catalog:
        """Index already-published definitions without copying their owners/identity."""
        result = object.__new__(cls)
        result.roots = tuple(dict.fromkeys(ref.owner.ref(ref.role, ref.index) for ref in declarations))
        if any(reference.role not in {"declarations", "primitive_sorts", "primitives"} for reference in result.roots):
            msg = "Catalog roots must be canonical declaration references"
            raise TypeError(msg)
        result.reindex()
        return result

    def reindex(self) -> None:  # noqa: C901, PLR0912
        """Discard all language indexes and derive them solely from owned records."""
        definitions: dict[tuple[str, str], Ref] = {}
        types: dict[tuple[str, ...], Ref] = {}
        functions: dict[tuple[str, ...], tuple[Ref, int]] = {}
        constants: dict[tuple[str, ...], tuple[Ref, int]] = {}
        members: dict[tuple[tuple[str, str] | None, str], tuple[Ref, int]] = {}
        for ref in self.roots:
            declaration: pb.Declaration | pb.Primitive | pb.PrimitiveSort = ref.read()
            key = _definition_key(declaration)
            assert key is not None
            if key in definitions:
                raise ValueError(f"Duplicate catalog definition: {key}")
            definitions[key] = ref
            if isinstance(declaration, pb.Declaration) and declaration.kind is None:
                msg = "Catalog declaration has no kind"
                raise ValueError(msg)
            if isinstance(declaration, pb.Declaration):
                assert declaration.kind is not None
                kind = declaration.kind.value
            else:
                kind = declaration
            if isinstance(kind, pb.PrimitiveSort | pb.EqSort):
                binding = presentation(ref).type_view
                if binding is not None:
                    # type() can return names that static Python class syntax
                    # cannot spell. Keep qualifiers strict and terminals exact;
                    # static generators validate the identifiers they emit.
                    path = tuple(binding.path)
                    if not path or not path[-1]:
                        raise ValueError(f"Invalid Python path: {path!r}")
                    if len(path) > 1:
                        _python_path(path[:-1])
                    if path in types or path in functions or path in constants:
                        raise ValueError(f"Python binding collision: {path}")
                    types[path] = ref
                continue
            views = presentation(ref).views
            application = (
                isinstance(kind, pb.Primitive) and kind.typing is not None and kind.typing.field == "application"
            )
            for view_index, view in enumerate(views):
                validate_view(ref, view)
                if (
                    isinstance(kind, pb.Primitive)
                    and kind.typing is not None
                    and kind.typing.field == "application"
                    and (view.owner is not None or view.kind != CallKind.METHOD)
                ):
                    msg = "Python member has an invalid structural function application owner"
                    raise ValueError(msg)
                if view.kind == CallKind.FUNCTION:
                    # Returned callables may have dynamic names that cannot be
                    # spelled as Python identifiers. Qualifiers remain strict;
                    # the static generator validates its own emitted final name.
                    path = tuple(view.path)
                    if not path:
                        raise ValueError(f"Invalid Python path: {path!r}")
                    if len(path) > 1:
                        _python_path(path[:-1])
                    if view.owner is not None or view.receiver is not None:
                        msg = "A free Python function cannot have an owner or receiver"
                        raise ValueError(msg)
                    if path in types or path in functions or path in constants:
                        raise ValueError(f"Python binding collision: {path}")
                    functions[path] = ref, view_index
                    continue
                if view.owner is None and view.kind == CallKind.CONSTANT:
                    # constant() accepts display-name strings, not Python identifiers.
                    path = tuple(view.path)
                    if not path or any(not part for part in path[:-1]):
                        raise ValueError(f"Invalid Python constant path: {path!r}")
                    if view.receiver is not None or view.params or view.mutates is not None:
                        msg = "A Python constant must be ownerless and nullary"
                        raise ValueError(msg)
                    if isinstance(kind, pb.Primitive):
                        if kind.typing is None or kind.typing.field != "signature":
                            msg = "A Python constant requires a concrete signature"
                            raise ValueError(msg)
                        signature = kind.typing.value
                        if signature.inputs or signature.type_params or signature.varargs:
                            msg = "A Python constant requires a concrete nullary signature"
                            raise ValueError(msg)
                    elif (
                        not isinstance(kind, pb.Constructor | pb.Function | pb.UserDefinedPrimitive | pb.Relation)
                        or kind.inputs
                    ):
                        msg = "A Python constant requires a nullary callable"
                        raise ValueError(msg)
                    if path in types or path in functions or path in constants:
                        raise ValueError(f"Python binding collision: {path}")
                    constants[path] = ref, view_index
                    continue
                if not application and view.owner is None:
                    msg = "Python member requires an owner"
                    raise ValueError(msg)
                if view.kind not in {
                    CallKind.INITIALIZER,
                    CallKind.METHOD,
                    CallKind.CLASS_METHOD,
                    CallKind.PROPERTY,
                    CallKind.CLASS_VARIABLE,
                }:
                    msg = "Python member has an invalid presentation kind"
                    raise ValueError(msg)
                owner_key = None
                if application:
                    if (
                        not isinstance(kind, pb.Primitive)
                        or kind.typing is None
                        or kind.typing.field != "application"
                        or view.kind != CallKind.METHOD
                        or list(view.path) != ["__call__"]
                        or view.receiver is None
                        or view.receiver != 0
                        or view.mutates is not None
                        or view.tail_repetitions is not None
                        or len(view.params) != 1
                        or view.params[0].core_input != 1
                        or view.params[0].name != "args"
                        or view.params[0].default is not None
                    ):
                        msg = "Invalid Python structural function application view"
                        raise ValueError(msg)
                    owner_key = "function", ""
                else:
                    assert view.owner is not None
                    owner_sort: pb.Sort = view.owner.read()
                    if owner_sort.kind is None:
                        msg = "Python member owner has no sort kind"
                        raise ValueError(msg)
                    match owner_sort.kind.field:
                        case "eq":
                            owner_key = "eq_sort", owner_sort.kind.value
                        case "application":
                            owner_key = "primitive_sort", owner_sort.kind.value.name
                        case _:
                            msg = "Python member requires an equality or host family owner"
                            raise ValueError(msg)
                if view.kind == CallKind.INITIALIZER:
                    if view.path:
                        msg = "Python initializer derives its name and cannot supply a path"
                        raise ValueError(msg)
                    name = "__init__"
                else:
                    path = _python_path(view.path)
                    if len(path) != 1:
                        msg = "Python member path must contain exactly one name"
                        raise ValueError(msg)
                    name = path[0]
                member = owner_key, name
                if member in members:
                    raise ValueError(f"Python binding collision: {member}")
                members[member] = ref, view_index
        self.definitions = definitions
        self.types = types
        self.functions = functions
        self.constants = constants
        self.members = members
