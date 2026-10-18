"""Derived Python presentation indexes over canonical declaration records."""

from __future__ import annotations

import keyword
from collections.abc import Iterable, Sequence
from functools import cache
from importlib.resources import files

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Builder, Ref, _definition_key


@cache
def builtin_catalog() -> Catalog:
    """Decode the native-generated package snapshot, without creating an engine."""
    path = files("egglog").joinpath("_builtin_catalog.pb")
    try:
        encoded = path.read_bytes()
    except FileNotFoundError as exc:
        msg = "The native-generated builtin catalog is missing; regenerate the package resource"
        raise RuntimeError(msg) from exc
    return Catalog(pb.Program.from_binary(encoded))


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

    def __init__(self, program: pb.Program) -> None:
        self.owner = Builder.from_program(program).publish()
        self.roots = tuple(self.owner.ref("declarations", index) for index in range(len(program.declarations)))
        self.reindex()

    @classmethod
    def from_refs(cls, declarations: Iterable[Ref]) -> Catalog:
        """Index already-published definitions without copying their owners/identity."""
        result = object.__new__(cls)
        result.roots = tuple(dict.fromkeys(ref.owner.ref(ref.role, ref.index) for ref in declarations))
        if any(reference.role != "declarations" for reference in result.roots):
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
            declaration: pb.Declaration = ref.read()
            key = _definition_key(declaration)
            assert key is not None
            if key in definitions:
                raise ValueError(f"Duplicate catalog definition: {key}")
            definitions[key] = ref
            assert declaration.kind is not None
            kind = declaration.kind.value
            if isinstance(kind, pb.HostSortFamily | pb.EqSort):
                binding = kind.bindings
                if binding is not None and binding.python is not None:
                    path = _python_path(binding.python.path)
                    if path in types or path in functions or path in constants:
                        raise ValueError(f"Python binding collision: {path}")
                    types[path] = ref
                continue
            binding = declaration.bindings
            if binding is None or binding.python is None:
                continue
            for view_index, view in enumerate(binding.python.views):
                if (
                    isinstance(kind, pb.HostPrimitive)
                    and kind.typing is not None
                    and kind.typing.field == "application"
                    and (view.owner is None or view.owner.kind is None or view.owner.kind.field != "function")
                ):
                    msg = "Python member has an invalid structural function application owner"
                    raise ValueError(msg)
                if view.kind == pb.PythonCallKind.FUNCTION:
                    # Returned callables may have dynamic names that cannot be
                    # spelled as Python identifiers. Qualifiers remain strict;
                    # the static generator validates its own emitted final name.
                    path = tuple(view.path)
                    if not path:
                        raise ValueError(f"Invalid Python path: {path!r}")
                    if len(path) > 1:
                        _python_path(path[:-1])
                    if view.owner is not None or view.has_field("receiver"):
                        msg = "A free Python function cannot have an owner or receiver"
                        raise ValueError(msg)
                    if path in types or path in functions or path in constants:
                        raise ValueError(f"Python binding collision: {path}")
                    functions[path] = ref, view_index
                    continue
                if view.owner is None and view.kind == pb.PythonCallKind.CONSTANT:
                    # constant() accepts display-name strings, not Python identifiers.
                    path = tuple(view.path)
                    if not path or any(not part for part in path[:-1]):
                        raise ValueError(f"Invalid Python constant path: {path!r}")
                    if view.has_field("receiver") or view.params or view.has_field("mutates"):
                        msg = "A Python constant must be ownerless and nullary"
                        raise ValueError(msg)
                    if isinstance(kind, pb.HostPrimitive):
                        if kind.typing is None or kind.typing.field != "signature":
                            msg = "A Python constant requires a concrete signature"
                            raise ValueError(msg)
                        signature = kind.typing.value
                        if signature.inputs or signature.type_params or signature.varargs:
                            msg = "A Python constant requires a concrete nullary signature"
                            raise ValueError(msg)
                    elif not isinstance(kind, pb.Constructor | pb.Function | pb.Primitive | pb.Relation) or kind.inputs:
                        msg = "A Python constant requires a nullary callable"
                        raise ValueError(msg)
                    if path in types or path in functions or path in constants:
                        raise ValueError(f"Python binding collision: {path}")
                    constants[path] = ref, view_index
                    continue
                if view.owner is None or view.owner.kind is None:
                    msg = "Python member requires an owner"
                    raise ValueError(msg)
                if view.kind not in {
                    pb.PythonCallKind.INITIALIZER,
                    pb.PythonCallKind.METHOD,
                    pb.PythonCallKind.CLASS_METHOD,
                    pb.PythonCallKind.PROPERTY,
                    pb.PythonCallKind.CLASS_VARIABLE,
                }:
                    msg = "Python member has an invalid presentation kind"
                    raise ValueError(msg)
                owner_key = None
                if view.owner.kind.field == "function":
                    if (
                        not isinstance(kind, pb.HostPrimitive)
                        or kind.typing is None
                        or kind.typing.field != "application"
                        or view.kind != pb.PythonCallKind.METHOD
                        or list(view.path) != ["__call__"]
                        or not view.has_field("receiver")
                        or view.receiver != 0
                        or view.has_field("mutates")
                        or view.has_field("tail_repetitions")
                        or len(view.params) != 1
                        or not view.params[0].has_field("core_input")
                        or view.params[0].core_input != 1
                        or view.params[0].name != "args"
                        or view.params[0].has_field("default_expr")
                    ):
                        msg = "Invalid Python structural function application view"
                        raise ValueError(msg)
                    owner_key = "function", ""
                elif view.owner.kind.field == "sort":
                    owner_sort: pb.Sort = ref.owner.ref("sorts", view.owner.kind.value).read()
                    if owner_sort.kind is None:
                        msg = "Python member owner has no sort kind"
                        raise ValueError(msg)
                    match owner_sort.kind.field:
                        case "eq":
                            owner_key = "eq_sort", owner_sort.kind.value
                        case "family":
                            owner_key = "host_sort_family", owner_sort.kind.value.name
                        case _:
                            msg = "Python member requires an equality or host family owner"
                            raise ValueError(msg)
                if view.kind == pb.PythonCallKind.INITIALIZER:
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
