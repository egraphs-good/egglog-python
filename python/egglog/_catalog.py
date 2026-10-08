"""Derived Python presentation indexes over canonical declaration records."""

from __future__ import annotations

import keyword
from collections.abc import Sequence
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
        self.definitions: dict[tuple[str, str], Ref] = {}
        self.types: dict[tuple[str, ...], Ref] = {}
        self.functions: dict[tuple[str, ...], tuple[Ref, int]] = {}
        self.members: dict[tuple[str | None, str], tuple[Ref, int]] = {}
        self.reindex()

    def reindex(self) -> None:  # noqa: C901, PLR0912
        """Discard all language indexes and derive them solely from owned records."""
        definitions: dict[tuple[str, str], Ref] = {}
        types: dict[tuple[str, ...], Ref] = {}
        functions: dict[tuple[str, ...], tuple[Ref, int]] = {}
        members: dict[tuple[str | None, str], tuple[Ref, int]] = {}
        for index in range(len(self.owner._slots["declarations"])):
            ref = self.owner.ref("declarations", index)
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
                    if path in types or path in functions:
                        raise ValueError(f"Python binding collision: {path}")
                    types[path] = ref
                continue
            binding = declaration.bindings
            if binding is None or binding.python is None:
                continue
            for view_index, view in enumerate(binding.python.views):
                if view.kind == pb.PythonCallKind.FUNCTION:
                    path = _python_path(view.path)
                    if view.owner is not None or view.has_field("receiver"):
                        msg = "A free Python function cannot have an owner or receiver"
                        raise ValueError(msg)
                    if path in types or path in functions:
                        raise ValueError(f"Python binding collision: {path}")
                    functions[path] = ref, view_index
                    continue
                if view.owner is None or view.owner.kind is None:
                    msg = "Python member requires an owner"
                    raise ValueError(msg)
                owner_name = None
                if view.owner.kind.field == "sort":
                    owner_sort: pb.Sort = self.owner.ref("sorts", view.owner.kind.value).read()
                    if owner_sort.kind is None:
                        msg = "Python member owner has no sort kind"
                        raise ValueError(msg)
                    match owner_sort.kind.field:
                        case "eq":
                            owner_name = owner_sort.kind.value
                        case "family":
                            owner_name = owner_sort.kind.value.name
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
                member = owner_name, name
                if member in members:
                    raise ValueError(f"Python binding collision: {member}")
                members[member] = ref, view_index
        self.definitions = definitions
        self.types = types
        self.functions = functions
        self.members = members
