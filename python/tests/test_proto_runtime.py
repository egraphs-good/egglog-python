"""Record-native authoring gates for the single public runtime migration."""

import gc
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from time import sleep
from typing import Literal
from weakref import ref as weakref

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog import ConvertError, Expr, bindings, eq, expr_parts, i64, method
from egglog._program import Builder, Ref, StructuralView, pack
from egglog.deconstruct import get_callable_args
from egglog.runtime import _CLASS_CACHE, RuntimeExpr, RuntimeFunction, _author_call, resolve_callable
from egglog.type_constraint_solver import TypeConstraintError, infer_sort, substitute_sort


def test_user_constructor_defaults_keywords_and_names_are_records(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Class authoring cannot create an engine")

    monkeypatch.setattr(bindings, "EGraph", forbidden)
    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    default = i64(5)

    class Num(Expr, egg_sort="exact sort"):
        @method(egg_fn="exact constructor")
        def __init__(self, value: i64 = default) -> None: ...

    first = Num()
    declaration = resolve_callable(Num)
    record = declaration.read()
    assert record.kind.value.name == "exact constructor"
    assert Num.__egg_definition__.read().kind.value.name == "exact sort"
    assert record.bindings.python.views[0].params[0].name == "value"
    assert get_callable_args(first, Num)[0].value == 5
    default.__replace_expr__(i64(9))
    assert get_callable_args(Num(), Num)[0].value == 5
    assert expr_parts(Num(value=3)) == expr_parts(Num(3))
    assert Num.__egg_context__.prepare is None
    Num.__egg_attr_cache__.clear()
    Num.__egg_context__.catalog.members.clear()
    Num.__egg_context__.catalog.reindex()
    assert resolve_callable(Num) == declaration
    assert get_callable_args(Num(), Num)[0].value == 5
    with pytest.raises(TypeError):
        Num(1, value=2)
    with pytest.raises(TypeError):
        Num(unknown=2)
    with pytest.raises(TypeError):
        Num(1, 2)
    with pytest.raises(ConvertError, match="Cannot convert"):
        Num("wrong sort")


def test_user_preserved_protocols_and_match_args():
    class Local(Expr):
        def __init__(self) -> None: ...

        __match_args__ = ("value",)

        @method(preserve=True)
        @property
        def value(self):
            return 4

        @method(preserve=True)
        def tag(self):
            return "tag"

        def __str__(self) -> str:
            return "local str"

        def __repr__(self) -> str:
            return "local repr"

        def __hash__(self) -> int:
            return 42

        @method(preserve=True)
        def __eq__(self, other) -> bool:
            return other == "sentinel"

        @method(preserve=True)
        def __ne__(self, other) -> bool:
            return other != "sentinel"

    expression = Local()
    assert expression.value == 4
    assert expression.tag() == "tag"
    assert str(expression) == "local str"
    assert repr(expression) == "local repr"
    assert hash(expression) == 42
    assert expression == "sentinel"
    assert (expression == "other") is False
    assert expression != "other"
    assert (expression != "sentinel") is False
    assert bool(eq(expression).to(copy(expression)))
    match expression:
        case Local(4):
            pass
        case _:
            pytest.fail("Explicit match_args and preserved property were lost")


def test_local_class_hooks_survive_caches_but_not_all_roots():
    def make(label):
        class Local(Expr):
            def __init__(self) -> None: ...

            @method(preserve=True)
            def tag(self):
                return label

        expression = Local()
        return expression, weakref(Local), weakref(Local.__egg_definition__.owner), weakref(Local.__egg_context__)

    left, cls, owner, context = make("left")
    right, other_cls, other_owner, other_context = make("right")
    retained = copy(left)
    _CLASS_CACHE.clear()
    gc.collect()
    assert left.tag() == retained.tag() == "left"
    assert right.tag() == "right"
    assert cls() is not other_cls()
    del left, right, retained
    gc.collect()
    assert cls() is owner() is context() is None
    assert other_cls() is other_owner() is other_context() is None


def test_recursive_class_sorts_resolve_without_recursive_member_compilation():
    class Left(Expr):
        def __init__(self, right: "Right") -> None: ...

    class Right(Expr):
        def __init__(self, left: Left) -> None: ...

    left = resolve_callable(Left)
    right = resolve_callable(Right)
    assert StructuralView(left.owner.ref("sorts", left.read().kind.value.inputs[0].sort)) == StructuralView(
        Right.__egg_sort__
    )
    assert StructuralView(right.owner.ref("sorts", right.read().kind.value.inputs[0].sort)) == StructuralView(
        Left.__egg_sort__
    )


def test_constructor_scope_publishes_once_across_authoring_threads():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    calls = []
    prepare = Num.__egg_context__.prepare

    def delayed_prepare(cls):
        calls.append(cls)
        sleep(0.02)  # Hold first compilation open while other authors enter.
        return prepare(cls)

    Num.__egg_context__.prepare = delayed_prepare
    with ThreadPoolExecutor(max_workers=4) as executor:
        expressions = list(executor.map(Num, range(8)))
    assert calls == [Num]
    assert [get_callable_args(expression, Num)[0].value for expression in expressions] == list(range(8))


@pytest.mark.parametrize(
    "options",
    [
        {"cost": 1},
        {"merge": lambda old, new: old},
        {"mutates_self": True},
        {"unextractable": True},
        {"subsume": True},
        {"reverse_args": True},
        {"preserve": True},
    ],
)
def test_unmigrated_constructor_options_never_publish_members(options):
    class Invalid(Expr):
        @method(**options)
        def __init__(self) -> None: ...

    with pytest.raises(NotImplementedError, match="options"):
        Invalid()
    assert Invalid.__egg_context__.catalog is None
    assert not Invalid.__egg_attr_cache__


@pytest.mark.parametrize("options", [{"cost": 1}, {}])
def test_wrapped_ignored_name_rejects_unsupported_members_before_publication(options):
    class Num(Expr):
        def __init__(self) -> None: ...

        @method(**options)
        def __radd__(self, other: i64) -> i64: ...

    with pytest.raises(NotImplementedError, match="__radd__"):
        Num()
    assert Num.__egg_context__.catalog is None
    assert not Num.__egg_attr_cache__


def test_preserved_wrapped_ignored_name_remains_a_local_hook():
    class Num(Expr):
        def __init__(self) -> None: ...

        @method(preserve=True)
        def __radd__(self, other: int) -> int:
            return other + 1

    expression = Num()
    assert Num.__radd__(expression, 2) == 3
    assert {name for _, name in Num.__egg_context__.catalog.members} == {"__init__"}


def test_unsupported_class_members_and_parameter_kinds_are_not_ignored():
    class HasMethod(Expr):
        def __init__(self) -> None: ...
        def other(self) -> i64: ...

    class KeywordOnly(Expr):
        def __init__(self, *, value: i64) -> None: ...

    for cls in (HasMethod, KeywordOnly):
        with pytest.raises(NotImplementedError):
            cls()
        assert cls.__egg_context__.catalog is None
        assert not cls.__egg_attr_cache__


def test_user_sort_cannot_borrow_builtin_codec_or_members_by_name():
    class Shadow(Expr, egg_sort="i64"):
        def __init__(self, value: i64) -> None: ...

    expression = Shadow(3)
    assert expression.__egg_ref__.read().kind.field == "call"
    assert get_callable_args(expression, Shadow)[0].value == 3
    with pytest.raises(AttributeError, match="member"):
        _ = Shadow.__add__


def test_unresolved_constructor_annotation_does_not_publish_members():
    class Missing(Expr):
        def __init__(self, value: "NotDefinedHere") -> None: ...  # noqa: F821 -- intentional unresolved annotation

    with pytest.raises(NameError, match="NotDefinedHere"):
        Missing(1)
    assert Missing.__egg_context__.catalog is None
    assert not Missing.__egg_attr_cache__


def test_user_default_expansion_has_fresh_union_identity_and_retains_sharing():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    child = Num(1)
    builder = Builder()
    member = builder.import_ref(child.__egg_ref__)
    root = builder.add(
        "nodes",
        pb.Node(
            sort_id=builder.import_ref(Num.__egg_sort__),
            kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[member, member])),
        ),
    )
    default = RuntimeExpr(builder.publish().ref("nodes", root))

    class Box(Expr):
        def __init__(self, value: Num = default) -> None: ...

    first = get_callable_args(Box(), Box)[0]
    second = get_callable_args(Box(), Box)[0]
    assert first.__egg_ref__ != second.__egg_ref__ != default.__egg_ref__
    assert expr_parts(first) != expr_parts(second)
    for expression in (first, second):
        members = expression.__egg_ref__.read().kind.value.members
        assert len(members) == 2
        assert members[0] == members[1]


def test_reconstructed_constructor_uses_changed_protobuf_parameter_not_python_signature():
    class Num(Expr):
        def __init__(self, value: i64) -> None: ...

    program = pack([Num(1).__egg_ref__]).program
    index, constructor = next(
        (index, declaration)
        for index, declaration in enumerate(program.declarations)
        if declaration.kind.field == "constructor"
    )
    constructor.bindings.python.views[0].params[0].name = "renamed"
    owner = Builder.from_program(program).publish()
    function = RuntimeFunction(owner.ref("declarations", index), 0, Num)
    result = function(renamed=8)
    assert get_callable_args(result)[0].value == 8
    with pytest.raises(TypeError):
        function(value=8)


def test_generic_inference_uses_binder_positions_and_is_transactional():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 1)))
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[0, 1])))
    )
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="String"))))
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[3, 4])))
    )
    builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[4, 3])))
    )
    owner = builder.publish()
    bindings: dict[int, Ref] = {}
    # GenericSignature labels may repeat: indices, not labels, identify binders.
    signature = pb.GenericSignature(type_params=["T", "T"], inputs=[pb.Arg(sort=2)], output=0)
    infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 5), len(signature.type_params), bindings)
    assert bindings == {0: owner.ref("sorts", 3), 1: owner.ref("sorts", 4)}
    assert StructuralView(substitute_sort(owner.ref("sorts", 2), bindings)) == StructuralView(owner.ref("sorts", 5))
    before = bindings.copy()
    with pytest.raises(TypeConstraintError):
        infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 6), len(signature.type_params), bindings)
    assert bindings == before
    independent: dict[int, Ref] = {}
    infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 6), len(signature.type_params), independent)
    assert independent == {0: owner.ref("sorts", 4), 1: owner.ref("sorts", 3)}


def test_sort_inference_rejects_missing_binders_arity_and_open_actuals():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Box")))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[1], result=1))))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(result=1))))
    owner = builder.publish()
    with pytest.raises(TypeConstraintError, match="binder"):
        infer_sort(owner.ref("sorts", 0), owner.ref("sorts", 1), 0, {})
    with pytest.raises(TypeConstraintError, match="closed"):
        infer_sort(owner.ref("sorts", 0), owner.ref("sorts", 0), 1, {})
    with pytest.raises(TypeConstraintError, match="arity"):
        infer_sort(owner.ref("sorts", 2), owner.ref("sorts", 3), 0, {})
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        substitute_sort(owner.ref("sorts", 0), {})


def test_sort_inference_and_substitution_are_iterative():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Leaf")))
    pattern, concrete = 0, 1
    for _ in range(10_000):
        pattern = builder.add(
            "sorts",
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[pattern]))),
        )
        concrete = builder.add(
            "sorts",
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[concrete]))),
        )
    owner = builder.publish()
    bindings: dict[int, Ref] = {}
    infer_sort(owner.ref("sorts", pattern), owner.ref("sorts", concrete), 1, bindings)
    result = substitute_sort(owner.ref("sorts", pattern), bindings)
    assert StructuralView(result) == StructuralView(owner.ref("sorts", concrete))


def test_call_authoring_binds_keywords_defaults_and_core_input_order():
    builder = Builder()
    sort = builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    default = builder.add(
        "nodes",
        pb.Node(
            sort_id=sort,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 4))
            ),
        ),
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="sample.subtract",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature", pb.GenericSignature(inputs=[pb.Arg(sort=sort), pb.Arg(sort=sort)], output=sort)
                    ),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.FUNCTION,
                            path=["sample", "subtract_reversed"],
                            params=[
                                pb.PythonParameter(core_input=1, name="right"),
                                pb.PythonParameter(core_input=0, name="left", default_expr=default),
                            ],
                        )
                    ]
                )
            ),
        ),
    )
    owner = builder.publish()

    def convert(expected: Ref, value: object) -> Ref:
        assert isinstance(value, int)
        literal = Builder()
        literal.add(
            "nodes",
            pb.Node(
                sort_id=literal.import_ref(expected),
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
        )
        return literal.publish().ref("nodes", 0)

    root, mutated = _author_call(
        owner.ref("declarations", declaration), 0, (), {"right": 9}, node_of=lambda _: None, convert=convert
    )
    assert mutated is None
    node = root.read()
    assert node.kind.value.func == "sample.subtract"
    args = [root.owner.ref("nodes", index) for index in node.kind.value.args]
    assert [arg.read().kind.value.value.value for arg in args] == [4, 9]
    assert args[0] != owner.ref("nodes", default)
    with pytest.raises(TypeError, match="unexpected keyword"):
        _author_call(
            owner.ref("declarations", declaration),
            0,
            (),
            {"right": 9, "bogus": 9},
            node_of=lambda _: None,
            convert=convert,
        )


def test_generic_empty_call_requires_result_owner_and_keeps_signature_unchanged():
    builder = Builder()
    variable = builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    pattern = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[variable])))
    )
    scalar = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    )
    concrete = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[scalar])))
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                "host_primitive",
                pb.HostPrimitive(
                    name="sample.vec.of",
                    typing=Oneof[Literal["signature"], pb.GenericSignature](
                        "signature",
                        pb.GenericSignature(type_params=["T"], output=pattern, varargs=pb.Arg(sort=variable)),
                    ),
                ),
            ),
            bindings=pb.CallableBindings(
                python=pb.PythonBindings(
                    views=[
                        pb.PythonCallable(
                            kind=pb.PythonCallKind.INITIALIZER,
                            owner=pb.BindingOwner(kind=Oneof[Literal["sort"], int]("sort", pattern)),
                            params=[pb.PythonParameter(core_input=0, name="args")],
                        )
                    ]
                )
            ),
        ),
    )
    owner = builder.publish()
    ref = owner.ref("declarations", declaration)
    before = ref.read().to_binary()

    def forbidden(*args):
        pytest.fail("An empty constructor must not convert an argument")

    root, mutated = _author_call(
        ref, 0, (), {}, node_of=forbidden, convert=forbidden, owner=owner.ref("sorts", concrete)
    )
    assert mutated is None
    assert list(root.read().kind.value.args) == []
    assert StructuralView(root.owner.ref("sorts", root.read().sort_id)) == StructuralView(owner.ref("sorts", concrete))
    assert ref.read().to_binary() == before
    with pytest.raises(TypeConstraintError, match="Unresolved"):
        _author_call(ref, 0, (), {}, node_of=forbidden, convert=forbidden)
