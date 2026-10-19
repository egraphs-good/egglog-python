"""Storage and relocation tests, separate from engine conformance."""

from __future__ import annotations

import gc
import struct
from copy import copy
from typing import Literal
from weakref import ref as weakref

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import DescFieldValueList, Oneof

from egglog import Bool, String, Unit, _program, bindings, join, rule, ruleset
from egglog._program import Builder, StructuralView, clone_nodes, pack
from egglog.egraph import Action, RewriteOrRule


def literal(value: int):
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    index = builder.add(
        "nodes",
        pb.Node(
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
            )
        ),
    )
    return builder.publish().ref("nodes", index)


AMBIENT = frozenset({("host_sort_family", "i64"), ("callable", "f"), ("callable", "g")})


def test_sort_text_alias_metadata_survives_copy_and_pack_without_affecting_identity():
    sorts = []
    for symbols in ([], ["A"], ["Z"], ["A", "Z"]):
        builder = Builder()
        scalar = builder.add(
            "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
        )
        index = builder.add(
            "sorts",
            pb.Sort(
                kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Vec", args=[scalar])),
                egglog_aliases=[pb.EgglogTypeBinding(symbol=s) for s in symbols],
            ),
        )
        sort = builder.publish().ref("sorts", index)
        sorts.append(StructuralView(sort))
        imported = Builder()
        imported.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "unused")))
        copied = imported.import_ref(sort)
        copied = imported.publish().ref("sorts", copied)
        assert [a.symbol for a in copied.read().egglog_aliases] == symbols
        packed = pack([copied], ambient={("host_sort_family", "i64"), ("host_sort_family", "Vec")}).program
        decoded = pb.Program.from_binary(packed.to_binary())
        assert [a.symbol for a in decoded.sorts[0].egglog_aliases] == symbols
        assert StructuralView(copied) == StructuralView(sort)
    assert len(set(sorts)) == 1


def test_print_function_limit_presence_survives_copy_and_pack_without_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Table command authoring accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    views = []
    encodings = []
    for limit in (None, 0, 1):
        requested = pb.PrintFunction(table="f")
        if limit is not None:
            requested.max_rows = limit
        builder = Builder()
        index = builder.add(
            "commands", pb.Command(kind=Oneof[Literal["print_function"], pb.PrintFunction]("print_function", requested))
        )
        reference = builder.publish().ref("commands", index)
        requested.max_rows = 99
        imported = Builder()
        imported_index = imported.import_ref(reference)
        imported_reference = imported.publish().ref("commands", imported_index)
        program = pack(commands=[imported_reference], ambient=AMBIENT).program
        encoded = program.to_binary()
        decoded = pb.Program.from_binary(encoded)
        assert decoded == program
        for command in (reference.read(), imported_reference.read(), program.commands[0], decoded.commands[0]):
            kind = command.kind
            assert kind is not None
            assert kind.field == "print_function"
            request = kind.value
            assert isinstance(request, pb.PrintFunction)
            assert request.table == "f"
            assert request.has_field("max_rows") == (limit is not None)
            assert request.max_rows == (0 if limit is None else limit)
        views.append(StructuralView(reference))
        encodings.append(encoded)
    assert len(set(views)) == len(set(encodings)) == 3


def test_schedule_composition_relocates_owned_inline_spans_without_engine(monkeypatch):
    from egglog import Schedule, seq  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Schedule relocation accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    builder = Builder()
    builder.add("files", pb.SourceFile(name="unused"))
    file = builder.add("files", pb.SourceFile(name="schedule.py", contents="run()"))
    index = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["run"], pb.Run](
                "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", "")))
            ),
            span=pb.Span(file=file, start=0, end=5),
        ),
    )
    original = Schedule(builder.publish().ref("commands", index), {})
    combined = seq(original, original.saturate() * 2)
    program = pack(commands=[combined._resolve(None)], ambient={("ruleset", "")}).program
    assert [file.name for file in program.files] == ["schedule.py"]
    sequence_kind = program.commands[0].kind
    assert sequence_kind is not None
    assert sequence_kind.field == "repeat"
    first, second = sequence_kind.value.body
    assert first.span == pb.Span(file=0, start=0, end=5)
    repeated_kind = second.kind
    assert repeated_kind is not None
    assert repeated_kind.field == "saturate"
    saturated_kind = repeated_kind.value.body[0].kind
    assert saturated_kind is not None
    assert saturated_kind.field == "saturate"
    assert saturated_kind.value.body[0].span == first.span
    assert pb.Program.from_binary(program.to_binary()) == program


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("seminaive", pb.RuleEvalMode.SEMINAIVE),
        ("naive", pb.RuleEvalMode.NAIVE),
        ("unsafe-seminaive", pb.RuleEvalMode.UNSAFE_SEMINAIVE),
    ],
)
def test_direct_rule_option_presence_and_empty_head(monkeypatch, mode, expected):
    def forbidden(*args, **kwargs):
        pytest.fail("Rule storage accessed the engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden, raising=False)
    authored = rule(eval_mode=mode).then()
    record = authored.__egg_ref__.read()
    assert record.eval_mode == expected
    assert not record.has_field("name")
    assert not record.no_decomp
    assert not record.include_subsumed
    assert record.kind.value.query == record.kind.value.head == []
    expected_str = "rule().then()" if mode == "seminaive" else f"rule(eval_mode={mode!r}).then()"
    assert str(authored) == expected_str
    program = pack([authored.__egg_ref__]).program
    assert not program.nodes
    assert not program.commands
    assert not program.rulesets
    assert pb.Program.from_binary(program.to_binary()).rules == [record]
    with pytest.raises(TypeError, match="canonical rule reference"):
        RewriteOrRule(literal(1))
    with pytest.raises(ValueError, match="evaluation mode"):
        rule(eval_mode="unknown").then()
    assert not rule(name="").then().__egg_ref__.read().has_field("name")
    assert str(rule(name="").then()) == "rule().then()"
    with pytest.raises(TypeError, match="strings"):
        rule(name=42).then()
    with pytest.raises(TypeError, match="destinations must be Rulesets"):
        rule(ruleset=object()).then()
    assert authored == authored  # noqa: PLR0124 -- reflexivity is part of the wrapper contract
    assert authored.__eq__(object()) is NotImplemented
    with pytest.raises(TypeError, match="unhashable"):
        hash(authored)
    assert authored == rule(eval_mode=mode).then()
    destination = ruleset()
    initial = destination._snapshot({})
    destination.append(authored)
    assert initial.read().kind.value.rules == []
    assert len(destination._snapshot({}).read().kind.value.rules) == 1
    calls = []

    def deferred():
        calls.append("called")
        return [authored]

    destination.register(deferred)
    pending = tuple(destination.deferred_rule_gens)
    with pytest.raises(NotImplementedError, match="legacy declaration projection"):
        _ = destination.__egg_decls__
    assert not calls
    assert tuple(destination.deferred_rule_gens) == pending
    assert len(destination._snapshot({}).read().kind.value.rules) == 2
    assert calls == ["called"]
    assert not destination.deferred_rule_gens


def test_nested_fail_pack_relocates_child_facts_and_all_spans():
    builder = Builder()
    builder.import_ref(literal(99))
    value = builder.import_ref(literal(7))
    builder.add("files", pb.SourceFile(name="unused"))
    file = builder.add("files", pb.SourceFile(name="fail.egg", contents="(fail (fail (check 7)))"))
    command = pb.Command(
        kind=Oneof[Literal["check"], pb.Check]("check", pb.Check(facts=[value])),
        span=pb.Span(file=file, start=12, end=21),
    )
    for start, end in [(6, 22), (0, 23)]:
        command = pb.Command(
            kind=Oneof[Literal["fail"], pb.Command]("fail", command),
            span=pb.Span(file=file, start=start, end=end),
        )
    index = builder.add("commands", command)
    reference = builder.publish().ref("commands", index)
    packed = pack(commands=[reference], ambient=AMBIENT).program
    decoded = pb.Program.from_binary(packed.to_binary())
    assert decoded == packed
    assert len(decoded.nodes) == 1
    assert len(decoded.files) == 1
    child = decoded.commands[0]
    for start, end in [(0, 23), (6, 22), (12, 21)]:
        assert child.span == pb.Span(file=0, start=start, end=end)
        assert child.kind is not None
        if child.kind.field == "fail":
            child = child.kind.value
    assert child.kind is not None
    assert child.kind.field == "check"
    assert child.kind.value.facts == [0]
    assert decoded.nodes[0].kind.value.value.value == 7
    assert reference.read().kind.value.kind.value.span.file == file


@pytest.mark.parametrize(
    ("left_bits", "right_bits", "equal"),
    [
        (0, 1 << 63, True),
        (0x7FF8000000000001, 0x7FF8000000000001, True),
        (0x7FF0000000000001, 0x7FF0000000000001, True),
        (0x7FF8000000000001, 0x7FF8000000000002, False),
        (0x7FF8000000000001, 0xFFF8000000000001, False),
        (0x7FF8000000000001, 0x7FF0000000000001, False),
        (0x7FF8000000000001, int.from_bytes(struct.pack("!d", float(0x7FF8000000000001))), False),
    ],
)
def test_structural_float_equality_hashes_and_exact_bits(left_bits, right_bits, equal):
    views = []
    for bits in (left_bits, right_bits):
        builder = Builder()
        builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="f64"))))
        index = builder.add(
            "nodes",
            pb.Node(
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["f64_bits"], int]("f64_bits", bits))
                )
            ),
        )
        reference = builder.publish().ref("nodes", index)
        assert reference.read().kind.value.value.value == bits
        views.append(StructuralView(reference))
    assert (views[0] == views[1]) is equal
    assert views[0] == views[0]
    if equal:
        assert hash(views[0]) == hash(views[1])


@pytest.fixture
def action_program():
    """A Set whose declaration closure contains a recursive named dependency."""
    return pb.Program(
        ir_version=1,
        sorts=[
            pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=name)))
            for name in ("i64", "f64")
        ],
        nodes=[
            pb.Node(
                sort_id=0,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            )
            for value in (1, 2)
        ]
        + [pb.Node(sort_id=0, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func=name))) for name in ("g", "f")],
        declarations=[
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name=name)
                )
            )
            for name in ("i64", "f64")
        ]
        + [
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name=name, output=0, merge=merge))
            )
            for name, merge in (("f", 2), ("g", 3))
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(kind=Oneof[Literal["set"], pb.Set]("set", pb.Set(target=pb.Call(func="f"), value=0))),
                )
            )
        ],
    )


def test_action_equality_ignores_unrelated_records_and_arena_layout(action_program):
    first = Action(Builder.from_program(action_program).publish().ref("commands", 0))
    action_program.declarations += [
        pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="unused")))
    ]
    action_program.nodes.append(pb.Node(sort_id=1, kind=Oneof[Literal["var"], str]("var", "unused")))
    assert first == Action(Builder.from_program(action_program).publish().ref("commands", 0))
    mappings = {
        role: {
            index: len(getattr(action_program, role)) - index - 1 for index in range(len(getattr(action_program, role)))
        }
        for role in ("nodes", "sorts", "declarations", "commands")
    }
    for role in mappings:
        original = list(getattr(action_program, role))
        relocated = [
            _program._copy_record(record, relocate=lambda arena, index: mappings[arena][index])
            for record in reversed(original)
        ]
        getattr(action_program, role)[:] = relocated
    assert first == Action(Builder.from_program(action_program).publish().ref("commands", 0))


@pytest.mark.parametrize("change", ["doc", "presentation", "hidden", "merge", "sort", "rhs", "action"])
def test_action_equality_observes_referenced_declaration_closure(action_program, change):
    first = Action(Builder.from_program(action_program).publish().ref("commands", 0))
    assert first == Action(Builder.from_program(action_program).publish().ref("commands", 0))
    # g is reached only through f's merge; checking the action root or f alone
    # would miss these changes to the recursive referenced declaration closure.
    declaration = action_program.declarations[3]
    if change == "doc":
        declaration.doc = "different"
    elif change == "presentation":
        declaration.bindings = pb.CallableBindings(
            python=pb.PythonBindings(views=[pb.PythonCallable(kind=pb.PythonCallKind.FUNCTION, path=["renamed"])])
        )
    elif change == "hidden":
        declaration.hidden = True
        visible = first.__egg_ref__.owner.ref("declarations", 3)
        hidden = Builder.from_program(action_program).publish().ref("declarations", 3)
        assert visible.read().hidden is False
        assert hidden.read().hidden is True
        assert StructuralView(visible) != StructuralView(hidden)
        assert len({StructuralView(visible), StructuralView(hidden)}) == 2
        with pytest.raises(ValueError, match="same-name"):
            pack([visible, hidden])
    elif change == "merge":
        declaration.kind.value.merge = 1
    elif change == "sort":
        action_program.declarations[0].kind.value.bindings = pb.SortBindings(python=pb.TypeBinding(path=["renamed"]))
    elif change == "rhs":
        action_program.commands[0].kind.value.kind.value.value = 1
    else:
        action_program.commands[0].kind.value.kind = Oneof[Literal["term"], int]("term", 0)
    assert first != Action(Builder.from_program(action_program).publish().ref("commands", 0))


@pytest.mark.parametrize("kind", ["ordinary", "empty", "cycle"])
def test_action_equality_compares_union_syntax_in_declaration_closure(action_program, kind):
    action_program.nodes.append(
        pb.Node(
            sort_id=0,
            kind=Oneof[Literal["union"], pb.Union](
                "union", pb.Union(members=[0, 1] if kind == "ordinary" else [4] if kind == "cycle" else [])
            ),
        )
    )
    action_program.declarations[2].kind.value.merge = 4
    owner = Builder.from_program(action_program).publish()
    first = Action(owner.ref("commands", 0))
    assert first == Action(owner.ref("commands", 0))
    second = Action(Builder.from_program(action_program).publish().ref("commands", 0))
    assert (first == second) is (kind == "ordinary")
    assert (second == first) is (kind == "ordinary")


@pytest.fixture
def union_action_graph():
    return pb.Program(
        ir_version=1,
        sorts=[pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E"))],
        declarations=[pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="E")))]
        + [
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name=name, inputs=[pb.Arg(sort=0) for _ in range(arity)], output=0)
                )
            )
            for name, arity in (("F", 1), ("Pair", 2), ("C", 0), ("D", 0))
        ],
    )


def test_action_equality_has_finite_shared_cycle_proofs(union_action_graph):
    union_action_graph.nodes = [
        pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[1]))),
        pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="F", args=[0]))),
    ]
    union_action_graph.commands = [
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", index))
            )
        )
        for index in range(2)
    ]
    source = Builder.from_program(union_action_graph).publish()
    builder = Builder()
    sort = builder.import_ref(source.ref("sorts", 0))
    u, f = [builder.import_ref(source.ref("nodes", index)) for index in range(2)]
    g0 = builder.add(
        "nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="F", args=[u])))
    )
    v = builder.add(
        "nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[f])))
    )
    g = builder.add(
        "nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="F", args=[v])))
    )
    w = builder.add(
        "nodes", pb.Node(sort_id=sort, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[g])))
    )
    for index in (u, f, g0, v, g, w):
        builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", index))
                )
            ),
        )
    owner = builder.publish()
    actions = [Action(owner.ref("commands", index)) for index in range(6)]
    for group in ([actions[index] for index in (0, 3, 5)], [actions[index] for index in (1, 2, 4)]):
        for left in reversed(group):
            for right in group:
                assert left == right
                assert right == left
    for index in range(2):
        assert actions[index] == Action(source.ref("commands", index)) == copy(actions[index])
    packed = pack(commands=[action.__egg_ref__ for action in actions]).program
    terms = []
    for command in packed.commands:
        command_kind = command.kind
        assert command_kind is not None
        assert command_kind.field == "action"
        term_kind = command_kind.value.kind
        assert term_kind is not None
        assert term_kind.field == "term"
        terms.append(term_kind.value)
    assert len(set(terms)) == 6
    assert pb.Program.from_binary(packed.to_binary()) == packed
    independent = Builder.from_program(union_action_graph).publish()
    for index in (0, 3, 5):
        assert actions[index] != Action(independent.ref("commands", 0))
        assert Action(independent.ref("commands", 0)) != actions[index]
    for index in (1, 2, 4):
        assert actions[index] != Action(independent.ref("commands", 1))

    def merge_action(merge):
        declaration = Builder()
        output = declaration.import_ref(source.ref("sorts", 0))
        merge_index = declaration.import_ref(owner.ref("nodes", merge))
        declaration.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="slot", output=output, merge=merge_index)
                )
            ),
        )
        root = declaration.add(
            "nodes", pb.Node(sort_id=output, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="slot")))
        )
        command = declaration.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", root))
                )
            ),
        )
        return Action(declaration.publish().ref("commands", command))

    first, second, third = [merge_action(index) for index in (u, v, w)]
    assert first == second == third
    assert third == first


def test_action_equality_keeps_empty_anchors_under_structural_parents(union_action_graph):
    union_action_graph.nodes = [
        pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())) for _ in range(2)
    ] + [pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="F", args=[index]))) for index in (0, 1, 0)]
    union_action_graph.commands = [
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action](
                "action", pb.Action(kind=Oneof[Literal["term"], int]("term", index))
            )
        )
        for index in range(5)
    ]
    owner = Builder.from_program(union_action_graph).publish()
    empty, other, first, different, same = [Action(owner.ref("commands", index)) for index in range(5)]
    assert empty != other
    assert empty == copy(empty)
    assert first != different
    assert first == same
    assert same == first
    imported = Builder()
    alias = imported.import_ref(empty.__egg_ref__)
    assert Action(imported.publish().ref("commands", alias)) == empty


@pytest.mark.parametrize("length", [1, 2])
def test_action_equality_does_not_validate_or_coinductively_merge_cycles(union_action_graph, length):
    # Addressable but native-invalid: comparison is not submission validation.
    union_action_graph.nodes = [
        pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="F", args=[(index + 1) % length])))
        for index in range(length)
    ]
    union_action_graph.commands = [
        pb.Command(
            kind=Oneof[Literal["action"], pb.Action]("action", pb.Action(kind=Oneof[Literal["term"], int]("term", 0)))
        )
    ]
    left, right = [Action(Builder.from_program(union_action_graph).publish().ref("commands", 0)) for _ in range(2)]
    assert left == copy(left)
    assert left != right
    assert right != left


@pytest.mark.parametrize("arity", [1, 2])
def test_action_equality_is_iterative_and_dag_safe(union_action_graph, monkeypatch, arity):
    depth = 2_000
    actions = []
    node_counts = []
    for width in (1, 2):
        builder = Builder.from_program(union_action_graph)
        roots = [builder.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="C"))))]
        for _ in range(depth):
            children = [roots[index % len(roots)] for index in range(arity)]
            roots = [
                builder.add(
                    "nodes",
                    pb.Node(
                        kind=Oneof[Literal["call"], pb.Call](
                            "call", pb.Call(func="F" if arity == 1 else "Pair", args=children)
                        )
                    ),
                )
                for _ in range(width)
            ]
        node_counts.append(1 + depth * width)
        command = builder.add(
            "commands",
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["term"], int]("term", roots[0]))
                )
            ),
        )
        actions.append(Action(builder.publish().ref("commands", command)))
    visits = 0
    original = _program._structural_parts

    def counted(reference, **kwargs):
        nonlocal visits
        visits += 1
        return original(reference, **kwargs)

    monkeypatch.setattr(_program, "_structural_parts", counted)
    assert actions[0] == actions[1]
    assert actions[1] == actions[0]
    assert visits <= 8 * sum(node_counts)


@pytest.mark.parametrize("reverse", [False, True])
def test_nominal_family_callable_names_keep_independent_closure(reverse):
    declarations = [
        pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Pair"))),
        pb.Declaration(
            kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                "host_sort_family", pb.HostSortFamily(name="Pair", arity=2)
            )
        ),
        pb.Declaration(
            kind=Oneof[Literal["constructor"], pb.Constructor]("constructor", pb.Constructor(name="Pair", output=0))
        ),
    ]
    owner = Builder.from_program(
        pb.Program(
            ir_version=1,
            sorts=[
                pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Pair")),
                pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair", args=[0, 0]))),
            ],
            declarations=declarations[::-1] if reverse else declarations,
            nodes=[
                pb.Node(sort_id=0, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="Pair"))),
                pb.Node(
                    sort_id=1,
                    kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                        "primitive_value",
                        pb.PrimitiveValue(
                            value=Oneof[Literal["pair"], pb.PairValue]("pair", pb.PairValue(first=0, second=0))
                        ),
                    ),
                ),
            ],
        )
    ).publish()
    nominal, family = owner.ref("nodes", 0), owner.ref("nodes", 1)
    for roots in ([nominal, family], [family, nominal], clone_nodes([family]), [nominal, *clone_nodes([family])]):
        result = pack(roots)
        assert {_program._definition_key(d) for d in result.program.declarations} == {
            ("eq_sort", "Pair"),
            ("host_sort_family", "Pair"),
            ("callable", "Pair"),
        }
        payload = next(n.kind.value.value.value for n in result.program.nodes if n.kind.field == "primitive_value")
        assert isinstance(payload, pb.PairValue)
        assert payload.first == payload.second
    assert {_program._definition_key(d) for d in pack([nominal]).program.declarations} == {
        ("eq_sort", "Pair"),
        ("callable", "Pair"),
    }


@pytest.mark.parametrize("nominal", [False, True])
@pytest.mark.parametrize("local", [False, True])
def test_sort_dependency_never_falls_back_to_other_kind(nominal, local):
    builder = Builder()
    sort = builder.add(
        "sorts",
        pb.Sort(
            kind=Oneof[Literal["eq"], str]("eq", "Pair")
            if nominal
            else Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="Pair"))
        ),
    )
    wrong_kind = "host_sort_family" if nominal else "eq_sort"
    if local:
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="Pair")
                )
                if nominal
                else Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Pair"))
            ),
        )
    reference = builder.publish().ref("sorts", sort)
    with pytest.raises(ValueError, match="missing definitions"):
        pack([reference], ambient=() if local else {(wrong_kind, "Pair")})
    correct_kind = "eq_sort" if nominal else "host_sort_family"
    assert not pack([reference], ambient={(correct_kind, "Pair")}).program.declarations


@pytest.mark.parametrize("nominal", [False, True])
def test_same_kind_sort_definition_conflicts_remain_errors(nominal):
    builder = Builder()
    for index in range(2):
        builder.add(
            "declarations",
            pb.Declaration(
                doc=str(index),
                kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Pair"))
                if nominal
                else Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="Pair", arity=index)
                ),
            ),
        )
    owner = builder.publish()
    with pytest.raises(ValueError, match="same-name"):
        pack([owner.ref("declarations", 0), owner.ref("declarations", 1)])


@pytest.mark.parametrize("root_kind", ["calls", "definitions", "late_owner"])
def test_pack_reconciles_independent_compatible_definitions(monkeypatch, root_kind):
    def forbidden(*args, **kwargs):
        pytest.fail("Definition composition cannot construct an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    producers = []
    for offset in range(2):
        builder = Builder()
        if offset:
            builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="f64"))))
        scalar = builder.add(
            "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
        )
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="i64")
                )
            ),
        )
        definition = builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="composed", inputs=[pb.Arg(sort=scalar)], output=scalar)
                )
            ),
        )
        value = builder.add(
            "nodes",
            pb.Node(
                sort_id=scalar,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 4 + offset))
                ),
            ),
        )
        call = builder.add(
            "nodes",
            pb.Node(
                sort_id=scalar, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="composed", args=[value]))
            ),
        )
        owner = builder.publish()
        producers.append((owner.ref("sorts", scalar), owner.ref("declarations", definition), owner.ref("nodes", call)))
    left, right = producers
    assert left[1] != right[1]
    assert left[1].read().to_binary() != right[1].read().to_binary()  # Arena IDs differ.
    assert StructuralView(left[1]) == StructuralView(right[1])
    before = [reference.read().to_binary() for _, reference, _ in producers]
    if root_kind == "calls":
        roots = [left[2], right[2]]
    elif root_kind == "definitions":
        roots = [left[1], right[1]]
    else:
        builder = Builder()
        wrapper = builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive",
                    pb.Primitive(
                        name="wrapper",
                        output=builder.import_ref(left[0]),
                        body=builder.import_ref(right[2]),
                    ),
                )
            ),
        )
        roots = [builder.publish().ref("declarations", wrapper), left[1]]
    packed = pack(roots)
    assert sum(declaration.kind.value.name == "composed" for declaration in packed.program.declarations) == 1
    assert sum(declaration.kind.field == "host_sort_family" for declaration in packed.program.declarations) == 1
    assert [reference.read().to_binary() for _, reference, _ in producers] == before
    if root_kind == "definitions":
        assert packed.indices[0] == packed.indices[1]
    elif root_kind == "calls":
        assert packed.indices[0] != packed.indices[1]
        assert len(packed.program.nodes) == 4
    restored = Builder.from_program(pb.Program.from_binary(packed.program.to_binary())).publish()
    role = "nodes" if root_kind == "calls" else "declarations"
    assert (
        pack([restored.ref(role, index) for index in packed.indices]).program.to_binary() == packed.program.to_binary()
    )


@pytest.mark.parametrize("foreign_body_owner", [False, True])
def test_compatible_definition_does_not_hide_conflicting_named_dependency(foreign_body_owner):
    roots = []
    for value in (4, 9):
        builder = Builder()
        scalar = builder.add(
            "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
        )
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="i64")
                )
            ),
        )
        literal = builder.add(
            "nodes",
            pb.Node(
                sort_id=scalar,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
        )
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive", pb.Primitive(name="dependency", output=scalar, body=literal)
                )
            ),
        )
        call = builder.add(
            "nodes", pb.Node(sort_id=scalar, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="dependency")))
        )
        if foreign_body_owner:
            foreign = builder.publish()
            builder = Builder()
            scalar = builder.import_ref(foreign.ref("sorts", scalar))
            call = builder.import_ref(foreign.ref("nodes", call))
        wrapper = builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive", pb.Primitive(name="wrapper", output=scalar, body=call)
                )
            ),
        )
        roots.append(builder.publish().ref("declarations", wrapper))
    assert StructuralView(roots[0]) == StructuralView(roots[1])
    for ordered in (roots, list(reversed(roots))):
        with pytest.raises(ValueError, match=r"same-name.*dependency"):
            pack(ordered)


@pytest.mark.parametrize("body_kind", ["signed_zero", "union"])
def test_definition_reconciliation_preserves_literal_bits_and_union_identity(body_kind):
    roots = []
    for index in range(2):
        builder = Builder()
        if body_kind == "signed_zero":
            scalar = builder.add(
                "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="f64")))
            )
            builder.add(
                "declarations",
                pb.Declaration(
                    kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                        "host_sort_family", pb.HostSortFamily(name="f64")
                    )
                ),
            )
            kind = Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["f64_bits"], int]("f64_bits", index << 63))
            )
        else:
            scalar = builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E")))
            builder.add(
                "declarations",
                pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="E"))),
            )
            kind = Oneof[Literal["union"], pb.Union]("union", pb.Union())
        body = builder.add("nodes", pb.Node(sort_id=scalar, kind=kind))
        definition = builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive", pb.Primitive(name="body", output=scalar, body=body)
                )
            ),
        )
        roots.append(builder.publish().ref("declarations", definition))
    if body_kind == "signed_zero":
        assert StructuralView(roots[0]) == StructuralView(roots[1])
    else:
        assert StructuralView(roots[0]) != StructuralView(roots[1])
    with pytest.raises(ValueError, match="same-name"):
        pack(roots)


def test_reconciliation_does_not_discover_unreachable_foreign_bodies():
    roots = []
    for value in (4, 9):
        foreign = Builder()
        scalar = foreign.add(
            "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
        )
        body = foreign.add(
            "nodes",
            pb.Node(
                sort_id=scalar,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
        )
        foreign.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive", pb.Primitive(name="dependency", output=scalar, body=body)
                )
            ),
        )
        call = foreign.add(
            "nodes", pb.Node(sort_id=scalar, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="dependency")))
        )
        foreign_owner = foreign.publish()
        builder = Builder()
        scalar = builder.add(
            "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
        )
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                    "host_sort_family", pb.HostSortFamily(name="i64")
                )
            ),
        )
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive",
                    pb.Primitive(
                        name="unused",
                        output=scalar,
                        body=builder.import_ref(foreign_owner.ref("nodes", call)),
                    ),
                )
            ),
        )
        literal = builder.add(
            "nodes",
            pb.Node(
                sort_id=scalar,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", value))
                ),
            ),
        )
        roots.append(builder.publish().ref("nodes", literal))
    packed = pack(roots)
    assert [declaration.kind.value.name for declaration in packed.program.declarations] == ["i64"]
    assert len(packed.program.nodes) == 2


@pytest.mark.parametrize("mapped_first", [False, True])
def test_late_definition_conflict_is_checked_before_dirty_demand_clears(monkeypatch, mapped_first):
    foreign = Builder()
    scalar = foreign.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    )
    foreign.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                "host_sort_family", pb.HostSortFamily(name="i64")
            )
        ),
    )
    nine = foreign.add(
        "nodes",
        pb.Node(
            sort_id=scalar,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 9))
            ),
        ),
    )
    foreign.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["primitive"], pb.Primitive](
                "primitive", pb.Primitive(name="f", output=scalar, body=nine)
            )
        ),
    )
    call = foreign.add(
        "nodes", pb.Node(sort_id=scalar, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f")))
    )
    foreign_owner = foreign.publish()
    builder = Builder()
    scalar = builder.add(
        "sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))
    )
    builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                "host_sort_family", pb.HostSortFamily(name="i64")
            )
        ),
    )
    four = builder.add(
        "nodes",
        pb.Node(
            sort_id=scalar,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 4))
            ),
        ),
    )
    first = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["primitive"], pb.Primitive](
                "primitive", pb.Primitive(name="f", output=scalar, body=four)
            )
        ),
    )
    wrapper = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["primitive"], pb.Primitive](
                "primitive",
                pb.Primitive(
                    name="wrapper",
                    output=scalar,
                    body=builder.import_ref(foreign_owner.ref("nodes", call)),
                ),
            )
        ),
    )
    owner = builder.publish()
    emitted = owner.ref("declarations", first)

    def candidate_order(values):
        # Both legal set iteration orders must reject, including the mapped
        # representative fast path that used to skip a dirty-key comparison.
        return iter(sorted(values, key=lambda reference: (reference == emitted) != mapped_first))

    monkeypatch.setattr(_program, "iter", candidate_order, raising=False)
    with pytest.raises(ValueError, match=r"same-name.*'f'"):
        pack([owner.ref("declarations", wrapper), emitted])


def test_template_expansion_freshens_nodes_not_definition_identity():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Box")))
    builder.add(
        "declarations", pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Box")))
    )
    declaration = builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["constructor"], pb.Constructor]("constructor", pb.Constructor(name="box", output=0))
        ),
    )
    call = builder.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="box"))))
    group = builder.add("nodes", pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[call]))))
    owner = builder.publish()
    first, alias = clone_nodes([owner.ref("nodes", group), owner.ref("nodes", group)])
    (second,) = clone_nodes([owner.ref("nodes", group)])
    assert first == alias
    assert first != second
    assert first != owner.ref("nodes", group)
    cloned_call = first.owner.ref("nodes", first.read().kind.value.members[0])
    assert cloned_call != owner.ref("nodes", call)
    assert first.owner.ref("declarations", 0) == owner.ref("declarations", declaration)
    combined = pack([first, second, owner.ref("nodes", call)])
    assert len(combined.program.declarations) == 2
    assert sum(node.kind is not None and node.kind.field == "union" for node in combined.program.nodes) == 2


@pytest.fixture(autouse=True)
def no_live_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        msg = "Protobuf ownership must not instantiate an engine"
        raise AssertionError(msg)

    monkeypatch.setattr("egglog.bindings._ProtoEngine", forbidden)


def test_slot_permutation_and_protobuf_fields_are_authoritative():
    a, b, unused = literal(10), literal(20), literal(30)

    def authored(imports, args, *, func="f", kind=None):
        builder = Builder()
        builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
        for imported in imports:
            builder.import_ref(imported)
        root = builder.add(
            "nodes", pb.Node(kind=kind or Oneof[Literal["call"], pb.Call]("call", pb.Call(func=func, args=args)))
        )
        return pack([builder.publish().ref("nodes", root)], ambient=AMBIENT)

    original = authored([a, b, unused], [0, 1, 0])
    permuted = authored([b, unused, a], [2, 0, 2])
    assert original.program == permuted.program
    assert len(original.program.nodes) == 3
    call = original.program.nodes[original.indices[0]].kind.value
    assert call.args[0] == call.args[2] != call.args[1]
    changed = authored([b, unused, a], [0], func="g")
    assert changed.program.nodes[changed.indices[0]].kind.value.func == "g"
    assert len(changed.program.nodes) == 2
    variable = authored([a, b], [0, 1], kind=Oneof[Literal["var"], str]("var", "x"))
    assert len(variable.program.nodes) == 1
    assert variable.program.nodes[0].kind == Oneof[Literal["var"], str]("var", "x")


def test_adoption_publication_and_inspection_are_isolated():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    supplied = pb.Node(
        kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
            "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 7))
        )
    )
    index = builder.add("nodes", supplied)
    assert supplied.kind is not None
    assert supplied.kind.field == "primitive_value"
    supplied.kind.value.value = Oneof[Literal["i64"], int]("i64", 99)
    owner = builder.publish()
    ref = owner.ref("nodes", index)
    detached = ref.read()
    detached.kind.value.value = Oneof[Literal["i64"], int]("i64", 100)
    assert ref.read().kind.value.value.value == 7
    with pytest.raises(RuntimeError, match="published"):
        builder.add("nodes", supplied)
    with pytest.raises(RuntimeError, match="published"):
        builder.publish()
    packet = pack([ref], ambient=AMBIENT).program
    packed = packet.nodes[0]
    assert packed.kind is not None
    assert packed.kind.field == "primitive_value"
    packed.kind.value.value = Oneof[Literal["i64"], int]("i64", 101)
    assert ref.read().kind.value.value.value == 7


def test_diamonds_distinct_equal_unions_and_cycles():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E")))
    first = builder.reserve("nodes")
    child = builder.add(
        "nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="cycle", args=[first])))
    )
    builder.fill("nodes", first, pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[child]))))
    fresh = builder.add("nodes", pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
    other = builder.add("nodes", pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
    source = builder.publish()
    branches = []
    for _ in range(2):
        branch = Builder()
        root = branch.import_ref(source.ref("nodes", first))
        branches.append(branch.publish().ref("nodes", root))
    result = pack(
        [*branches, source.ref("nodes", fresh), source.ref("nodes", other)],
        ambient={("eq_sort", "E"), ("callable", "cycle")},
    )
    assert result.indices[0] == result.indices[1]
    assert result.indices[2] != result.indices[3]
    union = result.program.nodes[result.indices[0]]
    assert union.kind is not None
    assert union.kind.field == "union"
    call = result.program.nodes[union.kind.value.members[0]]
    assert call.kind is not None
    assert call.kind.field == "call"
    assert call.kind.value.args == [result.indices[0]]
    assert pb.Program.from_binary(result.program.to_binary()) == result.program


def test_named_recursive_definition_closure_and_missing_names():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    builder.add("nodes", pb.Node(kind=Oneof[Literal["var"], str]("var", "_0")))
    builder.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f", args=[0]))))
    builder.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="g", args=[0]))))
    builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["host_sort_family"], pb.HostSortFamily](
                "host_sort_family", pb.HostSortFamily(name="i64")
            )
        ),
    )
    for name, body in [("f", 2), ("g", 1), ("unreachable", 0)]:
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["primitive"], pb.Primitive](
                    "primitive", pb.Primitive(name=name, inputs=[pb.Arg(sort=0)], output=0, body=body)
                )
            ),
        )
    owner = builder.publish()
    result = pack([owner.ref("nodes", 1)])
    assert all(d.kind is not None for d in result.program.declarations)
    assert {d.kind.value.name for d in result.program.declarations if d.kind is not None} == {"f", "g", "i64"}
    assert len(result.program.nodes) == 3
    with pytest.raises(ValueError, match=r"missing.*i64"):
        pack([literal(1)])


def test_rules_rulesets_commands_files_and_command_multiplicity():
    builder = Builder()
    builder.add("files", pb.SourceFile(name="source", contents="x"))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    builder.add("nodes", pb.Node(kind=Oneof[Literal["var"], str]("var", "x"), span=pb.Span(file=0, end=1)))
    rule = pb.RuleDecl(
        kind=Oneof[Literal["rule"], pb.Rule](
            "rule", pb.Rule(query=[0], head=[pb.Action(kind=Oneof[Literal["term"], int]("term", 0))])
        ),
        eval_mode=pb.RuleEvalMode.NAIVE,
    )
    builder.add("rules", rule)
    builder.add("rules", rule)
    builder.add(
        "rulesets",
        pb.Ruleset(name="r", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList(rules=[0, 0, 1]))),
    )
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["repeat"], pb.Repeat](
                "repeat",
                pb.Repeat(
                    times=2,
                    until=[0],
                    body=[
                        pb.Command(
                            kind=Oneof[Literal["run"], pb.Run](
                                "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", "r")))
                            )
                        )
                    ],
                ),
            )
        ),
    )
    owner = builder.publish()
    reference = owner.ref("commands", command)
    result = pack(commands=[reference, reference], ambient=AMBIENT)
    assert len(result.program.commands) == 2
    assert len(result.program.rules) == 2
    ruleset = result.program.rulesets[0]
    assert ruleset.kind is not None
    assert ruleset.kind.field == "rules"
    rules = ruleset.kind.value.rules
    assert rules[0] == rules[1] != rules[2]
    assert result.program.files[0].name == "source"
    assert result.program.nodes[0].span is not None
    assert result.program.nodes[0].span.file == 0


def test_binders_defaults_and_payloads_are_not_reinterpreted():
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    builder.add("nodes", pb.Node(sort_id=1, kind=Oneof[Literal["var"], str]("var", "_0")))
    for name in ("left", "right"):
        builder.add(
            "declarations",
            pb.Declaration(
                kind=Oneof[Literal["host_primitive"], pb.HostPrimitive](
                    "host_primitive",
                    pb.HostPrimitive(
                        name=name,
                        typing=Oneof[Literal["signature"], pb.GenericSignature](
                            "signature", pb.GenericSignature(type_params=["T"], inputs=[pb.Arg(sort=0)], output=0)
                        ),
                    ),
                ),
                bindings=pb.CallableBindings(
                    python=pb.PythonBindings(
                        views=[
                            pb.PythonCallable(
                                kind=pb.PythonCallKind.FUNCTION,
                                path=[name],
                                params=[pb.PythonParameter(core_input=0, name="x", default_expr=0)],
                            )
                        ]
                    )
                ),
            ),
        )
    owner = builder.publish()
    result = pack([owner.ref("declarations", 1), owner.ref("declarations", 0)], ambient=AMBIENT)
    assert result.program.sorts[0].kind == Oneof[Literal["var"], int]("var", 0)
    assert result.program.nodes[0].kind == Oneof[Literal["var"], str]("var", "_0")
    assert len(result.program.declarations) == 2
    for declaration in result.program.declarations:
        assert declaration.kind is not None
        assert declaration.kind.field == "host_primitive"
        typing = declaration.kind.value.typing
        assert typing is not None
        assert typing.field == "signature"
        assert typing.value.type_params == ["T"]
        assert declaration.bindings is not None
        assert declaration.bindings.python is not None
        assert declaration.bindings.python.views[0].params[0].core_input == 0
    # Storage does not claim this free-variable default is semantically valid;
    # shared adapter binding checks must reject it if submitted for execution.


def test_linear_authoring_and_pack(monkeypatch):
    source = literal(1)
    count = 0
    original = _program._copy_record

    def counted(record, *args, **kwargs):
        nonlocal count
        if isinstance(record, pb.Node):
            count += 1
        return original(record, *args, **kwargs)

    monkeypatch.setattr(_program, "_copy_record", counted)
    current = source
    owners = []
    for _ in range(10_000):
        builder = Builder()
        builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
        left = builder.import_ref(current)
        right = builder.import_ref(source)
        index = builder.add(
            "nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f", args=[left, right])))
        )
        current = builder.publish().ref("nodes", index)
        owners.append(weakref(current.owner))
    assert count == 10_000
    count = 0
    result = pack([current, current], ambient=AMBIENT)
    assert count == 10_001
    assert len(result.program.nodes) == 10_001
    assert result.indices[0] == result.indices[1]
    assert len(pb.Program.from_binary(result.program.to_binary()).nodes) == 10_001
    del current
    gc.collect()
    assert all(owner() is None for owner in owners)
    assert source.read().kind.value.value.value == 1


def test_inventory_covers_all_program_message_fields():
    _program._check_inventory()


@pytest.mark.parametrize("message", ["PythonCallable", "RustCallable"])
def test_tail_repetitions_must_be_classified_in_inventory(monkeypatch, message):
    monkeypatch.delitem(_program._FIELD_ROLES[message], "tail_repetitions")
    with pytest.raises(NotImplementedError, match="Unclassified protobuf fields"):
        _program._check_inventory()


@pytest.mark.parametrize("repetitions", [None, 0, 2])
def test_scalar_view_presence_survives_owned_closure_and_relocation(repetitions):
    packet = pack([join("a", "b").__egg_ref__])
    program = packet.program
    declaration = next(d for d in program.declarations if d.kind.value.name == "egglog.core.string.concat")
    declaration_bindings = declaration.bindings
    assert declaration_bindings is not None
    assert declaration_bindings.python is not None
    assert declaration_bindings.rust is not None
    python_view, rust_view = declaration_bindings.python.views[0], declaration_bindings.rust.views[0]
    if repetitions is not None:
        python_view.tail_repetitions = rust_view.tail_repetitions = repetitions
        python_view.params = [pb.PythonParameter(core_input=i, name=f"part{i}") for i in range(repetitions)]
        rust_view.params = [pb.RustParameter(core_input=i, name=f"part{i}") for i in range(repetitions)]
    owner = Builder.from_program(program).publish()
    reference = owner.ref("declarations", program.declarations.index(declaration))
    detached = reference.read()
    assert isinstance(detached, pb.Declaration)
    detached_bindings = detached.bindings
    assert detached_bindings is not None
    assert detached_bindings.python is not None
    assert detached_bindings.rust is not None
    detached_bindings.python.views[0].tail_repetitions = 7
    detached_bindings.rust.views[0].tail_repetitions = 7
    builder = Builder()
    for index in reversed(range(len(program.sorts))):
        builder.import_ref(owner.ref("sorts", index))
    for index in reversed(range(len(program.declarations))):
        builder.import_ref(owner.ref("declarations", index))
    cloned = clone_nodes([owner.ref("nodes", packet.indices[0])])[0]
    index = builder.import_ref(cloned)
    packed = pack([builder.publish().ref("nodes", index)]).program
    reencoded = pb.Program.from_binary(packed.to_binary())
    assert reencoded == packed
    for records in (
        reference.read(),
        next(d for d in reencoded.declarations if d.kind.value.name == "egglog.core.string.concat"),
    ):
        assert isinstance(records, pb.Declaration)
        record_bindings = records.bindings
        assert record_bindings is not None
        assert record_bindings.python is not None
        assert record_bindings.rust is not None
        for view in (record_bindings.python.views[0], record_bindings.rust.views[0]):
            assert view.has_field("tail_repetitions") is (repetitions is not None)
            if repetitions is not None:
                assert view.tail_repetitions == repetitions
        record_kind = records.kind
        assert record_kind is not None
        assert record_kind.field == "host_primitive"
        signature_kind = record_kind.value.typing
        assert signature_kind is not None
        assert signature_kind.field == "signature"
        signature = signature_kind.value
        assert not signature.inputs
        assert len(signature.varargs) == 1
    call = next(n.kind.value for n in reencoded.nodes if n.kind.field == "call")
    assert isinstance(call, pb.Call)
    assert [reencoded.nodes[i].kind.value.value.value for i in call.args] == ["a", "b"]


def test_scalar_literals_publish_and_pack_exact_payloads():
    values = [String("a\0b\n☃"), Bool(False), Unit()]
    result = pack([value.__egg_ref__ for value in values])
    decoded = pb.Program.from_binary(result.program.to_binary())
    payloads = [decoded.nodes[index].kind.value.value for index in result.indices]
    assert [value.field for value in payloads] == ["string", "bool", "unit"]
    assert payloads[0].value == "a\0b\n☃"
    assert payloads[1].value is False
    assert payloads[2].value == pb.Unit()


def test_owner_keepalive_is_lifetime_only_not_semantic_storage():
    class LocalContext:
        pass

    context = LocalContext()
    watched = weakref(context)
    builder = Builder()
    builder.import_ref(literal(7))
    owner = builder.publish(keepalive=(context,))
    del context
    gc.collect()
    assert watched() is not None
    assert pack([owner.ref("nodes", 0)], ambient=AMBIENT).program == pack([literal(7)], ambient=AMBIENT).program
    del owner
    gc.collect()
    assert watched() is None


def test_proof_command_relocation_and_constructor_closure():
    builder = Builder()
    unused = builder.import_ref(literal(999))
    fact = builder.import_ref(literal(7))
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "Box")))
    builder.add(
        "declarations", pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="Box")))
    )
    builder.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["constructor"], pb.Constructor]("constructor", pb.Constructor(name="box", output=0))
        ),
    )
    prove = builder.add(
        "commands", pb.Command(kind=Oneof[Literal["prove"], pb.Prove]("prove", pb.Prove(facts=[fact, fact])))
    )
    exists = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="box"))
        ),
    )
    missing = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["prove_exists"], pb.ProveExists]("prove_exists", pb.ProveExists(constructor="missing"))
        ),
    )
    owner = builder.publish()
    result = pack(commands=[owner.ref("commands", prove), owner.ref("commands", exists)], ambient=AMBIENT).program
    prove_kind = result.commands[0].kind
    assert prove_kind is not None
    assert prove_kind.field == "prove"
    assert prove_kind.value.facts == [0, 0]
    assert unused != fact == 1
    assert len(result.nodes) == 1
    scalar_kind = result.nodes[0].kind
    assert scalar_kind is not None
    assert scalar_kind.field == "primitive_value"
    scalar_value = scalar_kind.value.value
    assert scalar_value is not None
    assert scalar_value.field == "i64"
    assert scalar_value.value == 7
    exists_kind = result.commands[1].kind
    assert exists_kind is not None
    assert exists_kind.field == "prove_exists"
    assert exists_kind.value.constructor == "box"
    assert {declaration.kind.value.name for declaration in result.declarations} == {"box", "Box"}
    assert pb.Program.from_binary(result.to_binary()) == result
    with pytest.raises(ValueError, match=r"missing.*missing"):
        pack(commands=[owner.ref("commands", missing)])


@pytest.mark.parametrize(("message", "field", "role"), _program._index_fields())
def test_each_index_role_remaps_zero_and_preserves_presence(message, field, role):
    record = message()
    descriptor = next(f for f in message.desc().fields if f.name == field)
    repeated = isinstance(descriptor.value, DescFieldValueList)
    record[descriptor] = [0, 0] if repeated else 0
    seen = []

    def relocate(actual_role, index):
        seen.append((actual_role, index))
        return 17

    cloned = _program._copy_record(record, relocate=relocate)
    assert cloned[descriptor] == ([17, 17] if repeated else 17)
    assert (role, 0) in seen
    assert record[descriptor] == ([0, 0] if repeated else 0)


def test_rejects_invalid_records_slots_and_inline_cycles():
    builder = Builder()
    with pytest.raises(TypeError):
        builder.add("nodes", pb.Sort())
    builder.reserve("nodes")
    with pytest.raises(ValueError, match="unfilled"):
        builder.publish()
    malformed = Builder()
    malformed.add("sorts", pb.Sort(kind=Oneof[Literal["var"], int]("var", 0)))
    root = malformed.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f", args=[9]))))
    with pytest.raises(IndexError):
        pack([malformed.publish().ref("nodes", root)], ambient=AMBIENT)
    command = pb.Command(kind=Oneof[Literal["repeat"], pb.Repeat]("repeat", pb.Repeat(times=1)))
    assert command.kind is not None
    assert command.kind.field == "repeat"
    command.kind.value.body.append(command)
    with pytest.raises(ValueError, match=r"inline.*cycle"):
        Builder().add("commands", command)
    unknown = pb.Node.from_binary(b"\xf8\x07\x01")
    with pytest.raises(NotImplementedError, match="unknown"):
        Builder().add("nodes", unknown)


def test_detached_default_expansions_preserve_internal_identity():
    default = Builder()
    default.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E")))
    default.add("nodes", pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
    root = default.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f", args=[0, 0]))))
    owner = default.publish()
    packet = pack([owner.ref("nodes", root)], ambient={*AMBIENT, ("eq_sort", "E")})
    refs = []
    for _ in range(2):
        imported = Builder.from_program(packet.program).publish()
        refs.append(imported.ref("nodes", packet.indices[0]))
    result = pack(refs, ambient={*AMBIENT, ("eq_sort", "E")})
    args = []
    for index in result.indices:
        node = result.program.nodes[index]
        assert node.kind is not None
        assert node.kind.field == "call"
        args.append(node.kind.value.args)
    assert args[0][0] == args[0][1] != args[1][0] == args[1][1]


def test_import_cache_is_not_authoritative():
    value = literal(1)
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    first = builder.import_ref(value)
    builder._imports.clear()
    second = builder.import_ref(value)
    root = builder.add(
        "nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f", args=[first, second])))
    )
    result = pack([builder.publish().ref("nodes", root)], ambient=AMBIENT)
    node = result.program.nodes[result.indices[0]]
    assert node.kind is not None
    assert node.kind.field == "call"
    args = node.kind.value.args
    assert args[0] == args[1]
    assert len(result.program.nodes) == 2


def test_rejects_mutable_scalar_payload_instead_of_aliasing():
    payload = bytearray(b"payload")
    record = pb.Node(
        kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
            "primitive_value",
            pb.PrimitiveValue(
                value=Oneof[Literal["custom"], pb.CustomValue]("custom", pb.CustomValue(payload=payload))  # type: ignore[arg-type]
            ),
        )
    )
    with pytest.raises(TypeError, match="scalar"):
        Builder().add("nodes", record)


def test_late_discovered_definition_conflict_is_not_hidden():
    foreign = Builder()
    foreign.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    foreign.add("nodes", pb.Node(kind=Oneof[Literal["var"], str]("var", "_0")))
    foreign.add(
        "declarations",
        pb.Declaration(kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name="f", output=0))),
    )
    foreign_owner = foreign.publish()
    local = Builder()
    local.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
    body = local.import_ref(foreign_owner.ref("nodes", 0))
    root = local.add("nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f"))))
    local.add(
        "declarations",
        pb.Declaration(
            kind=Oneof[Literal["primitive"], pb.Primitive]("primitive", pb.Primitive(name="f", output=0, body=body))
        ),
    )
    with pytest.raises(ValueError, match="same-name"):
        pack([local.publish().ref("nodes", root)], ambient=AMBIENT)


def test_anonymous_rulesets_do_not_define_the_default_ruleset():
    builder = Builder()
    builder.add("rulesets", pb.Ruleset(kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList())))
    builder.add("rulesets", pb.Ruleset(kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList())))
    builder.add("rulesets", pb.Ruleset(name="", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList())))
    command = builder.add(
        "commands",
        pb.Command(
            kind=Oneof[Literal["run"], pb.Run](
                "run", pb.Run(ruleset=pb.RulesetRef(kind=Oneof[Literal["name"], str]("name", "")))
            )
        ),
    )
    owner = builder.publish()
    result = pack(commands=[owner.ref("commands", command)])
    assert len(result.program.rulesets) == 1
    assert result.program.rulesets[0].has_field("name")


@pytest.mark.parametrize("index", [True, -1, 2**32, "0"])
def test_rejects_invalid_index_scalars(index):
    with pytest.raises(TypeError, match="arena index"):
        Builder().add("nodes", pb.Node(sort_id=index))


def test_rejects_non_string_dependency_names():
    with pytest.raises(TypeError, match="definition name"):
        Builder().add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name=5))))  # type: ignore[arg-type]


def test_distinct_same_name_ruleset_roots_are_rejected():
    refs = []
    for _ in range(2):
        builder = Builder()
        root = builder.add(
            "rulesets", pb.Ruleset(name="r", kind=Oneof[Literal["rules"], pb.RuleList]("rules", pb.RuleList()))
        )
        refs.append(builder.publish().ref("rulesets", root))
    with pytest.raises(ValueError, match="same-name"):
        pack(refs)


def test_structural_view_ignores_addresses_and_preserves_identity():
    first, second = literal(10), literal(10)
    assert first != second
    assert StructuralView(first) == StructuralView(second)
    assert hash(StructuralView(first)) == hash(StructuralView(second))
    assert StructuralView(first) != StructuralView(literal(20))
    builder = Builder()
    builder.add("sorts", pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "E")))
    for _ in range(2):
        builder.add("nodes", pb.Node(kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())))
    owner = builder.publish()
    assert StructuralView(owner.ref("nodes", 0)) == StructuralView(owner.ref("nodes", 0))
    assert StructuralView(owner.ref("nodes", 0)) != StructuralView(owner.ref("nodes", 1))


def test_structural_view_zero_hash_and_origin_are_distinct_from_wire_bits():
    refs = []
    for bits in (0, 1 << 63):
        builder = Builder()
        builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="f64"))))
        root = builder.add(
            "nodes",
            pb.Node(
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["f64_bits"], int]("f64_bits", bits))
                )
            ),
        )
        refs.append(builder.publish().ref("nodes", root))
    assert refs[0].read() != refs[1].read()
    assert StructuralView(refs[0]) == StructuralView(refs[1])
    assert hash(StructuralView(refs[0])) == hash(StructuralView(refs[1]))


def test_structural_view_deep_hash_cache_is_erasable_and_does_not_retain_owners(monkeypatch):
    original = _program._structural_parts
    visits = 0

    def counted(reference):
        nonlocal visits
        visits += 1
        return original(reference)

    monkeypatch.setattr(_program, "_structural_parts", counted)
    _program._STRUCTURAL_HASHES.clear()
    root = literal(1)
    owners = []
    for _ in range(10_000):
        builder = Builder()
        builder.add("sorts", pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64"))))
        child = builder.import_ref(root)
        index = builder.add(
            "nodes", pb.Node(kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="f", args=[child, child])))
        )
        root = builder.publish().ref("nodes", index)
        owners.append(weakref(root.owner))
        hash(StructuralView(root))
    assert visits == 4 * 10_001
    first_hash = hash(StructuralView(root))
    assert visits == 4 * 10_001
    _program._STRUCTURAL_HASHES.clear()
    assert hash(StructuralView(root)) == first_hash
    assert visits == 8 * 10_001
    del root
    gc.collect()
    assert all(owner() is None for owner in owners)
