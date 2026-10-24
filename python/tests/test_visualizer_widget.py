"""Inert display projection, not native engine conformance."""

import json
from typing import Literal

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import Oneof

from egglog.visualizer_widget import _snapshot_to_data


@pytest.fixture
def snapshot():
    program = pb.Program(
        sorts=[
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "A")),
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="i64"))
            ),
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="String"))
            ),
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication](
                    "application", pb.SortApplication(name="Vec", args=[0])
                )
            ),
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="Unit"))
            ),
        ],
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="A"))),
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name="C", inputs=[pb.Arg(sort=1)], output=0)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["constructor"], pb.Constructor](
                    "constructor", pb.Constructor(name="Wrap", inputs=[pb.Arg(sort=0)], output=0)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function](
                    "function", pb.Function(name="cell", inputs=[pb.Arg(sort=0)], output=3)
                )
            ),
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name="scalar", output=2))
            ),
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name="unit", output=4))
            ),
            pb.Declaration(
                kind=Oneof[Literal["relation"], pb.Relation](
                    "relation", pb.Relation(name="member", inputs=[pb.Arg(sort=0)])
                )
            ),
        ],
        nodes=[
            pb.Node(sort_id=0, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union(members=[2, 3]))),
            pb.Node(
                sort_id=1,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 1))
                ),
            ),
            pb.Node(sort_id=0, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="C", args=[1]))),
            pb.Node(sort_id=0, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="Wrap", args=[0]))),
            pb.Node(sort_id=0, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())),
            pb.Node(sort_id=0, kind=Oneof[Literal["union"], pb.Union]("union", pb.Union())),
            pb.Node(
                sort_id=3,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(
                        value=Oneof[Literal["vec"], pb.ValueList]("vec", pb.ValueList(items=[0, 0, 4, 5]))
                    ),
                ),
            ),
            pb.Node(
                sort_id=2,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(value=Oneof[Literal["string"], str]("string", 'quote"\n</script>')),
                ),
            ),
            pb.Node(
                sort_id=4,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["unit"], pb.Unit]("unit", pb.Unit()))
                ),
            ),
            pb.Node(sort_id=4, kind=Oneof[Literal["call"], pb.Call]("call", pb.Call(func="member", args=[0]))),
        ],
        commands=[
            *[
                pb.Command(
                    kind=Oneof[Literal["action"], pb.Action](
                        "action", pb.Action(kind=Oneof[Literal["term"], int]("term", index))
                    )
                )
                for index in (0, 4, 5, 9)
            ],
            *[
                pb.Command(
                    kind=Oneof[Literal["action"], pb.Action](
                        "action",
                        pb.Action(
                            kind=Oneof[Literal["set"], pb.Set](
                                "set", pb.Set(target=pb.Call(func=name, args=args), value=value)
                            )
                        ),
                    )
                )
                for name, args, value in [("cell", [0], 6), ("scalar", [], 7), ("unit", [], 8)]
            ],
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action", pb.Action(kind=Oneof[Literal["subsume"], pb.Call]("subsume", pb.Call(func="C", args=[1])))
                )
            ),
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["set_cost"], pb.SetCost](
                            "set_cost", pb.SetCost(target=pb.Call(func="C", args=[1]), cost=1)
                        )
                    ),
                )
            ),
        ],
    )
    return pb.EGraphSnapshot(program=program, options=pb.EGraphOptions(cost_sort=1))


def test_snapshot_projection_preserves_classes_rows_cycles_and_subsumption(snapshot):
    before = snapshot.to_binary()
    data = _snapshot_to_data(snapshot, split_primitive_outputs=False)
    nodes = data["nodes"]
    by_op = {node["op"]: node for node in nodes.values()}
    constructor = by_op["C"]
    assert constructor["subsumed"]
    assert constructor["cost"] == "1"
    wrapper = by_op["Wrap"]
    assert wrapper["eclass"] == constructor["eclass"]
    assert nodes[wrapper["children"][0]]["eclass"] == constructor["eclass"]
    empty = [node["eclass"] for node in nodes.values() if node["op"] == "∅"]
    assert len(set(empty)) == 2
    vector = by_op["Vec[A]"]
    classes = [nodes[child]["eclass"] for child in vector["children"]]
    assert classes[:2] == [constructor["eclass"]] * 2
    assert set(classes[2:]) == set(empty)
    assert by_op["cell"]["eclass"] == vector["eclass"]
    assert by_op["scalar"]["eclass"] == by_op[repr('quote"\n</script>')]["eclass"]
    assert by_op["unit"]["eclass"] == by_op["Unit"]["eclass"]
    assert by_op["member"]["eclass"] != by_op["Unit"]["eclass"]
    assert json.loads(json.dumps(data)) == data
    assert snapshot.to_binary() == before


def test_snapshot_function_type_labels_use_only_owned_records(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Detached snapshot display attempted provider lookup")

    monkeypatch.setattr("egglog.runtime._definition_at", forbidden)
    program = pb.Program(
        sorts=[
            pb.Sort(kind=Oneof[Literal["eq"], str]("eq", "A")),
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="String"))
            ),
            pb.Sort(
                kind=Oneof[Literal["application"], pb.SortApplication](
                    "application", pb.SortApplication(name="Vec", args=[0])
                )
            ),
            pb.Sort(kind=Oneof[Literal["func"], pb.FuncSort]("func", pb.FuncSort(params=[2], result=1))),
        ],
        declarations=[
            pb.Declaration(kind=Oneof[Literal["eq_sort"], pb.EqSort]("eq_sort", pb.EqSort(name="A"))),
            pb.Declaration(
                kind=Oneof[Literal["function"], pb.Function]("function", pb.Function(name="callback", output=3))
            ),
        ],
        nodes=[
            pb.Node(
                sort_id=1,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["string"], str]("string", "result"))
                ),
            ),
            pb.Node(
                sort_id=3,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(value=Oneof[Literal["lambda"], pb.Lambda]("lambda", pb.Lambda(body=0))),
                ),
            ),
        ],
        commands=[
            pb.Command(
                kind=Oneof[Literal["action"], pb.Action](
                    "action",
                    pb.Action(
                        kind=Oneof[Literal["set"], pb.Set]("set", pb.Set(target=pb.Call(func="callback"), value=1))
                    ),
                )
            )
        ],
    )
    snapshot = pb.EGraphSnapshot(program=program)
    before = snapshot.to_binary()
    data = _snapshot_to_data(snapshot)
    assert data["class_data"]["value-1"]["type"] == "UnstableFn[String, Vec[A]]"
    assert snapshot.to_binary() == before


@pytest.mark.parametrize("assigned", [False, True])
def test_snapshot_costs_preserve_declared_value_and_assigned_override(snapshot, assigned):
    program = snapshot.program
    constructor = next(d.kind.value for d in program.declarations if d.kind.value.name == "C")
    constructor.cost = len(program.nodes)
    program.nodes.append(
        pb.Node(
            sort_id=1,
            kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                "primitive_value", pb.PrimitiveValue(value=Oneof[Literal["i64"], int]("i64", 7))
            ),
        )
    )
    if not assigned:
        program.commands = [command for command in program.commands if command.kind.value.kind.field != "set_cost"]
    data = _snapshot_to_data(snapshot)
    assert next(node["cost"] for node in data["nodes"].values() if node["op"] == "C") == ("1" if assigned else "7")


@pytest.mark.parametrize("assigned", [False, True])
def test_snapshot_custom_costs_never_invoke_codec_or_debug(snapshot, monkeypatch, assigned):
    def forbidden(*args, **kwargs):
        pytest.fail("Custom cost inspection must not materialize host values")

    monkeypatch.setattr("cloudpickle.loads", forbidden)
    program = snapshot.program
    cost_sort = len(program.sorts)
    program.sorts.append(
        pb.Sort(
            kind=Oneof[Literal["application"], pb.SortApplication]("application", pb.SortApplication(name="Opaque"))
        )
    )
    first = len(program.nodes)
    for payload in (b"declared", b"assigned!"):
        program.nodes.append(
            pb.Node(
                sort_id=cost_sort,
                kind=Oneof[Literal["primitive_value"], pb.PrimitiveValue](
                    "primitive_value",
                    pb.PrimitiveValue(
                        value=Oneof[Literal["custom"], pb.CustomValue](
                            "custom", pb.CustomValue(payload=payload, args=[4])
                        )
                    ),
                ),
            )
        )
    constructor = next(d.kind.value for d in program.declarations if d.kind.value.name == "C")
    constructor.cost = first
    snapshot.options.cost_sort = cost_sort
    if assigned:
        next(
            command.kind.value.kind.value for command in program.commands if command.kind.value.kind.field == "set_cost"
        ).cost = first + 1
    else:
        program.commands = [command for command in program.commands if command.kind.value.kind.field != "set_cost"]
    before = snapshot.to_binary()
    data = _snapshot_to_data(snapshot)
    cost = next(node["cost"] for node in data["nodes"].values() if node["op"] == "C")
    assert cost == f"Opaque(<{9 if assigned else 8} opaque bytes>)(@class-4)"
    assert "declared" not in json.dumps(data)
    assert "assigned!" not in json.dumps(data)
    assert snapshot.to_binary() == before


@pytest.mark.parametrize("kind", ["set", "multiset", "pair", "map", "maybe", "none", "custom"])
def test_snapshot_container_edges_are_inert_and_ordered(snapshot, monkeypatch, kind):
    def forbidden(*args, **kwargs):
        pytest.fail("Visualization must not materialize custom values")

    monkeypatch.setattr("cloudpickle.loads", forbidden)
    program = snapshot.program
    variants = {
        "set": ("Set", [0], pb.ValueList(items=[0, 4, 5])),
        "multiset": ("MultiSet", [0], pb.ValueList(items=[0, 0, 4, 5])),
        "pair": ("Pair", [0, 0], pb.PairValue(first=4, second=5)),
        "map": ("Map", [0, 0], pb.MapValue(entries=[pb.MapEntry(key=0, value=4), pb.MapEntry(key=0, value=5)])),
        "maybe": ("Maybe", [0], pb.MaybeValue(value=0)),
        "none": ("Maybe", [0], pb.MaybeValue()),
        "custom": ("Opaque", [], pb.CustomValue(payload=b"not a pickle", args=[0, 4, 5])),
    }
    name, args, value = variants[kind]
    program.sorts[3].kind.value.name, program.sorts[3].kind.value.args = name, args
    program.nodes[6].kind.value.value = Oneof("maybe" if kind == "none" else kind, value)
    before = snapshot.to_binary()
    data = _snapshot_to_data(snapshot, split_primitive_outputs=False)
    container = data["nodes"]["node-6"]
    expected = {
        "set": [0, 4, 5],
        "multiset": [0, 0, 4, 5],
        "pair": [4, 5],
        "map": [0, 4, 0, 5],
        "maybe": [0],
        "none": [],
        "custom": [0, 4, 5],
    }[kind]
    assert [data["nodes"][child]["eclass"] for child in container["children"]] == [f"class-{i}" for i in expected]
    assert "not a pickle" not in json.dumps(data)
    assert snapshot.to_binary() == before


@pytest.mark.parametrize(
    "options", [{"max_functions": 0}, {"max_calls_per_function": 0}, {"n_inline_leaves": 2}, {"split_functions": {"C"}}]
)
def test_snapshot_display_filters_keep_edges_valid_and_snapshot_exact(snapshot, options):
    before = snapshot.to_binary()
    if "max_functions" in options or "max_calls_per_function" in options:
        with pytest.warns(UserWarning, match="Omitted|Truncated"):
            data = _snapshot_to_data(snapshot, **options)
    else:
        data = _snapshot_to_data(snapshot, **options)
    assert sum(node["op"] == "∅" for node in data["nodes"].values()) == 2
    assert all(child in data["nodes"] for node in data["nodes"].values() for child in node["children"])
    assert snapshot.to_binary() == before
