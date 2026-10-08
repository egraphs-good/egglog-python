"""Storage and relocation tests, separate from engine conformance."""

from __future__ import annotations

import gc
from typing import Literal
from weakref import ref as weakref

import pytest
from egglog_proto.egglog.v1 import egglog_pb as pb
from protobuf import DescFieldValueList, Oneof

from egglog import _program
from egglog._program import Builder, pack


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


AMBIENT = frozenset({("sort", "i64"), ("callable", "f"), ("callable", "g")})


@pytest.fixture(autouse=True)
def no_live_engine(monkeypatch):
    def forbidden(*args, **kwargs):
        msg = "Protobuf ownership must not instantiate an engine"
        raise AssertionError(msg)

    monkeypatch.setattr("egglog.bindings.EGraph", forbidden)


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
        ambient={("sort", "E"), ("callable", "cycle")},
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
    packet = pack([owner.ref("nodes", root)], ambient={*AMBIENT, ("sort", "E")})
    refs = []
    for _ in range(2):
        imported = Builder.from_program(packet.program).publish()
        refs.append(imported.ref("nodes", packet.indices[0]))
    result = pack(refs, ambient={*AMBIENT, ("sort", "E")})
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
