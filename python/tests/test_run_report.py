# mypy: disable-error-code="empty-body"
from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pytest

from egglog import *
from egglog import bindings
from egglog.run_report import RuleKey


def test_proto_report_projection_preserves_grouping_rounding_and_ownership(monkeypatch):
    from typing import Literal  # noqa: PLC0415

    from egglog_proto.egglog.v1 import egglog_pb as pb  # noqa: PLC0415
    from protobuf import Oneof  # noqa: PLC0415

    from egglog._program import pack  # noqa: PLC0415
    from egglog.declarations import Ident  # noqa: PLC0415
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        pytest.fail("Report projection accessed an engine")

    monkeypatch.setattr(bindings, "_ProtoEngine", forbidden)
    roots = [ruleset(rule(name="same").then())._snapshot({}) for _ in range(2)]
    request = pack(roots)
    assert request.rulesets == tuple(roots)
    state = object.__new__(EGraphState)
    state.handle = None
    state.definitions = {}
    state._ruleset_identifiers = dict.fromkeys(roots, Ident("logical"))
    response = pb.RunProgramResponse(
        rules=[
            pb.RuleAttribution(
                root=pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", index)), position=0, name="same"
            )
            for index in range(2)
        ]
    )
    native = pb.NativeRunReport(
        updated=False,
        can_stop=True,
        search_and_apply_time_per_rule=[
            pb.RuleDurationEntry(
                rule=index,
                direction=pb.NativeRuleDirection.SINGLE,
                native_name=str(index),
                value=pb.ReportDuration(nanos=600),
            )
            for index in range(2)
        ],
        num_matches_per_rule=[
            pb.RuleCountEntry(
                rule=index, direction=pb.NativeRuleDirection.SINGLE, native_name=str(index), value=index + 2
            )
            for index in range(2)
        ],
        search_and_apply_time_per_ruleset=[
            pb.RulesetDurationEntry(
                ruleset=attribution.root, native_name=str(index), value=pb.ReportDuration(nanos=600)
            )
            for index, attribution in enumerate(response.rules)
        ],
        iterations=[
            pb.NativeIterationReport(
                rule_set_report=pb.NativeRuleSetReport(
                    changed=False,
                    search_and_apply_time=pb.ReportDuration(nanos=2200),
                    merge_time=pb.ReportDuration(),
                    rule_reports=[
                        pb.NativeRuleReportsEntry(
                            rule=index,
                            direction=pb.NativeRuleDirection.SINGLE,
                            native_name=str(index),
                            reports=[
                                pb.NativeRuleReport(
                                    plan=None if index == 0 else pb.ReportPlan(),
                                    search_and_apply_time=pb.ReportDuration(nanos=1600),
                                    num_matches=index + 2,
                                )
                            ],
                        )
                        for index in range(2)
                    ],
                ),
                rebuild_time=pb.ReportDuration(seconds=1, nanos=1999),
            )
        ],
    )
    result = RunReport._from_proto(native, response, request, state)
    assert result._decls is None
    assert not result.updated
    assert result.can_stop
    assert len(result.num_matches_per_rule) == 1
    (key,) = result.num_matches_per_rule
    assert isinstance(key, RuleKey)
    assert key.__egg_ref__.read().name == "same"
    assert result.num_matches_per_rule[key] == 5
    assert result.search_and_apply_time_per_rule[key] == timedelta()  # truncate each physical row first
    assert result.search_and_apply_time_per_ruleset == {"logical": timedelta(microseconds=1)}
    assert result.merge_time_per_ruleset == result.rebuild_time_per_ruleset == {}
    (iteration,) = result.iterations
    assert iteration.rebuild_time == timedelta(seconds=1, microseconds=1)
    assert iteration.rule_set_report.search_and_apply_time == timedelta(microseconds=2)
    assert [report.num_matches for report in iteration.rule_set_report.rule_reports[key]] == [2, 3]
    assert iteration.rule_set_report.rule_reports[key][0].plan is None
    planned = iteration.rule_set_report.rule_reports[key][1]
    assert planned.plan is not None
    assert planned.plan.stages == []
    assert "rule(name='same').then()" in repr(result)
    state._ruleset_identifiers.clear()
    with pytest.raises(KeyError):
        RunReport._from_proto(native, response, request, state)
    assert repr(key) == "rule(name='same').then()"  # Previously returned owner remains alive.
    state._ruleset_identifiers = dict.fromkeys(roots, Ident("logical"))
    native.num_matches_per_rule[0] = pb.RuleCountEntry(native_name="0", value=2)
    with pytest.raises(KeyError, match="0"):
        RunReport._from_proto(native, response, request, state)


def test_proto_report_duration_and_plan_controls_without_engine():
    from typing import Literal  # noqa: PLC0415

    from egglog_proto.egglog.v1 import egglog_pb as pb  # noqa: PLC0415
    from protobuf import Oneof  # noqa: PLC0415

    from egglog.run_report import _duration, _plan  # noqa: PLC0415

    assert _duration(pb.ReportDuration(seconds=1, nanos=1999)) == timedelta(seconds=1, microseconds=1)
    with pytest.raises(ValueError, match="duration"):
        _duration(None)
    with pytest.raises(ValueError, match="duration"):
        _duration(pb.ReportDuration(nanos=10**9))
    with pytest.raises(OverflowError, match="Rust Duration"):
        _duration(pb.ReportDuration(seconds=(999999999 + 1) * 86400))
    scan = pb.ReportScan(table="N", columns=[pb.ReportColumn(variable="x", column=2)])
    native = pb.ReportPlan(
        stages=[
            pb.ReportStage(
                kind=Oneof[Literal["intersect"], pb.ReportIntersect]("intersect", pb.ReportIntersect(scans=[scan])),
                stats=pb.ReportStageStats(num_candidates=3, num_succeeded=0),
                next=[1],
            ),
            pb.ReportStage(
                kind=Oneof[Literal["fused_intersect"], pb.ReportFusedIntersect](
                    "fused_intersect", pb.ReportFusedIntersect(cover=scan, to_intersect=[scan])
                )
            ),
        ]
    )
    plan = _plan(native)
    assert plan is not None
    assert plan.stages[0].kind is not None
    assert plan.stages[0].kind.field == "intersect"
    assert plan.stages[0].kind.value.scans[0].columns[0] == pb.ReportColumn(variable="x", column=2)
    assert plan.stages[0].stats is not None
    assert plan.stages[0].stats.num_candidates == 3
    assert plan.stages[0].stats.num_succeeded == 0
    assert plan.stages[0].next == [1]
    assert plan.stages[1].kind is not None
    assert plan.stages[1].kind.field == "fused_intersect"
    assert plan.stages[1].stats is None
    native.stages[0].next = [2]
    assert plan.stages[0].next == [1]
    with pytest.raises(ValueError, match="Malformed"):
        _plan(native)


@pytest.mark.parametrize("mutation", [None, "root", "route", "ordinal", "label", "direction", "missing"])
def test_proto_report_combined_routes_and_historical_names(mutation):
    from egglog_proto.egglog.v1 import egglog_pb as pb  # noqa: PLC0415
    from protobuf import Oneof  # noqa: PLC0415

    from egglog._program import pack  # noqa: PLC0415
    from egglog.egraph_state import EGraphState  # noqa: PLC0415

    combined = (ruleset(rule(name="first").then()) | ruleset(rule(name="selected").then()))._snapshot({})
    request = pack([combined])
    root = pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", request.indices[0]))
    attribution = pb.RuleAttribution(root=root, children=[1], position=0, name="selected")
    row = pb.RuleCountEntry(rule=0, direction=pb.NativeRuleDirection.SINGLE, native_name="physical", value=7)
    response = pb.RunProgramResponse(rules=[attribution])  # No response nodes/sorts stand in for request arenas.
    native = pb.NativeRunReport(num_matches_per_rule=[row])
    state = object.__new__(EGraphState)
    state.handle = None
    state.definitions = {("ruleset", "retained-version"): combined}
    if mutation == "root":
        root.kind = Oneof[Literal["index"], int]("index", len(request.rulesets))
    elif mutation == "route":
        attribution.children = [2]
    elif mutation == "ordinal":
        attribution.position = 1
    elif mutation == "label":
        attribution.name = "wrong"
    elif mutation == "direction":
        row.direction = pb.NativeRuleDirection.UNSPECIFIED
    elif mutation == "missing":
        attribution.root = None
    if mutation is not None:
        with pytest.raises(ValueError, match=r"Report (rule|is)"):
            RunReport._from_proto(native, response, request, state)
        return
    indexed = RunReport._from_proto(native, response, request, state)
    root.kind = Oneof[Literal["name"], str]("name", "retained-version")
    historical = RunReport._from_proto(native, response, pack([]), state)
    assert indexed.num_matches_per_rule == historical.num_matches_per_rule
    (key,) = indexed.num_matches_per_rule
    assert key.__egg_ref__.read().name == "selected"
    assert indexed.num_matches_per_rule[key] == 7
    state.definitions.clear()
    with pytest.raises(KeyError, match="retained-version"):
        RunReport._from_proto(native, response, pack([]), state)
    assert repr(key) == "rule(name='selected').then()"


def _setup_simple_egraph():
    egraph = EGraph()

    class Num(Expr):
        def __init__(self, n: i64Like) -> None: ...
        def __add__(self, other: Num) -> Num: ...

    x, y = vars_("x y", Num)
    egraph.register(rewrite(x + y).to(y + x))
    egraph.register(Num(1) + Num(2))
    return egraph


def test_run_returns_report():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)
    assert type(report).__name__ == "RunReport"


def test_stats_returns_report():
    egraph = _setup_simple_egraph()
    egraph.run(10)
    report = egraph.stats()
    assert type(report).__name__ == "RunReport"


def test_rule_names_translated_in_top_level_dicts():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)

    for key in report.search_and_apply_time_per_rule:
        assert isinstance(key, RuleKey)
        assert key.__egg_ref__.read().kind.field == "rewrite"

    for key in report.num_matches_per_rule:
        assert isinstance(key, RuleKey)
        assert key.__egg_ref__.read().kind.field == "rewrite"


def test_rule_names_translated_in_iterations():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)

    assert len(report.iterations) > 0
    for iteration in report.iterations:
        for key in iteration.rule_set_report.rule_reports:
            assert isinstance(key, RuleKey)
            assert key.__egg_ref__.read().kind.field == "rewrite"


def test_updated_field():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)
    assert isinstance(report.updated, bool)
    assert report.updated is True


def test_can_stop_field():
    report = EGraph().run(1)

    assert report.can_stop is True
    assert "can_stop=True" in repr(report)


def test_can_stop_preserves_positional_constructor():
    report = RunReport(None, [], False, {}, {}, {}, {}, {})

    assert report.can_stop is False


def test_num_matches():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)

    total_matches = sum(report.num_matches_per_rule.values())
    assert total_matches > 0


def test_timedelta_types():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)

    for v in report.search_and_apply_time_per_rule.values():
        assert isinstance(v, timedelta)
    for v in report.search_and_apply_time_per_ruleset.values():
        assert isinstance(v, timedelta)
    for v in report.merge_time_per_ruleset.values():
        assert isinstance(v, timedelta)
    for v in report.rebuild_time_per_ruleset.values():
        assert isinstance(v, timedelta)


def test_iteration_reports():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)

    for it in report.iterations:
        assert type(it).__name__ == "IterationReport"
        assert type(it.rule_set_report).__name__ == "RuleSetReport"
        for rule_reports in it.rule_set_report.rule_reports.values():
            for rr in rule_reports:
                assert type(rr).__name__ == "RuleReport"


def test_str_no_egglog_sexprs():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)
    output = str(report)

    assert "(rewrite" not in output, f"str() still contains egglog s-expressions:\n{output}"
    assert "__main__" not in output, f"str() still contains mangled names:\n{output}"
    assert "rewrite(" in output, f"Expected 'rewrite(' in:\n{output}"


def test_multiple_rules():
    egraph = EGraph()

    class Math(Expr):
        def __init__(self, value: i64Like) -> None: ...
        def __add__(self, other: Math) -> Math: ...
        def __mul__(self, other: Math) -> Math: ...

    a, b = vars_("a b", Math)
    egraph.register(
        rewrite(a + b).to(b + a),
        rewrite(a * b).to(b * a),
    )
    egraph.register(Math(1) + Math(2), Math(3) * Math(4))
    report = egraph.run(10)

    rule_keys = list(report.search_and_apply_time_per_rule.keys())
    assert len(rule_keys) == 2
    for key in rule_keys:
        assert isinstance(key, RuleKey)
        assert key.__egg_ref__.read().kind.field == "rewrite"


def test_empty_run():
    egraph = EGraph()
    report = egraph.run(1)
    assert type(report).__name__ == "RunReport"
    assert isinstance(report.updated, bool)


def test_named_rule():
    egraph = EGraph()

    class Num(Expr):
        def __init__(self, n: i64Like) -> None: ...
        def __add__(self, other: Num) -> Num: ...

    x, y = vars_("x y", Num)
    egraph.register(rule(x + y, name="comm").then(union(x + y).with_(y + x)))
    egraph.register(Num(1) + Num(2))
    report = egraph.run(10)

    output = str(report)
    assert "__main__" not in output, f"str() still contains mangled names:\n{output}"
    assert "rule(" in output, f"Expected 'rule(' in:\n{output}"
    assert "comm" in output, f"Expected rule name 'comm' in:\n{output}"


def _assert_saved_report_replay(egraph: EGraph, report: RunReport) -> None:  # noqa: C901, PLR0912
    """Replay real Source phases, checking state, loop identity, and report attribution."""
    from egglog_proto.egglog.v1 import egglog_pb as pb  # noqa: PLC0415
    from protobuf import Oneof  # noqa: PLC0415

    (key,) = report.num_matches_per_rule
    authored = key.__egg_ref__.read()
    source = bindings._ProtoEngine.source(egraph.as_egglog_string)
    engine = bindings._ProtoEngine()
    created = pb.CreateEGraphResponse.from_binary(
        engine.create(
            pb.CreateEGraphRequest(
                sorts=[pb.Sort(kind=Oneof[Literal["family"], pb.HostSort]("family", pb.HostSort(name="i64")))],
                options=pb.EGraphOptions(cost_sort=0, seminaive=True),
                threads=1,
            ).to_binary()
        )
    )
    rulesets = {}
    constructors: dict[str, tuple[pb.Program, pb.Declaration]] = {}
    loops = checks = 0
    try:
        while (submission := source.next_submission()) is not None:
            kind, encoded_program, count = submission
            assert kind == "program"
            assert type(encoded_program) is bytes
            assert count == 0
            program = pb.Program.from_binary(encoded_program)
            encoded_response, host_error = engine.run(
                pb.RunProgramRequest(
                    egraph_id=created.egraph_id, program=program, profile=True, include_run_reports=True
                ).to_binary()
            )
            assert host_error is None
            response = pb.RunProgramResponse.from_binary(encoded_response)
            assert response.error is None
            assert response.ruleset_publication is not None
            assert response.profile is not None
            assert response.profile.complete
            for ruleset in program.rulesets:
                if ruleset.has_field("name"):
                    rulesets[ruleset.name] = program, ruleset
            for declaration in program.declarations:
                if declaration.kind is not None and isinstance(declaration.kind.value, pb.Constructor):
                    constructors[declaration.kind.value.name] = program, declaration
            for command in program.commands:
                assert command.kind is not None
                if command.kind.field == "check":
                    checks += 1  # The same final-state assertion succeeded in the fresh engine.
            for output in response.outputs:
                assert output.location is not None
                (index,) = output.location.path
                assert not output.location.iterations
                command = program.commands[index].kind
                assert command is not None
                assert command.field == "saturate"
                assert command.value.has_field("max_iterations")
                assert command.value.max_iterations == 1
                assert not command.value.until
                (body,) = command.value.body
                assert body.kind is not None
                assert body.kind.field == "run"
                assert output.kind is not None
                assert output.kind.field == "loop"
                assert output.kind.value.iterations == 1
                assert output.kind.value.updated is report.updated
                assert output.kind.value.termination == (
                    pb.TerminationReason.SATURATED
                    if not report.updated and report.can_stop
                    else pb.TerminationReason.ITERATION_LIMIT
                )
                native = output.run_report
                assert native is not None
                assert native.updated is report.updated
                assert native.can_stop is report.can_stop
                assert len(native.iterations) == len(report.iterations)
                assert sum(row.value for row in native.num_matches_per_rule) == sum(
                    report.num_matches_per_rule.values()
                )
                (summary,) = response.profile.runs
                assert summary.command_path == [index, 0]
                assert summary.invocations == 1
                for field in ("search_and_apply_nanos", "merge_nanos", "rebuild_nanos"):
                    assert summary.has_field(field)
                (attribution,) = response.rules
                owner = program
                target = attribution.root
                for child in [*attribution.children, None]:
                    assert target is not None
                    assert target.kind is not None
                    if target.kind.field == "index":
                        ruleset = owner.rulesets[target.kind.value]
                    else:
                        owner, ruleset = rulesets[target.kind.value]
                    if child is not None:
                        assert ruleset.kind is not None
                        assert ruleset.kind.field == "combined"
                        target = ruleset.kind.value.rulesets[child]
                assert ruleset.kind is not None
                assert ruleset.kind.field == "rules"
                replayed = owner.rules[ruleset.kind.value.rules[attribution.position]]
                assert replayed.kind is not None
                assert replayed.kind.field == authored.kind.field
                assert replayed.has_field("name") == attribution.has_field("name")
                assert replayed.name == attribution.name
                if authored.has_field("name"):
                    assert replayed.has_field("name")
                    assert replayed.name == authored.name
                if authored.kind.field in {"rewrite", "birewrite"}:
                    for operand in ("lhs", "rhs"):
                        original_ref = key.__egg_ref__.owner.ref("nodes", getattr(authored.kind.value, operand))
                        original = original_ref.read()
                        replay = owner.nodes[getattr(replayed.kind.value, operand)]
                        assert replay.kind is not None
                        assert original.kind.field == replay.kind.field == "call"
                        assert isinstance(original.kind.value, pb.Call)
                        assert isinstance(replay.kind.value, pb.Call)
                        original_definition = egraph._state.definitions["callable", original.kind.value.func]
                        original_declaration: pb.Declaration = original_definition.read()
                        assert original_declaration.kind is not None
                        assert isinstance(original_declaration.kind.value, pb.Constructor)
                        declaration_owner, declaration = constructors[replay.kind.value.func]
                        assert declaration.kind is not None
                        assert isinstance(declaration.kind.value, pb.Constructor)
                        # Recording retains the original selector separately from its private callable key.
                        assert original_declaration.table_name is None
                        assert declaration.table_name == Oneof("default_name", original_declaration.kind.value.name)
                        original_constructor = original_declaration.kind.value
                        replay_constructor = declaration.kind.value
                        assert declaration.hidden == original_declaration.hidden
                        assert replay_constructor.unextractable == original_constructor.unextractable
                        assert not original_constructor.has_field("cost")
                        assert not replay_constructor.has_field("cost")
                        original_output = original_definition.owner.ref("sorts", original_constructor.output).read()
                        replay_output = declaration_owner.sorts[replay_constructor.output]
                        assert original_output.kind is not None
                        assert replay_output.kind is not None
                        assert original_output.kind.field == replay_output.kind.field == "eq"
                        assert original_ref.owner.ref("sorts", original.sort_id).read().kind == original_output.kind
                        assert owner.sorts[replay.sort_id].kind == replay_output.kind
                        (original_input,) = original_constructor.inputs
                        (replay_input,) = replay_constructor.inputs
                        original_input_sort = original_definition.owner.ref("sorts", original_input.sort).read()
                        replay_input_sort = declaration_owner.sorts[replay_input.sort]
                        assert original_input_sort.kind == replay_input_sort.kind
                        (original_arg,) = original.kind.value.args
                        (replay_arg,) = replay.kind.value.args
                        original_value_ref = original_ref.owner.ref("nodes", original_arg)
                        original_value = original_value_ref.read()
                        replay_value = owner.nodes[replay_arg]
                        original_value_sort = original_value_ref.owner.ref("sorts", original_value.sort_id).read()
                        assert original_value_sort.kind == original_input_sort.kind
                        assert owner.sorts[replay_value.sort_id].kind == replay_input_sort.kind
                        assert replay_value.kind is not None
                        assert original_value.kind.field == replay_value.kind.field == "primitive_value"
                        assert replay_value.kind.value == original_value.kind.value
                rows: list[pb.RuleDurationEntry | pb.RuleCountEntry | pb.NativeRuleReportsEntry] = [
                    *native.search_and_apply_time_per_rule,
                    *native.num_matches_per_rule,
                ]
                for iteration in native.iterations:
                    assert iteration.rule_set_report is not None
                    rows.extend(iteration.rule_set_report.rule_reports)
                assert rows
                for row in rows:
                    assert row.has_field("rule")
                    assert row.rule == 0
                    assert row.has_field("direction")
                    assert row.direction in (
                        {pb.NativeRuleDirection.FORWARD, pb.NativeRuleDirection.REVERSE}
                        if authored.kind.field == "birewrite"
                        else {pb.NativeRuleDirection.SINGLE}
                    )
                loops += 1
        assert loops == checks == 1
    finally:
        engine.destroy(pb.DestroyEGraphRequest(egraph_id=created.egraph_id).to_binary())


@pytest.mark.parametrize("name", ['a"b', r"a\b", 'a"b\\c'])
def test_saved_named_rule_round_trips_escaped_name(name: str) -> None:
    egraph = EGraph(save_egglog_string=True)
    seen = relation("seen", i64)
    x = var("x", i64)
    egraph.register(rule(seen(x), name=name).then(seen(x)), seen(i64(1)))
    report = egraph.run(1)

    assert any(
        isinstance(decl, RuleKey)
        and decl.__egg_ref__.read().kind.field == "rule"
        and decl.__egg_ref__.read().name == name
        for decl in report.num_matches_per_rule
    )
    egraph.check(seen(i64(1)))
    _assert_saved_report_replay(egraph, report)


def test_unnamed_rule():
    egraph = EGraph()

    class Num(Expr):
        def __init__(self, n: i64Like) -> None: ...
        def __add__(self, other: Num) -> Num: ...

    x, y = vars_("x y", Num)
    egraph.register(rule(x + y).then(union(x + y).with_(y + x)))
    egraph.register(Num(1) + Num(2))
    report = egraph.run(10)

    output = str(report)
    assert "__main__" not in output, f"Unnamed Rule key not translated:\n{output}"
    assert "rule(" in output, f"Expected 'rule(' in:\n{output}"
    rule_keys = list(report.search_and_apply_time_per_rule.keys())
    assert len(rule_keys) > 0
    for key in rule_keys:
        assert isinstance(key, RuleKey)
        assert key.__egg_ref__.read().kind.field == "rule"


def test_birewrite_decl():
    egraph = EGraph()

    class Num(Expr):
        def __init__(self, n: i64Like) -> None: ...
        def __add__(self, other: Num) -> Num: ...
        def __mul__(self, other: Num) -> Num: ...

    x, y = vars_("x y", Num)
    egraph.register(birewrite(x + y).to(y + x))
    egraph.register(Num(1) + Num(2))
    report = egraph.run(10)

    output = str(report)
    assert "__main__" not in output, f"BiRewriteDecl key not translated:\n{output}"
    assert "birewrite(" in output, f"Expected 'birewrite(' in:\n{output}"
    rule_keys = list(report.search_and_apply_time_per_rule.keys())
    assert len(rule_keys) > 0
    for key in rule_keys:
        assert isinstance(key, RuleKey)
        assert key.__egg_ref__.read().kind.field == "birewrite"


def test_saved_transcript_report_translates_rewrite_with_string_literal() -> None:
    class Word(Expr):
        @classmethod
        def named(cls, value: StringLike) -> Word: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(rewrite(Word.named("input")).to(Word.named("fixed")))
    egraph.register(Word.named("input"))

    report = egraph.run(1)

    assert report.search_and_apply_time_per_rule
    assert all(
        isinstance(key, RuleKey) and key.__egg_ref__.read().kind.field == "rewrite"
        for key in report.search_and_apply_time_per_rule
    )
    egraph.check(eq(Word.named("input")).to(Word.named("fixed")))
    _assert_saved_report_replay(egraph, report)


def test_saved_transcript_report_translates_birewrite_directions() -> None:
    class Word(Expr):
        @classmethod
        def named(cls, value: StringLike) -> Word: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(birewrite(Word.named("input")).to(Word.named("fixed")))
    egraph.register(Word.named("input"))

    report = egraph.run(1)

    assert report.search_and_apply_time_per_rule
    assert all(
        isinstance(key, RuleKey) and key.__egg_ref__.read().kind.field == "birewrite"
        for key in report.search_and_apply_time_per_rule
    )
    egraph.check(eq(Word.named("input")).to(Word.named("fixed")))
    _assert_saved_report_replay(egraph, report)
