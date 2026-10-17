# mypy: disable-error-code="empty-body"
from __future__ import annotations

from datetime import timedelta

import pytest

from egglog import *
from egglog.declarations import Declarations
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
                root=pb.RulesetRef(kind=Oneof[Literal["index"], int]("index", index)), rule=0, name="same"
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
    assert iteration.rule_set_report.rule_reports[key][1].plan.stages == []
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
    assert plan.stages[0].kind.field == "intersect"
    assert plan.stages[0].kind.value.scans[0].columns[0] == pb.ReportColumn(variable="x", column=2)
    assert plan.stages[0].stats.num_candidates == 3
    assert plan.stages[0].stats.num_succeeded == 0
    assert plan.stages[0].next == [1]
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
    root = pb.RulesetRef(kind=Oneof("index", request.indices[0]))
    attribution = pb.RuleAttribution(root=root, children=[1], rule=0, name="selected")
    row = pb.RuleCountEntry(rule=0, direction=pb.NativeRuleDirection.SINGLE, native_name="physical", value=7)
    response = pb.RunProgramResponse(rules=[attribution])  # No response nodes/sorts stand in for request arenas.
    native = pb.NativeRunReport(num_matches_per_rule=[row])
    state = object.__new__(EGraphState)
    state.handle = None
    state.definitions = {("ruleset", "retained-version"): combined}
    if mutation == "root":
        root.kind = Oneof("index", len(request.rulesets))
    elif mutation == "route":
        attribution.children = [2]
    elif mutation == "ordinal":
        attribution.rule = 1
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
    root.kind = Oneof("name", "retained-version")
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
    report = RunReport(Declarations(), [], False, {}, {}, {}, {}, {})

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


def test_unnamed_rule_decl():
    egraph = EGraph()

    class Num(Expr):
        def __init__(self, n: i64Like) -> None: ...
        def __add__(self, other: Num) -> Num: ...

    x, y = vars_("x y", Num)
    egraph.register(rule(x + y).then(union(x + y).with_(y + x)))
    egraph.register(Num(1) + Num(2))
    report = egraph.run(10)

    output = str(report)
    assert "__main__" not in output, f"Unnamed RuleDecl key not translated:\n{output}"
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
