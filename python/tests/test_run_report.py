# mypy: disable-error-code="empty-body"
from __future__ import annotations

from datetime import timedelta

import pytest

from egglog import *
from egglog.declarations import BiRewriteDecl, Declarations, RewriteDecl, RuleDecl


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
        assert isinstance(key, RewriteDecl)

    for key in report.num_matches_per_rule:
        assert isinstance(key, RewriteDecl)


def test_rule_names_translated_in_iterations():
    egraph = _setup_simple_egraph()
    report = egraph.run(10)

    assert len(report.iterations) > 0
    for iteration in report.iterations:
        for key in iteration.rule_set_report.rule_reports:
            assert isinstance(key, RewriteDecl)


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
        assert isinstance(key, RewriteDecl)


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

    assert any(isinstance(decl, RuleDecl) and decl.name == name for decl in report.num_matches_per_rule)


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
        assert isinstance(key, RuleDecl)


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
        assert isinstance(key, BiRewriteDecl)


def test_saved_transcript_report_translates_rewrite_with_string_literal() -> None:
    class Word(Expr):
        @classmethod
        def named(cls, value: StringLike) -> Word: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(rewrite(Word.named("input")).to(Word.named("fixed")))
    egraph.register(Word.named("input"))

    report = egraph.run(1)

    assert report.search_and_apply_time_per_rule
    assert all(isinstance(key, RewriteDecl) for key in report.search_and_apply_time_per_rule)


def test_saved_transcript_report_translates_birewrite_directions() -> None:
    class Word(Expr):
        @classmethod
        def named(cls, value: StringLike) -> Word: ...

    egraph = EGraph(save_egglog_string=True)
    egraph.register(birewrite(Word.named("input")).to(Word.named("fixed")))
    egraph.register(Word.named("input"))

    report = egraph.run(1)

    assert report.search_and_apply_time_per_rule
    assert all(isinstance(key, BiRewriteDecl) for key in report.search_and_apply_time_per_rule)
