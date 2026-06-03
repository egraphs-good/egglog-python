from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from . import bindings
from .declarations import BiRewriteDecl, Declarations, RewriteDecl, RuleDecl
from .egraph_state import EGraphState
from .pretty import pretty_decl

RewriteOrRuleDecl = RuleDecl | BiRewriteDecl | RewriteDecl


@dataclass
class RuleReport:
    plan: bindings.Plan | None
    search_and_apply_time: timedelta
    num_matches: int

    @classmethod
    def _from_bindings(cls, report: bindings.RuleReport) -> RuleReport:
        return cls(
            plan=report.plan,
            search_and_apply_time=report.search_and_apply_time,
            num_matches=report.num_matches,
        )


@dataclass
class RuleSetReport:
    _decls: Declarations = field(repr=False)
    changed: bool = False
    rule_reports: dict[RewriteOrRuleDecl, list[RuleReport]] = field(default_factory=dict)
    search_and_apply_time: timedelta = field(default_factory=timedelta)
    merge_time: timedelta = field(default_factory=timedelta)

    @classmethod
    def _from_bindings(
        cls, report: bindings.RuleSetReport, rule_map: dict[str, RewriteOrRuleDecl], decls: Declarations
    ) -> RuleSetReport:
        rule_reports: dict[RewriteOrRuleDecl, list[RuleReport]] = {}
        for k, v in report.rule_reports.items():
            translated = rule_map[k]
            reports = [RuleReport._from_bindings(rr) for rr in v]
            if translated in rule_reports:
                rule_reports[translated].extend(reports)
            else:
                rule_reports[translated] = reports
        return cls(
            _decls=decls,
            changed=report.changed,
            rule_reports=rule_reports,
            search_and_apply_time=report.search_and_apply_time,
            merge_time=report.merge_time,
        )

    def __repr__(self) -> str:
        rule_reports_str = {pretty_decl(self._decls, k): v for k, v in self.rule_reports.items()}
        return (
            f"RuleSetReport(changed={self.changed}, "
            f"rule_reports={rule_reports_str}, "
            f"search_and_apply_time={self.search_and_apply_time}, "
            f"merge_time={self.merge_time})"
        )


@dataclass
class IterationReport:
    rule_set_report: RuleSetReport
    rebuild_time: timedelta

    @classmethod
    def _from_bindings(
        cls, report: bindings.IterationReport, rule_map: dict[str, RewriteOrRuleDecl], decls: Declarations
    ) -> IterationReport:
        return cls(
            rule_set_report=RuleSetReport._from_bindings(report.rule_set_report, rule_map, decls),
            rebuild_time=report.rebuild_time,
        )


@dataclass
class RunReport:
    """Python-friendly wrapper around bindings.RunReport."""

    _decls: Declarations = field(repr=False)
    iterations: list[IterationReport] = field(default_factory=list)
    updated: bool = False
    search_and_apply_time_per_rule: dict[RewriteOrRuleDecl, timedelta] = field(default_factory=dict)
    num_matches_per_rule: dict[RewriteOrRuleDecl, int] = field(default_factory=dict)
    search_and_apply_time_per_ruleset: dict[str, timedelta] = field(default_factory=dict)
    merge_time_per_ruleset: dict[str, timedelta] = field(default_factory=dict)
    rebuild_time_per_ruleset: dict[str, timedelta] = field(default_factory=dict)

    def __repr__(self) -> str:
        time_per_rule = {pretty_decl(self._decls, k): v for k, v in self.search_and_apply_time_per_rule.items()}
        matches_per_rule = {pretty_decl(self._decls, k): v for k, v in self.num_matches_per_rule.items()}
        return (
            f"RunReport(iterations={self.iterations}, "
            f"updated={self.updated}, "
            f"search_and_apply_time_per_rule={time_per_rule}, "
            f"num_matches_per_rule={matches_per_rule}, "
            f"search_and_apply_time_per_ruleset={self.search_and_apply_time_per_ruleset}, "
            f"merge_time_per_ruleset={self.merge_time_per_ruleset}, "
            f"rebuild_time_per_ruleset={self.rebuild_time_per_ruleset})"
        )

    @classmethod
    def _from_bindings(cls, report: bindings.RunReport, state: EGraphState) -> RunReport:
        rule_map = state.rule_name_to_command_decl
        decls = state.__egg_decls__

        search_and_apply_time_per_rule: dict[RewriteOrRuleDecl, timedelta] = {}
        for k, v in report.search_and_apply_time_per_rule.items():
            translated = rule_map[k]
            if translated in search_and_apply_time_per_rule:
                search_and_apply_time_per_rule[translated] += v
            else:
                search_and_apply_time_per_rule[translated] = v

        num_matches_per_rule: dict[RewriteOrRuleDecl, int] = {}
        for k, v in report.num_matches_per_rule.items():
            translated = rule_map[k]
            if translated in num_matches_per_rule:
                num_matches_per_rule[translated] += v
            else:
                num_matches_per_rule[translated] = v

        return cls(
            _decls=decls,
            iterations=[IterationReport._from_bindings(it, rule_map, decls) for it in report.iterations],
            updated=report.updated,
            search_and_apply_time_per_rule=search_and_apply_time_per_rule,
            num_matches_per_rule=num_matches_per_rule,
            search_and_apply_time_per_ruleset=report.search_and_apply_time_per_ruleset,
            merge_time_per_ruleset=report.merge_time_per_ruleset,
            rebuild_time_per_ruleset=report.rebuild_time_per_ruleset,
        )
