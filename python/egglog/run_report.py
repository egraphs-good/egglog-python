from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from . import bindings
from .declarations import CommandDecl, Declarations
from .egraph_state import EGraphState
from .pretty import pretty_decl


def _format_rule_key(decls: Declarations, key: CommandDecl) -> str:
    return pretty_decl(decls, key)


@dataclass
class RuleReport:
    plan: bindings.Plan | None
    search_and_apply_time: timedelta
    num_matches: int

    @classmethod
    def from_bindings(cls, report: bindings.RuleReport) -> RuleReport:
        return cls(
            plan=report.plan,
            search_and_apply_time=report.search_and_apply_time,
            num_matches=report.num_matches,
        )


@dataclass
class RuleSetReport:
    changed: bool
    rule_reports: dict[CommandDecl, list[RuleReport]]
    search_and_apply_time: timedelta
    merge_time: timedelta
    _decls: Declarations = field(repr=False, default=None)

    @classmethod
    def from_bindings(
        cls, report: bindings.RuleSetReport, translate_key: callable, decls: Declarations
    ) -> RuleSetReport:
        return cls(
            changed=report.changed,
            rule_reports={
                translate_key(k): [RuleReport.from_bindings(rr) for rr in v] for k, v in report.rule_reports.items()
            },
            search_and_apply_time=report.search_and_apply_time,
            merge_time=report.merge_time,
            _decls=decls,
        )

    def __repr__(self) -> str:
        rule_reports_str = {_format_rule_key(self._decls, k): v for k, v in self.rule_reports.items()}
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
    def from_bindings(
        cls, report: bindings.IterationReport, translate_key: callable, decls: Declarations
    ) -> IterationReport:
        return cls(
            rule_set_report=RuleSetReport.from_bindings(report.rule_set_report, translate_key, decls),
            rebuild_time=report.rebuild_time,
        )


@dataclass
class RunReport:
    """Python-friendly wrapper around bindings.RunReport."""

    iterations: list[IterationReport]
    updated: bool
    search_and_apply_time_per_rule: dict[CommandDecl, timedelta]
    num_matches_per_rule: dict[CommandDecl, int]
    search_and_apply_time_per_ruleset: dict[str, timedelta]
    merge_time_per_ruleset: dict[str, timedelta]
    rebuild_time_per_ruleset: dict[str, timedelta]
    _decls: Declarations = field(repr=False, default=None)

    def __repr__(self) -> str:
        time_per_rule = {_format_rule_key(self._decls, k): v for k, v in self.search_and_apply_time_per_rule.items()}
        matches_per_rule = {_format_rule_key(self._decls, k): v for k, v in self.num_matches_per_rule.items()}
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
    def from_bindings(cls, report: bindings.RunReport, state: EGraphState) -> RunReport:
        return cls(
            iterations=[
                IterationReport.from_bindings(it, state.translate_rule_key, state.__egg_decls__)
                for it in report.iterations
            ],
            updated=report.updated,
            search_and_apply_time_per_rule={
                state.translate_rule_key(k): v for k, v in report.search_and_apply_time_per_rule.items()
            },
            num_matches_per_rule={state.translate_rule_key(k): v for k, v in report.num_matches_per_rule.items()},
            search_and_apply_time_per_ruleset=report.search_and_apply_time_per_ruleset,
            merge_time_per_ruleset=report.merge_time_per_ruleset,
            rebuild_time_per_ruleset=report.rebuild_time_per_ruleset,
            _decls=state.__egg_decls__,
        )
