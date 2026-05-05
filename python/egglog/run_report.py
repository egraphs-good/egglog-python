from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from . import bindings
from .egraph_state import EGraphState


@dataclass
class PrettyRuleReport:
    plan: bindings.Plan | None
    search_and_apply_time: timedelta
    num_matches: int

    @classmethod
    def from_bindings(cls, report: bindings.RuleReport) -> PrettyRuleReport:
        return cls(
            plan=report.plan,
            search_and_apply_time=report.search_and_apply_time,
            num_matches=report.num_matches,
        )


@dataclass
class PrettyRuleSetReport:
    changed: bool
    rule_reports: dict[str, list[PrettyRuleReport]]
    search_and_apply_time: timedelta
    merge_time: timedelta

    @classmethod
    def from_bindings(cls, report: bindings.RuleSetReport, translate_key: callable) -> PrettyRuleSetReport:
        return cls(
            changed=report.changed,
            rule_reports={
                translate_key(k): [PrettyRuleReport.from_bindings(rr) for rr in v]
                for k, v in report.rule_reports.items()
            },
            search_and_apply_time=report.search_and_apply_time,
            merge_time=report.merge_time,
        )


@dataclass
class PrettyIterationReport:
    rule_set_report: PrettyRuleSetReport
    rebuild_time: timedelta

    @classmethod
    def from_bindings(cls, report: bindings.IterationReport, translate_key: callable) -> PrettyIterationReport:
        return cls(
            rule_set_report=PrettyRuleSetReport.from_bindings(report.rule_set_report, translate_key),
            rebuild_time=report.rebuild_time,
        )


@dataclass
class PrettyRunReport:
    """Python-friendly wrapper around bindings.RunReport."""

    iterations: list[PrettyIterationReport]
    updated: bool
    search_and_apply_time_per_rule: dict[str, timedelta]
    num_matches_per_rule: dict[str, int]
    search_and_apply_time_per_ruleset: dict[str, timedelta]
    merge_time_per_ruleset: dict[str, timedelta]
    rebuild_time_per_ruleset: dict[str, timedelta]

    @classmethod
    def from_bindings(cls, report: bindings.RunReport, state: EGraphState) -> PrettyRunReport:
        return cls(
            iterations=[PrettyIterationReport.from_bindings(it, state.translate_rule_key) for it in report.iterations],
            updated=report.updated,
            search_and_apply_time_per_rule={
                state.translate_rule_key(k): v for k, v in report.search_and_apply_time_per_rule.items()
            },
            num_matches_per_rule={state.translate_rule_key(k): v for k, v in report.num_matches_per_rule.items()},
            search_and_apply_time_per_ruleset={
                state.translate_rule_key(k): v for k, v in report.search_and_apply_time_per_ruleset.items()
            },
            merge_time_per_ruleset={state.translate_rule_key(k): v for k, v in report.merge_time_per_ruleset.items()},
            rebuild_time_per_ruleset={
                state.translate_rule_key(k): v for k, v in report.rebuild_time_per_ruleset.items()
            },
        )
