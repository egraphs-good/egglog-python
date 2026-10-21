from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING

from egglog_proto.egglog.v1 import egglog_pb as pb

from ._program import Packed, Ref, _rule_key, rule_equal
from .pretty import _format_pyobject, pretty_ref

if TYPE_CHECKING:
    from .egraph_state import EGraphState


@dataclass(frozen=True, slots=True, eq=False)
class RuleKey:
    """Immutable authored-rule inspection key, independent of installation identity."""

    __egg_ref__: Ref

    def __post_init__(self) -> None:
        if self.__egg_ref__.role != "rules":
            msg = "Report keys require a canonical rule reference"
            raise TypeError(msg)
        record: pb.Rule = self.__egg_ref__.read()
        if record.kind is None:
            msg = "Report rule has no body"
            raise ValueError(msg)

    def __hash__(self) -> int:
        return hash(_rule_key(self.__egg_ref__))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuleKey):
            return NotImplemented
        return rule_equal(self.__egg_ref__, other.__egg_ref__)

    def __repr__(self) -> str:
        return pretty_ref(self.__egg_ref__, custom_literal=_format_pyobject)


def _duration(value: pb.ReportDuration | None) -> timedelta:
    """Preserve native integer microsecond truncation and Python duration limits."""
    if value is None or not 0 <= value.seconds < 2**64 or not 0 <= value.nanos < 10**9:
        msg = "Missing or invalid native report duration"
        raise ValueError(msg)
    try:
        return timedelta(seconds=value.seconds, microseconds=value.nanos // 1000)
    except OverflowError:
        msg = "Rust Duration is too large for datetime.timedelta"
        raise OverflowError(msg) from None


def _plan(value: pb.ReportPlan | None) -> pb.ReportPlan | None:
    """Project inert native plan observations, validating their local topology."""
    if value is None:
        return None
    for stage in value.stages:
        if stage.kind is None or any(not 0 <= successor < len(value.stages) for successor in stage.next):
            msg = "Malformed native report plan"
            raise ValueError(msg)
        if stage.kind.field == "intersect":
            for scan in stage.kind.value.scans:
                if len(scan.columns) != 1:
                    msg = "A native single scan must have exactly one column"
                    raise ValueError(msg)
        else:
            fused = stage.kind.value
            if fused.cover is None:
                msg = "A fused native plan requires a cover"
                raise ValueError(msg)
    return pb.ReportPlan.from_binary(value.to_binary())


@dataclass
class RuleReport:
    plan: pb.ReportPlan | None
    search_and_apply_time: timedelta
    num_matches: int


@dataclass
class RuleSetReport:
    # Retained positional observation slot; no semantic declarations are stored.
    _decls: object = field(repr=False)
    changed: bool = False
    rule_reports: dict[RuleKey, list[RuleReport]] = field(default_factory=dict)
    search_and_apply_time: timedelta = field(default_factory=timedelta)
    merge_time: timedelta = field(default_factory=timedelta)

    def __repr__(self) -> str:
        return (
            f"RuleSetReport(changed={self.changed}, rule_reports={self.rule_reports}, "
            f"search_and_apply_time={self.search_and_apply_time}, merge_time={self.merge_time})"
        )


@dataclass
class IterationReport:
    rule_set_report: RuleSetReport
    rebuild_time: timedelta


@dataclass
class RunReport:
    """Actual native report data with meaningful canonical authored-rule keys."""

    _decls: object = field(repr=False)
    iterations: list[IterationReport] = field(default_factory=list)
    updated: bool = False
    search_and_apply_time_per_rule: dict[RuleKey, timedelta] = field(default_factory=dict)
    num_matches_per_rule: dict[RuleKey, int] = field(default_factory=dict)
    search_and_apply_time_per_ruleset: dict[str, timedelta] = field(default_factory=dict)
    merge_time_per_ruleset: dict[str, timedelta] = field(default_factory=dict)
    rebuild_time_per_ruleset: dict[str, timedelta] = field(default_factory=dict)
    can_stop: bool = False

    @classmethod
    def _from_proto(  # noqa: C901
        cls, report: pb.NativeRunReport, response: pb.RunProgramResponse, request: Packed, state: EGraphState
    ) -> RunReport:
        def ruleset(reference: pb.RulesetRef | None) -> Ref:
            if reference is None or reference.kind is None:
                msg = "Report is missing canonical ruleset attribution"
                raise ValueError(msg)
            if reference.kind.field == "index":
                index = reference.kind.value
                if not 0 <= index < len(request.rulesets):
                    msg = "Report ruleset index is outside its request"
                    raise ValueError(msg)
                return request.rulesets[index]
            return state.definitions["ruleset", reference.kind.value]

        def rule(row: pb.RuleDurationEntry | pb.RuleCountEntry | pb.NativeRuleReportsEntry) -> RuleKey:
            if not row.has_field("rule"):
                raise KeyError(row.native_name)
            if not row.has_field("direction") or row.direction not in (
                pb.NativeRuleDirection.SINGLE,
                pb.NativeRuleDirection.FORWARD,
                pb.NativeRuleDirection.REVERSE,
            ):
                msg = "Report rule is missing a valid native direction"
                raise ValueError(msg)
            if not 0 <= row.rule < len(response.rules):
                msg = "Report rule index is outside its response"
                raise ValueError(msg)
            attribution = response.rules[row.rule]
            reference = ruleset(attribution.root)
            for ordinal in attribution.children:
                record: pb.Ruleset = reference.read()
                if (
                    record.kind is None
                    or record.kind.field != "combined"
                    or not 0 <= ordinal < len(record.kind.value.rulesets)
                ):
                    msg = "Report rule route is not a valid combined-ruleset path"
                    raise ValueError(msg)
                child = record.kind.value.rulesets[ordinal]
                if child.kind is not None and child.kind.field == "index":
                    reference = reference.owner.ref("rulesets", child.kind.value)
                else:
                    reference = ruleset(child)
            record = reference.read()
            if (
                record.kind is None
                or record.kind.field != "rules"
                or not 0 <= attribution.position < len(record.kind.value.rules)
            ):
                msg = "Report rule route does not end at a rule membership"
                raise ValueError(msg)
            result = reference.owner.ref("rules", record.kind.value.rules[attribution.position])
            canonical: pb.Rule = result.read()
            if attribution.has_field("name") != canonical.has_field("name") or attribution.name != canonical.name:
                msg = "Report rule label disagrees with its canonical occurrence"
                raise ValueError(msg)
            return RuleKey(result)

        result = cls(None, updated=report.updated, can_stop=report.can_stop)
        for row in report.search_and_apply_time_per_rule:
            key = rule(row)
            result.search_and_apply_time_per_rule[key] = result.search_and_apply_time_per_rule.get(
                key, timedelta()
            ) + _duration(row.value)
        for count in report.num_matches_per_rule:
            key = rule(count)
            result.num_matches_per_rule[key] = result.num_matches_per_rule.get(key, 0) + count.value
        for source, target in (
            (report.search_and_apply_time_per_ruleset, result.search_and_apply_time_per_ruleset),
            (report.merge_time_per_ruleset, result.merge_time_per_ruleset),
            (report.rebuild_time_per_ruleset, result.rebuild_time_per_ruleset),
        ):
            exact: dict[str, int] = {}
            for entry in source:
                if entry.ruleset is None:
                    raise KeyError(entry.native_name)
                ident = state._ruleset_identifiers[ruleset(entry.ruleset)]
                _duration(entry.value)  # Validate every captured duration before combining.
                assert entry.value is not None
                name = str(ident)
                exact[name] = exact.get(name, 0) + entry.value.seconds * 10**9 + entry.value.nanos
            for name, nanos in exact.items():
                seconds, subsecond = divmod(nanos, 10**9)
                target[name] = _duration(pb.ReportDuration(seconds=seconds, nanos=subsecond))
        for iteration in report.iterations:
            native = iteration.rule_set_report
            if native is None:
                msg = "Native iteration omitted its ruleset report"
                raise ValueError(msg)
            projected = RuleSetReport(
                None,
                changed=native.changed,
                search_and_apply_time=_duration(native.search_and_apply_time),
                merge_time=_duration(native.merge_time),
            )
            for group in native.rule_reports:
                projected.rule_reports.setdefault(rule(group), []).extend(
                    RuleReport(_plan(item.plan), _duration(item.search_and_apply_time), item.num_matches)
                    for item in group.reports
                )
            result.iterations.append(IterationReport(projected, _duration(iteration.rebuild_time)))
        return result
