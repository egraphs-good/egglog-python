"""
Implement conversion to/from egglog.
"""

from __future__ import annotations

import contextlib
import math
import re
import tempfile
import weakref
from base64 import standard_b64decode, standard_b64encode
from dataclasses import InitVar, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, TextIO, assert_never, cast, overload
from uuid import UUID

import cloudpickle
from opentelemetry import trace

from . import bindings
from ._tracing import call_with_current_trace
from .declarations import *
from .declarations import (
    _BUILTIN_EGG_FN_NAMES,
    _BUILTIN_EGG_SORT_NAMES,
    ConstructorDecl,
    is_callable_decl_constructor,
)
from .pretty import *
from .type_constraint_solver import *

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["EGraphState", "span"]


_TRACER = trace.get_tracer(__name__)
_VALIDATE_COST_PRIMITIVE = "@validate-dynamic-cost"

# These heads are interpreted as syntax before Egglog falls back to parsing a
# generic top-level call. Keep this aligned with Parser::parse_command,
# Parser::parse_action, Parser::parse_fact, and the extensions installed by
# egglog_experimental::new_experimental_egraph.
_EGGLOG_RESERVED_CALL_HEADS = frozenset({
    "=",
    "birewrite",
    "check",
    "constructor",
    "datatype",
    "datatype*",
    "delete",
    "extract",
    "fail",
    "for",
    "function",
    "include",
    "input",
    "keep-best",
    "let",
    "let-scheduler",
    "multi-extract",
    "output",
    "panic",
    "pop",
    "primitive",
    "print-function",
    "print-size",
    "print-stats",
    "print-table-stats",
    "prove",
    "prove-exists",
    "push",
    "relation",
    "rewrite",
    "rule",
    "ruleset",
    "run",
    "run-schedule",
    "set",
    "set-cost",
    "sort",
    "subsume",
    "union",
    "unstable-combined-ruleset",
    "unstable-fresh!",
    "with-dynamic-cost",
    "with-ruleset",
})
_EGGLOG_RESERVED_ACTION_HEADS = frozenset({"delete", "let", "panic", "set", "set-cost", "subsume", "union"})
_EGGLOG_RESERVED_COMMAND_OR_ACTION_HEADS = _EGGLOG_RESERVED_CALL_HEADS - {"="}
_EGGLOG_LITERAL_NAMES = frozenset({"false", "NaN", "inf", "-inf", "true"})
_EGGLOG_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")


@dataclass
class _SavedEgglogFile:
    """
    Shared state for file-backed command execution.

    When `save_egglog_string=True`, the Python wrapper keeps one cumulative temp
    `.egg` file per high-level egraph. Each command is reparsed-and-run from a
    synthetic one-command program padded with blank lines so egglog error spans
    point at the real file path and the command's true line numbers inside that
    cumulative source log.

    We keep the append handle open for performance and for easy post-failure
    inspection. Successful commands are saved normally; Egglog-reported
    failures are saved as expected failures with trailing error comments.
    Other execution failures invalidate the transcript because their partial
    effects cannot be represented by a replayable command.
    """

    path: str
    file: TextIO
    line_count: int = 0
    poisoned: bool = False
    _finalizer: weakref.finalize = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._finalizer = weakref.finalize(self, _close_saved_egglog_file, self.file, self.path)

    def close(self) -> None:
        self._finalizer()


def _close_saved_egglog_file(file: TextIO, path: str) -> None:
    try:
        file.close()
    finally:
        with contextlib.suppress(FileNotFoundError):
            Path(path).unlink()


def _normalize_global_let_name(name: str) -> str:
    return name if name.startswith("$") else f"${name}"


def _collect_explicit_backend_names(declarations: Declarations) -> set[str]:
    """Collect names that declarations explicitly reserve in Egglog's shared symbol namespace."""
    names = {
        decl.egg_name
        for decl in (*declarations._functions.values(), *declarations._constants.values())
        if decl.egg_name is not None
    }
    for class_decl in declarations._classes.values():
        if class_decl.egg_name is not None:
            names.add(class_decl.egg_name)
        class_callables = (
            *class_decl.class_methods.values(),
            *class_decl.class_variables.values(),
            *class_decl.methods.values(),
            *class_decl.properties.values(),
        )
        names.update(decl.egg_name for decl in class_callables if decl.egg_name is not None)
        if class_decl.init is not None and class_decl.init.egg_name is not None:
            names.add(class_decl.init.egg_name)
    return names


def _egg_name_is_source_safe_symbol(name: str) -> bool:
    """Return whether a name is safe to emit as an ordinary Egglog symbol."""
    return (
        bool(name)
        and name != "_"
        and not name.startswith(('"', "@"))
        and not any(c.isspace() or c in ";()" for c in name)
    )


def _saved_egglog_failure_message(error: bindings.EggSmolError) -> str:
    lines = [line.strip() for line in error.context.splitlines() if line.strip()]
    for message in lines:
        if "Failed" in message or "failed" in message:
            return message
    if len(lines) > 1 and lines[0].startswith("In "):
        return lines[1]
    if lines:
        return lines[-1]
    return str(error)


def _rule_variable_names(rule: RuleDecl) -> set[str]:  # noqa: C901, PLR0912
    """Collect source-level names that compiler-generated rule lets must not shadow."""
    names: set[str] = set()

    def visit_expr(expr: ExprDecl) -> None:
        match expr:
            case UnboundVarDecl(name, egg_name):
                emitted_name = egg_name or f"_{name}"
                names.add(emitted_name.removeprefix("$"))
            case LetRefDecl(name):
                names.add(_normalize_global_let_name(name).removeprefix("$"))
            case CallDecl(_, args) | GetCostDecl(_, args) | PartialCallDecl(CallDecl(_, args)):
                for arg in args:
                    visit_expr(arg.expr)
            case LitDecl() | PyObjectDecl() | ValueDecl() | DummyDecl():
                pass
            case _:
                assert_never(expr)

    for fact in rule.body:
        match fact:
            case EqDecl(_, left, right):
                visit_expr(left)
                visit_expr(right)
            case ExprFactDecl(typed_expr):
                visit_expr(typed_expr.expr)
            case _:
                assert_never(fact)
    for action in rule.head:
        match action:
            case LetDecl(name, typed_expr):
                names.add(_normalize_global_let_name(name).removeprefix("$"))
                visit_expr(typed_expr.expr)
            case SetDecl(_, call, rhs):
                visit_expr(call)
                visit_expr(rhs)
            case ExprActionDecl(typed_expr):
                visit_expr(typed_expr.expr)
            case ChangeDecl(_, call, _):
                visit_expr(call)
            case UnionDecl(_, left, right):
                visit_expr(left)
                visit_expr(right)
            case SetCostDecl(_, call, cost):
                visit_expr(call)
                visit_expr(cost)
            case PanicDecl():
                pass
            case _:
                assert_never(action)
    return names


def span(frame_index: int = 0) -> bindings.RustSpan:
    """
    Returns a span for the current file and line.

    If `frame_index` is passed, it will return the span for that frame in the stack, where 0 is the current frame
    this is called in and 1 is the parent.
    """
    # Currently disable this because it's too expensive.
    # import inspect

    # frame = inspect.stack()[frame_index + 1]
    return bindings.RustSpan("", 0, 0)


@dataclass
class EGraphState:
    """
    State of the EGraph declarations and rulesets, so when we pop/push the stack we know whats defined.

    Used for converting to/from egg and for pretty printing.
    """

    egraph: bindings.EGraph
    seminaive: bool = True
    # Opaque identity for values created in the current push scope. Descendant
    # scopes may use ancestor values, but values from a popped scope must not be
    # accepted if the backend later reuses their raw IDs.
    value_owner: object = field(default_factory=object, repr=False)
    valid_value_owners: frozenset[object] = field(default_factory=frozenset, repr=False)
    save_egglog_string: InitVar[bool] = False
    egglog_file_state: _SavedEgglogFile | None = field(default=None, repr=False)
    # The declarations we have added.
    __egg_decls__: Declarations = field(default_factory=Declarations)
    # Explicit names are cached as declarations are merged. Generated names
    # consult this set so a registration batch reserves all user-selected
    # backend symbols without rescanning the complete declaration graph for
    # every generated callable, sort, cost table, and synthetic let.
    _explicit_backend_names: set[str] = field(default_factory=set, init=False, repr=False)
    # Mapping of added rulesets to the added rules
    rulesets: dict[Ident, set[RewriteOrRuleDecl]] = field(default_factory=dict)
    # Persistent schedulers live outside a single run-schedule command; only emit their let once per active scope.
    registered_persistent_schedulers: set[UUID] = field(default_factory=set)

    # Bidirectional mapping between egg function names and python callable references.
    # Note that there are possibly multiple callable references for a single egg function name, like `+`
    # for both int and rational classes.
    egg_fn_to_callable_refs: dict[str, set[CallableRef]] = field(
        default_factory=lambda: {"!=": {FunctionRef(Ident.builtin("!="))}}
    )
    callable_ref_to_egg_fn: dict[CallableRef, tuple[str, bool]] = field(
        default_factory=lambda: {FunctionRef(Ident.builtin("!=")): ("!=", False)}
    )

    # Bidirectional mapping between egg sort names and python type references.
    type_ref_to_egg_sort: dict[JustTypeRef, str] = field(default_factory=dict)
    egg_sort_to_type_ref: dict[str, JustTypeRef] = field(default_factory=dict)

    # Cache of direct structural egg expressions for converting to egg.
    expr_to_egg_cache: dict[ExprDecl, bindings._Expr] = field(default_factory=dict)
    # Cache of top-level expressions lowered with any available synthetic let
    # references. Rules and rewrites must never read from this cache.
    expr_to_let_egg_cache: dict[ExprDecl, bindings._Expr] = field(default_factory=dict)
    # Cache of synthetic let references introduced for top-level command lowering.
    # This stays separate from `expr_to_egg_cache` so nested rule/rewrite lowering
    # can always rebuild structural surface syntax instead of leaking a previously
    # synthesized `$__expr_n` binding across contexts.
    expr_to_letref_cache: dict[ExprDecl, bindings.Var] = field(default_factory=dict)

    # Callables with custom-cost tables and the reserved backend names for those tables.
    cost_table_names: dict[CallableRef, str] = field(default_factory=dict)
    # Counter for deterministic synthetic let bindings created while lowering expressions to egg.
    expr_to_let_counter: int = 0
    # Explicit top-level lets waiting to be lowered in the current registration batch.
    # Synthetic lets must avoid them regardless of command order within that batch.
    pending_let_names: frozenset[str] = field(default_factory=frozenset, repr=False)
    # Counter for deterministic synthetic names assigned to unnamed functions.
    unnamed_function_counter: int = 0

    # Counter for numeric rule names
    rule_name_counter: int = 0
    # Mapping from numeric name (str) to command decl
    rule_name_to_command_decl: dict[str, RuleDecl | BiRewriteDecl | RewriteDecl] = field(default_factory=dict)

    def __post_init__(self, save_egglog_string: bool) -> None:
        if not self.valid_value_owners:
            self.valid_value_owners = frozenset((self.value_owner,))
        self._explicit_backend_names = _collect_explicit_backend_names(self.__egg_decls__)
        if save_egglog_string and self.egglog_file_state is None:
            # Keep one persistent temp `.egg` file per high-level egraph so parse errors
            # can point at a stable filename the user can open after a failure.
            egglog_file = tempfile.NamedTemporaryFile(  # noqa: SIM115 - kept open for incremental appends
                mode="w+", encoding="utf-8", newline="", suffix=".egg", delete=False
            )
            self.egglog_file_state = _SavedEgglogFile(egglog_file.name, cast("TextIO", egglog_file))

    def copy(self) -> EGraphState:
        """
        Returns a copy of the state. The egraph reference is kept the same. Used for pushing/popping.
        """
        value_owner = object()
        return EGraphState(
            egraph=self.egraph,
            seminaive=self.seminaive,
            value_owner=value_owner,
            valid_value_owners=self.valid_value_owners | {value_owner},
            save_egglog_string=self.egglog_file_state is not None,
            egglog_file_state=self.egglog_file_state,
            __egg_decls__=self.__egg_decls__.copy(),
            rulesets={k: v.copy() for k, v in self.rulesets.items()},
            registered_persistent_schedulers=self.registered_persistent_schedulers.copy(),
            egg_fn_to_callable_refs={k: v.copy() for k, v in self.egg_fn_to_callable_refs.items()},
            callable_ref_to_egg_fn=self.callable_ref_to_egg_fn.copy(),
            type_ref_to_egg_sort=self.type_ref_to_egg_sort.copy(),
            egg_sort_to_type_ref=self.egg_sort_to_type_ref.copy(),
            expr_to_egg_cache=self.expr_to_egg_cache.copy(),
            expr_to_let_egg_cache=self.expr_to_let_egg_cache.copy(),
            expr_to_letref_cache=self.expr_to_letref_cache.copy(),
            cost_table_names=self.cost_table_names.copy(),
            expr_to_let_counter=self.expr_to_let_counter,
            pending_let_names=self.pending_let_names,
            unnamed_function_counter=self.unnamed_function_counter,
            rule_name_counter=self.rule_name_counter,
            rule_name_to_command_decl=self.rule_name_to_command_decl.copy(),
        )

    def add_declarations(self, *declarations_like: DeclarationsLike) -> None:
        """Merge declarations while maintaining the explicit backend-name index."""
        attempted_update = False
        try:
            for declarations in declarations_like:
                if declarations is None:
                    continue
                attempted_update = True
                self.__egg_decls__ |= declarations
        finally:
            if attempted_update:
                # Rebuild after the batch rather than accumulating names: declaration
                # merges may replace an earlier declaration and release its explicit
                # name for later generated symbols. The finally path also keeps the
                # index aligned when a later lazy declaration fails to resolve after
                # an earlier declaration has already been merged.
                self._explicit_backend_names = _collect_explicit_backend_names(self.__egg_decls__)

    def egglog_string(self) -> str:
        if self.egglog_file_state is None:
            msg = "Can't get egglog string unless EGraph created with save_egglog_string=True"
            raise ValueError(msg)
        if self.egglog_file_state.file.closed:
            msg = "Can't get egglog string after the saved transcript has been closed"
            raise ValueError(msg)
        if self.egglog_file_state.poisoned:
            msg = "Can't use the saved Egglog transcript after an execution failure whose partial effects cannot be replayed"
            raise RuntimeError(msg)
        # The append handle stays open for execution, so flush before reading the saved source.
        self.egglog_file_state.file.flush()
        with open(self.egglog_file_state.path, encoding="utf-8", newline="") as saved_file:
            return saved_file.read()

    def close(self) -> None:
        if self.egglog_file_state is not None:
            self.egglog_file_state.close()

    def ensure_open(self) -> None:
        if self.egglog_file_state is not None and self.egglog_file_state.file.closed:
            msg = "Cannot run commands after the saved Egglog transcript has been closed"
            raise ValueError(msg)
        if self.egglog_file_state is not None and self.egglog_file_state.poisoned:
            msg = "Cannot run commands after an execution failure whose partial effects cannot be replayed"
            raise RuntimeError(msg)

    def run_program(self, *commands: bindings._Command) -> list[bindings._CommandOutput]:
        if not commands:
            return []
        self.ensure_open()
        if self.egglog_file_state is None:
            return call_with_current_trace(self.egraph.run_program, *commands)

        outputs: list[bindings._CommandOutput] = []
        for command in commands:
            command_text = str(command).rstrip("\n") + "\n"
            start_line = self.egglog_file_state.line_count + 1

            # Parse and run just this command in Rust, but pad it with blank lines
            # so the span lines match its location in the cumulative saved source file.
            padded_command = ("\n" * (start_line - 1)) + command_text
            try:
                command_outputs = call_with_current_trace(
                    self.egraph.parse_and_run_program, padded_command, filename=self.egglog_file_state.path
                )
            except bindings.EggSmolError as error:
                if not error.replayable_by_fail:
                    # Parsing, expansion, and typechecking can mutate backend
                    # metadata before they fail, while `(fail ...)` cannot
                    # reproduce those failures. Do not claim a replayable log.
                    self.egglog_file_state.poisoned = True
                    raise
                fail_command_text = str(bindings.Fail(span(), command)).rstrip("\n")
                saved_text = f"{fail_command_text} ; {_saved_egglog_failure_message(error)}\n"
                try:
                    self.egglog_file_state.file.write(saved_text)
                    self.egglog_file_state.file.flush()
                except BaseException:
                    self.egglog_file_state.poisoned = True
                    raise
                self.egglog_file_state.line_count += saved_text.count("\n")
                raise
            except BaseException:
                # A Python primitive or unrelated runtime failure can interrupt
                # a command after some actions have committed. There is no
                # replayable Egglog command for that partial state.
                self.egglog_file_state.poisoned = True
                raise
            try:
                self.egglog_file_state.file.write(command_text)
                self.egglog_file_state.file.flush()
            except BaseException:
                # Execution already succeeded, so omitting even part of this
                # command would make the cumulative source diverge.
                self.egglog_file_state.poisoned = True
                raise
            self.egglog_file_state.line_count += command_text.count("\n")
            outputs.extend(command_outputs)
        return outputs

    @staticmethod
    def _persistent_scheduler_name(scheduler: BackOffDecl) -> str:
        return f"_persistent_scheduler_{scheduler.id.hex}"

    @staticmethod
    def _local_scheduler_name(index: int) -> str:
        return f"_scheduler_{index}"

    @staticmethod
    def _back_off_scheduler_to_egg(scheduler: BackOffDecl) -> bindings.Call:
        """Serialize the shared option protocol for local and persistent backoff schedulers."""
        args: list[bindings._Expr] = []
        if scheduler.match_limit is not None:
            args.extend((
                bindings.Var(span(), ":match-limit"),
                bindings.Lit(span(), bindings.Int(scheduler.match_limit)),
            ))
        if scheduler.ban_length is not None:
            args.extend((
                bindings.Var(span(), ":ban-length"),
                bindings.Lit(span(), bindings.Int(scheduler.ban_length)),
            ))
        return bindings.Call(span(), "back-off", args)

    @_TRACER.start_as_current_span("run_schedule_to_egg")
    def run_schedule_to_egg(self, schedule: ScheduleDecl) -> bindings._Command:
        """
        Turn a run schedule into an egg command.

        If there exists any custom schedulers in the schedule, it will be turned into a custom extract command otherwise
        will be a normal run command.
        """
        processed_schedule, persistent_schedulers = self._process_schedule(schedule)
        if processed_schedule is None:
            return bindings.RunSchedule(self._schedule_to_egg(schedule))
        for scheduler in persistent_schedulers:
            if scheduler.id in self.registered_persistent_schedulers:
                continue
            self.run_program(self._persistent_scheduler_to_egg(scheduler))
            self.registered_persistent_schedulers.add(scheduler.id)
        top_level_schedules = self._schedule_with_scheduler_to_egg(processed_schedule, [])
        if len(top_level_schedules) == 1:
            schedule_expr = top_level_schedules[0]
        else:
            schedule_expr = bindings.Call(span(), "seq", top_level_schedules)
        return bindings.UserDefined(span(), "run-schedule", [schedule_expr])

    def _process_schedule(  # noqa: C901
        self, schedule: ScheduleDecl
    ) -> tuple[ScheduleDecl | None, tuple[BackOffDecl, ...]]:
        """
        Processes a schedule to determine if it contains any custom schedulers.

        If it does, it returns a new schedule with all the required let bindings added to the other scope.
        If not, returns none.

        Also processes all rulesets in the schedule to make sure they are registered.
        """
        bound_schedulers: list[UUID] = []
        unbound_schedulers: dict[UUID, BackOffDecl] = {}
        persistent_schedulers: dict[UUID, BackOffDecl] = {}
        has_bound_scheduler = False

        def helper(s: ScheduleDecl) -> None:
            nonlocal has_bound_scheduler
            match s:
                case LetSchedulerDecl(scheduler, inner):
                    has_bound_scheduler = True
                    bound_schedulers.append(scheduler.id)
                    try:
                        return helper(inner)
                    finally:
                        bound_schedulers.pop()
                case RunDecl(ruleset_name, _, scheduler):
                    self.ruleset_to_egg(ruleset_name)
                    if scheduler and scheduler.id not in bound_schedulers:
                        if scheduler.persistent:
                            persistent_schedulers[scheduler.id] = scheduler
                        else:
                            unbound_schedulers[scheduler.id] = scheduler
                case SaturateDecl(inner) | RepeatDecl(inner, _):
                    return helper(inner)
                case SequenceDecl(schedules):
                    for sc in schedules:
                        helper(sc)
                case _:
                    assert_never(s)
            return None

        helper(schedule)
        if not has_bound_scheduler and not unbound_schedulers and not persistent_schedulers:
            return None, ()
        for scheduler in unbound_schedulers.values():
            schedule = LetSchedulerDecl(scheduler, schedule)
        return schedule, tuple(persistent_schedulers.values())

    def _schedule_to_egg(self, schedule: ScheduleDecl) -> bindings._Schedule:
        msg = "Should never reach this, let schedulers should be handled by custom scheduler"
        match schedule:
            case SaturateDecl(schedule):
                return bindings.Saturate(span(), self._schedule_to_egg(schedule))
            case RepeatDecl(schedule, times):
                return bindings.Repeat(span(), times, self._schedule_to_egg(schedule))
            case SequenceDecl(schedules):
                return bindings.Sequence(span(), [self._schedule_to_egg(s) for s in schedules])
            case RunDecl(ruleset_ident, until, scheduler):
                if scheduler is not None:
                    raise ValueError(msg)
                config = bindings.RunConfig(
                    str(ruleset_ident), None if not until else list(map(self.fact_to_egg, until))
                )
                return bindings.Run(span(), config)
            case LetSchedulerDecl():
                raise ValueError(msg)
            case _:
                assert_never(schedule)

    def _schedule_with_scheduler_to_egg(  # noqa: C901, PLR0912
        self, schedule: ScheduleDecl, bound_schedulers: list[BackOffDecl]
    ) -> list[bindings._Expr]:
        """
        Turns a scheduler into an egg expression, to be used with a custom extract command.

        The bound_schedulers is a list of all the schedulers that have been bound. We can lookup their name as `_scheduler_{index}`.
        """
        match schedule:
            case LetSchedulerDecl(scheduler, inner):
                name = self._local_scheduler_name(len(bound_schedulers))
                bound_schedulers.append(scheduler)
                let_decl = bindings.Call(
                    span(),
                    "let-scheduler",
                    [bindings.Var(span(), name), self._back_off_scheduler_to_egg(scheduler)],
                )
                try:
                    inner_exprs = self._schedule_with_scheduler_to_egg(inner, bound_schedulers)
                finally:
                    bound_schedulers.pop()
                return [bindings.Call(span(), "seq", [let_decl, *inner_exprs])]
            case RunDecl(ruleset_ident, until, scheduler):
                args: list[bindings._Expr] = [bindings.Var(span(), str(ruleset_ident))]
                if scheduler:
                    name = "run-with"
                    scheduler_name = self._persistent_scheduler_name(scheduler)
                    for i in range(len(bound_schedulers) - 1, -1, -1):
                        if bound_schedulers[i].id == scheduler.id:
                            scheduler_name = self._local_scheduler_name(i)
                            break
                    args.insert(0, bindings.Var(span(), scheduler_name))
                else:
                    name = "run"
                if until:
                    if len(until) > 1:
                        msg = "Can only have one until fact with custom scheduler"
                        raise ValueError(msg)
                    args.append(bindings.Var(span(), ":until"))
                    fact_egg = self.fact_to_egg(until[0])
                    if isinstance(fact_egg, bindings.Eq):
                        msg = "Cannot use equality fact with custom scheduler"
                        raise ValueError(msg)
                    args.append(fact_egg.expr)
                return [bindings.Call(span(), name, args)]
            case SaturateDecl(inner):
                return [
                    bindings.Call(span(), "saturate", self._schedule_with_scheduler_to_egg(inner, bound_schedulers))
                ]
            case RepeatDecl(inner, times):
                return [
                    bindings.Call(
                        span(),
                        "repeat",
                        [
                            bindings.Lit(span(), bindings.Int(times)),
                            *self._schedule_with_scheduler_to_egg(inner, bound_schedulers),
                        ],
                    )
                ]
            case SequenceDecl(schedules):
                res = []
                for s in schedules:
                    res.extend(self._schedule_with_scheduler_to_egg(s, bound_schedulers))
                return res
            case _:
                assert_never(schedule)

    def _persistent_scheduler_to_egg(self, scheduler: BackOffDecl) -> bindings._Command:
        return bindings.UserDefined(
            span(),
            "let-scheduler",
            [
                bindings.Var(span(), self._persistent_scheduler_name(scheduler)),
                self._back_off_scheduler_to_egg(scheduler),
            ],
        )

    def ruleset_to_egg(self, ident: Ident) -> None:  # noqa: C901
        """
        Registers a ruleset if it's not already registered.
        """
        if ident.name == "" and ident not in self.__egg_decls__._rulesets:
            self.rulesets.setdefault(ident, set())
            return
        egg_name = str(ident)
        if self.egglog_file_state is not None and (
            not _egg_name_is_source_safe_symbol(egg_name) or _egg_name_is_parser_literal(egg_name)
        ):
            msg = (
                f"Ruleset name {egg_name!r} cannot be used with save_egglog_string=True because "
                "it does not serialize as one Egglog symbol"
            )
            raise ValueError(msg)
        match self.__egg_decls__._rulesets[ident]:
            case RulesetDecl(rules):
                if ident not in self.rulesets:
                    if str(ident):
                        self.run_program(bindings.AddRuleset(span(), str(ident)))
                    added_rules = self.rulesets[ident] = set()
                else:
                    added_rules = self.rulesets[ident]
                for rule in rules:
                    if rule in added_rules:
                        continue
                    commands = self.commands_to_egg(rule, ident)
                    self.run_program(*commands)
                    added_rules.add(rule)
            case CombinedRulesetDecl(rulesets):
                if ident in self.rulesets:
                    return
                self.rulesets[ident] = set()
                for ruleset in rulesets:
                    self.ruleset_to_egg(ruleset)
                self.run_program(bindings.UnstableCombinedRuleset(span(), str(ident), list(map(str, rulesets))))

    def commands_to_egg(  # noqa: C901, PLR0912
        self, cmd: CommandDecl, ruleset: Ident
    ) -> list[bindings._Command]:
        match cmd:
            case ActionCommandDecl(action):
                return [
                    bindings.ActionCommand(action_egg)
                    for action_egg in self.actions_to_egg(action, expr_to_let=True, standalone=True)
                ]
            case RewriteDecl(tp, lhs, rhs, conditions) | BiRewriteDecl(tp, lhs, rhs, conditions):
                self.type_ref_to_egg(tp)
                name = str(self.rule_name_counter)
                self.rule_name_counter += 1
                rewrite = bindings.Rewrite(
                    span(),
                    self._expr_to_egg(lhs),
                    self._expr_to_egg(rhs),
                    [self.fact_to_egg(c, expr_to_let=False) for c in conditions],
                    name,
                )
                egg_cmd: bindings._Command
                if isinstance(cmd, RewriteDecl):
                    egg_cmd = bindings.RewriteCommand(str(ruleset), rewrite, cmd.subsume)
                    serialized_name = str(egg_cmd)
                    reported_name = serialized_name.replace('"', "'")
                    if (
                        self.egglog_file_state is not None
                        and (not self.seminaive or self.egraph.no_decomp())
                        and (not cmd.subsume or isinstance(lhs, CallDecl))
                    ):
                        rule = self._rewrite_to_rule_decl(tp, lhs, rhs, conditions, reported_name, cmd.subsume)
                        commands = self.commands_to_egg(rule, ruleset)
                        self.rule_name_to_command_decl[reported_name] = cmd
                        return commands
                    self.rule_name_to_command_decl[name] = cmd
                    # Saving a transcript executes the serialized command, whose syntax does not
                    # preserve the internal rewrite name. The engine then reports the serialized
                    # rewrite itself as its name, so retain that alias for RunReport translation.
                    self.rule_name_to_command_decl[serialized_name] = cmd
                    # Backend report names render every Symbol with single quotes, while
                    # command serialization accepts symbols as double-quoted literals.
                    self.rule_name_to_command_decl[reported_name] = cmd
                else:
                    egg_cmd = bindings.BiRewriteCommand(str(ruleset), rewrite)
                    serialized_name = str(egg_cmd)
                    reported_name = serialized_name.replace('"', "'")
                    if self.egglog_file_state is not None and (not self.seminaive or self.egraph.no_decomp()):
                        report_names = (f"{reported_name}=>", f"{reported_name}<=")
                        rules = (
                            self._rewrite_to_rule_decl(tp, lhs, rhs, conditions, report_names[0], False),
                            self._rewrite_to_rule_decl(tp, rhs, lhs, conditions, report_names[1], False),
                        )
                        commands = [command for rule in rules for command in self.commands_to_egg(rule, ruleset)]
                        for report_name in report_names:
                            self.rule_name_to_command_decl[report_name] = cmd
                        return commands
                    self.rule_name_to_command_decl[f"{name}=>"] = cmd
                    self.rule_name_to_command_decl[f"{name}<="] = cmd
                    for suffix in ("=>", "<="):
                        self.rule_name_to_command_decl[f"{serialized_name}{suffix}"] = cmd
                        self.rule_name_to_command_decl[f"{reported_name}{suffix}"] = cmd
                return [egg_cmd]
            case RuleDecl(head, body, name, eval_mode, no_decomp):
                if not name:
                    name = str(self.rule_name_counter)
                    self.rule_name_counter += 1
                self.rule_name_to_command_decl[name] = cmd
                eval_modes = {
                    "seminaive": bindings.Seminaive(),
                    "naive": bindings.Naive(),
                    "unsafe-seminaive": bindings.UnsafeSeminaive(),
                }
                if eval_mode not in eval_modes:
                    msg = (
                        f"Unknown rule evaluation mode {eval_mode!r}; expected "
                        "'seminaive', 'naive', or 'unsafe-seminaive'"
                    )
                    raise ValueError(msg)
                binding_eval_mode = (
                    bindings.Naive()
                    if not self.seminaive
                    else cast(
                        "bindings.Seminaive | bindings.Naive | bindings.UnsafeSeminaive",
                        eval_modes[eval_mode],
                    )
                )
                used_variable_names = _rule_variable_names(cmd)
                return [
                    bindings.RuleCommand(
                        bindings.Rule(
                            span(),
                            [
                                action_egg
                                for action in head
                                for action_egg in self.actions_to_egg(action, used_variable_names=used_variable_names)
                            ],
                            [self.fact_to_egg(f, expr_to_let=False) for f in body],
                            name or "",
                            str(ruleset),
                            binding_eval_mode,
                            no_decomp or self.egraph.no_decomp(),
                        )
                    )
                ]
            case DefaultRewriteDecl(ref, expr, subsume):
                sig = self.__egg_decls__.get_callable_decl(ref).signature
                assert isinstance(sig, FunctionSignature)
                # Replace args with rule_var_name mapping
                arg_mapping = tuple(
                    TypedExprDecl(tp.to_just(), UnboundVarDecl(name, f"_{i}"))
                    for i, (name, tp) in enumerate(zip(sig.arg_names, sig.arg_types, strict=True))
                )
                rewrite_decl = RewriteDecl(
                    sig.semantic_return_type.to_just(), CallDecl(ref, arg_mapping), expr, (), subsume
                )
                return self.commands_to_egg(rewrite_decl, ruleset)
            case _:
                assert_never(cmd)

    def _rewrite_to_rule_decl(
        self,
        tp: JustTypeRef,
        lhs: ExprDecl,
        rhs: ExprDecl,
        conditions: tuple[FactDecl, ...],
        name: str,
        subsume: bool,
    ) -> RuleDecl:
        """Desugar a rewrite while retaining rule options that rewrite syntax cannot encode."""
        used_variable_names = _rule_variable_names(RuleDecl((), (EqDecl(tp, lhs, rhs), *conditions), None))
        fresh_name = self._allocate_synthetic_let_name().removeprefix("$")
        while fresh_name in used_variable_names:
            fresh_name = self._allocate_synthetic_let_name().removeprefix("$")
        fresh = UnboundVarDecl(fresh_name, fresh_name)
        head: list[ActionDecl] = [UnionDecl(tp, fresh, rhs)]
        if subsume:
            if not isinstance(lhs, CallDecl):
                msg = "subsumed rewrite must have a function call on the lhs"
                raise ValueError(msg)
            head.append(ChangeDecl(tp, lhs, "subsume"))
        return RuleDecl(tuple(head), (EqDecl(tp, fresh, lhs), *conditions), name)

    def actions_to_egg(  # noqa: C901, PLR0911, PLR0912
        self,
        action: ActionDecl,
        expr_to_let: bool = False,
        *,
        standalone: bool = False,
        used_variable_names: set[str] | None = None,
    ) -> list[bindings._Action]:
        match action:
            case LetDecl(name, typed_expr):
                normalized_name = _normalize_global_let_name(name)
                if self.egglog_file_state is not None and not _egg_name_is_source_safe_symbol(normalized_name):
                    msg = (
                        f"Let name {name!r} cannot be used with save_egglog_string=True because "
                        "it does not serialize as one Egglog symbol"
                    )
                    raise ValueError(msg)
                var_decl = LetRefDecl(name)
                var_egg = self._expr_to_egg(var_decl)
                self.expr_to_egg_cache[var_decl] = var_egg
                return [
                    bindings.Let(
                        span(),
                        var_egg.name,
                        self.typed_expr_to_egg(typed_expr, expr_to_let=expr_to_let),
                    )
                ]
            case SetDecl(tp, call, rhs):
                self.type_ref_to_egg(tp)
                egg_fn, typed_args = self.translate_call(call)
                return [
                    bindings.Set(
                        span(),
                        egg_fn,
                        [self.typed_expr_to_egg(arg, expr_to_let) for arg in typed_args],
                        self._expr_to_egg(rhs, expr_to_let=expr_to_let),
                    )
                ]
            case ExprActionDecl(typed_expr):
                if not isinstance(typed_expr.expr, CallDecl):
                    msg = "Top-level egglog expr commands must be calls"
                    raise ValueError(msg)  # noqa: TRY004 - preserve the public validation error
                callable_decl = self.__egg_decls__.get_callable_decl(typed_expr.expr.callable)
                if (
                    self.egglog_file_state is not None
                    and callable_decl.egg_name is not None
                    and callable_decl.egg_name
                    in (_EGGLOG_RESERVED_COMMAND_OR_ACTION_HEADS if standalone else _EGGLOG_RESERVED_ACTION_HEADS)
                ):
                    context = "top-level action" if standalone else "rule action"
                    msg = (
                        f"Explicit Egglog callable name {callable_decl.egg_name!r} cannot be used as a {context} "
                        "with save_egglog_string=True because it is parsed as Egglog syntax"
                    )
                    raise ValueError(msg)
                egg_expr = self.typed_expr_to_egg(typed_expr, expr_to_let=expr_to_let)
                if isinstance(egg_expr, bindings.Var):
                    return []
                assert isinstance(egg_expr, bindings.Call)
                return [bindings.Expr_(span(), egg_expr)]
            case ChangeDecl(tp, call, change):
                self.type_ref_to_egg(tp)
                egg_fn, typed_args = self.translate_call(call)
                egg_change: bindings._Change
                match change:
                    case "delete":
                        egg_change = bindings.Delete()
                    case "subsume":
                        egg_change = bindings.Subsume()
                    case _:
                        assert_never(change)
                return [
                    bindings.Change(
                        span(),
                        egg_change,
                        egg_fn,
                        [self.typed_expr_to_egg(arg, expr_to_let) for arg in typed_args],
                    )
                ]
            case UnionDecl(tp, lhs, rhs):
                self.type_ref_to_egg(tp)
                return [
                    bindings.Union(
                        span(),
                        self._expr_to_egg(lhs, expr_to_let=expr_to_let),
                        self._expr_to_egg(rhs, expr_to_let=expr_to_let),
                    )
                ]
            case PanicDecl(name):
                return [bindings.Panic(span(), name)]
            case SetCostDecl(tp, expr, cost):
                self.type_ref_to_egg(tp)
                egg_fn, typed_args = self.translate_call(expr)
                cost_table = self.create_cost_table(expr.callable)
                # Match egglog-experimental's set-cost action macro: bind each
                # argument once, materialize the target call, then write its
                # validated cost. Structural lowering here avoids evaluating a
                # cost expression early through top-level factoring.
                lowered: list[bindings._Action] = []
                args_egg: list[bindings._Expr] = []
                for typed_arg in typed_args:
                    name = self._allocate_synthetic_let_name()
                    if standalone:
                        # Mark implementation-only globals so freeze() omits them
                        # without making later expressions reuse their values.
                        var_egg = bindings.Var(span(), name)
                        self.expr_to_letref_cache[LetRefDecl(name)] = var_egg
                        self.expr_to_egg_cache[LetRefDecl(name)] = var_egg
                    else:
                        name = name.removeprefix("$")
                        while used_variable_names is not None and name in used_variable_names:
                            name = self._allocate_synthetic_let_name().removeprefix("$")
                        if used_variable_names is not None:
                            used_variable_names.add(name)
                        var_egg = bindings.Var(span(), name)
                    lowered.append(bindings.Let(span(), name, self.typed_expr_to_egg(typed_arg, False)))
                    args_egg.append(var_egg)
                lowered.append(bindings.Expr_(span(), bindings.Call(span(), egg_fn, args_egg)))
                cost_expr = self._expr_to_egg(cost, expr_to_let=False)
                validated_cost = bindings.Call(span(), _VALIDATE_COST_PRIMITIVE, [cost_expr])
                lowered.append(bindings.Set(span(), cost_table, args_egg, validated_cost))
                return lowered
            case _:
                assert_never(action)

    def create_cost_table(self, ref: CallableRef) -> str:
        """
        Creates the egg cost table if needed and gets the name of the table.
        """
        if ref in self.cost_table_names:
            return self.cost_table_names[ref]
        name = f"cost_table_{self.callable_ref_to_egg(ref)[0]}"
        signature = self.__egg_decls__.get_callable_decl(ref).signature
        assert isinstance(signature, FunctionSignature), "Can only add cost tables for functions"
        target_schema = self._signature_to_egg_schema(signature)
        schema = self._signature_to_egg_schema(replace(signature, return_type=TypeRefWithVars(Ident.builtin("i64"))))

        # egglog-experimental's DynamicCostModel probes this exact canonical
        # name, so choosing a generated suffix would silently ignore costs.
        # Aliases of the same backend callable share a cost table, but overloaded
        # callables with incompatible schemas cannot: the backend protocol names
        # cost tables only by the callable's backend symbol. A user-declared raw
        # cost table can also be reused, but only when it is a bodyless function
        # with the protocol's input sorts and i64 output.
        if not self._has_compatible_cost_table_target(name, target_schema):
            existing_refs = self.egg_fn_to_callable_refs.get(name, set())
            compatible_raw_table = bool(existing_refs) and all(
                self._raw_cost_table_matches_schema(existing_ref, schema) for existing_ref in existing_refs
            )
            if compatible_raw_table and not self._callable_is_table_backed(ref):
                raise ValueError(
                    f"Canonical dynamic-cost table {name!r} for an eager or builtin primitive "
                    "cannot also be a user-declared function"
                )
            if existing_refs and not compatible_raw_table:
                msg = (
                    f"Canonical dynamic-cost table {name!r} is already used by an incompatible callable; "
                    "it must be a bodyless function with the target's input sorts and i64 output"
                )
                raise ValueError(msg)
            if not compatible_raw_table:
                if self._backend_symbol_is_occupied(name):
                    raise ValueError(f"Canonical dynamic-cost table name {name!r} is already in use")
                self.run_program(bindings.FunctionCommand(span(), name, schema, None))
        self.cost_table_names[ref] = name
        return name

    def _callable_is_table_backed(self, ref: CallableRef) -> bool:
        """Return whether Egglog stores rows for this callable."""
        decl = self.__egg_decls__.get_callable_decl(ref)
        match decl:
            case RelationDecl() | ConstructorDecl() | ConstantDecl(body=None):
                return True
            case FunctionDecl(body=None, builtin=False):
                return not isinstance(ref, UnnamedFunctionRef)
            case ConstantDecl() | FunctionDecl():
                return False
            case _:
                assert_never(decl)

    def _raw_cost_table_matches_schema(self, ref: CallableRef, schema: bindings.Schema) -> bool:
        """Check the complete public declaration contract for a raw dynamic-cost table."""
        decl = self.__egg_decls__.get_callable_decl(ref)
        if not (
            isinstance(decl, FunctionDecl)
            and not decl.builtin
            and decl.body is None
            and decl.merge is None
            and isinstance(decl.signature, FunctionSignature)
        ):
            return False
        existing_schema = self._signature_to_egg_schema(decl.signature)
        return existing_schema.input == schema.input and existing_schema.output == schema.output

    def _has_compatible_cost_table_target(self, name: str, target_schema: bindings.Schema) -> bool:
        """Validate every callable already sharing a canonical dynamic-cost table."""
        existing_cost_refs = [ref for ref, existing_name in self.cost_table_names.items() if existing_name == name]
        for existing_ref in existing_cost_refs:
            existing_signature = self.__egg_decls__.get_callable_decl(existing_ref).signature
            assert isinstance(existing_signature, FunctionSignature)
            existing_schema = self._signature_to_egg_schema(existing_signature)
            if existing_schema.input != target_schema.input or existing_schema.output != target_schema.output:
                raise ValueError(
                    f"Canonical dynamic-cost table {name!r} already serves a callable with an incompatible schema"
                )
        return bool(existing_cost_refs)

    def fact_to_egg(self, fact: FactDecl, *, expr_to_let: bool = False) -> bindings._Fact:
        match fact:
            case EqDecl(tp, left, right):
                self.type_ref_to_egg(tp)
                return bindings.Eq(
                    span(),
                    self._expr_to_egg(left, expr_to_let=expr_to_let),
                    self._expr_to_egg(right, expr_to_let=expr_to_let),
                )
            case ExprFactDecl(typed_expr):
                if isinstance(typed_expr.expr, CallDecl) and typed_expr.expr.callable != FunctionRef(
                    Ident.builtin("!=")
                ):
                    callable_decl = self.__egg_decls__.get_callable_decl(typed_expr.expr.callable)
                    if (
                        self.egglog_file_state is not None
                        and callable_decl.egg_name is not None
                        and callable_decl.egg_name == "="
                    ):
                        msg = (
                            "Explicit Egglog callable name '=' cannot be used as a fact with "
                            "save_egglog_string=True because it is parsed as equality syntax"
                        )
                        raise ValueError(msg)
                return bindings.Fact(self.typed_expr_to_egg(typed_expr, expr_to_let=expr_to_let))
            case _:
                assert_never(fact)

    def callable_ref_to_egg(self, ref: CallableRef) -> tuple[str, bool]:  # noqa: C901, PLR0912
        """
        Returns the egg function name for a callable reference, registering it if it is not already registered.

        Also returns whether the args should be reversed
        """
        if ref in self.callable_ref_to_egg_fn:
            return self.callable_ref_to_egg_fn[ref]
        decl = self.__egg_decls__.get_callable_decl(ref)
        if (
            self.egglog_file_state is not None
            and decl.egg_name
            and (not _egg_name_is_source_safe_symbol(decl.egg_name) or _egg_name_is_parser_literal(decl.egg_name))
        ):
            msg = (
                f"Explicit Egglog callable name {decl.egg_name!r} cannot be used with "
                "save_egglog_string=True because it is parsed as Egglog syntax"
            )
            raise ValueError(msg)
        egg_name = decl.egg_name or self._allocate_name(
            self._generate_callable_egg_name(ref), avoid_reserved_call_heads=True
        )
        cost_table_targets = tuple(
            target_ref for target_ref, cost_table_name in self.cost_table_names.items() if cost_table_name == egg_name
        )
        reuse_cost_table = bool(cost_table_targets)
        existing_refs = self.egg_fn_to_callable_refs.get(egg_name, set())
        if existing_refs and not reuse_cost_table and not (isinstance(decl, FunctionDecl) and decl.builtin):
            msg = f"Explicit Egglog callable name {egg_name!r} is already registered"
            raise ValueError(msg)
        for target_ref in cost_table_targets:
            if not self._callable_is_table_backed(target_ref):
                raise ValueError(
                    f"Canonical dynamic-cost table {egg_name!r} for an eager or builtin primitive "
                    "cannot also be a user-declared function"
                )
            target_signature = self.__egg_decls__.get_callable_decl(target_ref).signature
            assert isinstance(target_signature, FunctionSignature)
            cost_schema = self._signature_to_egg_schema(
                replace(target_signature, return_type=TypeRefWithVars(Ident.builtin("i64")))
            )
            if not self._raw_cost_table_matches_schema(ref, cost_schema):
                msg = (
                    f"Canonical dynamic-cost table {egg_name!r} is already used by an incompatible callable; "
                    "it must be a bodyless function with the target's input sorts and i64 output"
                )
                raise ValueError(msg)
        if reuse_cost_table and self.egg_fn_to_callable_refs.get(egg_name):
            msg = f"Canonical dynamic-cost table {egg_name!r} already has a raw callable alias"
            raise ValueError(msg)
        callable_signature = decl.signature
        reverse_args = callable_signature.reverse_args if isinstance(callable_signature, FunctionSignature) else False
        match decl:
            case RelationDecl(arg_types, _, _):
                self.run_program(bindings.Relation(span(), egg_name, [self.type_ref_to_egg(a) for a in arg_types]))
            case ConstantDecl(tp, _, body, merge):
                if body is not None:
                    self.run_program(self._primitive_command_to_egg(egg_name, decl.signature, body))
                else:
                    # Egglog v3 has no constant command, so lower Python constants
                    # and class variables as zero-argument functions or constructors.
                    is_function = self.__egg_decls__._classes[tp.ident].builtin or merge is not None
                    schema = bindings.Schema([], self.type_ref_to_egg(tp))
                    if is_function:
                        self.run_program(
                            bindings.FunctionCommand(
                                span(),
                                egg_name,
                                schema,
                                self._expr_to_egg(merge) if merge else None,
                            )
                        )
                    else:
                        self.run_program(bindings.Constructor(span(), egg_name, schema, None, False))
            case FunctionDecl(signature=signature, builtin=builtin, body=body, merge=merge):
                if not builtin and not reuse_cost_table:
                    assert isinstance(signature, FunctionSignature), "Cannot turn special function to egg"
                    if body is None and isinstance(ref, UnnamedFunctionRef):
                        body = ref.res
                    if body is not None:
                        self.run_program(self._primitive_command_to_egg(egg_name, signature, body))
                    else:
                        # Compile functions that return unit to relations, because these show up in methods where you
                        # cant use the relation helper
                        schema = self._signature_to_egg_schema(signature)
                        if signature.return_type == TypeRefWithVars(Ident.builtin("Unit")):
                            if merge:
                                msg = "Cannot specify a merge function for a function that returns unit"
                                raise ValueError(msg)
                            self.run_program(bindings.Relation(span(), egg_name, schema.input))
                        else:
                            self.run_program(
                                bindings.FunctionCommand(
                                    span(),
                                    egg_name,
                                    schema,
                                    self._expr_to_egg(merge) if merge else None,
                                ),
                            )
            case ConstructorDecl(signature, _, cost, unextractable):
                self.run_program(
                    bindings.Constructor(
                        span(),
                        egg_name,
                        self._signature_to_egg_schema(signature),
                        cost,
                        unextractable,
                    ),
                )
            case _:
                assert_never(decl)
        # Publish the reverse mapping only after any backend declaration has
        # succeeded; otherwise a failed registration corrupts extraction and
        # freeze by claiming an alias the backend never accepted.
        self.egg_fn_to_callable_refs.setdefault(egg_name, set()).add(ref)
        self.callable_ref_to_egg_fn[ref] = egg_name, reverse_args
        return egg_name, reverse_args

    def _primitive_command_to_egg(
        self,
        egg_name: str,
        signature: FunctionSignature,
        body: TypedExprDecl,
    ) -> bindings.UserDefined:
        backend_arg_types = signature.arg_types[::-1] if signature.reverse_args else signature.arg_types
        input_sort_expr = self._primitive_input_sorts_to_egg([
            self.type_ref_to_egg(arg_type.to_just()) for arg_type in backend_arg_types
        ])
        output_sort_expr = bindings.Var(span(), self.type_ref_to_egg(signature.semantic_return_type.to_just()))
        return bindings.UserDefined(
            span(),
            "primitive",
            [
                bindings.Var(span(), egg_name),
                input_sort_expr,
                output_sort_expr,
                self.typed_expr_to_egg(body, expr_to_let=False),
            ],
        )

    def _primitive_input_sorts_to_egg(self, sort_names: list[str]) -> bindings._Expr:
        if not sort_names:
            return bindings.Lit(span(), bindings.Unit())
        if len(sort_names) == 1:
            return bindings.Var(span(), sort_names[0])
        first, *rest = sort_names
        return bindings.Call(span(), first, [bindings.Var(span(), sort_name) for sort_name in rest])

    def _signature_to_egg_schema(self, signature: FunctionSignature) -> bindings.Schema:
        backend_arg_types = signature.arg_types[::-1] if signature.reverse_args else signature.arg_types
        return bindings.Schema(
            [self.type_ref_to_egg(a.to_just()) for a in backend_arg_types],
            self.type_ref_to_egg(signature.semantic_return_type.to_just()),
        )

    def type_ref_to_egg(self, ref: JustTypeRef) -> str:
        """
        Returns the egg sort name for a type reference, registering it not already registered, and also recursively
        any type args are registered.
        """
        try:
            return self.type_ref_to_egg_sort[ref]
        except KeyError:
            pass
        decl = self.__egg_decls__._classes[ref.ident]
        if (
            self.egglog_file_state is not None
            and decl.egg_name
            and (not _egg_name_is_source_safe_symbol(decl.egg_name) or _egg_name_is_parser_literal(decl.egg_name))
        ):
            msg = (
                f"Explicit Egglog sort name {decl.egg_name!r} cannot be used with "
                "save_egglog_string=True because it is not parsed as an Egglog symbol"
            )
            raise ValueError(msg)
        if not decl.builtin and not ref.args and decl.egg_name and self._backend_symbol_is_occupied(decl.egg_name):
            msg = f"Explicit Egglog sort name {decl.egg_name!r} is already registered"
            raise ValueError(msg)
        arg_names = [self.type_ref_to_egg(arg) for arg in ref.args]
        self.type_ref_to_egg_sort[ref] = egg_name = (not ref.args and decl.egg_name) or self._allocate_name(
            self._generate_type_egg_name(ref, decl, arg_names)
        )
        self.egg_sort_to_type_ref[egg_name] = ref

        if decl.builtin:
            # If this has args, create a new parameterized version of the builtin class
            if ref.args:
                if ref.ident == Ident.builtin("UnstableFn"):
                    type_args: list[bindings._Expr] = [
                        bindings.Call(
                            span(),
                            self.type_ref_to_egg(ref.args[1]),
                            [bindings.Var(span(), self.type_ref_to_egg(a)) for a in ref.args[2:]],
                        )
                        if len(ref.args) > 1
                        else bindings.Lit(span(), bindings.Unit()),
                        bindings.Var(span(), self.type_ref_to_egg(ref.args[0])),
                    ]
                else:
                    type_args = [bindings.Var(span(), self.type_ref_to_egg(a)) for a in ref.args]
                assert decl.egg_name
                self.run_program(bindings.Sort(span(), egg_name, (decl.egg_name, type_args)))

            # For builtin classes, let's also make sure we have the mapping of all egg fn names for class methods.
            # these can be created even without adding them to the e-graph, like `vec-empty` which can be extracted
            # even if you never use that function.
            for method_name in decl.class_methods:
                self.callable_ref_to_egg(ClassMethodRef(ref.ident, method_name))
            if decl.init:
                self.callable_ref_to_egg(InitRef(ref.ident))
        else:
            self.run_program(bindings.Sort(span(), egg_name, None))

        return egg_name

    def op_mapping(self) -> dict[str, str]:
        """
        Create a mapping of egglog function name to Python function name, for use in the serialized format
        for better visualization.

        Includes cost tables
        """
        return {
            k: pretty_callable_ref(self.__egg_decls__, next(iter(v)))
            for k, v in self.egg_fn_to_callable_refs.items()
            if len(v) == 1
        } | {
            name: f"cost({pretty_callable_ref(self.__egg_decls__, ref, include_all_args=True)})"
            for ref, name in self.cost_table_names.items()
        }

    def possible_egglog_functions(self, names: list[str]) -> Iterable[str]:
        """
        Given a list of egglog functions, returns all the possible Python function strings
        """
        for name in names:
            for c in self.egg_fn_to_callable_refs.get(name, ()):
                yield pretty_callable_ref(self.__egg_decls__, c)

    def typed_expr_to_egg(
        self,
        typed_expr_decl: TypedExprDecl,
        expr_to_let: bool = True,
    ) -> bindings._Expr:
        # transform all expressions with multiple parents into a let binding, so that less expressions
        # are sent to egglog. Only for performance reasons.
        if expr_to_let:
            have_multiple_parents = _exprs_multiple_parents(typed_expr_decl)
            for expr in reversed(have_multiple_parents):
                self._transform_let(expr)

        self.type_ref_to_egg(typed_expr_decl.tp)
        return self._expr_to_egg(typed_expr_decl.expr, expr_to_let=expr_to_let)

    def _transform_let(self, typed_expr: TypedExprDecl) -> TypedExprDecl | None:
        """
        Rewrites this expression as a let binding if it's not already a let binding.
        """
        if not isinstance(typed_expr.expr, CallDecl):
            return typed_expr
        if not is_callable_decl_constructor(
            self.__egg_decls__, self.__egg_decls__.get_callable_decl(typed_expr.expr.callable)
        ):
            return typed_expr
        # A synthetic let is a top-level binding, so it cannot capture a rule
        # variable (or an invalid top-level unbound variable). Leave those
        # expressions inline for their enclosing command to validate instead.
        if _contains_unbound_var(typed_expr):
            return typed_expr
        if typed_expr.expr in self.expr_to_letref_cache:
            return None
        var_decl = LetRefDecl(self._allocate_synthetic_let_name())
        var_egg = self._expr_to_egg(var_decl)
        cmd = bindings.ActionCommand(bindings.Let(span(), var_egg.name, self.typed_expr_to_egg(typed_expr, True)))
        self.run_program(cmd)
        self.expr_to_letref_cache[typed_expr.expr] = var_egg
        self.expr_to_egg_cache[var_decl] = var_egg
        return None

    @overload
    def _expr_to_egg(self, expr_decl: CallDecl, *, expr_to_let: bool = ...) -> bindings.Call: ...

    @overload
    def _expr_to_egg(self, expr_decl: UnboundVarDecl | LetRefDecl, *, expr_to_let: bool = ...) -> bindings.Var: ...

    @overload
    def _expr_to_egg(self, expr_decl: ExprDecl, *, expr_to_let: bool = ...) -> bindings._Expr: ...

    def _expr_to_egg(self, expr_decl: ExprDecl, *, expr_to_let: bool = False) -> bindings._Expr:  # noqa: PLR0912,C901
        """
        Convert an ExprDecl to an egg expression.
        """
        if expr_to_let:
            try:
                return self.expr_to_letref_cache[expr_decl]
            except KeyError:
                pass
        cache = self.expr_to_let_egg_cache if expr_to_let else self.expr_to_egg_cache
        try:
            return cache[expr_decl]
        except KeyError:
            pass
        res: bindings._Expr
        match expr_decl:
            case LetRefDecl(name):
                res = bindings.Var(span(), _normalize_global_let_name(name))
            case UnboundVarDecl(name, egg_name):
                emitted_name = egg_name or f"_{name}"
                if self.egglog_file_state is not None and (
                    emitted_name == "_"
                    or emitted_name.startswith("@")
                    or not _egg_name_is_source_safe_symbol(emitted_name)
                    or _egg_name_is_parser_literal(emitted_name)
                ):
                    msg = (
                        f"Egglog variable name {emitted_name!r} cannot be used with "
                        "save_egglog_string=True because it is not parsed as an Egglog symbol"
                    )
                    raise ValueError(msg)
                res = bindings.Var(span(), emitted_name)
            case LitDecl(value):
                l: bindings._Literal
                match value:
                    case None:
                        l = bindings.Unit()
                    case bool(i):
                        l = bindings.Bool(i)
                    case int(i):
                        l = bindings.Int(i)
                    case float(f):
                        l = bindings.Float(f)
                    case str(s):
                        l = bindings.String(s)
                    case _:
                        assert_never(value)
                res = bindings.Lit(span(), l)
            case CallDecl() | GetCostDecl():
                egg_fn, typed_args = self.translate_call(expr_decl)
                egg_args = [self.typed_expr_to_egg(a, expr_to_let) for a in typed_args]
                res = bindings.Call(span(), egg_fn, egg_args)
            case PyObjectDecl(value):
                res = bindings.Call(
                    span(),
                    "py-object",
                    [bindings.Lit(span(), bindings.String(standard_b64encode(value).decode("utf-8")))],
                )
            case PartialCallDecl(call_decl):
                egg_fn, typed_args = self.translate_call(call_decl)
                res = bindings.Call(
                    span(),
                    "unstable-fn",
                    [
                        bindings.Lit(span(), bindings.String(egg_fn)),
                        *[self.typed_expr_to_egg(arg, expr_to_let) for arg in typed_args],
                    ],
                )
            case ValueDecl():
                msg = "Cannot turn a Value into an expression"
                raise ValueError(msg)
            case DummyDecl():
                msg = "Cannot turn a DummyDecl into an expression"
                raise ValueError(msg)
            case _:
                assert_never(expr_decl.expr)
        cache[expr_decl] = res
        return res

    def translate_call(self, expr: CallDecl | GetCostDecl) -> tuple[str, list[TypedExprDecl]]:
        """
        Handle get cost and call decl, turn into egg table name and typed expr decls.
        """
        match expr:
            case CallDecl(ref, args, _):
                egg_fn, reverse_args = self.callable_ref_to_egg(ref)
            case GetCostDecl(ref, args):
                egg_fn = self.create_cost_table(ref)
                _, reverse_args = self.callable_ref_to_egg(ref)
            case _:
                assert_never(expr)
        args_list = list(args)
        if reverse_args:
            args_list.reverse()
        return egg_fn, args_list

    def exprs_from_egg(self, termdag: bindings.TermDag, terms: list[int], tp: JustTypeRef) -> Iterable[TypedExprDecl]:
        """
        Create a function that can convert from an egg term to a typed expr.
        """
        state = FromEggState(self, termdag)
        return [state.resolve_term(term_id, tp) for term_id in terms]

    def _get_possible_types(self, cls_ident: Ident) -> frozenset[JustTypeRef]:
        """
        Given a class name, returns all possible registered types that it can be.
        """
        return frozenset(tp for tp in self.type_ref_to_egg_sort if tp.ident == cls_ident)

    def _generate_callable_egg_name(self, ref: CallableRef) -> str:
        """
        Generate a fully-qualified Egglog name for a callable reference.

        Qualification separates same-named Python declarations; explicit names
        in the same registration batch are reserved by `_allocate_name`.
        """
        match ref:
            case FunctionRef(ident):
                return _sanitize_egg_ident(str(ident))
            case ConstantRef(ident):
                # Prefix to avoid name collisions with local vars
                return _sanitize_egg_ident(f"%{ident}")
            case (
                MethodRef(cls_ident, name)
                | ClassMethodRef(cls_ident, name)
                | ClassVariableRef(cls_ident, name)
                | PropertyRef(cls_ident, name)
            ):
                return _sanitize_egg_ident(f"{cls_ident}.{name}")
            case InitRef(cls_ident):
                return _sanitize_egg_ident(f"{cls_ident}.__init__")
            case UnnamedFunctionRef():
                name = f"_lambda_{self.unnamed_function_counter}"
                self.unnamed_function_counter += 1
                return name
            case _:
                assert_never(ref)

    def _backend_symbol_is_occupied(self, name: str) -> bool:
        """Check Egglog's shared namespace and replay-sensitive syntax names."""
        return (
            not _egg_name_is_source_safe_symbol(name)
            or _egg_name_is_parser_literal(name)
            or bool(self.egg_fn_to_callable_refs.get(name))
            or name in self.egg_sort_to_type_ref
            or name in self.cost_table_names.values()
            or name in _BUILTIN_EGG_FN_NAMES
            or name in _BUILTIN_EGG_SORT_NAMES
        )

    def _generate_type_egg_name(self, ref: JustTypeRef, decl: ClassDecl, arg_names: list[str]) -> str:
        if decl.egg_name:
            base = decl.egg_name
        else:
            # Preserve the readable dotted qualification while sanitizing each
            # Python identifier component into a source-safe Egglog symbol.
            parts = (*ref.ident.module.split("."), ref.ident.name) if ref.ident.module else (ref.ident.name,)
            base = ".".join(_sanitize_egg_ident(part) or "_" for part in parts)
        if not ref.args:
            return base
        args = ",".join(arg_names)
        return f"{base}[{args}]"

    def _allocate_synthetic_let_name(self) -> str:
        existing_let_names = self.pending_let_names | {
            egg_expr.name
            for decl, egg_expr in self.expr_to_egg_cache.items()
            if isinstance(decl, LetRefDecl) and isinstance(egg_expr, bindings.Var)
        }
        while True:
            candidate = f"$__expr_{self.expr_to_let_counter}"
            self.expr_to_let_counter += 1
            name = self._allocate_name(candidate)
            if name not in existing_let_names:
                return name

    def _allocate_name(self, candidate: str, *, avoid_reserved_call_heads: bool = False) -> str:
        # All declarations for a register(...) batch are merged before any
        # command is lowered. Reserve their explicit backend names up front so
        # generated names do not depend on action order within that batch.
        if (
            candidate not in self._explicit_backend_names
            and not self._backend_symbol_is_occupied(candidate)
            and (not avoid_reserved_call_heads or candidate not in _EGGLOG_RESERVED_CALL_HEADS)
        ):
            return candidate

        index = 1
        while (
            f"{candidate}_{index}" in self._explicit_backend_names
            or self._backend_symbol_is_occupied(f"{candidate}_{index}")
            or (avoid_reserved_call_heads and f"{candidate}_{index}" in _EGGLOG_RESERVED_CALL_HEADS)
        ):
            index += 1
        return f"{candidate}_{index}"

    def typed_expr_to_value(self, typed_expr: TypedExprDecl) -> bindings.Value:
        if isinstance(typed_expr.expr, ValueDecl):
            if typed_expr.expr.owner not in self.valid_value_owners:
                msg = "Cannot use a value that belongs to a different EGraph or inactive push scope"
                raise ValueError(msg)
            return typed_expr.expr.value
        egg_expr = self.typed_expr_to_egg(typed_expr, False)
        return call_with_current_trace(self.egraph.eval_expr, egg_expr)[1]

    def value_to_expr(self, tp: JustTypeRef, value: bindings.Value) -> ExprDecl:  # noqa: C901, PLR0911, PLR0912
        if tp.ident.module != Ident.builtin("").module:
            return ValueDecl(value, self.value_owner)

        match tp.ident.name:
            # Should match list in egraph bindings
            case "i64":
                return LitDecl(self.egraph.value_to_i64(value))
            case "f64":
                return LitDecl(self.egraph.value_to_f64(value))
            case "Bool":
                return LitDecl(self.egraph.value_to_bool(value))
            case "String":
                return LitDecl(self.egraph.value_to_string(value))
            case "Unit":
                return LitDecl(None)
            case "PyObject":
                val = self.egraph.value_to_pyobject(value)
                return PyObjectDecl(cloudpickle.dumps(val))
            case "Rational":
                fraction = self.egraph.value_to_rational(value)
                return CallDecl(
                    InitRef(Ident.builtin("Rational")),
                    (
                        TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(fraction.numerator)),
                        TypedExprDecl(JustTypeRef(Ident.builtin("i64")), LitDecl(fraction.denominator)),
                    ),
                )
            case "BigInt":
                i = self.egraph.value_to_bigint(value)
                return CallDecl(
                    ClassMethodRef(Ident.builtin("BigInt"), "from_string"),
                    (TypedExprDecl(JustTypeRef(Ident.builtin("String")), LitDecl(str(i))),),
                )
            case "BigRat":
                fraction = self.egraph.value_to_bigrat(value)
                return CallDecl(
                    InitRef(Ident.builtin("BigRat")),
                    (
                        TypedExprDecl(
                            JustTypeRef(Ident.builtin("BigInt")),
                            CallDecl(
                                ClassMethodRef(Ident.builtin("BigInt"), "from_string"),
                                (
                                    TypedExprDecl(
                                        JustTypeRef(Ident.builtin("String")), LitDecl(str(fraction.numerator))
                                    ),
                                ),
                            ),
                        ),
                        TypedExprDecl(
                            JustTypeRef(Ident.builtin("BigInt")),
                            CallDecl(
                                ClassMethodRef(Ident.builtin("BigInt"), "from_string"),
                                (
                                    TypedExprDecl(
                                        JustTypeRef(Ident.builtin("String")), LitDecl(str(fraction.denominator))
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            case "Map":
                k_tp, v_tp = tp.args
                expr = CallDecl(ClassMethodRef(Ident.builtin("Map"), "empty"), (), (k_tp, v_tp))
                for k, v in self.egraph.value_to_map(value).items():
                    expr = CallDecl(
                        MethodRef(Ident.builtin("Map"), "insert"),
                        (
                            TypedExprDecl(tp, expr),
                            TypedExprDecl(k_tp, self.value_to_expr(k_tp, k)),
                            TypedExprDecl(v_tp, self.value_to_expr(v_tp, v)),
                        ),
                    )
                return expr
            case "Set":
                xs_ = self.egraph.value_to_set(value)
                (v_tp,) = tp.args
                return CallDecl(
                    InitRef(Ident.builtin("Set")),
                    tuple(TypedExprDecl(v_tp, self.value_to_expr(v_tp, x)) for x in xs_),
                    (v_tp,) if not xs_ else (),
                )
            case "Vec":
                xs = self.egraph.value_to_vec(value)
                (v_tp,) = tp.args
                return CallDecl(
                    InitRef(Ident.builtin("Vec")),
                    tuple(TypedExprDecl(v_tp, self.value_to_expr(v_tp, x)) for x in xs),
                    (v_tp,) if not xs else (),
                )
            case "MultiSet":
                xs = self.egraph.value_to_multiset(value)
                (v_tp,) = tp.args
                return CallDecl(
                    InitRef(Ident.builtin("MultiSet")),
                    tuple(TypedExprDecl(v_tp, self.value_to_expr(v_tp, x)) for x in xs),
                    (v_tp,) if not xs else (),
                )
            case "UnstableFn":
                _names, _args = self.egraph.value_to_function(value)
                return_tp, *arg_types = tp.args
                return self._unstable_fn_value_to_expr(_names, _args, return_tp, arg_types)
            case "Pair" | "Maybe":
                termdag, term, _cost = call_with_current_trace(
                    self.egraph.extract_value, value, self.type_ref_to_egg(tp)
                )
                return FromEggState(self, termdag).resolve_term(term, tp).expr
            case _:
                # If this is not a builtin type, or we don't know how to convert it, just return as value
                return ValueDecl(value, self.value_owner)

    def _unstable_fn_value_to_expr(
        self, name: str, partial_args: list[bindings.Value], return_tp: JustTypeRef, _arg_types: list[JustTypeRef]
    ) -> PartialCallDecl:
        # Similar to FromEggState::from_call but reconstructs a partial application from serialized values.
        # Find first callable ref whose return type matches and fill in arg types.
        for callable_ref in self.egg_fn_to_callable_refs.get(name, ()):
            signature = self.__egg_decls__.get_callable_decl(callable_ref).signature
            if not isinstance(signature, FunctionSignature):
                continue
            if signature.semantic_return_type.ident != return_tp.ident:
                continue
            arg_types = TypeConstraintSolver().infer_arg_types(
                signature.arg_types, signature.semantic_return_type, signature.var_arg_type, return_tp
            )
            args = tuple(
                TypedExprDecl(tp, self.value_to_expr(tp, v)) for tp, v in zip(arg_types, partial_args, strict=False)
            )
            call_decl = CallDecl(callable_ref, args)
            return PartialCallDecl(call_decl)
        raise ValueError(f"Function '{name}' not found")


# https://chatgpt.com/share/9ab899b4-4e17-4426-a3f2-79d67a5ec456
_EGGLOG_INVALID_IDENT = re.compile(r"[^\w\-+*/?!=<>&|^/%]")


def _egg_name_is_parser_literal(name: str) -> bool:
    if not name or name in _EGGLOG_LITERAL_NAMES:
        return True
    # Egglog parses every finite Rust f64 spelling as a literal rather than an
    # atom. Callable names are sanitized first, but sort names can retain dots.
    return bool(_EGGLOG_NUMBER.fullmatch(name)) and math.isfinite(float(name))


def _sanitize_egg_ident(input_string: str) -> str:
    """
    Replaces all invalid characters in an egg identifier with an underscore.
    """
    return _EGGLOG_INVALID_IDENT.sub("_", input_string)


def _exprs_multiple_parents(typed_expr: TypedExprDecl) -> list[TypedExprDecl]:
    """
    Returns all expressions that have multiple parents (a list but semantically just an ordered set).
    """
    parent_counts: dict[TypedExprDecl, int] = {}
    traversal_order: list[TypedExprDecl] = []
    traversed: set[TypedExprDecl] = set()
    stack = [typed_expr]
    while stack:
        node = stack.pop()
        if node in traversed:
            continue
        traversed.add(node)
        if node is not typed_expr:
            traversal_order.append(node)
        match node.expr:
            case CallDecl(args=args) | PartialCallDecl(CallDecl(args=args)):
                for child in args:
                    parent_counts[child] = parent_counts.get(child, 0) + 1
                stack.extend(reversed(args))
            case _:
                pass
    return [node for node in traversal_order if parent_counts[node] > 1]


def _contains_unbound_var(typed_expr: TypedExprDecl) -> bool:
    """Check for an unbound variable without recursively hashing a deep expression DAG."""
    seen: set[int] = set()
    stack = [typed_expr]
    while stack:
        node = stack.pop()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        match node.expr:
            case UnboundVarDecl():
                return True
            case CallDecl(args=args) | PartialCallDecl(CallDecl(args=args)):
                stack.extend(args)
            case _:
                pass
    return False


@dataclass
class FromEggState:
    """
    Dataclass containing state used when converting from an egg term to a typed expr.
    """

    state: EGraphState
    termdag: bindings.TermDag
    # Cache of termdag ID and expected type to TypedExprDecl. Polymorphic
    # zero-argument terms like map-empty can appear once in a termdag but be
    # decoded at multiple concrete types.
    cache: dict[tuple[int, JustTypeRef], TypedExprDecl] = field(default_factory=dict)

    @property
    def decls(self) -> Declarations:
        return self.state.__egg_decls__

    def from_expr(self, tp: JustTypeRef, term: bindings._Term) -> TypedExprDecl:
        """
        Convert an egg term to a typed expr.
        """
        # Extracted builtin values can use canonical constructors that were not
        # present in the original Python expression, such as BigInt.from_string
        # inside an extracted BigRat. Seed the expected type's callable mapping
        # before resolving the term.
        self.state.type_ref_to_egg(tp)
        expr_decl: ExprDecl
        if isinstance(term, bindings.TermVar):
            expr_decl = LetRefDecl(term.name)
        elif isinstance(term, bindings.TermLit):
            value = term.value
            expr_decl = LitDecl(None if isinstance(value, bindings.Unit) else value.value)
        elif isinstance(term, bindings.TermApp):
            if term.name == "map-of" and tp.ident == Ident.builtin("Map"):
                if len(term.args) % 2:
                    raise ValueError(f"Expected alternating key/value terms in map-of, got {len(term.args)} terms")
                key_tp, value_tp = tp.args
                expr_decl = CallDecl(ClassMethodRef(Ident.builtin("Map"), "empty"), (), (key_tp, value_tp))
                for index in range(0, len(term.args), 2):
                    expr_decl = CallDecl(
                        MethodRef(Ident.builtin("Map"), "insert"),
                        (
                            TypedExprDecl(tp, expr_decl),
                            self.resolve_term(term.args[index], key_tp),
                            self.resolve_term(term.args[index + 1], value_tp),
                        ),
                    )
            elif term.name == "py-object":
                (str_term,) = term.args
                call = self.termdag.get(str_term)
                assert isinstance(call, bindings.TermLit)
                assert isinstance(call.value, bindings.String)
                expr_decl = PyObjectDecl(standard_b64decode(call.value.value))
            elif term.name == "unstable-fn":
                # Get function name
                fn_term, *arg_terms = term.args
                fn_value = self.resolve_term(fn_term, JustTypeRef(Ident.builtin("String")))
                assert isinstance(fn_value.expr, LitDecl)
                fn_name = fn_value.expr.value
                assert isinstance(fn_name, str)

                # Resolve what types the partially applied args are
                assert tp.ident == Ident.builtin("UnstableFn")
                call_decl = self.from_call(tp.args[0], bindings.TermApp(fn_name, arg_terms))
                expr_decl = PartialCallDecl(call_decl)
            else:
                expr_decl = self.from_call(tp, term)
        else:
            assert_never(term)
        return TypedExprDecl(tp, expr_decl)

    def from_call(self, tp: JustTypeRef, term: bindings.TermApp) -> CallDecl:
        """
        Convert a call to a CallDecl.

        There could be Python call refs which match the call, so we need to find the correct one.

        The additional_arg_tps are known types for arguments that come after the term args, used to infer types
        for partially applied functions, where we know the types of the later args, but not of the earlier ones where
        we have values for.
        """
        # Find the first callable ref that matches the call
        possible_callable_refs = self.state.egg_fn_to_callable_refs.get(term.name, ())
        for callable_ref in possible_callable_refs:
            # If this is a classmethod, we might need the type params that were bound for this type
            # This could be multiple types if the classmethod is ambiguous, like map create.
            possible_types: Iterable[JustTypeRef | None]
            signature = self.decls.get_callable_decl(callable_ref).signature
            assert isinstance(signature, FunctionSignature)
            term_args = term.args[::-1] if signature.reverse_args else term.args
            if isinstance(callable_ref, ClassMethodRef | InitRef | MethodRef):
                # Need OR in case we have class method whose class was never added as a sort, which would happen
                # if the class method didn't return that type and no other function did. In this case, we don't need
                # to care about the type vars and we don't need to bind any possible type.
                possible_types = self.state._get_possible_types(callable_ref.ident) or [None]
            else:
                possible_types = [None]
            for possible_type in possible_types:
                tcs = TypeConstraintSolver()
                if possible_type and possible_type.args:
                    tcs.bind_class(possible_type, self.decls)
                    bound_args = possible_type.args
                else:
                    bound_args = ()
                try:
                    arg_types = tcs.infer_arg_types(
                        signature.arg_types, signature.semantic_return_type, signature.var_arg_type, tp
                    )
                    # Include this in try because of iterable
                    a_tp = list(zip(term_args, arg_types, strict=False))
                except TypeConstraintError:
                    continue
                args = tuple(self.resolve_term(a, tp) for a, tp in a_tp)
                # Only save bound tp params if needed for inferring return type
                # this is true if the set of set of type vars in the return are not a subset of those in the args
                bound_tp_params = () if signature.semantic_return_type.vars.issubset(signature.arg_vars) else bound_args
                return CallDecl(callable_ref, args, bound_tp_params)
        raise ValueError(
            f"Could not find callable ref for call {term}. None of these refs matched the types: {possible_callable_refs}"
        )

    def resolve_term(self, term_id: int, tp: JustTypeRef) -> TypedExprDecl:
        key = (term_id, tp)
        try:
            return self.cache[key]
        except KeyError:
            res = self.cache[key] = self.from_expr(tp, self.termdag.get(term_id))
            return res
