"""
Implement conversion to/from egglog.
"""

from __future__ import annotations

import re
import tempfile
from base64 import standard_b64decode, standard_b64encode
from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, TextIO, assert_never, overload
from uuid import UUID

import cloudpickle
from opentelemetry import trace

from . import bindings
from ._tracing import call_with_current_trace
from .declarations import *
from .declarations import ConstructorDecl, is_callable_decl_constructor
from .pretty import *
from .type_constraint_solver import *

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["EGraphState", "span"]


_TRACER = trace.get_tracer(__name__)


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
    inspection. Successful commands are saved normally; failed commands are
    saved as expected failures with trailing error comments.
    """

    path: str
    file: TextIO
    line_count: int = 0


def _normalize_global_let_name(name: str) -> str:
    return name if name.startswith("$") else f"${name}"


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
    save_egglog_string: InitVar[bool] = True
    egglog_file_state: _SavedEgglogFile | None = field(default=None, repr=False)
    # The declarations we have added.
    __egg_decls__: Declarations = field(default_factory=Declarations)
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
    # Cache of synthetic let references introduced for top-level command lowering.
    # This stays separate from `expr_to_egg_cache` so nested rule/rewrite lowering
    # can always rebuild structural surface syntax instead of leaking a previously
    # synthesized `$_n` binding across contexts.
    expr_to_letref_cache: dict[ExprDecl, bindings.Var] = field(default_factory=dict)

    # Callables which have cost tables associated with them
    cost_callables: set[CallableRef] = field(default_factory=set)
    # Counter for deterministic synthetic let bindings created while lowering expressions to egg.
    expr_to_let_counter: int = 0
    # Counter for deterministic synthetic names assigned to unnamed functions.
    unnamed_function_counter: int = 0

    def __post_init__(self, save_egglog_string: bool) -> None:
        if save_egglog_string and self.egglog_file_state is None:
            # Keep one persistent temp `.egg` file per high-level egraph so parse errors
            # can point at a stable filename the user can open after a failure.
            egglog_file = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8", suffix=".egg", delete=False)
            self.egglog_file_state = _SavedEgglogFile(egglog_file.name, egglog_file)

    def copy(self) -> EGraphState:
        """
        Returns a copy of the state. The egraph reference is kept the same. Used for pushing/popping.
        """
        return EGraphState(
            egraph=self.egraph,
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
            expr_to_letref_cache=self.expr_to_letref_cache.copy(),
            cost_callables=self.cost_callables.copy(),
            expr_to_let_counter=self.expr_to_let_counter,
            unnamed_function_counter=self.unnamed_function_counter,
        )

    def egglog_string(self) -> str:
        if self.egglog_file_state is None:
            msg = "Can't get egglog string unless EGraph created with save_egglog_string=True"
            raise ValueError(msg)
        # The append handle stays open for execution, so flush before reading the saved source.
        self.egglog_file_state.file.flush()
        with open(self.egglog_file_state.path, encoding="utf-8") as saved_file:
            return saved_file.read()

    def run_program(self, *commands: bindings._Command) -> list[bindings._CommandOutput]:
        if not commands:
            return []
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
                fail_command_text = str(bindings.Fail(span(), command)).rstrip("\n")
                saved_text = f"{fail_command_text} ; {_saved_egglog_failure_message(error)}\n"
                self.egglog_file_state.file.write(saved_text)
                self.egglog_file_state.file.flush()
                self.egglog_file_state.line_count += saved_text.count("\n")
                raise
            self.egglog_file_state.file.write(command_text)
            self.egglog_file_state.file.flush()
            self.egglog_file_state.line_count += command_text.count("\n")
            outputs.extend(command_outputs)
        return outputs

    @staticmethod
    def _persistent_scheduler_name(scheduler: BackOffDecl) -> str:
        return f"_persistent_scheduler_{scheduler.id.hex}"

    @staticmethod
    def _local_scheduler_name(index: int) -> str:
        return f"_scheduler_{index}"

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

    def _process_schedule(self, schedule: ScheduleDecl) -> tuple[ScheduleDecl | None, tuple[BackOffDecl, ...]]:
        """
        Processes a schedule to determine if it contains any custom schedulers.

        If it does, it returns a new schedule with all the required let bindings added to the other scope.
        If not, returns none.

        Also processes all rulesets in the schedule to make sure they are registered.
        """
        bound_schedulers: list[BackOffDecl] = []
        unbound_schedulers: list[BackOffDecl] = []
        persistent_schedulers: dict[UUID, BackOffDecl] = {}

        def helper(s: ScheduleDecl) -> None:
            match s:
                case LetSchedulerDecl(scheduler, inner):
                    bound_schedulers.append(scheduler)
                    return helper(inner)
                case RunDecl(ruleset_name, _, scheduler):
                    self.ruleset_to_egg(ruleset_name)
                    if scheduler and scheduler.id not in {s.id for s in bound_schedulers}:
                        if scheduler.persistent:
                            persistent_schedulers[scheduler.id] = scheduler
                        else:
                            unbound_schedulers.append(scheduler)
                case SaturateDecl(inner) | RepeatDecl(inner, _):
                    return helper(inner)
                case SequenceDecl(schedules):
                    for sc in schedules:
                        helper(sc)
                case _:
                    assert_never(s)
            return None

        helper(schedule)
        if not bound_schedulers and not unbound_schedulers and not persistent_schedulers:
            return None, ()
        for scheduler in unbound_schedulers:
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
                match_limit = scheduler.match_limit
                ban_length = scheduler.ban_length
                fresh_rematch = scheduler.fresh_rematch
                name = self._local_scheduler_name(len(bound_schedulers))
                bound_schedulers.append(scheduler)
                args: list[bindings._Expr] = []
                if match_limit is not None:
                    args.append(bindings.Var(span(), ":match-limit"))
                    args.append(bindings.Lit(span(), bindings.Int(match_limit)))
                if ban_length is not None:
                    args.append(bindings.Var(span(), ":ban-length"))
                    args.append(bindings.Lit(span(), bindings.Int(ban_length)))
                scheduler_name = "back-off-fresh" if fresh_rematch else "back-off"
                back_off_decl = bindings.Call(span(), scheduler_name, args)
                let_decl = bindings.Call(span(), "let-scheduler", [bindings.Var(span(), name), back_off_decl])
                return [let_decl, *self._schedule_with_scheduler_to_egg(inner, bound_schedulers)]
            case RunDecl(ruleset_ident, until, scheduler):
                args = [bindings.Var(span(), str(ruleset_ident))]
                if scheduler:
                    name = "run-with"
                    scheduler_name = self._persistent_scheduler_name(scheduler)
                    for i, bound in enumerate(bound_schedulers):
                        if bound.id == scheduler.id:
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
                args = self._schedule_with_scheduler_to_egg(inner, bound_schedulers)
                return [bindings.Call(span(), "saturate", args)]
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
        args: list[bindings._Expr] = []
        if scheduler.match_limit is not None:
            args.append(bindings.Var(span(), ":match-limit"))
            args.append(bindings.Lit(span(), bindings.Int(scheduler.match_limit)))
        if scheduler.ban_length is not None:
            args.append(bindings.Var(span(), ":ban-length"))
            args.append(bindings.Lit(span(), bindings.Int(scheduler.ban_length)))
        scheduler_name = "back-off-fresh" if scheduler.fresh_rematch else "back-off"
        back_off_decl = bindings.Call(span(), scheduler_name, args)
        return bindings.UserDefined(
            span(),
            "let-scheduler",
            [bindings.Var(span(), self._persistent_scheduler_name(scheduler)), back_off_decl],
        )

    def ruleset_to_egg(self, ident: Ident) -> None:
        """
        Registers a ruleset if it's not already registered.
        """
        if ident.name == "" and ident not in self.__egg_decls__._rulesets:
            self.rulesets.setdefault(ident, set())
            return
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
                    cmd = self.command_to_egg(rule, ident)
                    if cmd is not None:
                        self.run_program(cmd)
                    added_rules.add(rule)
            case CombinedRulesetDecl(rulesets):
                if ident in self.rulesets:
                    return
                self.rulesets[ident] = set()
                for ruleset in rulesets:
                    self.ruleset_to_egg(ruleset)
                self.run_program(bindings.UnstableCombinedRuleset(span(), str(ident), list(map(str, rulesets))))

    def command_to_egg(self, cmd: CommandDecl, ruleset: Ident) -> bindings._Command | None:
        match cmd:
            case ActionCommandDecl(action):
                action_egg = self.action_to_egg(action, expr_to_let=True)
                if not action_egg:
                    return None
                return bindings.ActionCommand(action_egg)
            case RewriteDecl(tp, lhs, rhs, conditions) | BiRewriteDecl(tp, lhs, rhs, conditions):
                self.type_ref_to_egg(tp)
                rewrite = bindings.Rewrite(
                    span(),
                    self._expr_to_egg(lhs),
                    self._expr_to_egg(rhs),
                    [self.fact_to_egg(c, expr_to_let=False) for c in conditions],
                )
                return (
                    bindings.RewriteCommand(str(ruleset), rewrite, cmd.subsume)
                    if isinstance(cmd, RewriteDecl)
                    else bindings.BiRewriteCommand(str(ruleset), rewrite)
                )
            case RuleDecl(head, body, name):
                return bindings.RuleCommand(
                    bindings.Rule(
                        span(),
                        [self.action_to_egg(a, expr_to_let=False) for a in head],
                        [self.fact_to_egg(f, expr_to_let=False) for f in body],
                        name or "",
                        str(ruleset),
                    )
                )
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
                return self.command_to_egg(rewrite_decl, ruleset)
            case _:
                assert_never(cmd)

    @overload
    def action_to_egg(self, action: ActionDecl) -> bindings._Action: ...

    @overload
    def action_to_egg(
        self,
        action: ActionDecl,
        expr_to_let: Literal[True] = ...,
    ) -> bindings._Action | None: ...

    def action_to_egg(  # noqa: C901, PLR0911
        self,
        action: ActionDecl,
        expr_to_let: bool = False,
    ) -> bindings._Action | None:
        match action:
            case LetDecl(name, typed_expr):
                var_decl = LetRefDecl(name)
                var_egg = self._expr_to_egg(var_decl)
                self.expr_to_egg_cache[var_decl] = var_egg
                return bindings.Let(
                    span(),
                    var_egg.name,
                    self.typed_expr_to_egg(typed_expr, expr_to_let=expr_to_let),
                )
            case SetDecl(tp, call, rhs):
                self.type_ref_to_egg(tp)
                egg_fn, typed_args = self.translate_call(call)
                return bindings.Set(
                    span(),
                    egg_fn,
                    [self.typed_expr_to_egg(arg, expr_to_let) for arg in typed_args],
                    self._expr_to_egg(rhs, expr_to_let=expr_to_let),
                )
            case ExprActionDecl(typed_expr):
                if not isinstance(typed_expr.expr, CallDecl):
                    msg = "Top-level egglog expr commands must be calls"
                    raise ValueError(msg)
                egg_expr = self.typed_expr_to_egg(typed_expr, expr_to_let=expr_to_let)
                if isinstance(egg_expr, bindings.Var):
                    return None
                assert isinstance(egg_expr, bindings.Call)
                return bindings.Expr_(span(), egg_expr)
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
                return bindings.Change(
                    span(),
                    egg_change,
                    egg_fn,
                    [self.typed_expr_to_egg(arg, expr_to_let) for arg in typed_args],
                )
            case UnionDecl(tp, lhs, rhs):
                self.type_ref_to_egg(tp)
                return bindings.Union(
                    span(),
                    self._expr_to_egg(lhs, expr_to_let=expr_to_let),
                    self._expr_to_egg(rhs, expr_to_let=expr_to_let),
                )
            case PanicDecl(name):
                return bindings.Panic(span(), name)
            case SetCostDecl(tp, expr, cost):
                self.type_ref_to_egg(tp)
                cost_table = self.create_cost_table(expr.callable)
                args_egg = [self.typed_expr_to_egg(x, expr_to_let) for x in expr.args]
                return bindings.Set(span(), cost_table, args_egg, self._expr_to_egg(cost, expr_to_let=expr_to_let))
            case _:
                assert_never(action)

    def create_cost_table(self, ref: CallableRef) -> str:
        """
        Creates the egg cost table if needed and gets the name of the table.
        """
        name = self.cost_table_name(ref)
        if ref not in self.cost_callables:
            self.cost_callables.add(ref)
            signature = self.__egg_decls__.get_callable_decl(ref).signature
            assert isinstance(signature, FunctionSignature), "Can only add cost tables for functions"
            signature = replace(signature, return_type=TypeRefWithVars(Ident.builtin("i64")))
            self.run_program(bindings.FunctionCommand(span(), name, self._signature_to_egg_schema(signature), None))
        return name

    def cost_table_name(self, ref: CallableRef) -> str:
        return f"cost_table_{self.callable_ref_to_egg(ref)[0]}"

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
        egg_name = decl.egg_name or self._allocate_callable_egg_name(ref)
        self.egg_fn_to_callable_refs.setdefault(egg_name, set()).add(ref)
        reverse_args = False
        match decl:
            case RelationDecl(arg_types, _, _):
                self.run_program(bindings.Relation(span(), egg_name, [self.type_ref_to_egg(a) for a in arg_types]))
            case ConstantDecl(tp, _, body, merge):
                if body is not None:
                    self.run_program(self._primitive_command_to_egg(egg_name, decl.signature, body))
                else:
                    # Use constructor declaration instead of constant b/c constants cannot be extracted
                    # https://github.com/egraphs-good/egglog/issues/334
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
                if isinstance(signature, FunctionSignature):
                    reverse_args = signature.reverse_args
                if not builtin:
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
        self.callable_ref_to_egg_fn[ref] = egg_name, reverse_args
        return egg_name, reverse_args

    def _primitive_command_to_egg(
        self,
        egg_name: str,
        signature: FunctionSignature,
        body: TypedExprDecl,
    ) -> bindings.UserDefined:
        input_sort_expr = self._primitive_input_sorts_to_egg(
            [self.type_ref_to_egg(arg_type.to_just()) for arg_type in signature.arg_types]
        )
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
        return bindings.Schema(
            [self.type_ref_to_egg(a.to_just()) for a in signature.arg_types],
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
        arg_names = [self.type_ref_to_egg(arg) for arg in ref.args]
        self.type_ref_to_egg_sort[ref] = egg_name = (not ref.args and decl.egg_name) or self._allocate_type_egg_name(
            ref, decl, arg_names
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
            self.cost_table_name(ref): f"cost({pretty_callable_ref(self.__egg_decls__, ref, include_all_args=True)})"
            for ref in self.cost_callables
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
        if not is_callable_decl_constructor(self.__egg_decls__.get_callable_decl(typed_expr.expr.callable)):
            return typed_expr
        if typed_expr.expr in self.expr_to_letref_cache:
            return None
        var_decl = LetRefDecl(self._allocate_synthetic_let_name())
        var_egg = self._expr_to_egg(var_decl)
        cmd = bindings.ActionCommand(bindings.Let(span(), var_egg.name, self.typed_expr_to_egg(typed_expr, False)))
        try:
            self.run_program(cmd)
        # errors when creating let bindings for things like `(vec-empty)`
        except bindings.EggSmolError:
            return typed_expr
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
        try:
            return self.expr_to_egg_cache[expr_decl]
        except KeyError:
            pass
        res: bindings._Expr
        match expr_decl:
            case LetRefDecl(name):
                res = bindings.Var(span(), _normalize_global_let_name(name))
            case UnboundVarDecl(name, egg_name):
                res = bindings.Var(span(), egg_name or f"_{name}")
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
                egg_fn_call = self._expr_to_egg(call_decl, expr_to_let=expr_to_let)
                res = bindings.Call(
                    span(),
                    "unstable-fn",
                    [bindings.Lit(span(), bindings.String(egg_fn_call.name)), *egg_fn_call.args],
                )
            case ValueDecl():
                msg = "Cannot turn a Value into an expression"
                raise ValueError(msg)
            case DummyDecl():
                msg = "Cannot turn a DummyDecl into an expression"
                raise ValueError(msg)
            case _:
                assert_never(expr_decl.expr)
        self.expr_to_egg_cache[expr_decl] = res
        return res

    def translate_call(self, expr: CallDecl | GetCostDecl) -> tuple[str, list[TypedExprDecl]]:
        """
        Handle get cost and call decl, turn into egg table name and typed expr decls.
        """
        match expr:
            case CallDecl(ref, args, _):
                egg_fn, reverse_args = self.callable_ref_to_egg(ref)
                args_list = list(args)
                if reverse_args:
                    args_list.reverse()
                return egg_fn, args_list
            case GetCostDecl(ref, args):
                cost_table = self.create_cost_table(ref)
                return cost_table, list(args)
            case _:
                assert_never(expr)

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

    def _allocate_callable_egg_name(self, ref: CallableRef) -> str:
        return self._allocate_name(
            self._generate_callable_egg_name_candidates(ref),
            lambda name: bool(self.egg_fn_to_callable_refs.get(name)) or name in BUILTIN_EGG_FN_NAMES,
        )

    def _generate_callable_egg_name_candidates(self, ref: CallableRef) -> tuple[str, ...]:
        """
        Generates short and fully-qualified egg function name candidates for a callable reference.
        """
        match ref:
            case FunctionRef(ident):
                return _name_candidates(ident.name, str(ident), sanitize=True)
            case ConstantRef(ident):
                # Prefix to avoid name collisions with local vars
                return _name_candidates(f"%{ident.name}", f"%{ident}", sanitize=True)
            case (
                MethodRef(cls_ident, name)
                | ClassMethodRef(cls_ident, name)
                | ClassVariableRef(cls_ident, name)
                | PropertyRef(cls_ident, name)
            ):
                return _name_candidates(f"{cls_ident.name}.{name}", f"{cls_ident}.{name}", sanitize=True)
            case InitRef(cls_ident):
                return _name_candidates(f"{cls_ident.name}.__init__", f"{cls_ident}.__init__", sanitize=True)
            case UnnamedFunctionRef():
                name = f"_lambda_{self.unnamed_function_counter}"
                self.unnamed_function_counter += 1
                return (name,)
            case _:
                assert_never(ref)

    def _allocate_type_egg_name(self, ref: JustTypeRef, decl: ClassDecl, arg_names: list[str]) -> str:
        return self._allocate_name(
            self._generate_type_egg_name_candidates(ref, decl, arg_names),
            lambda name: name in self.egg_sort_to_type_ref or name in BUILTIN_EGG_SORT_NAMES,
        )

    def _generate_type_egg_name_candidates(
        self, ref: JustTypeRef, decl: ClassDecl, arg_names: list[str]
    ) -> tuple[str, ...]:
        base_short = decl.egg_name or ref.ident.name
        base_full = decl.egg_name or str(ref.ident)
        if not ref.args:
            return _name_candidates(base_short, base_full, sanitize=False)
        args = ",".join(arg_names)
        return _name_candidates(f"{base_short}[{args}]", f"{base_full}[{args}]", sanitize=False)

    def _allocate_synthetic_let_name(self) -> str:
        while True:
            name = f"$__expr_{self.expr_to_let_counter}"
            self.expr_to_let_counter += 1
            if name not in self._known_let_names():
                return name

    @staticmethod
    def _allocate_name(candidates: Iterable[str], is_taken: Callable[[str], bool]) -> str:
        candidate_list = tuple(dict.fromkeys(candidates))
        for candidate in candidate_list:
            if not is_taken(candidate):
                return candidate

        fallback = candidate_list[-1]
        index = 1
        while is_taken(f"{fallback}_{index}"):
            index += 1
        return f"{fallback}_{index}"

    def _known_let_names(self) -> set[str]:
        return {
            egg_expr.name
            for decl, egg_expr in self.expr_to_egg_cache.items()
            if isinstance(decl, LetRefDecl) and isinstance(egg_expr, bindings.Var)
        }

    def synthetic_let_names(self) -> set[str]:
        return {var.name for var in self.expr_to_letref_cache.values()}

    def typed_expr_to_value(self, typed_expr: TypedExprDecl) -> bindings.Value:
        if isinstance(typed_expr.expr, ValueDecl):
            return typed_expr.expr.value
        egg_expr = self.typed_expr_to_egg(typed_expr, False)
        return call_with_current_trace(self.egraph.eval_expr, egg_expr)[1]

    def value_to_expr(self, tp: JustTypeRef, value: bindings.Value) -> ExprDecl:  # noqa: C901, PLR0911, PLR0912
        if tp.ident.module != Ident.builtin("").module:
            return ValueDecl(value)

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
            case _:
                # If this is not a builtin type, or we don't know how to convert it, just return as value
                return ValueDecl(value)

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


def _sanitize_egg_ident(input_string: str) -> str:
    """
    Replaces all invalid characters in an egg identifier with an underscore.
    """
    return _EGGLOG_INVALID_IDENT.sub("_", input_string)


def _name_candidates(short: str, full: str, *, sanitize: bool) -> tuple[str, ...]:
    if sanitize:
        short = _sanitize_egg_ident(short)
        full = _sanitize_egg_ident(full)
    return short, full


def _exprs_multiple_parents(typed_expr: TypedExprDecl) -> list[TypedExprDecl]:
    """
    Returns all expressions that have multiple parents (a list but semantically just an ordered set).
    """
    to_traverse = {typed_expr}
    traversed = set[TypedExprDecl]()
    traversed_twice = list[TypedExprDecl]()
    while to_traverse:
        typed_expr = to_traverse.pop()
        if typed_expr in traversed:
            traversed_twice.append(typed_expr)
            continue
        traversed.add(typed_expr)
        expr = typed_expr.expr
        if isinstance(expr, CallDecl):
            to_traverse.update(expr.args)
        elif isinstance(expr, PartialCallDecl):
            to_traverse.update(expr.call.args)
    return traversed_twice


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
            if term.name == "py-object":
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
                    a_tp = list(zip(term.args, arg_types, strict=False))
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
