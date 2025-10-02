"""
Implement conversion to/from egglog.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, overload
from uuid import UUID

from typing_extensions import assert_never

from . import bindings
from .declarations import *
from .declarations import ConstructorDecl
from .pretty import *
from .type_constraint_solver import *

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["GLOBAL_PY_OBJECT_SORT", "EGraphState", "span"]

# Create a global sort for python objects, so we can store them without an e-graph instance
# Needed when serializing commands to egg commands when creating modules
GLOBAL_PY_OBJECT_SORT = bindings.PyObjectSort()


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
    State of the EGraph declerations and rulesets, so when we pop/push the stack we know whats defined.

    Used for converting to/from egg and for pretty printing.
    """

    egraph: bindings.EGraph
    # The decleratons we have added.
    __egg_decls__: Declarations = field(default_factory=Declarations)
    # Mapping of added rulesets to the added rules
    rulesets: dict[str, set[RewriteOrRuleDecl]] = field(default_factory=dict)

    # Bidirectional mapping between egg function names and python callable references.
    # Note that there are possibly mutliple callable references for a single egg function name, like `+`
    # for both int and rational classes.
    egg_fn_to_callable_refs: dict[str, set[CallableRef]] = field(
        default_factory=lambda: defaultdict(set, {"!=": {FunctionRef("!=")}})
    )
    callable_ref_to_egg_fn: dict[CallableRef, tuple[str, bool]] = field(
        default_factory=lambda: {FunctionRef("!="): ("!=", False)}
    )

    # Bidirectional mapping between egg sort names and python type references.
    type_ref_to_egg_sort: dict[JustTypeRef, str] = field(default_factory=dict)
    egg_sort_to_type_ref: dict[str, JustTypeRef] = field(default_factory=dict)

    # Cache of egg expressions for converting to egg
    expr_to_egg_cache: dict[ExprDecl, bindings._Expr] = field(default_factory=dict)

    # Callables which have cost tables associated with them
    cost_callables: set[CallableRef] = field(default_factory=set)

    def copy(self) -> EGraphState:
        """
        Returns a copy of the state. Th egraph reference is kept the same. Used for pushing/popping.
        """
        return EGraphState(
            egraph=self.egraph,
            __egg_decls__=self.__egg_decls__.copy(),
            rulesets={k: v.copy() for k, v in self.rulesets.items()},
            egg_fn_to_callable_refs=defaultdict(set, {k: v.copy() for k, v in self.egg_fn_to_callable_refs.items()}),
            callable_ref_to_egg_fn=self.callable_ref_to_egg_fn.copy(),
            type_ref_to_egg_sort=self.type_ref_to_egg_sort.copy(),
            egg_sort_to_type_ref=self.egg_sort_to_type_ref.copy(),
            expr_to_egg_cache=self.expr_to_egg_cache.copy(),
            cost_callables=self.cost_callables.copy(),
        )

    def run_schedule_to_egg(self, schedule: ScheduleDecl) -> bindings._Command:
        """
        Turn a run schedule into an egg command.

        If there exists any custom schedulers in the schedule, it will be turned into a custom extract command otherwise
        will be a normal run command.
        """
        processed_schedule = self._process_schedule(schedule)
        if processed_schedule is None:
            return bindings.RunSchedule(self._schedule_to_egg(schedule))
        top_level_schedules = self._schedule_with_scheduler_to_egg(processed_schedule, [])
        if len(top_level_schedules) == 1:
            schedule_expr = top_level_schedules[0]
        else:
            schedule_expr = bindings.Call(span(), "seq", top_level_schedules)
        return bindings.UserDefined(span(), "run-schedule", [schedule_expr])

    def _process_schedule(self, schedule: ScheduleDecl) -> ScheduleDecl | None:
        """
        Processes a schedule to determine if it contains any custom schedulers.

        If it does, it returns a new schedule with all the required let bindings added to the other scope.
        If not, returns none.

        Also processes all rulesets in the schedule to make sure they are registered.
        """
        bound_schedulers: list[UUID] = []
        unbound_schedulers: list[BackOffDecl] = []

        def helper(s: ScheduleDecl) -> None:
            match s:
                case LetSchedulerDecl(scheduler, inner):
                    bound_schedulers.append(scheduler.id)
                    return helper(inner)
                case RunDecl(ruleset_name, _, scheduler):
                    self.ruleset_to_egg(ruleset_name)
                    if scheduler and scheduler.id not in bound_schedulers:
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
        if not bound_schedulers and not unbound_schedulers:
            return None
        for scheduler in unbound_schedulers:
            schedule = LetSchedulerDecl(scheduler, schedule)
        return schedule

    def _schedule_to_egg(self, schedule: ScheduleDecl) -> bindings._Schedule:
        msg = "Should never reach this, let schedulers should be handled by custom scheduler"
        match schedule:
            case SaturateDecl(schedule):
                return bindings.Saturate(span(), self._schedule_to_egg(schedule))
            case RepeatDecl(schedule, times):
                return bindings.Repeat(span(), times, self._schedule_to_egg(schedule))
            case SequenceDecl(schedules):
                return bindings.Sequence(span(), [self._schedule_to_egg(s) for s in schedules])
            case RunDecl(ruleset_name, until, scheduler):
                if scheduler is not None:
                    raise ValueError(msg)
                config = bindings.RunConfig(ruleset_name, None if not until else list(map(self.fact_to_egg, until)))
                return bindings.Run(span(), config)
            case LetSchedulerDecl():
                raise ValueError(msg)
            case _:
                assert_never(schedule)

    def _schedule_with_scheduler_to_egg(  # noqa: C901, PLR0912
        self, schedule: ScheduleDecl, bound_schedulers: list[UUID]
    ) -> list[bindings._Expr]:
        """
        Turns a scheduler into an egg expression, to be used with a custom extract command.

        The bound_schedulers is a list of all the schedulers that have been bound. We can lookup their name as `_scheduler_{index}`.
        """
        match schedule:
            case LetSchedulerDecl(BackOffDecl(id, match_limit, ban_length), inner):
                name = f"_scheduler_{len(bound_schedulers)}"
                bound_schedulers.append(id)
                args: list[bindings._Expr] = []
                if match_limit is not None:
                    args.append(bindings.Var(span(), ":match-limit"))
                    args.append(bindings.Lit(span(), bindings.Int(match_limit)))
                if ban_length is not None:
                    args.append(bindings.Var(span(), ":ban-length"))
                    args.append(bindings.Lit(span(), bindings.Int(ban_length)))
                back_off_decl = bindings.Call(span(), "back-off", args)
                let_decl = bindings.Call(span(), "let-scheduler", [bindings.Var(span(), name), back_off_decl])
                return [let_decl, *self._schedule_with_scheduler_to_egg(inner, bound_schedulers)]
            case RunDecl(ruleset_name, until, scheduler):
                args = [bindings.Var(span(), ruleset_name)]
                if scheduler:
                    name = "run-with"
                    scheduler_name = f"_scheduler_{bound_schedulers.index(scheduler.id)}"
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

    def ruleset_to_egg(self, name: str) -> None:
        """
        Registers a ruleset if it's not already registered.
        """
        match self.__egg_decls__._rulesets[name]:
            case RulesetDecl(rules):
                if name not in self.rulesets:
                    if name:
                        self.egraph.run_program(bindings.AddRuleset(span(), name))
                    added_rules = self.rulesets[name] = set()
                else:
                    added_rules = self.rulesets[name]
                for rule in rules:
                    if rule in added_rules:
                        continue
                    cmd = self.command_to_egg(rule, name)
                    if cmd is not None:
                        self.egraph.run_program(cmd)
                    added_rules.add(rule)
            case CombinedRulesetDecl(rulesets):
                if name in self.rulesets:
                    return
                self.rulesets[name] = set()
                for ruleset in rulesets:
                    self.ruleset_to_egg(ruleset)
                self.egraph.run_program(bindings.UnstableCombinedRuleset(span(), name, list(rulesets)))

    def command_to_egg(self, cmd: CommandDecl, ruleset: str) -> bindings._Command | None:
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
                    [self.fact_to_egg(c) for c in conditions],
                )
                return (
                    bindings.RewriteCommand(ruleset, rewrite, cmd.subsume)
                    if isinstance(cmd, RewriteDecl)
                    else bindings.BiRewriteCommand(ruleset, rewrite)
                )
            case RuleDecl(head, body, name):
                rule = bindings.Rule(
                    span(),
                    [self.action_to_egg(a) for a in head],
                    [self.fact_to_egg(f) for f in body],
                )
                return bindings.RuleCommand(name or "", ruleset, rule)
            # TODO: Replace with just constants value and looking at REF of function
            case DefaultRewriteDecl(ref, expr, subsume):
                sig = self.__egg_decls__.get_callable_decl(ref).signature
                assert isinstance(sig, FunctionSignature)
                # Replace args with rule_var_name mapping
                arg_mapping = tuple(
                    TypedExprDecl(tp.to_just(), UnboundVarDecl(name))
                    for name, tp in zip(sig.arg_names, sig.arg_types, strict=False)
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
    def action_to_egg(self, action: ActionDecl, expr_to_let: Literal[True] = ...) -> bindings._Action | None: ...

    def action_to_egg(self, action: ActionDecl, expr_to_let: bool = False) -> bindings._Action | None:  # noqa: C901, PLR0911, PLR0912
        match action:
            case LetDecl(name, typed_expr):
                var_decl = LetRefDecl(name)
                var_egg = self._expr_to_egg(var_decl)
                self.expr_to_egg_cache[var_decl] = var_egg
                return bindings.Let(span(), var_egg.name, self.typed_expr_to_egg(typed_expr))
            case SetDecl(tp, call, rhs):
                self.type_ref_to_egg(tp)
                call_ = self._expr_to_egg(call)
                return bindings.Set(span(), call_.name, call_.args, self._expr_to_egg(rhs))
            case ExprActionDecl(typed_expr):
                if expr_to_let:
                    maybe_typed_expr = self._transform_let(typed_expr)
                    if maybe_typed_expr:
                        typed_expr = maybe_typed_expr
                    else:
                        return None
                return bindings.Expr_(span(), self.typed_expr_to_egg(typed_expr))
            case ChangeDecl(tp, call, change):
                self.type_ref_to_egg(tp)
                call_ = self._expr_to_egg(call)
                egg_change: bindings._Change
                match change:
                    case "delete":
                        egg_change = bindings.Delete()
                    case "subsume":
                        egg_change = bindings.Subsume()
                    case _:
                        assert_never(change)
                return bindings.Change(span(), egg_change, call_.name, call_.args)
            case UnionDecl(tp, lhs, rhs):
                self.type_ref_to_egg(tp)
                return bindings.Union(span(), self._expr_to_egg(lhs), self._expr_to_egg(rhs))
            case PanicDecl(name):
                return bindings.Panic(span(), name)
            case SetCostDecl(tp, expr, cost):
                self.type_ref_to_egg(tp)
                cost_table = self.create_cost_table(expr.callable)
                args_egg = [self.typed_expr_to_egg(x, False) for x in expr.args]
                return bindings.Set(span(), cost_table, args_egg, self._expr_to_egg(cost))
            case _:
                assert_never(action)

    def create_cost_table(self, ref: CallableRef) -> str:
        """
        Creates the egg cost table if needed and gets the name of the table.
        """
        name = self.cost_table_name(ref)
        print(name, self.cost_callables)
        if ref not in self.cost_callables:
            self.cost_callables.add(ref)
            signature = self.__egg_decls__.get_callable_decl(ref).signature
            assert isinstance(signature, FunctionSignature), "Can only add cost tables for functions"
            signature = replace(signature, return_type=TypeRefWithVars("i64"))
            self.egraph.run_program(
                bindings.FunctionCommand(span(), name, self._signature_to_egg_schema(signature), None)
            )
        return name

    def cost_table_name(self, ref: CallableRef) -> str:
        return f"cost_table_{self.callable_ref_to_egg(ref)[0]}"

    def fact_to_egg(self, fact: FactDecl) -> bindings._Fact:
        match fact:
            case EqDecl(tp, left, right):
                self.type_ref_to_egg(tp)
                return bindings.Eq(span(), self._expr_to_egg(left), self._expr_to_egg(right))
            case ExprFactDecl(typed_expr):
                return bindings.Fact(self.typed_expr_to_egg(typed_expr, False))
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
        egg_name = decl.egg_name or _sanitize_egg_ident(self._generate_callable_egg_name(ref))
        self.egg_fn_to_callable_refs[egg_name].add(ref)
        reverse_args = False
        match decl:
            case RelationDecl(arg_types, _, _):
                self.egraph.run_program(
                    bindings.Relation(span(), egg_name, [self.type_ref_to_egg(a) for a in arg_types])
                )
            case ConstantDecl(tp, _):
                # Use constructor decleration instead of constant b/c constants cannot be extracted
                # https://github.com/egraphs-good/egglog/issues/334
                is_function = self.__egg_decls__._classes[tp.name].builtin
                schema = bindings.Schema([], self.type_ref_to_egg(tp))
                if is_function:
                    self.egraph.run_program(bindings.FunctionCommand(span(), egg_name, schema, None))
                else:
                    self.egraph.run_program(bindings.Constructor(span(), egg_name, schema, None, False))
            case FunctionDecl(signature, builtin, _, merge):
                if isinstance(signature, FunctionSignature):
                    reverse_args = signature.reverse_args
                if not builtin:
                    assert isinstance(signature, FunctionSignature), "Cannot turn special function to egg"
                    # Compile functions that return unit to relations, because these show up in methods where you
                    # cant use the relation helper
                    schema = self._signature_to_egg_schema(signature)
                    if signature.return_type == TypeRefWithVars("Unit"):
                        if merge:
                            msg = "Cannot specify a merge function for a function that returns unit"
                            raise ValueError(msg)
                        self.egraph.run_program(bindings.Relation(span(), egg_name, schema.input))
                    else:
                        self.egraph.run_program(
                            bindings.FunctionCommand(
                                span(),
                                egg_name,
                                self._signature_to_egg_schema(signature),
                                self._expr_to_egg(merge) if merge else None,
                            )
                        )
            case ConstructorDecl(signature, _, cost, unextractable):
                self.egraph.run_program(
                    bindings.Constructor(
                        span(),
                        egg_name,
                        self._signature_to_egg_schema(signature),
                        cost,
                        unextractable,
                    )
                )

            case _:
                assert_never(decl)
        self.callable_ref_to_egg_fn[ref] = egg_name, reverse_args
        return egg_name, reverse_args

    def _signature_to_egg_schema(self, signature: FunctionSignature) -> bindings.Schema:
        return bindings.Schema(
            [self.type_ref_to_egg(a.to_just()) for a in signature.arg_types],
            self.type_ref_to_egg(signature.semantic_return_type.to_just()),
        )

    def type_ref_to_egg(self, ref: JustTypeRef) -> str:  # noqa: C901, PLR0912
        """
        Returns the egg sort name for a type reference, registering it if it is not already registered.
        """
        try:
            return self.type_ref_to_egg_sort[ref]
        except KeyError:
            pass
        decl = self.__egg_decls__._classes[ref.name]
        self.type_ref_to_egg_sort[ref] = egg_name = decl.egg_name or _generate_type_egg_name(ref)
        self.egg_sort_to_type_ref[egg_name] = ref
        if not decl.builtin or ref.args:
            if ref.args:
                if ref.name == "UnstableFn":
                    # UnstableFn is a special case, where the rest of args are collected into a call
                    if len(ref.args) < 2:
                        msg = "Zero argument higher order functions not supported"
                        raise NotImplementedError(msg)
                    type_args: list[bindings._Expr] = [
                        bindings.Call(
                            span(),
                            self.type_ref_to_egg(ref.args[1]),
                            [bindings.Var(span(), self.type_ref_to_egg(a)) for a in ref.args[2:]],
                        ),
                        bindings.Var(span(), self.type_ref_to_egg(ref.args[0])),
                    ]
                else:
                    # If any of methods have another type ref in them process all those first with substituted vars
                    # so that things like multiset - mapp will be added. Function type must be added first.
                    # Find all args of all methods and find any with type args themselves that are not this type and add them
                    tcs = TypeConstraintSolver(self.__egg_decls__)
                    tcs.bind_class(ref)
                    for method in decl.methods.values():
                        if not isinstance((signature := method.signature), FunctionSignature):
                            continue
                        for arg_tp in signature.arg_types:
                            if isinstance(arg_tp, TypeRefWithVars) and arg_tp.args and arg_tp.name != ref.name:
                                self.type_ref_to_egg(tcs.substitute_typevars(arg_tp, ref.name))

                    type_args = [bindings.Var(span(), self.type_ref_to_egg(a)) for a in ref.args]
                args = (self.type_ref_to_egg(JustTypeRef(ref.name)), type_args)
            else:
                args = None
            self.egraph.run_program(bindings.Sort(span(), egg_name, args))
        # For builtin classes, let's also make sure we have the mapping of all egg fn names for class methods, because
        # these can be created even without adding them to the e-graph, like `vec-empty` which can be extracted
        # even if you never use that function.
        if decl.builtin:
            for method_name in decl.class_methods:
                self.callable_ref_to_egg(ClassMethodRef(ref.name, method_name))
            if decl.init:
                self.callable_ref_to_egg(InitRef(ref.name))

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
            for c in self.egg_fn_to_callable_refs[name]:
                yield pretty_callable_ref(self.__egg_decls__, c)

    def typed_expr_to_egg(self, typed_expr_decl: TypedExprDecl, transform_let: bool = True) -> bindings._Expr:
        # transform all expressions with multiple parents into a let binding, so that less expressions
        # are sent to egglog. Only for performance reasons.
        if transform_let:
            have_multiple_parents = _exprs_multiple_parents(typed_expr_decl)
            for expr in reversed(have_multiple_parents):
                self._transform_let(expr)

        self.type_ref_to_egg(typed_expr_decl.tp)
        return self._expr_to_egg(typed_expr_decl.expr)

    def _transform_let(self, typed_expr: TypedExprDecl) -> TypedExprDecl | None:
        """
        Rewrites this expression as a let binding if it's not already a let binding.
        """
        # TODO: Replace with counter so that it works with hash collisions and is more stable
        var_decl = LetRefDecl(f"__expr_{hash(typed_expr)}")
        if var_decl in self.expr_to_egg_cache:
            return None
        var_egg = self._expr_to_egg(var_decl)
        cmd = bindings.ActionCommand(bindings.Let(span(), var_egg.name, self.typed_expr_to_egg(typed_expr)))
        try:
            self.egraph.run_program(cmd)
        # errors when creating let bindings for things like `(vec-empty)`
        except bindings.EggSmolError:
            return typed_expr
        self.expr_to_egg_cache[typed_expr.expr] = var_egg
        self.expr_to_egg_cache[var_decl] = var_egg
        return None

    @overload
    def _expr_to_egg(self, expr_decl: CallDecl) -> bindings.Call: ...

    @overload
    def _expr_to_egg(self, expr_decl: UnboundVarDecl | LetRefDecl) -> bindings.Var: ...

    @overload
    def _expr_to_egg(self, expr_decl: ExprDecl) -> bindings._Expr: ...

    def _expr_to_egg(self, expr_decl: ExprDecl) -> bindings._Expr:  # noqa: PLR0912,C901
        """
        Convert an ExprDecl to an egg expression.
        """
        try:
            return self.expr_to_egg_cache[expr_decl]
        except KeyError:
            pass
        res: bindings._Expr
        match expr_decl:
            case LetRefDecl(name):
                res = bindings.Var(span(), f"{name}")
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
                egg_args = [self.typed_expr_to_egg(a, False) for a in typed_args]
                res = bindings.Call(span(), egg_fn, egg_args)
            case PyObjectDecl(value):
                res = GLOBAL_PY_OBJECT_SORT.store(value)
            case PartialCallDecl(call_decl):
                egg_fn_call = self._expr_to_egg(call_decl)
                res = bindings.Call(
                    span(),
                    "unstable-fn",
                    [bindings.Lit(span(), bindings.String(egg_fn_call.name)), *egg_fn_call.args],
                )
            case ValueDecl():
                msg = "Cannot turn a Value into an expression"
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

    def exprs_from_egg(
        self, termdag: bindings.TermDag, terms: list[bindings._Term], tp: JustTypeRef
    ) -> Iterable[TypedExprDecl]:
        """
        Create a function that can convert from an egg term to a typed expr.
        """
        state = FromEggState(self, termdag)
        return [state.from_expr(tp, term) for term in terms]

    def _get_possible_types(self, cls_name: str) -> frozenset[JustTypeRef]:
        """
        Given a class name, returns all possible registered types that it can be.
        """
        return frozenset(tp for tp in self.type_ref_to_egg_sort if tp.name == cls_name)

    def _generate_callable_egg_name(self, ref: CallableRef) -> str:
        """
        Generates a valid egg function name for a callable reference.
        """
        match ref:
            case FunctionRef(name):
                return name

            case ConstantRef(name):
                # Prefix to avoid name collisions with local vars
                return f"%{name}"
            case (
                MethodRef(cls_name, name)
                | ClassMethodRef(cls_name, name)
                | ClassVariableRef(cls_name, name)
                | PropertyRef(cls_name, name)
            ):
                return f"{cls_name}.{name}"
            case InitRef(cls_name):
                return f"{cls_name}.__init__"
            case UnnamedFunctionRef(args, val):
                parts = [str(self._expr_to_egg(a.expr)) + "-" + str(self.type_ref_to_egg(a.tp)) for a in args] + [
                    str(self.typed_expr_to_egg(val, False))
                ]
                return "_".join(parts)
            case _:
                assert_never(ref)

    def typed_expr_to_value(self, typed_expr: TypedExprDecl) -> bindings.Value:
        egg_expr = self.typed_expr_to_egg(typed_expr, False)
        return self.egraph.eval_expr(egg_expr)[1]

    def value_to_expr(self, tp: JustTypeRef, value: bindings.Value) -> ExprDecl:  # noqa: C901, PLR0911, PLR0912
        match tp.name:
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
                return PyObjectDecl(self.egraph.value_to_pyobject(GLOBAL_PY_OBJECT_SORT, value))
            case "Rational":
                fraction = self.egraph.value_to_rational(value)
                return CallDecl(
                    InitRef("Rational"),
                    (
                        TypedExprDecl(JustTypeRef("i64"), LitDecl(fraction.numerator)),
                        TypedExprDecl(JustTypeRef("i64"), LitDecl(fraction.denominator)),
                    ),
                )
            case "BigInt":
                i = self.egraph.value_to_bigint(value)
                return CallDecl(
                    ClassMethodRef("BigInt", "from_string"),
                    (TypedExprDecl(JustTypeRef("String"), LitDecl(str(i))),),
                )
            case "BigRat":
                fraction = self.egraph.value_to_bigrat(value)
                return CallDecl(
                    InitRef("BigRat"),
                    (
                        TypedExprDecl(
                            JustTypeRef("BigInt"),
                            CallDecl(
                                ClassMethodRef("BigInt", "from_string"),
                                (TypedExprDecl(JustTypeRef("String"), LitDecl(str(fraction.numerator))),),
                            ),
                        ),
                        TypedExprDecl(
                            JustTypeRef("BigInt"),
                            CallDecl(
                                ClassMethodRef("BigInt", "from_string"),
                                (TypedExprDecl(JustTypeRef("String"), LitDecl(str(fraction.denominator))),),
                            ),
                        ),
                    ),
                )
            case "Map":
                k_tp, v_tp = tp.args
                expr = CallDecl(ClassMethodRef("Map", "empty"), (), (k_tp, v_tp))
                for k, v in self.egraph.value_to_map(value).items():
                    expr = CallDecl(
                        MethodRef("Map", "insert"),
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
                    InitRef("Set"), tuple(TypedExprDecl(v_tp, self.value_to_expr(v_tp, x)) for x in xs_), (v_tp,)
                )
            case "Vec":
                xs = self.egraph.value_to_vec(value)
                (v_tp,) = tp.args
                return CallDecl(
                    InitRef("Vec"), tuple(TypedExprDecl(v_tp, self.value_to_expr(v_tp, x)) for x in xs), (v_tp,)
                )
            case "MultiSet":
                xs = self.egraph.value_to_multiset(value)
                (v_tp,) = tp.args
                return CallDecl(
                    InitRef("MultiSet"), tuple(TypedExprDecl(v_tp, self.value_to_expr(v_tp, x)) for x in xs), (v_tp,)
                )
            case "UnstableFn":
                _names, _args = self.egraph.value_to_function(value)
                return_tp, *arg_types = tp.args
                return self._unstable_fn_value_to_expr(_names, _args, return_tp, arg_types)
        return ValueDecl(value)

    def _unstable_fn_value_to_expr(
        self, name: str, partial_args: list[bindings.Value], return_tp: JustTypeRef, _arg_types: list[JustTypeRef]
    ) -> PartialCallDecl:
        # Similar to FromEggState::from_call but accepts partial list of args and returns in values
        # Find first callable ref whose return type matches and fill in arg types.
        for callable_ref in self.egg_fn_to_callable_refs[name]:
            signature = self.__egg_decls__.get_callable_decl(callable_ref).signature
            if not isinstance(signature, FunctionSignature):
                continue
            if signature.semantic_return_type.name != return_tp.name:
                continue
            tcs = TypeConstraintSolver(self.__egg_decls__)

            arg_types, bound_tp_params = tcs.infer_arg_types(
                signature.arg_types, signature.semantic_return_type, signature.var_arg_type, return_tp, None
            )

            args = tuple(
                TypedExprDecl(tp, self.value_to_expr(tp, v)) for tp, v in zip(arg_types, partial_args, strict=False)
            )

            call_decl = CallDecl(
                callable_ref,
                args,
                # Don't include bound type params if this is just a method, we only needed them for type resolution
                # but dont need to store them
                bound_tp_params if isinstance(callable_ref, ClassMethodRef | InitRef) else (),
            )
            return PartialCallDecl(call_decl)
        raise ValueError(f"Function '{name}' not found")


# https://chatgpt.com/share/9ab899b4-4e17-4426-a3f2-79d67a5ec456
_EGGLOG_INVALID_IDENT = re.compile(r"[^\w\-+*/?!=<>&|^/%]")


def _sanitize_egg_ident(input_string: str) -> str:
    """
    Replaces all invalid characters in an egg identifier with an underscore.
    """
    return _EGGLOG_INVALID_IDENT.sub("_", input_string)


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


def _generate_type_egg_name(ref: JustTypeRef) -> str:
    """
    Generates an egg sort name for this type reference by linearizing the type.
    """
    name = ref.name
    if not ref.args:
        return name
    return f"{name}_{'_'.join(map(_generate_type_egg_name, ref.args))}"


@dataclass
class FromEggState:
    """
    Dataclass containing state used when converting from an egg term to a typed expr.
    """

    state: EGraphState
    termdag: bindings.TermDag
    # Cache of termdag ID to TypedExprDecl
    cache: dict[int, TypedExprDecl] = field(default_factory=dict)

    @property
    def decls(self) -> Declarations:
        return self.state.__egg_decls__

    def from_expr(self, tp: JustTypeRef, term: bindings._Term) -> TypedExprDecl:
        """
        Convert an egg term to a typed expr.
        """
        expr_decl: ExprDecl
        if isinstance(term, bindings.TermVar):
            expr_decl = LetRefDecl(term.name)
        elif isinstance(term, bindings.TermLit):
            value = term.value
            expr_decl = LitDecl(None if isinstance(value, bindings.Unit) else value.value)
        elif isinstance(term, bindings.TermApp):
            if term.name == "py-object":
                call = self.termdag.term_to_expr(term, span())
                expr_decl = PyObjectDecl(GLOBAL_PY_OBJECT_SORT.load(call))
            elif term.name == "unstable-fn":
                # Get function name
                fn_term, *arg_terms = term.args
                fn_value = self.resolve_term(fn_term, JustTypeRef("String"))
                assert isinstance(fn_value.expr, LitDecl)
                fn_name = fn_value.expr.value
                assert isinstance(fn_name, str)

                # Resolve what types the partially applied args are
                assert tp.name == "UnstableFn"
                call_decl = self.from_call(tp.args[0], bindings.TermApp(fn_name, arg_terms))
                expr_decl = PartialCallDecl(call_decl)
            else:
                expr_decl = self.from_call(tp, term)
        else:
            assert_never(term)
        return TypedExprDecl(tp, expr_decl)

    def from_call(
        self,
        tp: JustTypeRef,
        term: bindings.TermApp,  # additional_arg_tps: tuple[JustTypeRef, ...]
    ) -> CallDecl:
        """
        Convert a call to a CallDecl.

        There could be Python call refs which match the call, so we need to find the correct one.

        The additional_arg_tps are known types for arguments that come after the term args, used to infer types
        for partially applied functions, where we know the types of the later args, but not of the earlier ones where
        we have values for.
        """
        # Find the first callable ref that matches the call
        for callable_ref in self.state.egg_fn_to_callable_refs[term.name]:
            # If this is a classmethod, we might need the type params that were bound for this type
            # This could be multiple types if the classmethod is ambiguous, like map create.
            possible_types: Iterable[JustTypeRef | None]
            signature = self.decls.get_callable_decl(callable_ref).signature
            assert isinstance(signature, FunctionSignature)
            if isinstance(callable_ref, ClassMethodRef | InitRef | MethodRef):
                # Need OR in case we have class method whose class whas never added as a sort, which would happen
                # if the class method didn't return that type and no other function did. In this case, we don't need
                # to care about the type vars and we we don't need to bind any possible type.
                possible_types = self.state._get_possible_types(callable_ref.class_name) or [None]
                cls_name = callable_ref.class_name
            else:
                possible_types = [None]
                cls_name = None
            for possible_type in possible_types:
                tcs = TypeConstraintSolver(self.decls)
                if possible_type and possible_type.args:
                    tcs.bind_class(possible_type)
                try:
                    arg_types, bound_tp_params = tcs.infer_arg_types(
                        signature.arg_types, signature.semantic_return_type, signature.var_arg_type, tp, cls_name
                    )
                except TypeConstraintError:
                    continue
                args = tuple(self.resolve_term(a, tp) for a, tp in zip(term.args, arg_types, strict=False))

                return CallDecl(
                    callable_ref,
                    args,
                    # Don't include bound type params if this is just a method, we only needed them for type resolution
                    # but dont need to store them
                    bound_tp_params if isinstance(callable_ref, ClassMethodRef | InitRef) else (),
                )
        raise ValueError(
            f"Could not find callable ref for call {term}. None of these refs matched the types: {self.state.egg_fn_to_callable_refs[term.name]}"
        )

    def resolve_term(self, term_id: int, tp: JustTypeRef) -> TypedExprDecl:
        try:
            return self.cache[term_id]
        except KeyError:
            res = self.cache[term_id] = self.from_expr(tp, self.termdag.get(term_id))
            return res
