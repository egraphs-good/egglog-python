"""
Implement conversion to/from egglog.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, assert_never, overload
from weakref import WeakKeyDictionary

from . import bindings
from .declarations import *
from .pretty import *
from .type_constraint_solver import TypeConstraintError, TypeConstraintSolver

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["EGraphState", "GLOBAL_PY_OBJECT_SORT"]

# Create a global sort for python objects, so we can store them without an e-graph instance
# Needed when serializing commands to egg commands when creating modules
GLOBAL_PY_OBJECT_SORT = bindings.PyObjectSort()


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
    callable_ref_to_egg_fn: dict[CallableRef, str] = field(default_factory=lambda: {FunctionRef("!="): "!="})

    # Bidirectional mapping between egg sort names and python type references.
    type_ref_to_egg_sort: dict[JustTypeRef, str] = field(default_factory=dict)

    # Cache of egg expressions for converting to egg
    expr_to_egg_cache: WeakKeyDictionary[ExprDecl, bindings._Expr] = field(default_factory=WeakKeyDictionary)

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
            expr_to_egg_cache=self.expr_to_egg_cache.copy(),
        )

    def schedule_to_egg(self, schedule: ScheduleDecl) -> bindings._Schedule:
        match schedule:
            case SaturateDecl(schedule):
                return bindings.Saturate(self.schedule_to_egg(schedule))
            case RepeatDecl(schedule, times):
                return bindings.Repeat(times, self.schedule_to_egg(schedule))
            case SequenceDecl(schedules):
                return bindings.Sequence([self.schedule_to_egg(s) for s in schedules])
            case RunDecl(ruleset_name, until):
                self.ruleset_to_egg(ruleset_name)
                config = bindings.RunConfig(ruleset_name, None if not until else list(map(self.fact_to_egg, until)))
                return bindings.Run(config)
            case _:
                assert_never(schedule)

    def ruleset_to_egg(self, name: str) -> None:
        """
        Registers a ruleset if it's not already registered.
        """
        if name not in self.rulesets:
            if name:
                self.egraph.run_program(bindings.AddRuleset(name))
            rules = self.rulesets[name] = set()
        else:
            rules = self.rulesets[name]
        for rule in self.__egg_decls__._rulesets[name].rules:
            if rule in rules:
                continue
            self.egraph.run_program(self.command_to_egg(rule, name))
            rules.add(rule)

    def command_to_egg(self, cmd: CommandDecl, ruleset: str) -> bindings._Command:
        match cmd:
            case ActionCommandDecl(action):
                return bindings.ActionCommand(self.action_to_egg(action))
            case RewriteDecl(tp, lhs, rhs, conditions) | BiRewriteDecl(tp, lhs, rhs, conditions):
                self.type_ref_to_egg(tp)
                rewrite = bindings.Rewrite(
                    self.expr_to_egg(lhs),
                    self.expr_to_egg(rhs),
                    [self.fact_to_egg(c) for c in conditions],
                )
                return (
                    bindings.RewriteCommand(ruleset, rewrite)
                    if isinstance(cmd, RewriteDecl)
                    else bindings.BiRewriteCommand(ruleset, rewrite)
                )
            case RuleDecl(head, body, name):
                rule = bindings.Rule(
                    [self.action_to_egg(a) for a in head],
                    [self.fact_to_egg(f) for f in body],
                )
                return bindings.RuleCommand(name or "", ruleset, rule)
            case _:
                assert_never(cmd)

    def action_to_egg(self, action: ActionDecl) -> bindings._Action:
        match action:
            case LetDecl(name, typed_expr):
                return bindings.Let(name, self.typed_expr_to_egg(typed_expr))
            case SetDecl(tp, call, rhs):
                self.type_ref_to_egg(tp)
                call_ = self.expr_to_egg(call)
                return bindings.Set(call_.name, call_.args, self.expr_to_egg(rhs))
            case ExprActionDecl(typed_expr):
                return bindings.Expr_(self.typed_expr_to_egg(typed_expr))
            case DeleteDecl(tp, call):
                self.type_ref_to_egg(tp)
                call_ = self.expr_to_egg(call)
                return bindings.Delete(call_.name, call_.args)
            case UnionDecl(tp, lhs, rhs):
                self.type_ref_to_egg(tp)
                return bindings.Union(self.expr_to_egg(lhs), self.expr_to_egg(rhs))
            case PanicDecl(name):
                return bindings.Panic(name)
            case _:
                assert_never(action)

    def fact_to_egg(self, fact: FactDecl) -> bindings._Fact:
        match fact:
            case EqDecl(tp, exprs):
                self.type_ref_to_egg(tp)
                return bindings.Eq([self.expr_to_egg(e) for e in exprs])
            case ExprFactDecl(typed_expr):
                return bindings.Fact(self.typed_expr_to_egg(typed_expr))
            case _:
                assert_never(fact)

    def callable_ref_to_egg(self, ref: CallableRef) -> str:
        """
        Returns the egg function name for a callable reference, registering it if it is not already registered.
        """
        if ref in self.callable_ref_to_egg_fn:
            return self.callable_ref_to_egg_fn[ref]
        decl = self.__egg_decls__.get_callable_decl(ref)
        self.callable_ref_to_egg_fn[ref] = egg_name = decl.egg_name or _generate_callable_egg_name(ref)
        self.egg_fn_to_callable_refs[egg_name].add(ref)
        match decl:
            case RelationDecl(arg_types, _, _):
                self.egraph.run_program(bindings.Relation(egg_name, [self.type_ref_to_egg(a) for a in arg_types]))
            case ConstantDecl(tp, _):
                # Use function decleration instead of constant b/c constants cannot be extracted
                # https://github.com/egraphs-good/egglog/issues/334
                self.egraph.run_program(
                    bindings.Function(bindings.FunctionDecl(egg_name, bindings.Schema([], self.type_ref_to_egg(tp))))
                )
            case FunctionDecl():
                if not decl.builtin:
                    egg_fn_decl = bindings.FunctionDecl(
                        egg_name,
                        bindings.Schema(
                            [self.type_ref_to_egg(a.to_just()) for a in decl.arg_types],
                            self.type_ref_to_egg(decl.semantic_return_type.to_just()),
                        ),
                        self.expr_to_egg(decl.default) if decl.default else None,
                        self.expr_to_egg(decl.merge) if decl.merge else None,
                        [self.action_to_egg(a) for a in decl.on_merge],
                        decl.cost,
                        decl.unextractable,
                    )
                    self.egraph.run_program(bindings.Function(egg_fn_decl))
            case _:
                assert_never(decl)
        return egg_name

    def type_ref_to_egg(self, ref: JustTypeRef) -> str:
        """
        Returns the egg sort name for a type reference, registering it if it is not already registered.
        """
        if ref in self.type_ref_to_egg_sort:
            return self.type_ref_to_egg_sort[ref]
        decl = self.__egg_decls__._classes[ref.name]
        self.type_ref_to_egg_sort[ref] = egg_name = decl.egg_name or _generate_type_egg_name(ref)
        if not decl.builtin or ref.args:
            self.egraph.run_program(
                bindings.Sort(
                    egg_name,
                    (
                        (
                            self.type_ref_to_egg(JustTypeRef(ref.name)),
                            [bindings.Var(self.type_ref_to_egg(a)) for a in ref.args],
                        )
                        if ref.args
                        else None
                    ),
                )
            )
        # For builtin classes, let's also make sure we have the mapping of all egg fn names for class methods, because
        # these can be created even without adding them to the e-graph, like `vec-empty` which can be extracted
        # even if you never use that function.
        if decl.builtin:
            for method in decl.class_methods:
                self.callable_ref_to_egg(ClassMethodRef(ref.name, method))

        return egg_name

    def op_mapping(self) -> dict[str, str]:
        """
        Create a mapping of egglog function name to Python function name, for use in the serialized format
        for better visualization.
        """
        return {
            k: pretty_callable_ref(self.__egg_decls__, next(iter(v)))
            for k, v in self.egg_fn_to_callable_refs.items()
            if len(v) == 1
        }

    def typed_expr_to_egg(self, typed_expr_decl: TypedExprDecl) -> bindings._Expr:
        self.type_ref_to_egg(typed_expr_decl.tp)
        return self.expr_to_egg(typed_expr_decl.expr)

    @overload
    def expr_to_egg(self, expr_decl: CallDecl) -> bindings.Call: ...

    @overload
    def expr_to_egg(self, expr_decl: ExprDecl) -> bindings._Expr: ...

    def expr_to_egg(self, expr_decl: ExprDecl) -> bindings._Expr:
        """
        Convert an ExprDecl to an egg expression.

        Cached using weakrefs to avoid memory leaks.
        """
        if expr_decl in self.expr_to_egg_cache:
            return self.expr_to_egg_cache[expr_decl]

        res: bindings._Expr
        match expr_decl:
            case VarDecl(name):
                res = bindings.Var(name)
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
                        l = bindings.F64(f)
                    case str(s):
                        l = bindings.String(s)
                    case _:
                        assert_never(value)
                res = bindings.Lit(l)
            case CallDecl(ref, args, _):
                res = bindings.Call(self.callable_ref_to_egg(ref), [self.typed_expr_to_egg(a) for a in args])
            case PyObjectDecl(value):
                res = GLOBAL_PY_OBJECT_SORT.store(value)
            case _:
                assert_never(expr_decl.expr)

        self.expr_to_egg_cache[expr_decl] = res
        return res

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


def _generate_type_egg_name(ref: JustTypeRef) -> str:
    """
    Generates an egg sort name for this type reference by linearizing the type.
    """
    name = ref.name
    if not ref.args:
        return name
    return f"{name}_{"_".join(map(_generate_type_egg_name, ref.args))}"


def _generate_callable_egg_name(ref: CallableRef) -> str:
    """
    Generates a valid egg function name for a callable reference.
    """
    match ref:
        case FunctionRef(name) | ConstantRef(name):
            return name
        case (
            MethodRef(cls_name, name)
            | ClassMethodRef(cls_name, name)
            | ClassVariableRef(cls_name, name)
            | PropertyRef(cls_name, name)
        ):
            return f"{cls_name}_{name}"
        case _:
            assert_never(ref)


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
            expr_decl = VarDecl(term.name)
        elif isinstance(term, bindings.TermLit):
            value = term.value
            expr_decl = LitDecl(None if isinstance(value, bindings.Unit) else value.value)
        elif isinstance(term, bindings.TermApp):
            if term.name == "py-object":
                call = bindings.termdag_term_to_expr(self.termdag, term)
                expr_decl = PyObjectDecl(self.state.egraph.eval_py_object(call))
            else:
                expr_decl = self.from_call(tp, term)
        else:
            assert_never(term)
        return TypedExprDecl(tp, expr_decl)

    def from_call(self, tp: JustTypeRef, term: bindings.TermApp) -> CallDecl:
        """
        Convert a call to a CallDecl.

        There could be Python call refs which match the call, so we need to find the correct one.
        """
        # Find the first callable ref that matches the call
        for callable_ref in self.state.egg_fn_to_callable_refs[term.name]:
            # If this is a classmethod, we might need the type params that were bound for this type
            # This could be multiple types if the classmethod is ambiguous, like map create.
            possible_types: Iterable[JustTypeRef | None]
            fn_decl = self.decls.get_callable_decl(callable_ref).to_function_decl()
            if isinstance(callable_ref, ClassMethodRef):
                possible_types = self.state._get_possible_types(callable_ref.class_name)
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
                        fn_decl.arg_types, fn_decl.semantic_return_type, fn_decl.var_arg_type, tp, cls_name
                    )
                except TypeConstraintError:
                    continue
                args: list[TypedExprDecl] = []
                for a, tp in zip(term.args, arg_types, strict=False):
                    try:
                        res = self.cache[a]
                    except KeyError:
                        res = self.cache[a] = self.from_expr(tp, self.termdag.nodes[a])
                    args.append(res)
                return CallDecl(callable_ref, tuple(args), bound_tp_params)
        raise ValueError(f"Could not find callable ref for call {term}")
