// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egglog types
use crate::utils::*;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use pyo3::types::PyDeltaAccess;
convert_enums!(
    egglog::ast::Literal: "{:}" => Literal {
        Int(value: i64)
            i -> egglog::ast::Literal::Int(i.value),
            egglog::ast::Literal::Int(i) => Int { value: i.clone() };
        F64(value: WrappedOrderedF64)
            f -> egglog::ast::Literal::F64(f.value.0),
            egglog::ast::Literal::F64(f) => F64 { value: WrappedOrderedF64(*f) };
        String_[name="String"](value: String)
            s -> egglog::ast::Literal::String((&s.value).into()),
            egglog::ast::Literal::String(s) => String_ { value: s.to_string() };
        Unit()
            _x -> egglog::ast::Literal::Unit,
            egglog::ast::Literal::Unit => Unit {}
    };
    egglog::ast::Expr: "{:}" => Expr {
        Lit(value: Literal)
            l -> egglog::ast::Expr::Lit((&l.value).into()),
            egglog::ast::Expr::Lit(l) => Lit { value: l.into() };
        Var(name: String)
            v -> egglog::ast::Expr::Var((&v.name).into()),
            egglog::ast::Expr::Var(v) => Var { name: v.to_string() };
        Call(name: String, args: Vec<Expr>)
            c -> egglog::ast::Expr::Call((&c.name).into(), (&c.args).into_iter().map(|e| e.into()).collect()),
            egglog::ast::Expr::Call(c, a) => Call {
                name: c.to_string(),
                args: a.into_iter().map(|e| e.into()).collect()
            }
    };
    egglog::ast::Fact: "{:}" => Fact_ {
        Eq(exprs: Vec<Expr>)
           eq -> egglog::ast::Fact::Eq((&eq.exprs).into_iter().map(|e| e.into()).collect()),
           egglog::ast::Fact::Eq(e) => Eq { exprs: e.into_iter().map(|e| e.into()).collect() };
        Fact(expr: Expr)
            f -> egglog::ast::Fact::Fact((&f.expr).into()),
            egglog::ast::Fact::Fact(e) => Fact { expr: e.into() }
    };
    egglog::ast::Action: "{:}" => Action {
        Let(lhs: String, rhs: Expr)
            d -> egglog::ast::Action::Let((&d.lhs).into(), (&d.rhs).into()),
            egglog::ast::Action::Let(n, e) => Let { lhs: n.to_string(), rhs: e.into() };
        Set(lhs: String, args: Vec<Expr>, rhs: Expr)
            s -> egglog::ast::Action::Set((&s.lhs).into(), (&s.args).into_iter().map(|e| e.into()).collect(), (&s.rhs).into()),
            egglog::ast::Action::Set(n, a, e) => Set {
                lhs: n.to_string(),
                args: a.into_iter().map(|e| e.into()).collect(),
                rhs: e.into()
            };
        Delete(sym: String, args: Vec<Expr>)
            d -> egglog::ast::Action::Delete((&d.sym).into(), (&d.args).into_iter().map(|e| e.into()).collect()),
            egglog::ast::Action::Delete(n, a) => Delete {
                sym: n.to_string(),
                args: a.into_iter().map(|e| e.into()).collect()
            };
        Union(lhs: Expr, rhs: Expr)
            u -> egglog::ast::Action::Union((&u.lhs).into(), (&u.rhs).into()),
            egglog::ast::Action::Union(l, r) => Union { lhs: l.into(), rhs: r.into() };
        Panic(msg: String)
            p -> egglog::ast::Action::Panic(p.msg.to_string()),
            egglog::ast::Action::Panic(msg) => Panic { msg: msg.to_string()  };
        Expr_(expr: Expr)
            e -> egglog::ast::Action::Expr((&e.expr).into()),
            egglog::ast::Action::Expr(e) => Expr_ { expr: e.into() }
    };
    egglog::ast::Schedule: "{:}" => Schedule {
        Saturate(schedule: Box<Schedule>)
            s -> (&s.schedule).into(),
            egglog::ast::Schedule::Saturate(s) => Saturate { schedule: Box::new((s).into()) };
        Repeat(length: usize, schedule: Box<Schedule>)
            r -> egglog::ast::Schedule::Repeat(r.length, Box::new((&r.schedule).into())),
            egglog::ast::Schedule::Repeat(l, s) => Repeat { length: *l, schedule: Box::new((s).into()) };
        Run(config: RunConfig)
            r -> egglog::ast::Schedule::Run((&r.config).into()),
            egglog::ast::Schedule::Run(c) => Run { config: c.into() };
        Sequence(schedules: Vec<Schedule>)
            s -> egglog::ast::Schedule::Sequence((&s.schedules).into_iter().map(|s| s.into()).collect()),
            egglog::ast::Schedule::Sequence(s) => Sequence { schedules: s.into_iter().map(|s| s.into()).collect() }
    };
    egglog::ast::Command: "{:?}" => Command {
        SetOption(name: String, value: Expr)
            s -> egglog::ast::Command::SetOption{
                name: (&s.name).into(),
                value: (&s.value).into()
            },
            egglog::ast::Command::SetOption {name, value} => SetOption {
                name: name.to_string(),
                value: value.into()
            };
        Datatype(name: String, variants: Vec<Variant>)
            d -> egglog::ast::Command::Datatype {
                name: (&d.name).into(),
                variants: (&d.variants).into_iter().map(|v| v.into()).collect()
            },
            egglog::ast::Command::Datatype {name, variants} => Datatype {
                name: name.to_string(),
                variants: variants.into_iter().map(|v| v.into()).collect()
            };
        Declare(name: String, sort: String)
            d -> egglog::ast::Command::Declare {
                name: (&d.name).into(),
                sort: (&d.sort).into()
            },
            egglog::ast::Command::Declare {name, sort} => Declare {
                name: name.to_string(),
                sort: sort.to_string()
            };
        Sort(name: String, presort_and_args: Option<(String, Vec<Expr>)>)
            s -> egglog::ast::Command::Sort(
                (&s.name).into(),
                (&s.presort_and_args).as_ref().map(|(p, a)| (p.into(), a.into_iter().map(|e| e.into()).collect()))
            ),
            egglog::ast::Command::Sort(n, presort_and_args) => Sort {
                name: n.to_string(),
                presort_and_args: presort_and_args.as_ref().map(|(p, a)| (p.to_string(), a.into_iter().map(|e| e.into()).collect()))
            };
        Function(decl: FunctionDecl)
            f -> egglog::ast::Command::Function((&f.decl).into()),
            egglog::ast::Command::Function(f) => Function { decl: f.into() };
        Define(name: String, expr: Expr, cost: Option<usize>)
            d -> egglog::ast::Command::Define {
                name: (&d.name).into(),
                expr: (&d.expr).into(),
                cost: d.cost
            },
            egglog::ast::Command::Define {name, expr, cost} => Define {
                name: name.to_string(),
                expr: expr.into(),
                cost: *cost
            };
        AddRuleset(name: String)
            a -> egglog::ast::Command::AddRuleset((&a.name).into()),
            egglog::ast::Command::AddRuleset(n) => AddRuleset { name: n.to_string() };
        RuleCommand(name: String, ruleset: String, rule: Rule)
            r -> egglog::ast::Command::Rule {
                name: (&r.name).into(),
                ruleset: (&r.ruleset).into(),
                rule: (&r.rule).into()
            },
            egglog::ast::Command::Rule {name, ruleset, rule} => RuleCommand {
                name: name.to_string(),
                ruleset: ruleset.to_string(),
                rule: rule.into()
            };
        RewriteCommand(name: String, rewrite: Rewrite)
            r -> egglog::ast::Command::Rewrite((&r.name).into(), (&r.rewrite).into()),
            egglog::ast::Command::Rewrite(name, r) => RewriteCommand {
                name: name.to_string(),
                rewrite: r.into()
            };
        BiRewriteCommand(name: String, rewrite: Rewrite)
            r -> egglog::ast::Command::BiRewrite((&r.name).into(), (&r.rewrite).into()),
            egglog::ast::Command::BiRewrite(name, r) => BiRewriteCommand {
                name: name.to_string(),
                rewrite: r.into()
            };
        ActionCommand(action: Action)
            a -> egglog::ast::Command::Action((&a.action).into()),
            egglog::ast::Command::Action(a) => ActionCommand { action: a.into() };
        RunCommand(config: RunConfig)
            r -> egglog::ast::Command::Run((&r.config).into()),
            egglog::ast::Command::Run(config) => RunCommand { config: config.into() };
        RunScheduleCommand(schedule: Schedule)
            r -> egglog::ast::Command::RunSchedule((&r.schedule).into()),
            egglog::ast::Command::RunSchedule(s) => RunScheduleCommand { schedule: s.into() };
        Simplify(expr: Expr, config: RunConfig)
            s -> egglog::ast::Command::Simplify {
                expr: (&s.expr).into(),
                config: (&s.config).into()
            },
            egglog::ast::Command::Simplify {expr, config} => Simplify {
                expr: expr.into(),
                config: config.into()
            };
        Calc(identifiers: Vec<IdentSort>, exprs: Vec<Expr>)
            c -> egglog::ast::Command::Calc(
                (&c.identifiers).into_iter().map(|i| i.into()).collect(),
                (&c.exprs).into_iter().map(|e| e.into()).collect()
            ),
            egglog::ast::Command::Calc(identifiers, exprs) => Calc {
                identifiers: identifiers.into_iter().map(|i| i.into()).collect(),
                exprs: exprs.into_iter().map(|e| e.into()).collect()
            };
        Extract(variants: usize, expr: Expr)
            e -> egglog::ast::Command::Extract {
                variants: e.variants,
                e: (&e.expr).into()
            },
            egglog::ast::Command::Extract {variants, e} => Extract {
                variants: *variants,
                expr: e.into()
            };
        Check(facts: Vec<Fact_>)
            c -> egglog::ast::Command::Check((&c.facts).into_iter().map(|f| f.into()).collect()),
            egglog::ast::Command::Check(facts) => Check { facts: facts.into_iter().map(|f| f.into()).collect() };
        Print(name: String, length: usize)
            p -> egglog::ast::Command::Print((&p.name).into(), p.length),
            egglog::ast::Command::Print(n, l) => Print {
                name: n.to_string(),
                length: *l
            };
        PrintSize(name: String)
            p -> egglog::ast::Command::PrintSize((&p.name).into()),
            egglog::ast::Command::PrintSize(n) => PrintSize { name: n.to_string() };
        Output(file: String, exprs: Vec<Expr>)
            o -> egglog::ast::Command::Output {
                file: (&o.file).into(),
                exprs: (&o.exprs).into_iter().map(|e| e.into()).collect()
            },
            egglog::ast::Command::Output {file, exprs} => Output {
                file: file.to_string(),
                exprs: exprs.into_iter().map(|e| e.into()).collect()
            };
        Input(name: String, file: String)
            i -> egglog::ast::Command::Input {
                name: (&i.name).into(),
                file: (&i.file).into()
            },
            egglog::ast::Command::Input {name, file} => Input {
                name: name.to_string(),
                file: file.to_string()
            };
        Push(length: usize)
            p -> egglog::ast::Command::Push(p.length),
            egglog::ast::Command::Push(l) => Push { length: *l };
        Pop(length: usize)
            p -> egglog::ast::Command::Pop(p.length),
            egglog::ast::Command::Pop(l) => Pop { length: *l };
        Fail(command: Box<Command>)
            f -> egglog::ast::Command::Fail(Box::new((&f.command).into())),
            egglog::ast::Command::Fail(c) => Fail { command: Box::new((c).into()) };
        Include(path: String)
            i -> egglog::ast::Command::Include((&i.path).into()),
            egglog::ast::Command::Include(p) => Include { path: p.to_string() }

    }
);

convert_struct!(
    egglog::ast::FunctionDecl: "{:?}" => FunctionDecl(
        name: String,
        schema: Schema,
        default: Option<Expr> = None,
        merge: Option<Expr> = None,
        merge_action: Vec<Action> = Vec::new(),
        cost: Option<usize> = None
    )
        f -> egglog::ast::FunctionDecl {
            name: (&f.name).into(),
            schema: (&f.schema).into(),
            default: f.default.as_ref().map(|e| e.into()),
            merge: f.merge.as_ref().map(|e| e.into()),
            merge_action: f.merge_action.iter().map(|a| a.into()).collect(),
            cost: f.cost
        },
        f -> FunctionDecl {
            name: f.name.to_string(),
            schema: (&f.schema).into(),
            default: f.default.as_ref().map(|e| e.into()),
            merge: f.merge.as_ref().map(|e| e.into()),
            merge_action: f.merge_action.iter().map(|a| a.into()).collect(),
            cost: f.cost
        };
    egglog::ast::Variant: "{:?}" => Variant(
        name: String,
        types: Vec<String>,
        cost: Option<usize> = None
    )
        v -> egglog::ast::Variant {name: (&v.name).into(), types: (&v.types).into_iter().map(|v| v.into()).collect(), cost: v.cost},
        v -> Variant {name: v.name.to_string(), types: v.types.iter().map(|v| v.to_string()).collect(), cost: v.cost};
    egglog::ast::Schema: "{:?}" => Schema(
        input: Vec<String>,
        output: String
    )
        s -> egglog::ast::Schema {input: (&s.input).into_iter().map(|v| v.into()).collect(), output: (&s.output).into()},
        s -> Schema {input: s.input.iter().map(|v| v.to_string()).collect(), output: s.output.to_string()};
    egglog::ast::Rule: "{}" => Rule(
        head: Vec<Action>,
        body: Vec<Fact_>
    )
        r -> egglog::ast::Rule {head: (&r.head).into_iter().map(|v| v.into()).collect(), body: (&r.body).into_iter().map(|v| v.into()).collect()},
        r -> Rule {head: r.head.iter().map(|v| v.into()).collect(), body: r.body.iter().map(|v| v.into()).collect()};
    egglog::ast::Rewrite: "{:?}" => Rewrite(
        lhs: Expr,
        rhs: Expr,
        conditions: Vec<Fact_> = Vec::new()
    )
        r -> egglog::ast::Rewrite {lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: (&r.conditions).into_iter().map(|v| v.into()).collect()},
        r -> Rewrite {lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect()};
    egglog::ast::RunConfig: "{:?}" => RunConfig(
        ruleset: String,
        limit: usize,
        until: Option<Vec<Fact_>>
    )
        r -> egglog::ast::RunConfig {ruleset: (&r.ruleset).into(), limit: r.limit, until: r.until.as_ref().map(|v| v.into_iter().map(|v| v.into()).collect())},
        r -> RunConfig {ruleset: r.ruleset.to_string(), limit: r.limit, until: r.until.as_ref().map(|v| v.into_iter().map(|v| v.into()).collect())};
    egglog::ast::IdentSort: "{:?}" => IdentSort(
        ident: String,
        sort: String
    )
        i -> egglog::ast::IdentSort {ident: (&i.ident).into(), sort: (&i.sort).into()},
        i -> IdentSort {ident: i.ident.to_string(), sort: i.sort.to_string()};
    egglog::RunReport: "{:?}" => RunReport(
        updated: bool,
        search_time: WrappedDuration,
        apply_time: WrappedDuration,
        rebuild_time: WrappedDuration
    )
        r -> egglog::RunReport {updated: r.updated, search_time: r.search_time.0, apply_time: r.apply_time.0, rebuild_time: r.rebuild_time.0},
        r -> RunReport {updated: r.updated, search_time: r.search_time.into(), apply_time: r.apply_time.into(), rebuild_time: r.rebuild_time.into()};
    egglog::ExtractReport: "{:?}" => ExtractReport(
        cost: usize,
        expr: Expr,
        variants: Vec<Expr>
    )
        r -> egglog::ExtractReport {cost: r.cost, expr: (&r.expr).into(), variants: r.variants.iter().map(|v| v.into()).collect()},
        r -> ExtractReport {cost: r.cost, expr: (&r.expr).into(), variants: r.variants.iter().map(|v| v.into()).collect()}
);

impl FromPyObject<'_> for Box<Schedule> {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        ob.extract::<Schedule>().map(|f| Box::new(f))
    }
}

impl IntoPy<PyObject> for Box<Schedule> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (*self).into_py(py)
    }
}

impl FromPyObject<'_> for Box<Command> {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        ob.extract::<Command>().map(|f| Box::new(f))
    }
}

impl IntoPy<PyObject> for Box<Command> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        (*self).into_py(py)
    }
}

// Wrapped version of ordered float
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WrappedOrderedF64(ordered_float::OrderedFloat<f64>);

impl From<ordered_float::OrderedFloat<f64>> for WrappedOrderedF64 {
    fn from(other: ordered_float::OrderedFloat<f64>) -> Self {
        WrappedOrderedF64(other)
    }
}

impl FromPyObject<'_> for WrappedOrderedF64 {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        ob.extract::<f64>()
            .map(|f| WrappedOrderedF64(OrderedFloat(f)))
    }
}

impl IntoPy<PyObject> for WrappedOrderedF64 {
    fn into_py(self, py: Python<'_>) -> PyObject {
        self.0.into_inner().into_py(py)
    }
}

// Wrapped version of Duration
// Converts from a rust duration to a python timedelta
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct WrappedDuration(std::time::Duration);

impl From<std::time::Duration> for WrappedDuration {
    fn from(other: std::time::Duration) -> Self {
        WrappedDuration(other)
    }
}

impl FromPyObject<'_> for WrappedDuration {
    fn extract(ob: &'_ PyAny) -> PyResult<Self> {
        let py_delta = <pyo3::types::PyDelta as PyTryFrom>::try_from(ob)?;

        Ok(WrappedDuration(std::time::Duration::new(
            py_delta.get_days() as u64 * 24 * 60 * 60 + py_delta.get_seconds() as u64,
            py_delta.get_microseconds() as u32 * 1000,
        ))
        .into())
    }
}

impl IntoPy<PyObject> for WrappedDuration {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let d = self.0;
        pyo3::types::PyDelta::new(
            py,
            0,
            0,
            d.as_millis()
                .try_into()
                .expect("Failed to convert miliseconds to int32 when converting duration"),
            true,
        )
        .expect("Failed to contruct timedelta")
        .into()
    }
}
