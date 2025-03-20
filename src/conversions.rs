// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egg_smol types
use crate::utils::*;
use egglog::ast::Symbol;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use pyo3::types::{PyDelta, PyDeltaAccess};
use std::collections::HashMap;
use std::sync::Arc;

use crate::termdag::TermDag;

convert_enums!(
    egglog::ast::Literal: "{:}" Hash => Literal {
        Int[trait=Hash](value: i64)
            i -> egglog::ast::Literal::Int(i.value),
            egglog::ast::Literal::Int(i) => Int { value: *i };
        Float[trait=Hash](value: WrappedOrderedF64)
            f -> egglog::ast::Literal::Float(f.value.0),
            egglog::ast::Literal::Float(f) => Float { value: WrappedOrderedF64(*f) };
        String_[name="String"][trait=Hash](value: String)
            s -> egglog::ast::Literal::String((&s.value).into()),
            egglog::ast::Literal::String(s) => String_ { value: s.to_string() };
        Bool[trait=Hash](value: bool)
            b -> egglog::ast::Literal::Bool(b.value),
            egglog::ast::Literal::Bool(b) => Bool { value: *b };
        Unit[trait=Hash]()
            _x -> egglog::ast::Literal::Unit,
            egglog::ast::Literal::Unit => Unit {}
    };
    egglog::ast::Expr: "{}" => Expr {
        Lit(span: Span, value: Literal)
            l -> egglog::ast::Expr::Lit(l.span.clone().into(), (&l.value).into()),
            egglog::ast::Expr::Lit(span, l) => Lit { span: span.into(), value: l.into() };
        Var(span: Span, name: String)
            v -> egglog::ast::Expr::Var(v.span.clone().into(), (&v.name).into()),
            egglog::ast::Expr::Var(span, v) => Var { span: span.into(), name: v.to_string() };
        Call(span: Span, name: String, args: Vec<Expr>)
            c -> egglog::ast::Expr::Call(c.span.clone().into(), (&c.name).into(), c.args.iter().map(|e| e.into()).collect()),
            egglog::ast::Expr::Call(span, c, a) => Call {
                span: span.into(),
                name: c.to_string(),
                args: a.iter().map(|e| e.into()).collect()
            }
    };
    egglog::ast::Fact: "{}" => Fact_ {
        Eq(span: Span, left: Expr, right: Expr)
           eq -> egglog::ast::Fact::Eq(eq.span.clone().into(), eq.left.clone().into(), eq.right.clone().into()),
           egglog::ast::Fact::Eq(span, left, right) => Eq { span: span.into(), left: left.into(), right: right.into() };
        Fact(expr: Expr)
            f -> egglog::ast::Fact::Fact((&f.expr).into()),
            egglog::ast::Fact::Fact(e) => Fact { expr: e.into() }
    };
    egglog::ast::Change: "{:?}" => _Change {
        Delete()
            _d -> egglog::ast::Change::Delete,
            egglog::ast::Change::Delete => Delete {};
        Subsume()
            _d -> egglog::ast::Change::Subsume,
            egglog::ast::Change::Subsume => Subsume {}
    };
    egglog::ast::Action: "{}" => Action {
        Let(span: Span, lhs: String, rhs: Expr)
            d -> egglog::ast::Action::Let(d.span.clone().into(), (&d.lhs).into(), (&d.rhs).into()),
            egglog::ast::Action::Let(span, n, e) => Let { span: span.into(), lhs: n.to_string(), rhs: e.into() };
        Set(span: Span, lhs: String, args: Vec<Expr>, rhs: Expr)
            s -> egglog::ast::Action::Set(s.span.clone().into(), (&s.lhs).into(), s.args.iter().map(|e| e.into()).collect(), (&s.rhs).into()),
            egglog::ast::Action::Set(span, n, a, e) => Set {
                span: span.into(),
                lhs: n.to_string(),
                args: a.iter().map(|e| e.into()).collect(),
                rhs: e.into()
            };
        Change(span: Span, change: _Change, sym: String, args: Vec<Expr>)
            d -> egglog::ast::Action::Change(d.span.clone().into(), (&d.change).into(), (&d.sym).into(), d.args.iter().map(|e| e.into()).collect()),
            egglog::ast::Action::Change(span, c, n, a) => Change {
                span: span.into(),
                change: c.into(),
                sym: n.to_string(),
                args: a.iter().map(|e| e.into()).collect()
            };
        Union(span: Span, lhs: Expr, rhs: Expr)
            u -> egglog::ast::Action::Union(u.span.clone().into(), (&u.lhs).into(), (&u.rhs).into()),
            egglog::ast::Action::Union(span, l, r) => Union { span: span.into(), lhs: l.into(), rhs: r.into() };
        Panic(span: Span, msg: String)
            p -> egglog::ast::Action::Panic(p.span.clone().into(), p.msg.to_string()),
            egglog::ast::Action::Panic(span, msg) => Panic { span: span.into(), msg: msg.to_string()  };
        Expr_(span: Span, expr: Expr)
            e -> egglog::ast::Action::Expr(e.span.clone().into(), (&e.expr).into()),
            egglog::ast::Action::Expr(span, e) => Expr_ {span: span.into(), expr: e.into() };
        Extract(span: Span, expr: Expr, variants: Expr)
            e -> egglog::ast::Action::Extract(e.span.clone().into(), (&e.expr).into(), (&e.variants).into()),
            egglog::ast::Action::Extract(span, e, v) => Extract {
                span: span.into(),
                expr: e.into(),
                variants: v.into()
            }
    };
    egglog::ast::Schedule: "{}" => Schedule {
        Saturate(span: Span, schedule: Box<Schedule>)
            s -> egglog::ast::Schedule::Saturate(s.span.clone().into(), Box::new((&s.schedule).into())),
            egglog::ast::Schedule::Saturate(span, s) => Saturate { span: span.into(), schedule: Box::new((s).into()) };
        Repeat(span: Span, length: usize, schedule: Box<Schedule>)
            r -> egglog::ast::Schedule::Repeat(r.span.clone().into(), r.length, Box::new((&r.schedule).into())),
            egglog::ast::Schedule::Repeat(span, l, s) => Repeat { span: span.into(), length: *l, schedule: Box::new((s).into()) };
        Run(span: Span, config: RunConfig)
            r -> egglog::ast::Schedule::Run(r.span.clone().into(), (&r.config).into()),
            egglog::ast::Schedule::Run(span, c) => Run { span: span.into(), config: c.into() };
        Sequence(span: Span, schedules: Vec<Schedule>)
            s -> egglog::ast::Schedule::Sequence(s.span.clone().into(), s.schedules.iter().map(|s| s.into()).collect()),
            egglog::ast::Schedule::Sequence(span, s) => Sequence { span: span.into(), schedules: s.iter().map(|s| s.into()).collect() }
    };
    egglog::Term: "{:?}" Hash => Term {
        TermLit[trait=Hash](value: Literal)
            l -> egglog::Term::Lit((&l.value).into()),
            egglog::Term::Lit(l) => TermLit { value: l.into() };
        TermVar[trait=Hash](name: String)
            v -> egglog::Term::Var((&v.name).into()),
            egglog::Term::Var(v) => TermVar { name: v.to_string() };
        TermApp[trait=Hash](name: String, args: Vec<usize>)
            a -> egglog::Term::App(a.name.clone().into(), a.args.to_vec()),
            egglog::Term::App(s, a) => TermApp {
                name: s.to_string(),
                args: a.to_vec()
            }
    };
    egglog::ast::Command: "{}" => Command {
        SetOption(name: String, value: Expr)
            s -> egglog::ast::Command::SetOption{
                name: (&s.name).into(),
                value: (&s.value).into()
            },
            egglog::ast::Command::SetOption {name, value} => SetOption {
                name: name.to_string(),
                value: value.into()
            };
        Datatype(span: Span, name: String, variants: Vec<Variant>)
            d -> egglog::ast::Command::Datatype {
                span: d.span.clone().into(),
                name: (&d.name).into(),
                variants: d.variants.iter().map(|v| v.into()).collect()
            },
            egglog::ast::Command::Datatype {span, name, variants} => Datatype {
                span: span.into(),
                name: name.to_string(),
                variants: variants.iter().map(|v| v.into()).collect()
            };
        Sort(span: Span, name: String, presort_and_args: Option<(String, Vec<Expr>)>)
            s -> egglog::ast::Command::Sort(
                s.span.clone().into(),
                (&s.name).into(),
                s.presort_and_args.as_ref().map(|(p, a)| (p.into(), a.iter().map(|e| e.into()).collect()))
            ),
            egglog::ast::Command::Sort(span, n, presort_and_args) => Sort {
                name: n.to_string(),
                presort_and_args: presort_and_args.as_ref().map(|(p, a)| (p.to_string(), a.iter().map(|e| e.into()).collect())),
                span: span.into()
            };
        Function(span: Span, name: String, schema: Schema, merge: Option<Expr>)
            f -> egglog::ast::Command::Function{
                span: f.span.clone().into(),
                name: (&f.name).into(),
                schema: (&f.schema).into(),
                merge: f.merge.as_ref().map(|e| e.into())
            },
            egglog::ast::Command::Function {span, name, schema, merge} => Function {
                span: span.into(),
                name: name.to_string(),
                schema: schema.into(),
                merge: merge.as_ref().map(|e| e.into())
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
        RewriteCommand(name: String, rewrite: Rewrite, subsume: bool)
            r -> egglog::ast::Command::Rewrite((&r.name).into(), (&r.rewrite).into(), r.subsume),
            egglog::ast::Command::Rewrite(name, r, subsume) => RewriteCommand {
                name: name.to_string(),
                rewrite: r.into(),
                subsume: *subsume
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
        RunSchedule(schedule: Schedule)
            r -> egglog::ast::Command::RunSchedule((&r.schedule).into()),
            egglog::ast::Command::RunSchedule(s) => RunSchedule { schedule: s.into() };
        Simplify(span: Span, expr: Expr, schedule: Schedule)
            s -> egglog::ast::Command::Simplify {
                span: s.span.clone().into(),
                expr: (&s.expr).into(),
                schedule: (&s.schedule).into()
            },
            egglog::ast::Command::Simplify {span, expr, schedule} => Simplify {
                span: span.clone().into(),
                expr: expr.into(),
                schedule: schedule.into()
            };
        QueryExtract(span: Span, variants: usize, expr: Expr)
            e -> egglog::ast::Command::QueryExtract {
                span: e.span.clone().into(),
                variants: e.variants,
                expr: (&e.expr).into()
            },
            egglog::ast::Command::QueryExtract {span, variants, expr} => QueryExtract {
                span: span.into(),
                variants: *variants,
                expr: expr.into()
            };
        Check(span: Span, facts: Vec<Fact_>)
            c -> egglog::ast::Command::Check(c.span.clone().into(), c.facts.iter().map(|f| f.into()).collect()),
            egglog::ast::Command::Check(span, facts) => Check { span: span.into(), facts: facts.iter().map(|f| f.into()).collect() };
        PrintFunction(span: Span, name: String, length: usize)
            p -> egglog::ast::Command::PrintFunction(p.span.clone().into(), (&p.name).into(), p.length),
            egglog::ast::Command::PrintFunction(span, n, l) => PrintFunction {
                span: span.into(),
                name: n.to_string(),
                length: *l
            };
        PrintSize(span: Span, name: Option<String>)
            p -> egglog::ast::Command::PrintSize(p.span.clone().into(), p.name.as_ref().map(|n| n.into())),
            egglog::ast::Command::PrintSize(span, n) => PrintSize { span: span.into(), name: n.map(|n| n.to_string()) };
        Output(span: Span, file: String, exprs: Vec<Expr>)
            o -> egglog::ast::Command::Output {
                span: o.span.clone().into(),
                file: (&o.file).into(),
                exprs: o.exprs.iter().map(|e| e.into()).collect()
            },
            egglog::ast::Command::Output {span, file, exprs} => Output {
                span: span.into(),
                file: file.to_string(),
                exprs: exprs.iter().map(|e| e.into()).collect()
            };
        Input(span: Span, name: String, file: String)
            i -> egglog::ast::Command::Input {
                span: i.span.clone().into(),
                name: (&i.name).into(),
                file: (&i.file).into()
            },
            egglog::ast::Command::Input {span, name, file} => Input {
                span: span.into(),
                name: name.to_string(),
                file: file.to_string()
            };
        Push(length: usize)
            p -> egglog::ast::Command::Push(p.length),
            egglog::ast::Command::Push(l) => Push { length: *l };
        Pop(span: Span, length: usize)
            p -> egglog::ast::Command::Pop(p.span.clone().into(), p.length),
            egglog::ast::Command::Pop(span, l) => Pop { span: span.into(), length: *l };
        Fail(span: Span, command: Box<Command>)
            f -> egglog::ast::Command::Fail(f.span.clone().into(), Box::new((&f.command).into())),
            egglog::ast::Command::Fail(span, c) => Fail { span: span.into(), command: Box::new((c).into()) };
        Include(span: Span, path: String)
            i -> egglog::ast::Command::Include(i.span.clone().into(), (&i.path).into()),
            egglog::ast::Command::Include(span, p) => Include { span: span.into(), path: p.to_string() };
        Constructor(span: Span, name: String, schema: Schema, cost: Option<usize>, unextractable: bool)
            c -> egglog::ast::Command::Constructor {
                span: c.span.clone().into(),
                name: (&c.name).into(),
                schema: (&c.schema).into(),
                cost: c.cost,
                unextractable: c.unextractable
            },
            egglog::ast::Command::Constructor {span, name, schema, cost, unextractable} => Constructor {
                span: span.into(),
                name: name.to_string(),
                schema: schema.into(),
                cost: *cost,
                unextractable: *unextractable
            };
        Relation(span: Span, name: String, inputs: Vec<String>)
            r -> egglog::ast::Command::Relation {
                span: r.span.clone().into(),
                name: (&r.name).into(),
                inputs: r.inputs.iter().map(|i| i.into()).collect()
            },
            egglog::ast::Command::Relation {span, name, inputs} => Relation {
                span: span.into(),
                name: name.to_string(),
                inputs: inputs.iter().map(|i| i.to_string()).collect()
            };
        PrintOverallStatistics()
            _c -> egglog::ast::Command::PrintOverallStatistics,
            egglog::ast::Command::PrintOverallStatistics => PrintOverallStatistics {};
        Datatypes(span: Span, datatypes: Vec<(Span, String, Subdatatypes)>)
            d -> egglog::ast::Command::Datatypes {
                span: d.span.clone().into(),
                datatypes: d.datatypes.iter().map(|(s, n, d)| (s.clone().into(), n.into(), d.clone().into())).collect()
            },
            egglog::ast::Command::Datatypes {span, datatypes} => Datatypes {
                span: span.into(),
                datatypes: datatypes.iter().map(|(s, n, d)| (s.into(), n.to_string(), d.into())).collect()
            };
        UnstableCombinedRuleset(name: String, rulesets: Vec<String>)
            r -> egglog::ast::Command::UnstableCombinedRuleset(
                (&r.name).into(),
                r.rulesets.iter().map(|i| i.into()).collect()
            ),
            egglog::ast::Command::UnstableCombinedRuleset(name, rulesets) => UnstableCombinedRuleset {
                name: name.to_string(),
                rulesets: rulesets.iter().map(|i| i.to_string()).collect()
            }

    };
    egglog::ast::Subdatatypes: "{:?}" => Subdatatypes {
        SubVariants(variants: Vec<Variant>)
            v -> egglog::ast::Subdatatypes::Variants(v.variants.iter().map(|v| v.into()).collect()),
            egglog::ast::Subdatatypes::Variants(v) => SubVariants { variants: v.iter().map(|v| v.into()).collect() };
        NewSort(name: String, args: Vec<Expr>)
            n -> egglog::ast::Subdatatypes::NewSort((&n.name).into(), n.args.iter().map(|e| e.into()).collect()),
            egglog::ast::Subdatatypes::NewSort(name, args) => NewSort { name: name.to_string(), args: args.iter().map(|e| e.into()).collect() }
    };
    egglog::ExtractReport: "{:?}" => ExtractReport {
        Best(termdag: TermDag, cost: usize, term: Term)
            b -> egglog::ExtractReport::Best {
                termdag: (&b.termdag).into(),
                cost: b.cost,
                term: (&b.term).into()
            },
            egglog::ExtractReport::Best {termdag, cost, term} => Best {
                termdag: termdag.into(),
                cost: *cost,
                term: term.into()
            };
        Variants(termdag: TermDag, terms: Vec<Term>)
            v -> egglog::ExtractReport::Variants {
                termdag: (&v.termdag).into(),
                terms: v.terms.iter().map(|v| v.into()).collect()
            },
            egglog::ExtractReport::Variants {termdag, terms} => Variants {
                termdag: termdag.into(),
                terms: terms.iter().map(|v| v.into()).collect()
            }
    };
    egglog::ast::Span: "{:?}" => Span {
        PanicSpan()
            _p -> egglog::ast::Span::Panic,
            egglog::ast::Span::Panic => PanicSpan {};
        EgglogSpan(file: SrcFile, i: usize, j: usize)
            e -> egglog::ast::Span::Egglog(Arc::new({
                egglog::ast::EgglogSpan {
                    file: Arc::new(e.file.clone().into()),
                    i: e.i,
                    j: e.j
                }
            })),
            egglog::ast::Span::Egglog(e) => EgglogSpan {
                file: (*e.file.clone()).clone().into(),
                i: e.i,
                j: e.j
            };
        RustSpan(file: String, line: u32, column: u32)
            r -> egglog::ast::Span::Rust(Arc::new(egglog::ast::RustSpan {
                file: Box::leak(r.file.clone().into_boxed_str()),
                line: r.line,
                column: r.column
            })),
            egglog::ast::Span::Rust(r) => RustSpan {file: r.file.to_string(), line: r.line, column: r.column}
    }
);

convert_struct!(
    egglog::ast::SrcFile: "{:?}" => SrcFile(
        name: Option<String>,
        contents: String
    )
        s -> egglog::ast::SrcFile {name: s.name.clone(), contents: s.contents.clone()},
        s -> SrcFile {name: s.name.clone(), contents: s.contents.clone()};
    egglog::ast::Variant: "{:?}" => Variant(
        span: Span,
        name: String,
        types: Vec<String>,
        cost: Option<usize> = None
    )
        v -> egglog::ast::Variant {span: v.span.clone().into(), name: (&v.name).into(), types: v.types.iter().map(|v| v.into()).collect(), cost: v.cost},
        v -> Variant {span: v.span.clone().into(), name: v.name.to_string(), types: v.types.iter().map(|v| v.to_string()).collect(), cost: v.cost};
    egglog::ast::Schema: "{:?}" => Schema(
        input: Vec<String>,
        output: String
    )
        s -> egglog::ast::Schema {input: s.input.iter().map(|v| v.into()).collect(), output: (&s.output).into()},
        s -> Schema {input: s.input.iter().map(|v| v.to_string()).collect(), output: s.output.to_string()};
    egglog::ast::GenericRule<Symbol, Symbol>: "{:?}" => Rule(
        span: Span,
        head: Vec<Action>,
        body: Vec<Fact_>
    )
        r -> egglog::ast::GenericRule {span: r.span.clone().into(), head: egglog::ast::GenericActions(r.head.iter().map(|v| v.into()).collect()), body: r.body.iter().map(|v| v.into()).collect()},
        r -> Rule {span: r.span.clone().into(), head: r.head.0.iter().map(|v| v.into()).collect(), body: r.body.iter().map(|v| v.into()).collect()};
    egglog::ast::GenericRewrite<Symbol, Symbol>: "{:?}" => Rewrite(
        span: Span,
        lhs: Expr,
        rhs: Expr,
        conditions: Vec<Fact_> = Vec::new()
    )
        r -> egglog::ast::GenericRewrite {span: r.span.clone().into(), lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect()},
        r -> Rewrite {span: r.span.clone().into(), lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect()};
    egglog::ast::RunConfig: "{:?}" => RunConfig(
        ruleset: String,
        until: Option<Vec<Fact_>> = None
    )
        r -> egglog::ast::RunConfig {ruleset: (&r.ruleset).into(), until: r.until.as_ref().map(|v| v.iter().map(|v| v.into()).collect())},
        r -> RunConfig {ruleset: r.ruleset.to_string(), until: r.until.as_ref().map(|v| v.iter().map(|v| v.into()).collect())};
    egglog::ast::IdentSort: "{:?}" => IdentSort(
        ident: String,
        sort: String
    )
        i -> egglog::ast::IdentSort {ident: (&i.ident).into(), sort: (&i.sort).into()},
        i -> IdentSort {ident: i.ident.to_string(), sort: i.sort.to_string()};
    egglog::RunReport: "{:?}" => RunReport(
        updated: bool,
        search_time_per_rule: HashMap<String, WrappedDuration>,
        apply_time_per_rule: HashMap<String, WrappedDuration>,
        search_time_per_ruleset: HashMap<String, WrappedDuration>,
        apply_time_per_ruleset: HashMap<String, WrappedDuration>,
        rebuild_time_per_ruleset: HashMap<String, WrappedDuration>,
        num_matches_per_rule: HashMap<String, usize>
    )
        r -> egglog::RunReport {
            updated: r.updated,
            search_time_per_rule: r.search_time_per_rule.iter().map(|(k, v)| (k.clone().into(), v.clone().0)).collect(),
            apply_time_per_rule: r.apply_time_per_rule.iter().map(|(k, v)| (k.clone().into(), v.clone().0)).collect(),
            search_time_per_ruleset: r.search_time_per_ruleset.iter().map(|(k, v)| (k.clone().into(), v.clone().0)).collect(),
            apply_time_per_ruleset: r.apply_time_per_ruleset.iter().map(|(k, v)| (k.clone().into(), v.clone().0)).collect(),
            rebuild_time_per_ruleset: r.rebuild_time_per_ruleset.iter().map(|(k, v)| (k.clone().into(), v.clone().0)).collect(),
            num_matches_per_rule: r.num_matches_per_rule.iter().map(|(k, v)| (k.clone().into(), *v)).collect()
        },
        r -> RunReport {
            updated: r.updated,
            search_time_per_rule: r.search_time_per_rule.iter().map(|(k, v)| (k.clone().to_string(), (*v).into())).collect(),
            apply_time_per_rule: r.apply_time_per_rule.iter().map(|(k, v)| (k.clone().to_string(), (*v).into())).collect(),
            search_time_per_ruleset: r.search_time_per_ruleset.iter().map(|(k, v)| (k.clone().to_string(), (*v).into())).collect(),
            apply_time_per_ruleset: r.apply_time_per_ruleset.iter().map(|(k, v)| (k.clone().to_string(), (*v).into())).collect(),
            rebuild_time_per_ruleset: r.rebuild_time_per_ruleset.iter().map(|(k, v)| (k.clone().to_string(), (*v).into())).collect(),
            num_matches_per_rule: r.num_matches_per_rule.iter().map(|(k, v)| (k.clone().to_string(), *v)).collect()
        }
);

impl FromPyObject<'_> for Box<Schedule> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        ob.extract::<Schedule>().map(Box::new)
    }
}

impl<'py> IntoPyObject<'py> for Box<Schedule> {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok((*self).into_pyobject(py)?.as_any().clone())
    }
}

impl FromPyObject<'_> for Box<Command> {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        ob.extract::<Command>().map(Box::new)
    }
}

impl<'py> IntoPyObject<'py> for Box<Command> {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok((*self).into_pyobject(py)?.as_any().clone())
    }
}

// Wrapped version of ordered float
#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct WrappedOrderedF64(ordered_float::OrderedFloat<f64>);

impl From<ordered_float::OrderedFloat<f64>> for WrappedOrderedF64 {
    fn from(other: ordered_float::OrderedFloat<f64>) -> Self {
        WrappedOrderedF64(other)
    }
}

impl FromPyObject<'_> for WrappedOrderedF64 {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        ob.extract::<f64>()
            .map(|f| WrappedOrderedF64(OrderedFloat(f)))
    }
}

impl<'py> IntoPyObject<'py> for WrappedOrderedF64 {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok((self.0.into_inner()).into_pyobject(py)?.as_any().clone())
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
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py_delta = ob.downcast::<pyo3::types::PyDelta>()?;

        Ok(WrappedDuration(std::time::Duration::new(
            py_delta.get_days() as u64 * 24 * 60 * 60 + py_delta.get_seconds() as u64,
            py_delta.get_microseconds() as u32 * 1000,
        )))
    }
}
impl<'py> IntoPyObject<'py> for WrappedDuration {
    type Target = PyDelta; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let d = self.0;
        Ok(pyo3::types::PyDelta::new(
            py,
            0,
            0,
            d.as_millis()
                .try_into()
                .expect("Failed to convert miliseconds to int32 when converting duration"),
            true,
        )?
        .clone())
    }
}
