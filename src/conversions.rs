// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egg_smol types
use crate::utils::*;
use egglog::extract::DefaultCost;
use ordered_float::OrderedFloat;
use pyo3::exceptions::{PyOverflowError, PyValueError};
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
            egglog::ast::Action::Expr(span, e) => Expr_ {span: span.into(), expr: e.into() }
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
            a -> egglog::Term::App(a.name.clone(), a.args.to_vec()),
            egglog::Term::App(s, a) => TermApp {
                name: s.to_string(),
                args: a.to_vec()
            }
    };
    egglog::ast::PrintFunctionMode: "{}" => PrintFunctionMode {
        DefaultPrintFunctionMode()
            _d -> egglog::ast::PrintFunctionMode::Default,
            egglog::ast::PrintFunctionMode::Default => DefaultPrintFunctionMode {};
        CSVPrintFunctionMode()
            _d -> egglog::ast::PrintFunctionMode::CSV,
            egglog::ast::PrintFunctionMode::CSV => CSVPrintFunctionMode {}
    };
    egglog::ast::RuleEvalMode: "{:?}" => RuleEvalMode {
        Seminaive()
            _s -> egglog::ast::RuleEvalMode::Seminaive,
            egglog::ast::RuleEvalMode::Seminaive => Seminaive {};
        Naive()
            _n -> egglog::ast::RuleEvalMode::Naive,
            egglog::ast::RuleEvalMode::Naive => Naive {};
        UnsafeSeminaive()
            _u -> egglog::ast::RuleEvalMode::UnsafeSeminaive,
            egglog::ast::RuleEvalMode::UnsafeSeminaive => UnsafeSeminaive {}
    };
    egglog::ast::Command: "{}" => Command {
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
        Sort(
            span: Span,
            name: String,
            presort_and_args: Option<(String, Vec<Expr>)>,
            uf: Option<(String, Option<String>)> = None,
            proof_func: Option<String> = None,
            container_rebuild: Option<ContainerRebuildSpec> = None,
            proof_constructors: Option<ProofConstructorNames> = None
        )
            s -> egglog::ast::Command::Sort {
                span: s.span.clone().into(),
                name: (&s.name).into(),
                presort_and_args: s.presort_and_args.as_ref().map(|(p, a)| (p.into(), a.iter().map(|e| e.into()).collect())),
                uf: s.uf.clone(),
                proof_func: s.proof_func.clone(),
                container_rebuild: s.container_rebuild.as_ref().map(Into::into),
                proof_constructors: s.proof_constructors.as_ref().map(Into::into),
                unionable: true
            },
            egglog::ast::Command::Sort {
                span,
                name,
                presort_and_args,
                uf,
                proof_func,
                container_rebuild,
                proof_constructors,
                unionable: _
            } => Sort {
                name: name.to_string(),
                presort_and_args: presort_and_args.as_ref().map(|(p, a)| (p.to_string(), a.iter().map(|e| e.into()).collect())),
                span: span.into(),
                uf: uf.clone(),
                proof_func: proof_func.clone(),
                container_rebuild: container_rebuild.as_ref().map(Into::into),
                proof_constructors: proof_constructors.as_ref().map(Into::into)
            };
        FunctionCommand(
            span: Span,
            name: String,
            schema: Schema,
            merge: Option<Expr>,
            term_constructor: Option<String> = None,
            unextractable: bool = false,
            hidden: bool = false,
            let_binding: bool = false
        )
            f -> egglog::ast::Command::Function{
                span: f.span.clone().into(),
                name: (&f.name).into(),
                schema: (&f.schema).into(),
                merge: f.merge.as_ref().map(|e| e.into()),
                hidden: f.hidden,
                let_binding: f.let_binding,
                term_constructor: f.term_constructor.clone(),
                unextractable: f.unextractable
            },
            egglog::ast::Command::Function {
                span,
                name,
                schema,
                merge,
                hidden,
                let_binding,
                term_constructor,
                unextractable
            } => FunctionCommand {
                span: span.into(),
                name: name.to_string(),
                schema: schema.into(),
                merge: merge.as_ref().map(|e| e.into()),
                term_constructor: term_constructor.clone(),
                unextractable: *unextractable,
                hidden: *hidden,
                let_binding: *let_binding
            };
        AddRuleset(span: Span, name: String)
            a -> egglog::ast::Command::AddRuleset(
                a.span.clone().into(),
                (&a.name).into()
            ),
            egglog::ast::Command::AddRuleset(span, n) => AddRuleset {
                span: span.into(),
                name: n.to_string()
            };
        RuleCommand(rule: Rule)
            r -> egglog::ast::Command::Rule {
                rule: (&r.rule).into()
            },
            egglog::ast::Command::Rule {rule} => RuleCommand {
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
        Extract(span: Span, expr: Expr, variants: Expr)
            e -> egglog::ast::Command::Extract(
                e.span.clone().into(),
                (&e.expr).into(),
                (&e.variants).into()
            ),
            egglog::ast::Command::Extract(span, expr, variants) => Extract {
                span: span.into(),
                expr: expr.into(),
                variants: variants.into()
            };
        Check(span: Span, facts: Vec<Fact_>)
            c -> egglog::ast::Command::Check(c.span.clone().into(), c.facts.iter().map(|f| f.into()).collect()),
            egglog::ast::Command::Check(span, facts) => Check { span: span.into(), facts: facts.iter().map(|f| f.into()).collect() };
        ProveCommand[name="Prove"](span: Span, facts: Vec<Fact_>)
            p -> egglog::ast::Command::Prove(p.span.clone().into(), p.facts.iter().map(|f| f.into()).collect()),
            egglog::ast::Command::Prove(span, facts) => ProveCommand {
                span: span.into(),
                facts: facts.iter().map(|f| f.into()).collect()
            };
        ProveExistsCommand[name="ProveExists"](span: Span, expr: String)
            p -> egglog::ast::Command::ProveExists(p.span.clone().into(), (&p.expr).into()),
            egglog::ast::Command::ProveExists(span, expr) => ProveExistsCommand {
                span: span.into(),
                expr: expr.to_string()
            };
        PrintFunction(span: Span, name: String, length: Option<usize>, filename: Option<String>, mode: PrintFunctionMode)
            p -> egglog::ast::Command::PrintFunction(p.span.clone().into(), (&p.name).into(), p.length, p.filename.clone(), p.mode.clone().into()),
            egglog::ast::Command::PrintFunction(span, n, l, f, m) => PrintFunction {
                span: span.into(),
                name: n.to_string(),
                length: *l,
                filename: f.clone(),
                mode: m.into()
            };
        PrintSize(span: Span, name: Option<String>)
            p -> egglog::ast::Command::PrintSize(p.span.clone().into(), p.name.as_ref().map(|n| n.into())),
            egglog::ast::Command::PrintSize(span, n) => PrintSize { span: span.into(), name: n.clone().map(|n| n.to_string()) };
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
        Constructor(
            span: Span,
            name: String,
            schema: Schema,
            cost: Option<DefaultCost>,
            unextractable: bool,
            hidden: bool = false,
            let_binding: bool = false
        )
            c -> egglog::ast::Command::Constructor {
                span: c.span.clone().into(),
                name: (&c.name).into(),
                schema: (&c.schema).into(),
                cost: c.cost,
                unextractable: c.unextractable,
                hidden: c.hidden,
                let_binding: c.let_binding,
                term_constructor: None
            },
            egglog::ast::Command::Constructor {
                span,
                name,
                schema,
                cost,
                unextractable,
                hidden,
                let_binding,
                term_constructor: _
            } => Constructor {
                span: span.into(),
                name: name.to_string(),
                schema: schema.into(),
                cost: *cost,
                unextractable: *unextractable,
                hidden: *hidden,
                let_binding: *let_binding
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
        PrintOverallStatistics(span: Span, file: Option<String>)
            c -> egglog::ast::Command::PrintOverallStatistics(
                c.span.clone().into(),
                c.file.as_ref().map(|f| f.clone().into())
            ),
            egglog::ast::Command::PrintOverallStatistics(span, file) => PrintOverallStatistics {
                span: span.into(),
                file: file.clone()
            };
        Datatypes(span: Span, datatypes: Vec<(Span, String, Subdatatypes)>)
            d -> egglog::ast::Command::Datatypes {
                span: d.span.clone().into(),
                datatypes: d.datatypes.iter().map(|(s, n, d)| (s.clone().into(), n.into(), d.clone().into())).collect()
            },
            egglog::ast::Command::Datatypes {span, datatypes} => Datatypes {
                span: span.into(),
                datatypes: datatypes.iter().map(|(s, n, d)| (s.into(), n.to_string(), d.into())).collect()
            };
        UserDefined(span: Span, name: String, args: Vec<Expr>)
            u -> egglog::ast::Command::UserDefined(u.span.clone().into(), (&u.name).into(), u.args.iter().map(|e| e.into()).collect()),
            egglog::ast::Command::UserDefined(span, n, a) => UserDefined {
                span: span.into(),
                name: n.to_string(),
                args: a.iter().map(|e| e.into()).collect()
            };
        UnstableCombinedRuleset(span: Span, name: String, rulesets: Vec<String>)
            r -> egglog::ast::Command::UnstableCombinedRuleset(
                r.span.clone().into(),
                (&r.name).into(),
                r.rulesets.iter().map(|i| i.into()).collect()
            ),
            egglog::ast::Command::UnstableCombinedRuleset(span, name, rulesets) => UnstableCombinedRuleset {
                span: span.into(),
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
    egglog_reports::Stage: "{:?}" => Stage {
        Intersect(scans: Vec<SingleScan>)
            s -> egglog_reports::Stage::Intersect {
                scans: s.scans.iter().map(|scan| scan.into()).collect()
            },
            egglog_reports::Stage::Intersect { scans } => Intersect {
                scans: scans.iter().map(|scan| scan.into()).collect()
            };
        FusedIntersect(cover: Scan, to_intersect: Vec<Scan>)
            f -> egglog_reports::Stage::FusedIntersect {
                cover: (&f.cover).into(),
                to_intersect: f.to_intersect.iter().map(|scan| scan.into()).collect()
            },
            egglog_reports::Stage::FusedIntersect { cover, to_intersect } => FusedIntersect {
                cover: cover.clone().into(),
                to_intersect: to_intersect.iter().map(|scan| scan.into()).collect()
            }
    };
    egglog_reports::ReportLevel: "{:?}" => ReportLevel {
        TimeOnly()
            _r -> egglog_reports::ReportLevel::TimeOnly,
            egglog_reports::ReportLevel::TimeOnly => TimeOnly {};
        WithPlan()
            _r -> egglog_reports::ReportLevel::WithPlan,
            egglog_reports::ReportLevel::WithPlan => WithPlan {};
        StageInfo()
            _r -> egglog_reports::ReportLevel::StageInfo,
            egglog_reports::ReportLevel::StageInfo => StageInfo {}
    };
    egglog::CommandOutput: "{}" => CommandOutput {
        PrintFunctionSize(size: usize)
            b -> egglog::CommandOutput::PrintFunctionSize(b.size),
            egglog::CommandOutput::PrintFunctionSize(size) => PrintFunctionSize {size: *size};
        PrintAllFunctionsSize(sizes: Vec<(String, usize)>)
            b -> egglog::CommandOutput::PrintAllFunctionsSize(b.sizes.clone()),
            egglog::CommandOutput::PrintAllFunctionsSize(sizes) => PrintAllFunctionsSize {sizes: sizes.clone()};
        ExtractBest(termdag: TermDag, cost: DefaultCost, term: usize)
            b -> egglog::CommandOutput::ExtractBest(
                b.termdag.0.clone(),
                b.cost,
                b.term
            ),
            egglog::CommandOutput::ExtractBest(termdag, cost, term) => ExtractBest {
                termdag: TermDag(termdag.clone()),
                cost: *cost,
                term: *term
            };
        ExtractVariants(termdag: TermDag, terms: Vec<usize>)
            v -> egglog::CommandOutput::ExtractVariants(
                v.termdag.0.clone(),
                v.terms.clone()
            ),
            egglog::CommandOutput::ExtractVariants(termdag, terms) => ExtractVariants {
                termdag: TermDag(termdag.clone()),
                terms: terms.clone()
            };
        ProveExistsOutput(proof: String)
            _p -> panic!("Converting Python proof output back into egglog is unsupported"),
            egglog::CommandOutput::ProveExists { proof_store, proof_id } => ProveExistsOutput {
                proof: proof_store.proof_to_string(*proof_id)
            };
        OverallStatistics(report: RunReport)
            b -> egglog::CommandOutput::OverallStatistics(b.report.clone().into()),
            egglog::CommandOutput::OverallStatistics(report) => OverallStatistics {report: report.into()};
        RunScheduleOutput(report: RunReport)
            b -> egglog::CommandOutput::RunSchedule(b.report.clone().into()),
            egglog::CommandOutput::RunSchedule(report) => RunScheduleOutput {report: report.into()};
        PrintFunctionOutput(function: Function, termdag: TermDag, terms: Vec<(usize, usize)>, mode: PrintFunctionMode)
            v -> egglog::CommandOutput::PrintFunction(
                v.function.0.clone(),
                v.termdag.0.clone(),
                v.terms.clone(),
                v.mode.clone().into()
            ),
            egglog::CommandOutput::PrintFunction(function, termdag, terms, mode) => PrintFunctionOutput {
                function: Function(function.clone()),
                termdag:  TermDag(termdag.clone()),
                terms: terms.clone(),
                mode: mode.into()
            };
        UserDefinedOutput(output: UserDefinedCommandOutput)
            b -> egglog::CommandOutput::UserDefined(b.output.0.clone()),
            egglog::CommandOutput::UserDefined(output) => UserDefinedOutput {output: UserDefinedCommandOutput(output.clone())}
    };
    egglog_ast::span::Span: "{:?}" => Span {
        PanicSpan()
            _p -> egglog_ast::span::Span::Panic,
            egglog_ast::span::Span::Panic => PanicSpan {};
        EgglogSpan(file: SrcFile, i: usize, j: usize)
            e -> egglog_ast::span::Span::Egglog(Arc::new({
                egglog_ast::span::EgglogSpan {
                    file: e.file.0.clone(),
                    i: e.i,
                    j: e.j
                }
            })),
            egglog_ast::span::Span::Egglog(e) => EgglogSpan {
                file: SrcFile(e.file.clone()),
                i: e.i,
                j: e.j
            };
        RustSpan(file: String, line: u32, column: u32)
            r -> egglog_ast::span::Span::Rust(Arc::new(egglog::ast::RustSpan {
                file: Box::leak(r.file.clone().into_boxed_str()),
                line: r.line,
                column: r.column
            })),
            egglog_ast::span::Span::Rust(r) => RustSpan {file: r.file.to_string(), line: r.line, column: r.column}
    }
);

impl Default for RuleEvalMode {
    fn default() -> Self {
        Self::Seminaive(Seminaive {})
    }
}

#[pyclass(frozen)]
#[derive(Clone, PartialEq, Eq)]
pub struct SrcFile(Arc<egglog_ast::span::SrcFile>);

#[pymethods]
impl SrcFile {
    #[new]
    fn new(name: Option<String>, contents: String) -> Self {
        Self(Arc::new(egglog_ast::span::SrcFile { name, contents }))
    }

    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    #[getter]
    fn contents(&self) -> &str {
        &self.0.contents
    }

    fn __repr__(slf: PyRef<'_, Self>, py: Python) -> PyResult<String> {
        data_repr(py, slf, vec!["name", "contents"])
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __richcmp__(
        &self,
        other: &Self,
        op: pyo3::basic::CompareOp,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        Ok(match op {
            pyo3::basic::CompareOp::Eq => {
                (self == other).into_pyobject(py)?.as_any().clone().unbind()
            }
            pyo3::basic::CompareOp::Ne => {
                (self != other).into_pyobject(py)?.as_any().clone().unbind()
            }
            _ => py.NotImplemented(),
        })
    }
}

convert_struct!(
    egglog::ast::ContainerRebuildSpec: "{}" => ContainerRebuildSpec(
        internal_rebuild_prim: String,
        internal_rebuild_proof_prim: Option<String> = None
    )
        s -> egglog::ast::ContainerRebuildSpec {
            internal_rebuild_prim: s.internal_rebuild_prim.clone(),
            internal_rebuild_proof_prim: s.internal_rebuild_proof_prim.clone()
        },
        s -> ContainerRebuildSpec {
            internal_rebuild_prim: s.internal_rebuild_prim.clone(),
            internal_rebuild_proof_prim: s.internal_rebuild_proof_prim.clone()
        };
    egglog::ast::ProofConstructorNames: "{:?}" => ProofConstructorNames(
        congr: String,
        trans: String,
        sym: String,
        normalize: String
    )
        n -> egglog::ast::ProofConstructorNames {
            congr: n.congr.clone(),
            trans: n.trans.clone(),
            sym: n.sym.clone(),
            normalize: n.normalize.clone()
        },
        n -> ProofConstructorNames {
            congr: n.congr.clone(),
            trans: n.trans.clone(),
            sym: n.sym.clone(),
            normalize: n.normalize.clone()
        };
    egglog::ast::Variant: "{:?}" => Variant(
        span: Span,
        name: String,
        types: Vec<String>,
        cost: Option<DefaultCost> = None,
        unextractable: bool = false
    )
        v -> egglog::ast::Variant {span: v.span.clone().into(), name: (&v.name).into(), types: v.types.iter().map(|v| v.into()).collect(), cost: v.cost, unextractable: v.unextractable},
        v -> Variant {span: v.span.clone().into(), name: v.name.to_string(), types: v.types.iter().map(|v| v.to_string()).collect(), cost: v.cost, unextractable: v.unextractable};
    egglog::ast::Schema: "{:?}" => Schema(
        input: Vec<String>,
        output: String
    )
        s -> egglog::ast::Schema {input: s.input.iter().map(|v| v.into()).collect(), output: (&s.output).into()},
        s -> Schema {input: s.input.iter().map(|v| v.to_string()).collect(), output: s.output.to_string()};
    egglog::ast::GenericRule<String, String>: "{:?}" => Rule(
        span: Span,
        head: Vec<Action>,
        body: Vec<Fact_>,
        name: String,
        ruleset: String,
        eval_mode: RuleEvalMode = RuleEvalMode::default(),
        no_decomp: bool = false,
        include_subsumed: bool = false
    )
        r -> egglog::ast::GenericRule {
            span: r.span.clone().into(),
            head: egglog::ast::GenericActions(r.head.iter().map(|v| v.into()).collect()),
            body: r.body.iter().map(|v| v.into()).collect(),
            name: (&r.name).into(),
            ruleset: (&r.ruleset).into(),
            eval_mode: (&r.eval_mode).into(),
            no_decomp: r.no_decomp,
            include_subsumed: r.include_subsumed
        },
        r -> Rule {
            span: r.span.clone().into(),
            head: r.head.0.iter().map(|v| v.into()).collect(),
            body: r.body.iter().map(|v| v.into()).collect(),
            name: r.name.to_string(),
            ruleset: r.ruleset.to_string(),
            eval_mode: (&r.eval_mode).into(),
            no_decomp: r.no_decomp,
            include_subsumed: r.include_subsumed
        };
    egglog::ast::GenericRewrite<String, String>: "{:?}" => Rewrite(
        span: Span,
        lhs: Expr,
        rhs: Expr,
        conditions: Vec<Fact_> = Vec::new(),
        name: String = String::new()
    )
        r -> egglog::ast::GenericRewrite {span: r.span.clone().into(), lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect(), name: (&r.name).into()},
        r -> Rewrite {span: r.span.clone().into(), lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect(), name: r.name.to_string()};
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
    egglog_reports::SingleScan: "{:?}" => SingleScan(
        atom: String,
        column: (String, i64)
    )
        s -> egglog_reports::SingleScan(
            s.atom.clone(),
            (s.column.0.clone(), s.column.1)
        ),
        s -> SingleScan {
            atom: s.0.to_string(),
            column: (s.1 .0.to_string(), s.1 .1)
        };
    egglog_reports::Scan: "{:?}" => Scan(
        atom: String,
        columns: Vec<(String, i64)>
    )
        s -> egglog_reports::Scan(
            s.atom.clone(),
            s.columns.iter().map(|(n, i)| (n.clone(), *i)).collect()
        ),
        s -> Scan {
            atom: s.0.to_string(),
            columns: s.1.iter().map(|(n, i)| (n.to_string(), *i)).collect()
        };
    egglog_reports::StageStats: "{:?}" => StageStats(
        num_candidates: usize,
        num_succeeded: usize
    )
        s -> egglog_reports::StageStats {
            num_candidates: s.num_candidates,
            num_succeeded: s.num_succeeded
        },
        s -> StageStats {
            num_candidates: s.num_candidates,
            num_succeeded: s.num_succeeded
        };
    egglog_reports::Plan: "{:?}" => Plan(
        stages: Vec<(Stage, Option<StageStats>, Vec<usize>)>
    )
        p -> egglog_reports::Plan {
            stages: p
                .stages
                .iter()
                .map(|(s, ss, ts)| {
                    (
                        s.clone().into(),
                        ss.as_ref().map(|st| st.clone().into()),
                        ts.clone(),
                    )
                })
                .collect()
        },
        p -> Plan {
            stages: p
                .stages
                .iter()
                .map(|(s, ss, ts)| {
                    (
                        s.clone().into(),
                        ss.as_ref().map(|st| st.clone().into()),
                        ts.clone(),
                    )
                })
                .collect()
        };
    egglog_reports::RuleReport: "{:?}" => RuleReport(
        plan: Option<Plan>,
        search_and_apply_time: WrappedDuration,
        num_matches: usize
    )
        r -> egglog_reports::RuleReport {
            plan: r.plan.as_ref().map(|p| p.clone().into()),
            search_and_apply_time: r.search_and_apply_time.clone().0,
            num_matches: r.num_matches
        },
        r -> RuleReport {
            plan: r.plan.as_ref().map(|p| p.clone().into()),
            search_and_apply_time: r.search_and_apply_time.clone().into(),
            num_matches: r.num_matches
        };
    egglog_reports::RuleSetReport: "{:?}" => RuleSetReport(
        changed: bool,
        rule_reports: HashMap<String, Vec<RuleReport>>,
        search_and_apply_time: WrappedDuration,
        merge_time: WrappedDuration
    )
        r -> egglog_reports::RuleSetReport {
            changed: r.changed,
            rule_reports: r
                .rule_reports
                .iter()
                .map(|(k, v)| {
                    (
                        Arc::<str>::from(k.clone()),
                        v.iter().map(|rr| rr.clone().into()).collect(),
                    )
                })
                .collect(),
            search_and_apply_time: r.search_and_apply_time.clone().0,
            merge_time: r.merge_time.clone().0,
        },
        r -> RuleSetReport {
            changed: r.changed,
            rule_reports: r
                .rule_reports
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        v.iter().map(|rr| rr.clone().into()).collect(),
                    )
                })
                .collect(),
            search_and_apply_time: r.search_and_apply_time.clone().into(),
            merge_time: r.merge_time.clone().into(),
        };
    egglog_reports::IterationReport: "{:?}" => IterationReport(
        rule_set_report: RuleSetReport,
        rebuild_time: WrappedDuration
    )
        r -> egglog_reports::IterationReport {
            rule_set_report: (&r.rule_set_report).into(),
            rebuild_time: r.rebuild_time.clone().0
        },
        r -> IterationReport {
            rule_set_report: (&r.rule_set_report).into(),
            rebuild_time: r.rebuild_time.clone().into()
        };
    egglog_reports::RunReport: "{:?}" => RunReport(
        iterations: Vec<IterationReport>,
        updated: bool,
        can_stop: bool,
        search_and_apply_time_per_rule: HashMap<String, WrappedDuration>,
        num_matches_per_rule: HashMap<String, usize>,
        search_and_apply_time_per_ruleset: HashMap<String, WrappedDuration>,
        merge_time_per_ruleset: HashMap<String, WrappedDuration>,
        rebuild_time_per_ruleset: HashMap<String, WrappedDuration>
    )
        r -> egglog_reports::RunReport {
            iterations: r
                .iterations
                .iter()
                .map(|i| Arc::new(i.clone().into()))
                .collect(),
            updated: r.updated,
            can_stop: r.can_stop,
            search_and_apply_time_per_rule: r
                .search_and_apply_time_per_rule
                .iter()
                .map(|(k, v)| (Arc::<str>::from(k.clone()), v.clone().0))
                .collect(),
            num_matches_per_rule: r
                .num_matches_per_rule
                .iter()
                .map(|(k, v)| (Arc::<str>::from(k.clone()), *v))
                .collect(),
            search_and_apply_time_per_ruleset: r
                .search_and_apply_time_per_ruleset
                .iter()
                .map(|(k, v)| (Arc::<str>::from(k.clone()), v.clone().0))
                .collect(),
            merge_time_per_ruleset: r
                .merge_time_per_ruleset
                .iter()
                .map(|(k, v)| (Arc::<str>::from(k.clone()), v.clone().0))
                .collect(),
            rebuild_time_per_ruleset: r
                .rebuild_time_per_ruleset
                .iter()
                .map(|(k, v)| (Arc::<str>::from(k.clone()), v.clone().0))
                .collect(),
        },
        r -> RunReport {
            iterations: r.iterations.iter().map(|i| i.as_ref().into()).collect(),
            updated: r.updated,
            can_stop: r.can_stop,
            search_and_apply_time_per_rule: r
                .search_and_apply_time_per_rule
                .iter()
                .map(|(k, v)| (k.to_string(), (*v).into()))
                .collect(),
            num_matches_per_rule: r
                .num_matches_per_rule
                .iter()
                .map(|(k, v)| (k.to_string(), *v))
                .collect(),
            search_and_apply_time_per_ruleset: r
                .search_and_apply_time_per_ruleset
                .iter()
                .map(|(k, v)| (k.to_string(), (*v).into()))
                .collect(),
            merge_time_per_ruleset: r
                .merge_time_per_ruleset
                .iter()
                .map(|(k, v)| (k.to_string(), (*v).into()))
                .collect(),
            rebuild_time_per_ruleset: r
                .rebuild_time_per_ruleset
                .iter()
                .map(|(k, v)| (k.to_string(), (*v).into()))
                .collect(),
        }
);

impl<'py> FromPyObject<'_, 'py> for Box<Schedule> {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        obj.extract::<Schedule>().map(Box::new)
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

impl<'py> FromPyObject<'_, 'py> for Box<Command> {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        obj.extract::<Command>().map(Box::new)
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

impl<'py> FromPyObject<'_, 'py> for WrappedOrderedF64 {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        obj.extract::<f64>()
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

impl<'py> FromPyObject<'_, 'py> for WrappedDuration {
    type Error = PyErr;
    fn extract(obj: Borrowed<'_, 'py, PyAny>) -> Result<Self, Self::Error> {
        let py_delta = obj.cast::<pyo3::types::PyDelta>()?;
        let days = py_delta.get_days();
        let seconds = py_delta.get_seconds();
        let microseconds = py_delta.get_microseconds();
        if days < 0 {
            return Err(PyValueError::new_err(
                "negative timedeltas cannot be converted to Rust Duration",
            ));
        }
        if seconds < 0 || microseconds < 0 {
            return Err(PyValueError::new_err("invalid timedelta components"));
        }

        let seconds = (days as u64)
            .checked_mul(SECONDS_PER_DAY)
            .and_then(|day_seconds| day_seconds.checked_add(seconds as u64))
            .ok_or_else(|| PyOverflowError::new_err("timedelta is too large for Rust Duration"))?;
        let nanoseconds = (microseconds as u32).checked_mul(1_000).ok_or_else(|| {
            PyOverflowError::new_err("timedelta is too precise for Rust Duration")
        })?;
        Ok(WrappedDuration(std::time::Duration::new(
            seconds,
            nanoseconds,
        )))
    }
}

const SECONDS_PER_DAY: u64 = 24 * 60 * 60;

fn duration_to_py_parts(duration: std::time::Duration) -> Option<(i32, i32, i32)> {
    let total_seconds = duration.as_secs();
    Some((
        (total_seconds / SECONDS_PER_DAY).try_into().ok()?,
        (total_seconds % SECONDS_PER_DAY).try_into().ok()?,
        duration.subsec_micros().try_into().ok()?,
    ))
}

impl<'py> IntoPyObject<'py> for WrappedDuration {
    type Target = PyDelta; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let (days, seconds, microseconds) = duration_to_py_parts(self.0).ok_or_else(|| {
            PyOverflowError::new_err("Rust Duration is too large for datetime.timedelta")
        })?;
        Ok(pyo3::types::PyDelta::new(py, days, seconds, microseconds, true)?.clone())
    }
}

#[cfg(test)]
mod duration_tests {
    use super::*;

    #[test]
    fn rejects_duration_with_too_many_days_for_python() {
        let seconds = (i32::MAX as u64 + 1) * SECONDS_PER_DAY;
        assert!(duration_to_py_parts(std::time::Duration::from_secs(seconds)).is_none());
    }
}

#[pyclass()]
#[derive(Clone)]
pub struct UserDefinedCommandOutput(Arc<dyn egglog::UserDefinedCommandOutput>);

#[pymethods]
impl UserDefinedCommandOutput {
    fn __str__(&self) -> String {
        format!("{}", self.0)
    }
    fn __repr__(&self) -> String {
        format!("{:}", self.0)
    }
}

impl PartialEq for UserDefinedCommandOutput {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl std::cmp::Eq for UserDefinedCommandOutput {}

#[pyclass()]
#[derive(Clone)]
pub struct Function(egglog::Function);

#[pymethods]
impl Function {
    fn __str__(&self) -> String {
        format!("Function(name={})", self.0.name())
    }

    fn name(&self) -> String {
        self.0.name().to_string()
    }
}

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.0.name() == other.0.name()
    }
}

impl std::cmp::Eq for Function {}

#[cfg(test)]
mod span_tests {
    use super::*;

    #[test]
    fn egglog_span_conversion_shares_its_source_file() {
        let file = Arc::new(egglog_ast::span::SrcFile {
            name: Some("large.egg".to_owned()),
            contents: "(relation R (i64))\n".repeat(1_000),
        });
        let span = egglog_ast::span::Span::Egglog(Arc::new(egglog_ast::span::EgglogSpan {
            file: file.clone(),
            i: 0,
            j: 18,
        }));

        let Span::EgglogSpan(converted) = Span::from(&span) else {
            panic!("expected an egglog span");
        };
        assert!(Arc::ptr_eq(&converted.file.0, &file));

        let egglog_ast::span::Span::Egglog(round_tripped) =
            egglog_ast::span::Span::from(&Span::EgglogSpan(converted))
        else {
            panic!("expected an egglog span");
        };
        assert!(Arc::ptr_eq(&round_tripped.file, &file));
    }
}
