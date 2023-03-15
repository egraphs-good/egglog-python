// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egg_smol types
use crate::utils::*;
use pyo3::prelude::*;

// TODO: Resolve display issue
convert_enums!(
    egg_smol::ast::Literal: "{:}" => Literal {
        Int(value: i64)
            i -> egg_smol::ast::Literal::Int(i.value),
            egg_smol::ast::Literal::Int(i) => Int { value: i.clone() };
        String_[name="String"](value: String)
            s -> egg_smol::ast::Literal::String((&s.value).into()),
            egg_smol::ast::Literal::String(s) => String_ { value: s.to_string() };
        Unit()
            _x -> egg_smol::ast::Literal::Unit,
            egg_smol::ast::Literal::Unit => Unit {}
    };
    egg_smol::ast::Expr: "{:}" => Expr {
        Lit(value: Literal)
            l -> egg_smol::ast::Expr::Lit((&l.value).into()),
            egg_smol::ast::Expr::Lit(l) => Lit { value: l.into() };
        Var(name: String)
            v -> egg_smol::ast::Expr::Var((&v.name).into()),
            egg_smol::ast::Expr::Var(v) => Var { name: v.to_string() };
        Call(name: String, args: Vec<Expr>)
            c -> egg_smol::ast::Expr::Call((&c.name).into(), (&c.args).into_iter().map(|e| e.into()).collect()),
            egg_smol::ast::Expr::Call(c, a) => Call {
                name: c.to_string(),
                args: a.into_iter().map(|e| e.into()).collect()
            }
    };
    egg_smol::ast::Fact: "{:}" => Fact_ {
        Eq(exprs: Vec<Expr>)
           eq -> egg_smol::ast::Fact::Eq((&eq.exprs).into_iter().map(|e| e.into()).collect()),
           egg_smol::ast::Fact::Eq(e) => Eq { exprs: e.into_iter().map(|e| e.into()).collect() };
        Fact(expr: Expr)
            f -> egg_smol::ast::Fact::Fact((&f.expr).into()),
            egg_smol::ast::Fact::Fact(e) => Fact { expr: e.into() }
    };
    egg_smol::ast::Action: "{:}" => Action {
        Let(lhs: String, rhs: Expr)
            d -> egg_smol::ast::Action::Let((&d.lhs).into(), (&d.rhs).into()),
            egg_smol::ast::Action::Let(n, e) => Let { lhs: n.to_string(), rhs: e.into() };
        Set(lhs: String, args: Vec<Expr>, rhs: Expr)
            s -> egg_smol::ast::Action::Set((&s.lhs).into(), (&s.args).into_iter().map(|e| e.into()).collect(), (&s.rhs).into()),
            egg_smol::ast::Action::Set(n, a, e) => Set {
                lhs: n.to_string(),
                args: a.into_iter().map(|e| e.into()).collect(),
                rhs: e.into()
            };
        Delete(sym: String, args: Vec<Expr>)
            d -> egg_smol::ast::Action::Delete((&d.sym).into(), (&d.args).into_iter().map(|e| e.into()).collect()),
            egg_smol::ast::Action::Delete(n, a) => Delete {
                sym: n.to_string(),
                args: a.into_iter().map(|e| e.into()).collect()
            };
        Union(lhs: Expr, rhs: Expr)
            u -> egg_smol::ast::Action::Union((&u.lhs).into(), (&u.rhs).into()),
            egg_smol::ast::Action::Union(l, r) => Union { lhs: l.into(), rhs: r.into() };
        Panic(msg: String)
            p -> egg_smol::ast::Action::Panic(p.msg.to_string()),
            egg_smol::ast::Action::Panic(msg) => Panic { msg: msg.to_string()  };
        Expr_(expr: Expr)
            e -> egg_smol::ast::Action::Expr((&e.expr).into()),
            egg_smol::ast::Action::Expr(e) => Expr_ { expr: e.into() }
    };
    egg_smol::ast::Command: "{:?}" => Command {
        Datatype(name: String, variants: Vec<Variant>)
            d -> egg_smol::ast::Command::Datatype {
                name: (&d.name).into(),
                variants: (&d.variants).into_iter().map(|v| v.into()).collect()
            },
            egg_smol::ast::Command::Datatype {name, variants} => Datatype {
                name: name.to_string(),
                variants: variants.into_iter().map(|v| v.into()).collect()
            };
        Sort(name: String, presort: String, args: Vec<Expr>)
            s -> egg_smol::ast::Command::Sort(
                (&s.name).into(),
                (&s.presort).into(),
                (&s.args).into_iter().map(|e| e.into()).collect()
            ),
            egg_smol::ast::Command::Sort(n, p, a) => Sort {
                name: n.to_string(),
                presort: p.to_string(),
                args: a.into_iter().map(|e| e.into()).collect()
            };
        Function(decl: FunctionDecl)
            f -> egg_smol::ast::Command::Function((&f.decl).into()),
            egg_smol::ast::Command::Function(f) => Function { decl: f.into() };
        Define(name: String, expr: Expr, cost: Option<usize>)
            d -> egg_smol::ast::Command::Define {
                name: (&d.name).into(),
                expr: (&d.expr).into(),
                cost: d.cost
            },
            egg_smol::ast::Command::Define {name, expr, cost} => Define {
                name: name.to_string(),
                expr: expr.into(),
                cost: *cost
            };
        RuleCommand(rule: Rule)
            r -> egg_smol::ast::Command::Rule((&r.rule).into()),
            egg_smol::ast::Command::Rule(r) => RuleCommand { rule: r.into() };
        RewriteCommand(rewrite: Rewrite)
            r -> egg_smol::ast::Command::Rewrite((&r.rewrite).into()),
            egg_smol::ast::Command::Rewrite(r) => RewriteCommand { rewrite: r.into() };
        ActionCommand(action: Action)
            a -> egg_smol::ast::Command::Action((&a.action).into()),
            egg_smol::ast::Command::Action(a) => ActionCommand { action: a.into() };
        Run(length: usize)
            r -> egg_smol::ast::Command::Run(r.length),
            egg_smol::ast::Command::Run(l) => Run { length: *l };
        Extract(variants: usize, expr: Expr)
            e -> egg_smol::ast::Command::Extract {
                variants: e.variants,
                e: (&e.expr).into()
            },
            egg_smol::ast::Command::Extract {variants, e} => Extract {
                variants: *variants,
                expr: e.into()
            };
        Check(fact: Fact_)
            c -> egg_smol::ast::Command::Check((&c.fact).into()),
            egg_smol::ast::Command::Check(f) => Check { fact: f.into() };
        ClearRules()
            c -> egg_smol::ast::Command::ClearRules,
            egg_smol::ast::Command::ClearRules => ClearRules {};
        Clear()
            c -> egg_smol::ast::Command::Clear,
            egg_smol::ast::Command::Clear => Clear {};
        Print(name: String, length: usize)
            p -> egg_smol::ast::Command::Print((&p.name).into(), p.length),
            egg_smol::ast::Command::Print(n, l) => Print {
                name: n.to_string(),
                length: *l
            };
        PrintSize(name: String)
            p -> egg_smol::ast::Command::PrintSize((&p.name).into()),
            egg_smol::ast::Command::PrintSize(n) => PrintSize { name: n.to_string() };
        Input(name: String, file: String)
            i -> egg_smol::ast::Command::Input {
                name: (&i.name).into(),
                file: (&i.file).into()
            },
            egg_smol::ast::Command::Input {name, file} => Input {
                name: name.to_string(),
                file: file.to_string()
            };
        Query(facts: Vec<Fact_>)
            q -> egg_smol::ast::Command::Query((&q.facts).into_iter().map(|f| f.into()).collect()),
            egg_smol::ast::Command::Query(f) => Query {
                facts: f.into_iter().map(|f| f.into()).collect()
            };
        Push(length: usize)
            p -> egg_smol::ast::Command::Push(p.length),
            egg_smol::ast::Command::Push(l) => Push { length: *l };
        Pop(length: usize)
            p -> egg_smol::ast::Command::Pop(p.length),
            egg_smol::ast::Command::Pop(l) => Pop { length: *l }

    }
);

convert_struct!(
    egg_smol::ast::FunctionDecl: "{:?}" => FunctionDecl(
        name: String,
        schema: Schema,
        default: Option<Expr> = None,
        merge: Option<Expr> = None,
        cost: Option<usize> = None
    )
        f -> egg_smol::ast::FunctionDecl {
            name: (&f.name).into(),
            schema: (&f.schema).into(),
            default: f.default.as_ref().map(|e| e.into()),
            merge: f.merge.as_ref().map(|e| e.into()),
            cost: f.cost
        },
        f -> FunctionDecl {
            name: f.name.to_string(),
            schema: (&f.schema).into(),
            default: f.default.as_ref().map(|e| e.into()),
            merge: f.merge.as_ref().map(|e| e.into()),
            cost: f.cost
        };
    egg_smol::ast::Variant: "{:?}" => Variant(
        name: String,
        types: Vec<String>,
        cost: Option<usize> = None
    )
        v -> egg_smol::ast::Variant {name: (&v.name).into(), types: (&v.types).into_iter().map(|v| v.into()).collect(), cost: v.cost},
        v -> Variant {name: v.name.to_string(), types: v.types.iter().map(|v| v.to_string()).collect(), cost: v.cost};
    egg_smol::ast::Schema: "{:?}" => Schema(
        input: Vec<String>,
        output: String
    )
        s -> egg_smol::ast::Schema {input: (&s.input).into_iter().map(|v| v.into()).collect(), output: (&s.output).into()},
        s -> Schema {input: s.input.iter().map(|v| v.to_string()).collect(), output: s.output.to_string()};
    egg_smol::ast::Rule: "{}" => Rule(
        head: Vec<Action>,
        body: Vec<Fact_>
    )
        r -> egg_smol::ast::Rule {head: (&r.head).into_iter().map(|v| v.into()).collect(), body: (&r.body).into_iter().map(|v| v.into()).collect()},
        r -> Rule {head: r.head.iter().map(|v| v.into()).collect(), body: r.body.iter().map(|v| v.into()).collect()};
    egg_smol::ast::Rewrite: "{:?}" => Rewrite(
        lhs: Expr,
        rhs: Expr,
        conditions: Vec<Fact_> = Vec::new()
    )
        r -> egg_smol::ast::Rewrite {lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: (&r.conditions).into_iter().map(|v| v.into()).collect()},
        r -> Rewrite {lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect()}
);

// // Wrapped version of command, which ends up being a union of many types
// pub struct Command(egg_smol::ast::Command);

// impl From<egg_smol::ast::Command> for Command {
//     fn from(other: egg_smol::ast::Command) -> Self {
//         Self { command: other }
//     }
// }

// impl IntoPy<PyObject> for Command {
//     fn into_py(self, py: Python<'_>) -> PyObject {
//         match self.command {
//             egg_smol::ast::Command::Function(decl) => {
//                 return FunctionDecl::from(decl).into_py(py);
//             }
//         }
//     }
// }

// Wrapped version of Duration
// Converts from a rust duration to a python timedelta
pub struct WrappedDuration(std::time::Duration);

impl From<std::time::Duration> for WrappedDuration {
    fn from(other: std::time::Duration) -> Self {
        WrappedDuration(other)
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
