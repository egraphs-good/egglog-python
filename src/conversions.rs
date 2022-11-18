// Create wrappers around input types so that convert from pyobjects to them
// and then from them to the egg_smol types
use crate::utils::*;
use pyo3::prelude::*;

convert_enums!(
    egg_smol::ast::Literal => Literal {
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
    egg_smol::ast::Expr => Expr {
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
    egg_smol::ast::Fact => Fact_ {
        Eq(exprs: Vec<Expr>)
           eq -> egg_smol::ast::Fact::Eq((&eq.exprs).into_iter().map(|e| e.into()).collect()),
           egg_smol::ast::Fact::Eq(e) => Eq { exprs: e.into_iter().map(|e| e.into()).collect() };
        Fact(expr: Expr)
            f -> egg_smol::ast::Fact::Fact((&f.expr).into()),
            egg_smol::ast::Fact::Fact(e) => Fact { expr: e.into() }
    };
    egg_smol::ast::Action => Action {
        Define(lhs: String, rhs: Expr)
            d -> egg_smol::ast::Action::Define((&d.lhs).into(), (&d.rhs).into()),
            egg_smol::ast::Action::Define(n, e) => Define { lhs: n.to_string(), rhs: e.into() };
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
    }
);

convert_struct!(
    egg_smol::ast::FunctionDecl => FunctionDecl(
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
    egg_smol::ast::Variant => Variant(
        name: String,
        types: Vec<String>,
        cost: Option<usize> = None
    )
        v -> egg_smol::ast::Variant {name: (&v.name).into(), types: (&v.types).into_iter().map(|v| v.into()).collect(), cost: v.cost},
        v -> Variant {name: v.name.to_string(), types: v.types.iter().map(|v| v.to_string()).collect(), cost: v.cost};
    egg_smol::ast::Schema => Schema(
        input: Vec<String>,
        output: String
    )
        s -> egg_smol::ast::Schema {input: (&s.input).into_iter().map(|v| v.into()).collect(), output: (&s.output).into()},
        s -> Schema {input: s.input.iter().map(|v| v.to_string()).collect(), output: s.output.to_string()};
    egg_smol::ast::Rule[display] => Rule(
        head: Vec<Action>,
        body: Vec<Fact_>
    )
        r -> egg_smol::ast::Rule {head: (&r.head).into_iter().map(|v| v.into()).collect(), body: (&r.body).into_iter().map(|v| v.into()).collect()},
        r -> Rule {head: r.head.iter().map(|v| v.into()).collect(), body: r.body.iter().map(|v| v.into()).collect()};
    egg_smol::ast::Rewrite => Rewrite(
        lhs: Expr,
        rhs: Expr,
        conditions: Vec<Fact_> = Vec::new()
    )
        r -> egg_smol::ast::Rewrite {lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: (&r.conditions).into_iter().map(|v| v.into()).collect()},
        r -> Rewrite {lhs: (&r.lhs).into(), rhs: (&r.rhs).into(), conditions: r.conditions.iter().map(|v| v.into()).collect()}
);

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
