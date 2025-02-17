use crate::conversions::{Expr, Literal, Span, Term};
use egglog::TermId;
use pyo3::prelude::*;

#[pyclass()]
#[derive(Clone, PartialEq, Eq)]
pub struct TermDag {
    pub termdag: egglog::TermDag,
}

#[pymethods]
impl TermDag {
    /// Create a new, empty TermDag.
    #[new]
    fn new() -> Self {
        Self {
            termdag: egglog::TermDag::default(),
        }
    }

    /// Returns the number of nodes in this DAG.
    pub fn size(&self) -> usize {
        self.termdag.size()
    }

    /// Convert the given term to its id.
    ///
    /// Panics if the term does not already exist in this [TermDag].
    pub fn lookup(&self, node: Term) -> TermId {
        self.termdag.lookup(&node.into()).into()
    }

    /// Convert the given id to the corresponding term.
    ///
    /// Panics if the id is not valid.
    pub fn get(&self, id: TermId) -> Term {
        self.termdag.get(id).into()
    }
    /// Make and return a App with the given head symbol and children,
    /// and insert into the DAG if it is not already present.
    ///
    /// Panics if any of the children are not already in the DAG.
    pub fn app(&mut self, sym: String, children: Vec<Term>) -> Term {
        self.termdag
            .app(sym.into(), children.into_iter().map(|c| c.into()).collect())
            .into()
    }

    /// Make and return a [`Term::Lit`] with the given literal, and insert into
    /// the DAG if it is not already present.
    pub fn lit(&mut self, lit: Literal) -> Term {
        self.termdag.lit(lit.into()).into()
    }

    /// Make and return a [`Term::Var`] with the given symbol, and insert into
    /// the DAG if it is not already present.
    pub fn var(&mut self, sym: String) -> Term {
        self.termdag.var(sym.into()).into()
    }

    /// Recursively converts the given expression to a term.
    ///
    /// This involves inserting every subexpression into this DAG. Because
    /// TermDags are hashconsed, the resulting term is guaranteed to maximally
    /// share subterms.
    pub fn expr_to_term(&mut self, expr: Expr) -> Term {
        self.termdag.expr_to_term(&expr.into()).into()
    }

    /// Recursively converts the given term to an expression.
    ///
    /// Panics if the term contains subterms that are not in the DAG.
    pub fn term_to_expr(&self, term: Term, span: Span) -> Expr {
        self.termdag.term_to_expr(&term.into(), span.into()).into()
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: Term) -> String {
        self.termdag.to_string(&term.into())
    }
}

impl From<&egglog::TermDag> for TermDag {
    fn from(termdag: &egglog::TermDag) -> Self {
        Self {
            termdag: termdag.clone(),
        }
    }
}
impl From<&TermDag> for egglog::TermDag {
    fn from(termdag: &TermDag) -> Self {
        termdag.termdag.clone()
    }
}

impl From<egglog::TermDag> for TermDag {
    fn from(termdag: egglog::TermDag) -> Self {
        Self { termdag }
    }
}
impl From<TermDag> for egglog::TermDag {
    fn from(termdag: TermDag) -> Self {
        termdag.termdag
    }
}
