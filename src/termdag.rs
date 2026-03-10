use crate::conversions::{Expr, Literal, Span, Term};
use egglog::TermId;
use pyo3::prelude::*;

#[pyclass(eq, str = "{0:?}")]
#[derive(PartialEq, Eq, Clone)]
pub struct TermDag(pub egglog::TermDag);

#[pymethods]
impl TermDag {
    /// Create a new, empty TermDag.
    #[new]
    fn new() -> Self {
        Self(egglog::TermDag::default())
    }

    /// Returns the number of nodes in this DAG.
    pub fn size(&self) -> usize {
        self.0.size()
    }

    /// Convert the given term to its id.
    ///
    /// Panics if the term does not already exist in this [TermDag].
    pub fn lookup(&self, node: Term) -> TermId {
        self.0.lookup(&node.into())
    }

    /// Convert the given id to the corresponding term.
    ///
    /// Panics if the id is not valid.
    pub fn get(&self, id: TermId) -> Term {
        self.0.get(id).into()
    }
    /// Make and return a App with the given head symbol and children,
    /// and insert into the DAG if it is not already present.
    ///
    /// Panics if any of the children are not already in the DAG.
    pub fn app(&mut self, sym: String, children: Vec<TermId>) -> TermId {
        self.0.app(sym, children)
    }

    /// Make a [`Term::Lit`] with the given literal and return its id,
    /// inserting it into the DAG if it is not already present.
    pub fn lit(&mut self, lit: Literal) -> TermId {
        self.0.lit(lit.into())
    }

    /// Make and return a [`Term::Var`] id with the given symbol, and insert into
    /// the DAG if it is not already present.
    pub fn var(&mut self, sym: String) -> TermId {
        self.0.var(sym)
    }

    /// Recursively converts the given expression to a term.
    ///
    /// This involves inserting every subexpression into this DAG. Because
    /// TermDags are hashconsed, the resulting term is guaranteed to maximally
    /// share subterms.
    pub fn expr_to_term(&mut self, expr: Expr) -> TermId {
        self.0.expr_to_term(&expr.into())
    }

    /// Recursively converts the given term to an expression.
    ///
    /// Panics if the term contains subterms that are not in the DAG.
    pub fn term_to_expr(&self, term: TermId, span: Span) -> Expr {
        self.0.term_to_expr(&term, span.into()).into()
    }

    /// Converts the given term to a string.
    ///
    /// Panics if the term or any of its subterms are not in the DAG.
    pub fn to_string(&self, term: TermId) -> String {
        self.0.to_string(term)
    }
}
