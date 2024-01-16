# Reference Documentation

```{module} egglog

```

This is a definitive reference of `egglog` module and the concepts in it.

## Terms

First we can start with an overview of

EGraph ([](egglog.EGraph))
: An equivalence relation over a set of expressions.
: A collection of expressions where each expression is part of a distinct equivalence class.

Expression ([](egglog.Expr))
: Either a function called with some number of argument expressions or a literal integer, float, or string, with a particular type.

Function ([](egglog.function))
: Defined by a unique name and a typing relation that will specify the return type based on the types of the argument expressions.
: These can either be builtin functions, which are implemented in Rust, or user defined function which have types for each argument and the return type.
: Relations ([](egglog.relation)), constants ([](egglog.constant)), methods, classmethods, and class variables are all syntactic sugar for defining functions.

Type (called a "sort" in the rust bindings)
: A uniquely named entity, that is used to constraint the composition of functions.
: Can be either a primitive type if it is defined in Rust or a user defined type.

```{class} EGraph

The `EGraph` is the main data structure in `egglog`, which can be thought of as storing an equivalence relation over
a set of expressions (`Expr` objects). Another way to say this is that every expression in the EGraph is part of a distinct
equivalence class, where all expressions in each equivalence class are considered equivalent to each other.
```

https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc
https://www.sphinx-doc.org/en/master/usage/domains/python.html#signatures
https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html

```{class} Expr

Subclass an expression to create a new type.


```

```{decorator} function

```

```{decorator} method

```

```{function} relation

```

```{function} constant[T](name: str, tp: T) -> T

```
