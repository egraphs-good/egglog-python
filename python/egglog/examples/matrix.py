"""
Matrix multiplication and Kronecker product.
============================================
"""

from __future__ import annotations

from egglog import *

egraph = EGraph()


class Dim(Expr):
    """
    A dimension of a matix.

    >>> Dim(3) * Dim.named("n")
    Dim(3) * Dim.named("n")
    """

    @method(egg_fn="Lit")
    def __init__(self, value: i64Like) -> None: ...

    @method(egg_fn="NamedDim")
    @classmethod
    def named(cls, name: StringLike) -> Dim:  # type: ignore[empty-body]
        ...

    @method(egg_fn="Times")
    def __mul__(self, other: Dim) -> Dim:  # type: ignore[empty-body]
        ...


a, b, c, n = vars_("a b c n", Dim)
i, j = vars_("i j", i64)
egraph.register(
    rewrite(a * (b * c)).to((a * b) * c),
    rewrite((a * b) * c).to(a * (b * c)),
    rewrite(Dim(i) * Dim(j)).to(Dim(i * j)),
    rewrite(a * b).to(b * a),
)


class Matrix(Expr, egg_sort="MExpr"):
    @method(egg_fn="Id")
    @classmethod
    def identity(cls, dim: Dim) -> Matrix:  # type: ignore[empty-body]
        """
        Create an identity matrix of the given dimension.
        """

    @method(egg_fn="NamedMat")
    @classmethod
    def named(cls, name: StringLike) -> Matrix:  # type: ignore[empty-body]
        """
        Create a named matrix.
        """

    @method(egg_fn="MMul")
    def __matmul__(self, other: Matrix) -> Matrix:  # type: ignore[empty-body]
        """
        Matrix multiplication.
        """

    @method(egg_fn="nrows")
    def nrows(self) -> Dim:  # type: ignore[empty-body]
        """
        Number of rows in the matrix.
        """

    @method(egg_fn="ncols")
    def ncols(self) -> Dim:  # type: ignore[empty-body]
        """
        Number of columns in the matrix.
        """


@function(egg_fn="Kron")
def kron(a: Matrix, b: Matrix) -> Matrix:  # type: ignore[empty-body]
    """
    Kronecker product of two matrices.

    https://en.wikipedia.org/wiki/Kronecker_product#Definition
    """


A, B, C, D = vars_("A B C D", Matrix)
egraph.register(
    # The dimensions of a kronecker product are the product of the dimensions
    rewrite(kron(A, B).nrows()).to(A.nrows() * B.nrows()),
    rewrite(kron(A, B).ncols()).to(A.ncols() * B.ncols()),
    # The dimensions of a matrix multiplication are the number of rows of the first
    # matrix and the number of columns of the second matrix.
    rewrite((A @ B).nrows()).to(A.nrows()),
    rewrite((A @ B).ncols()).to(B.ncols()),
    # The dimensions of an identity matrix are the input dimension
    rewrite(Matrix.identity(n).nrows()).to(n),
    rewrite(Matrix.identity(n).ncols()).to(n),
)
egraph.register(
    # Multiplication by an identity matrix is the same as the other matrix
    rewrite(Matrix.identity(n) @ A).to(A),
    rewrite(A @ Matrix.identity(n)).to(A),
    # Matrix multiplication is associative
    rewrite(A @ (B @ C)).to((A @ B) @ C),
    rewrite((A @ B) @ C).to(A @ (B @ C)),
    # Kronecker product is associative
    rewrite(kron(A, kron(B, C))).to(kron(kron(A, B), C)),
    rewrite(kron(kron(A, B), C)).to(kron(A, kron(B, C))),
    # Kronecker product distributes over matrix multiplication
    rewrite(kron(A @ C, B @ D)).to(kron(A, B) @ kron(C, D)),
    rewrite(kron(A, B) @ kron(C, D)).to(
        kron(A @ C, B @ D),
        # Only when the dimensions match
        eq(A.ncols()).to(C.nrows()),
        eq(B.ncols()).to(D.nrows()),
    ),
)
egraph.register(
    # demand rows and columns when we multiply matrices
    rule(eq(C).to(A @ B)).then(
        A.ncols(),
        A.nrows(),
        B.ncols(),
        B.nrows(),
    ),
    # demand rows and columns when we take the kronecker product
    rule(eq(C).to(kron(A, B))).then(
        A.ncols(),
        A.nrows(),
        B.ncols(),
        B.nrows(),
    ),
)


# Define a number of dimensions
n = egraph.let("n", Dim.named("n"))
m = egraph.let("m", Dim.named("m"))
p = egraph.let("p", Dim.named("p"))

# Define a number of matrices
A = egraph.let("A", Matrix.named("A"))
B = egraph.let("B", Matrix.named("B"))
C = egraph.let("C", Matrix.named("C"))

# Set each to be a square matrix of the given dimension
egraph.register(
    union(A.nrows()).with_(n),
    union(A.ncols()).with_(n),
    union(B.nrows()).with_(m),
    union(B.ncols()).with_(m),
    union(C.nrows()).with_(p),
    union(C.ncols()).with_(p),
)
# Create an example which should equal the kronecker product of A and B
ex1 = egraph.let("ex1", kron(Matrix.identity(n), B) @ kron(A, Matrix.identity(m)))
rows = egraph.let("rows", ex1.nrows())
cols = egraph.let("cols", ex1.ncols())

egraph.run(20)

egraph.check(eq(B.nrows()).to(m))
egraph.check(eq(kron(Matrix.identity(n), B).nrows()).to(n * m))

# Verify it matches the expected result
simple_ex1 = egraph.let("simple_ex1", kron(A, B))
egraph.check(eq(ex1).to(simple_ex1))

ex2 = egraph.let("ex2", kron(Matrix.identity(p), C) @ kron(A, Matrix.identity(m)))

egraph.run(10)
# Verify it is not simplified
egraph.check_fail(eq(ex2).to(kron(A, C)))
egraph
