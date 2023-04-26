"""
Matrix multiplication and Kronecker product.
============================================
"""
from __future__ import annotations

from egg_smol import *

egraph = EGraph()


@egraph.class_
class Dim(BaseExpr):
    """
    A dimension of a matix.

    >>> Dim(3) * Dim.named("n")
    Dim(3) * Dim.named("n")
    """

    @egraph.method(egg_fn="Lit")
    def __init__(self, value: i64Like) -> None:
        ...

    @egraph.method(egg_fn="NamedDim")
    @classmethod
    def named(cls, name: StringLike) -> Dim:  # type: ignore[empty-body]
        ...

    @egraph.method(egg_fn="Times")
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


@egraph.class_(egg_sort="MExpr")
class Matrix(BaseExpr):
    @egraph.method(egg_fn="Id")
    @classmethod
    def identity(cls, dim: Dim) -> Matrix:  # type: ignore[empty-body]
        """
        Create an identity matrix of the given dimension.
        """
        ...

    @egraph.method(egg_fn="NamedMat")
    @classmethod
    def named(cls, name: StringLike) -> Matrix:  # type: ignore[empty-body]
        """
        Create a named matrix.
        """
        ...

    @egraph.method(egg_fn="MMul")
    def __matmul__(self, other: Matrix) -> Matrix:  # type: ignore[empty-body]
        """
        Matrix multiplication.
        """
        ...

    @egraph.method(egg_fn="nrows")
    def nrows(self) -> Dim:  # type: ignore[empty-body]
        """
        Number of rows in the matrix.
        """
        ...

    @egraph.method(egg_fn="ncols")
    def ncols(self) -> Dim:  # type: ignore[empty-body]
        """
        Number of columns in the matrix.
        """
        ...


@egraph.function(egg_fn="Kron")
def kron(a: Matrix, b: Matrix) -> Matrix:  # type: ignore[empty-body]
    """
    Kronecker product of two matrices.

    https://en.wikipedia.org/wiki/Kronecker_product#Definition
    """
    ...


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
        let("demand1", A.ncols()),
        let("demand2", A.nrows()),
        let("demand3", B.ncols()),
        let("demand4", B.nrows()),
    ),
    # demand rows and columns when we take the kronecker product
    rule(eq(C).to(kron(A, B))).then(
        let("demand1", A.ncols()),
        let("demand2", A.nrows()),
        let("demand3", B.ncols()),
        let("demand4", B.nrows()),
    ),
)


# Define a number of dimensions
n = egraph.define("n", Dim.named("n"))
m = egraph.define("m", Dim.named("m"))
p = egraph.define("p", Dim.named("p"))

# Define a number of matrices
A = egraph.define("A", Matrix.named("A"))
B = egraph.define("B", Matrix.named("B"))
C = egraph.define("C", Matrix.named("C"))

# Set each to be a square matrix of the given dimension
egraph.register(
    set_(A.nrows()).to(n),
    set_(A.ncols()).to(n),
    set_(B.nrows()).to(m),
    set_(B.ncols()).to(m),
    set_(C.nrows()).to(p),
    set_(C.ncols()).to(p),
)
# Create an example which should equal the kronecker product of A and B
ex1 = egraph.define("ex1", kron(Matrix.identity(n), B) @ kron(A, Matrix.identity(m)))
rows = egraph.define("rows", ex1.nrows())
cols = egraph.define("cols", ex1.ncols())

egraph.run(20)

egraph.check(eq(B.nrows()).to(m))
egraph.check(eq(kron(Matrix.identity(n), B).nrows()).to(n * m))

# Verify it matches the expected result
# TODO
simple_ex1 = egraph.define("simple_ex1", kron(A, B))
egraph.check(eq(ex1).to(simple_ex1))

ex2 = egraph.define("ex2", kron(Matrix.identity(p), C) @ kron(A, Matrix.identity(m)))

egraph.run(10)
# Verify it is not simplified
egraph.check_fail(eq(ex2).to(kron(A, C)))
