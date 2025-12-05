from sympy import Array, Symbol, tensorcontraction, tensorproduct

# Parameters
Bp = Array([Symbol(f"bp{i}") for i in range(1, 5)])
Bpp = Array([Symbol(f"bpp{i}") for i in range(1, 5)])

Q = Array([Symbol(f"q{i}") for i in range(1, 13)])
QM = Q.reshape(4, 3).transpose()

# Projections (3-vector each)
yip = tensorcontraction(tensorproduct(QM, Bp), (1, 2))
yipp = tensorcontraction(tensorproduct(QM, Bpp), (1, 2))


def cross(a: Array, b: Array):
    """
    3d cross product
    https://reference.wolfram.com/language/ref/Cross.html
    """
    return Array([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])


def norm3(arr: Array):
    x, y, z = arr[0], arr[1], arr[2]
    return (x**2 + y**2 + z**2) ** (0.5)


FunctionBending = (norm3(cross(yip, yipp)) / (norm3(yip) ** 3)) ** 2
print(FunctionBending)
