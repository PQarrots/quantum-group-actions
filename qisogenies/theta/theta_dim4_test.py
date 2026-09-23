from random import randint

from qarton.modular_arithmetic import ModInt
from qarton.tests import run_real
from sympy import nextprime as _nextprime  # type: ignore

from qisogenies.theta.theta_util import CoordsDim4  # type: ignore

from .theta_dim4 import (
    ThetaArithmeticPrecomputation,
    ThetaDouble,
    ThetaPointDim4,
    ThetaStructureDim4,
)


def nextprime(n: int, ith: int = 1) -> int:
    """Typed wrapper around sympy's ``nextprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _nextprime(n, ith)
    if not isinstance(result, int):
        raise TypeError(f"sympy.nextprime returned {result!r}, expected an int")
    return result


@run_real(ThetaArithmeticPrecomputation)
def test_ThetaArithmeticPrecomputation() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))
    inv_dual_np = tuple(ModInt(i + 1, p) for i in range(16))

    qc = ThetaArithmeticPrecomputation(p)

    inputs = CoordsDim4(inv_dual_np)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    assert simulated == classical


@run_real(ThetaDouble)
def test_ThetaDouble() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))
    while True:
        try:
            null_point = tuple(ModInt(randint(1, p - 1), p) for _ in range(16))
            theta_str = ThetaStructureDim4(null_point)
            break
        except ZeroDivisionError:
            continue
    P_coords = CoordsDim4(ModInt(randint(0, p - 1), p) for _ in range(16))
    P = ThetaPointDim4(P_coords, theta_str)

    qc = ThetaDouble(p)

    inputs = P, theta_str.inverse_null_point, theta_str.inverse_dual_null_point_codomain
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    assert simulated == classical
