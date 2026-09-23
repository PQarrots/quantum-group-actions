from random import randint

from qarton.circuit import QartonBool, UInt
from qarton.modular_arithmetic import ModInt
from qarton.tests import run_real
from sympy import nextprime as _nextprime  # type: ignore

from .splitting_dim4 import (
    SplittingDim4,
)
from .theta_dim4 import (
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


@run_real(SplittingDim4)
def test_SplittingDim4() -> None:
    for _ in range(20):
        p: int = nextprime((1 << 30), ith=randint(1, 10))

        coords = [ModInt(x, p) for x in (0, 1, 2, p - 3)] + [
            ModInt(randint(1, p - 1), p) for _ in range(12)
        ]
        P = ThetaStructureDim4(tuple(coords))
        splitting_type = UInt(randint(0, 2))

        inputs = (P, splitting_type, QartonBool(0))

        qc = SplittingDim4(p)
        classical = qc.dummy_classical_function(inputs)
        simulated = qc.simulate(inputs)

        assert inputs == qc.dummy_classical_function_inverse(classical)
        # print(classical)
        # print(simulated)
        assert simulated[0].coords == classical[0].coords
        assert simulated[1:] == classical[1:]
