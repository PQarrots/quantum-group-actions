from random import randint

from qarton.circuit import QartonBool
from qarton.modular_arithmetic import ModInt
from qarton.tests import run_real
from sympy import nextprime as _nextprime  # type: ignore

from qisogenies.theta.theta_util import CoordsDim4  # type: ignore

from .isogeny_dim4 import (
    Theta2IsogenyDim4Generic_Codomain,
    Theta2IsogenyDim4Generic_Evaluation,
    Theta2IsogenyDim4Second_Codomain,
)
from .theta_dim4 import (
    ThetaPointDim4,
)


def nextprime(n: int, ith: int = 1) -> int:
    """Typed wrapper around sympy's ``nextprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _nextprime(n, ith)
    if not isinstance(result, int):
        raise TypeError(f"sympy.nextprime returned {result!r}, expected an int")
    return result


@run_real(Theta2IsogenyDim4Generic_Evaluation)
def test_Theta2IsogenyDim4Generic_Evaluation() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))

    P = ThetaPointDim4(CoordsDim4(ModInt(randint(1, p - 1), p) for _ in range(16)))
    inv_np_cod = CoordsDim4(ModInt(randint(1, p - 1), p) for _ in range(16))
    inputs = (P, inv_np_cod)

    qc = Theta2IsogenyDim4Generic_Evaluation(p)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)

    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(Theta2IsogenyDim4Generic_Codomain)
def test_Theta2IsogenyDim4Generic_Codomain() -> None:
    p = nextprime((1 << 10), ith=randint(1, 10))

    HSK = tuple(
        CoordsDim4(ModInt(randint(1, p - 1), p) for _ in range(16)) for _ in range(4)
    )
    inputs = HSK
    assert len(inputs) == 4

    qc = Theta2IsogenyDim4Generic_Codomain(p)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)

    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


#############################
def test_Theta2IsogenyDim4Generic_Codomain_2() -> None:
    ## now with find_aut
    p = nextprime((1 << 10), ith=randint(1, 10))

    HSK = (
        tuple(ModInt(x, p) for x in (0, 1, 2, 0, 4, 3, 0))
        + tuple(ModInt(randint(1, p - 1), p) for _ in range(9)),
        tuple(
            ModInt(x, p) for x in (1, 1, 2, 2, 1, 1, 2, 3, 4, 5, 1, 8, 10, 2, 12, 13)
        ),
        tuple(ModInt(randint(1, p - 1), p) for _ in range(16)),
        tuple(ModInt(randint(1, p - 1), p) for _ in range(16)),
    )
    inputs = tuple(CoordsDim4(_x) for _x in HSK)
    assert len(inputs) == 4

    qc = Theta2IsogenyDim4Generic_Codomain(p, find_aut=True)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)

    assert inputs == qc.dummy_classical_function_inverse(classical)

    assert simulated == classical, "\n" + str(simulated[1]) + "\n" + str(classical[1])


@run_real(Theta2IsogenyDim4Second_Codomain)
def test_Theta2IsogenyDim4Second_Codomain() -> None:
    p = nextprime((1 << 10), ith=randint(1, 10))

    HSK = [[ModInt(randint(1, p - 1), p) for _ in range(16)] for _ in range(5)]
    Taux_is_T3p4 = QartonBool(randint(0, 1))
    for _ in range(2):
        HSK[randint(0, 4)][randint(0, 15)] = ModInt(0, p)

    _t = tuple(CoordsDim4(x for x in T) for T in HSK)
    assert len(_t) == 5
    inputs = (_t, Taux_is_T3p4)

    qc = Theta2IsogenyDim4Second_Codomain(p)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)

    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical[-1])
    # print(simulated[-1])
    assert simulated == classical


# for _ in range(50):
# test_Theta2IsogenyDim4Generic_Codomain()
# test_Theta2IsogenyDim4Generic_Codomain_2()
# test_Theta2IsogenyDim4Generic_Evaluation()
# test_Theta2IsogenyDim4Second_Codomain()
