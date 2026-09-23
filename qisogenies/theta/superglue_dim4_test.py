from random import choice, randint

from qarton.circuit import QartonBool, UInt
from qarton.modular_arithmetic import ModInt
from qarton.tests import run_real
from sympy import nextprime as _nextprime  # type: ignore

from qisogenies.montgomery import (
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariableType,
    ECMontgomery,
)
from qisogenies.theta.theta_util import CoordsDim4

from .superglue_dim4 import (
    EllipticPointsToBarycentric,
    N2Matrix_Superglue4D,
    Superglue4DEvaluation,
    Superglue4DRabbitCodomain,
)


def nextprime(n: int, ith: int = 1) -> int:
    """Typed wrapper around sympy's ``nextprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _nextprime(n, ith)
    if not isinstance(result, int):
        raise TypeError(f"sympy.nextprime returned {result!r}, expected an int")
    return result


@run_real(EllipticPointsToBarycentric)
def test_EllipticPointsToBarycentric() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))
    E = ECMontgomery(p, randint(0, p - 1), 1)
    P = AffMontgomeryPointVariable(randint(1, p - 1), randint(1, p - 1), True, p)
    Q = AffMontgomeryPointVariable(randint(1, p - 1), randint(1, p - 1), True, p)
    cP = QartonBool(randint(0, 1))
    cQ = QartonBool(randint(0, 1))
    inputs = (P, Q, cP, cQ)

    qc = EllipticPointsToBarycentric(E)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)

    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(N2Matrix_Superglue4D)
def test_N2Matrix_Superglue4D() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))

    x = CoordsDim4(ModInt(randint(0, p - 1), p) for _ in range(16))

    import random

    zero_idx = random.randint(0, 3)
    sigmas = [random.choice([1, 3]) for _ in range(3)]
    sigmas.insert(zero_idx, 0)
    sigmas_as_tuple = tuple(UInt(x) for x in sigmas)
    assert len(sigmas_as_tuple) == 4

    inputs = (x, sigmas_as_tuple)

    qc = N2Matrix_Superglue4D(p)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(Superglue4DEvaluation)
def test_Superglue4DEvaluation() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))

    E = ECMontgomery(p, randint(0, p - 1), 1)
    np_1D = tuple(ModInt(randint(1, p - 1), p) for _ in range(2))
    import itertools

    theta_prod_np = CoordsDim4(
        np_1D[i1] * np_1D[i2] * np_1D[i3] * np_1D[i4]
        for i1, i2, i3, i4 in itertools.product(range(2), range(2), range(2), range(2))
    )

    PP = tuple(AffMontgomeryPointVariableType(p).random_value() for _ in range(4))
    TT = tuple(AffMontgomeryPointVariableType(p).random_value() for _ in range(4))
    cP = QartonBool(randint(0, 1))
    cQ = QartonBool(randint(0, 1))
    inv_np_cod = tuple(ModInt(randint(1, p - 1), p) for _ in range(16))

    zero_idx = randint(0, 3)
    sigmas = [choice([1, 3]) for _ in range(3)]
    sigmas.insert(zero_idx, 0)

    inputs = (PP, TT, cP, cQ, inv_np_cod, tuple(UInt(x) for x in sigmas))

    qc = Superglue4DEvaluation(theta_prod_np, E)
    classical = qc.dummy_classical_function(inputs)  # type: ignore
    simulated = qc.simulate(inputs)  # type: ignore
    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(Superglue4DEvaluation)
def test_Superglue4DEvaluation_2() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))

    E = ECMontgomery(p, randint(0, p - 1), 1)
    np_1D = tuple(ModInt(randint(1, p - 1), p) for _ in range(2))
    import itertools

    theta_prod_np = CoordsDim4(
        np_1D[i1] * np_1D[i2] * np_1D[i3] * np_1D[i4]
        for i1, i2, i3, i4 in itertools.product(range(2), range(2), range(2), range(2))
    )

    PP = tuple(AffMontgomeryPointVariableType(p).random_value() for _ in range(4))
    assert len(PP) == 4

    tt = list(PP)
    for i in range(4):
        if randint(0, 1):
            tt[i] = AffMontgomeryPointVariableType(p).random_value()
    tt_as_tuple = tuple(tt)
    assert len(tt_as_tuple) == 4

    cP = QartonBool(randint(0, 1))
    cQ = QartonBool(randint(0, 1))
    inv_np_cod = CoordsDim4(ModInt(randint(1, p - 1), p) for _ in range(16))

    zero_idx = randint(0, 3)
    sigmas = [choice([1, 3]) for _ in range(3)]
    sigmas.insert(zero_idx, 0)
    sigmas_as_tuple = tuple(UInt(x) for x in sigmas)
    assert len(sigmas_as_tuple) == 4

    inputs = (PP, tt_as_tuple, cP, cQ, inv_np_cod, sigmas_as_tuple)

    qc = Superglue4DEvaluation(theta_prod_np, E)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(Superglue4DRabbitCodomain)
def test_Superglue4DRabbitCodomain() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))

    HSK = tuple(
        CoordsDim4(ModInt(randint(1, p - 1), p) for _ in range(16)) for _ in range(5)
    )
    assert len(HSK) == 5
    rabbit_type = UInt(randint(0, 3))

    inputs = (HSK, rabbit_type)

    qc = Superglue4DRabbitCodomain(p)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    assert inputs == qc.dummy_classical_function_inverse(classical)
    # print(classical)
    # print(simulated)
    assert simulated == classical


# test_EllipticPointsToBarycentric()
# test_N2Matrix_Superglue4D()
# test_Superglue4DEvaluation()
# test_Superglue4DEvaluation_2()
# test_Superglue4DRabbitCodomain()
