from random import randint

from qarton.circuit import QartonBool
from qarton.modular_arithmetic import ModInt, ModIntType
from qarton.tests import run_real
from sympy import nextprime as _nextprime  # type: ignore

############# hadamard #######################
from .theta_util import (
    BatchedInversionAdd,
    CoordsDim2,
    CoordsDim4,
    IsAnyCoordZero,
    IsSquare,
    MontgomeryToJInvariantDiv256,
    ThetaHadamard,
    ThetaHadamardBitflipped,
    ThetaHadamardDim2,
)


def nextprime(n: int, ith: int = 1) -> int:
    """Typed wrapper around sympy's ``nextprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _nextprime(n, ith)
    if not isinstance(result, int):
        raise TypeError(f"sympy.nextprime returned {result!r}, expected an int")
    return result


@run_real(ThetaHadamardDim2)
def test_ThetaHadamardDim2() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))
    coords = CoordsDim2(ModInt(randint(0, p - 1), p) for _ in range(4))

    qc = ThetaHadamardDim2(p)
    classical = qc.dummy_classical_function(coords)
    simulated = qc.simulate(coords)
    assert simulated == classical


@run_real(ThetaHadamard)
def test_ThetaHadamard() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))
    coords = CoordsDim4(ModInt(randint(0, p - 1), p) for _ in range(16))

    qc = ThetaHadamard(p)
    classical = qc.dummy_classical_function(coords)
    simulated = qc.simulate(coords)
    assert simulated == classical


@run_real(ThetaHadamardBitflipped)
def test_ThetaHadamrdBitflipped() -> None:
    p = nextprime((1 << 30), ith=randint(1, 10))
    coords = CoordsDim4(ModInt(randint(0, p - 1), p) for _ in range(16))

    qc = ThetaHadamardBitflipped(p)
    classical = qc.dummy_classical_function(coords)
    simulated = qc.simulate(coords)
    assert simulated == classical


################ inversions #########################
@run_real(BatchedInversionAdd)
def test_BatchedInversionAdd() -> None:
    p = nextprime((1 << 30), ith=randint(1, 20))
    l = randint(10, 20)
    x = tuple(ModInt(randint(0, p), p) for _ in range(l))
    y = tuple(ModInt(randint(0, p), p) for _ in range(l))
    inputs = (x, y)

    qc = BatchedInversionAdd(p, l)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(IsAnyCoordZero)
def test_IsAnyCoordZero() -> None:
    p = 7
    l = randint(10, 20)
    which_coords = tuple(i for i in range(l) if randint(0, 1))
    x = tuple(ModInt(randint(0, p - 1), p) for _ in range(l))

    c = QartonBool(randint(0, 1))
    inputs = (x, c)

    qc = IsAnyCoordZero(p, l, which_coords)
    classical = qc.dummy_classical_function(inputs)
    simulated = qc.simulate(inputs)
    # print(classical)
    # print(simulated)
    assert simulated == classical


@run_real(IsSquare)
def test_IsSquare() -> None:

    for _ in range(50):
        p = nextprime((1 << 30), ith=randint(1, 10))

        x = ModIntType(p).random_value()
        inputs = x

        qc = IsSquare(p)
        classical = qc.dummy_classical_function(inputs)
        simulated = qc.simulate(inputs)
        # print(classical)
        # print(simulated)
        assert simulated == classical


@run_real(MontgomeryToJInvariantDiv256)
def test_MontgomeryToJInvariantDiv256() -> None:
    for _ in range(1):
        p = nextprime((1 << 30), ith=randint(1, 10))

        A = ModIntType(p).random_value()
        inputs = A

        qc = MontgomeryToJInvariantDiv256(p)
        classical = qc.dummy_classical_function(inputs)
        simulated = qc.simulate(inputs)
        # print(classical)
        # print(simulated)
        assert simulated == classical, f"{simulated = }, {classical = }"
