import random
import time

from qarton.circuit import UInt
from qarton.tests import auto_test
from sympy import legendre_symbol  # type: ignore
from sympy import randprime as _randprime  # type: ignore

from .mod_sqrt import FindNonResidue, MinusOneSqrtGeneral, minusone_sqrt_general


def randprime(a: int, b: int) -> int:
    """Typed wrapper around sympy's ``randprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _randprime(a, b)
    if not isinstance(result, int):
        raise TypeError(f"sympy.randprime returned {result!r}, expected an int")
    return result


def test_minusone_sqrt_general() -> None:
    """
    This test checks that our computation of the square root of -1 works in all
    congruence cases that we are interested in, including p = 1 mod 8.
    """

    random.seed(0)
    cong1 = 0
    cong2 = 0
    for _ in range(200):
        p = random.randrange(1 << 50)
        if not (p % 8 == 1 or p % 16 == 2):
            continue
        x, _ = minusone_sqrt_general(p, 50)
        if (x**2 + 1) % p == 0:
            if p % 8 == 1:
                cong1 += 1
            elif p % 16 == 2:
                cong2 += 1
    assert cong2 > 0
    assert cong1 > 0


def test_FindNonResidue() -> None:

    qc = FindNonResidue(100)

    for _ in range(10):
        p: int = randprime(1 << 98, 1 << 100)
        if p % 4 == 1:
            _, x = qc.simulate(UInt(p))
            assert legendre_symbol(x, p) == -1


def test_FindNonResidue_2() -> None:

    qc = FindNonResidue(50)
    t1 = time.time()
    auto_test(qc, nbr=20, stop_on_first_exception=True)
    print(time.time() - t1)


def test_MinusOneSqrtEven() -> None:

    qc = MinusOneSqrtGeneral(100)
    for _ in range(10):
        p: UInt = UInt(randprime(1 << 98, 1 << 99))
        if p % 4 == 1:
            assert qc.simulate(p) == qc.dummy_classical_function(p)
            assert qc.simulate(UInt(2 * p)) == qc.dummy_classical_function(UInt(p * 2))

    t1 = time.time()
    auto_test(qc, nbr=20, stop_on_first_exception=True)
    print(time.time() - t1)
