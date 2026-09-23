import random

from qarton.arithmetic import (
    ControlledCDKMAdder,
    ControlledCDKMComparator,
    ControlledGidneyConstantAdder,
    ControlledGidneyConstantComparator,
    GidneyConstantAdder,
    GidneyConstantComparator,
    GidneyConstantComparatorAdder,
)
from qarton.binary_operations import MCXWithBorrowedBits
from qarton.circuit import PriorityBackends, QartonBool
from qarton.modular_arithmetic import ModInt
from qarton.tests import auto_test

from .efficient_mod_arithmetic import (
    EfficientControlledModularAdder,
    EfficientModularDouble,
    SpecialControlledModularAdder,
    SpecialModularDouble,
)


def test_EfficientModularDouble() -> None:

    # secp256k1 curve
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    # p = 2**255 - 19
    # p = 17
    qc = EfficientModularDouble(p)
    auto_test(qc, nbr=1000)

    # must also work for small vcalues
    assert qc.simulate(ModInt(0, p)) == qc.dummy_classical_function(ModInt(0, p))
    assert qc.simulate(ModInt(2, p)) == qc.dummy_classical_function(ModInt(2, p))


def test_SpecialModularDouble() -> None:

    # 27 * 2**500 - 1 is prime
    p = 27 * 2**500 - 1

    qc = SpecialModularDouble(p)
    auto_test(qc, nbr=1000)

    # edge cases: small values, and values around p/2 and p-1, where the
    # conditional reduction step is/isn't triggered
    for x in (0, 1, 2, (p - 1) // 2, (p - 1) // 2 + 1, p - 1):
        assert qc.simulate(ModInt(x, p)) == qc.dummy_classical_function(ModInt(x, p))


def test_SpecialControlledModularAdder() -> None:

    random.seed(0)

    # 27 * 2**500 - 1 is prime
    p = 27 * 2**500 - 1

    qc = SpecialControlledModularAdder(p)

    auto_test(qc, nbr=1000)

    # corner case: x + y = p exactly. The high-order bits of y then equal f - 1
    # (not bigger than f), but the low-order bits are all 1; the result must
    # reduce to 0, not stay at p.
    xs = [(p - 1) // 2, (p - 1) // 2 + 1, p - 1] + [
        random.randrange(1, p) for _ in range(5)
    ]
    for x in xs:
        y = p - x
        t = (QartonBool(1), ModInt(x, p), ModInt(y, p))
        assert qc.simulate(t) == qc.dummy_classical_function(t)

    # corner case: x = 0
    for _ in range(20):
        y = random.randrange(1, p)
        t = (QartonBool(1), ModInt(0, p), ModInt(y, p))
        assert qc.simulate(t) == qc.dummy_classical_function(t)

    # corner case: y = 0
    for _ in range(20):
        y = random.randrange(1, p)
        t = (QartonBool(1), ModInt(y, p), ModInt(0, p))
        assert qc.simulate(t) == qc.dummy_classical_function(t)


BACKENDS = PriorityBackends(
    GidneyConstantAdder,
    ControlledGidneyConstantAdder,
    GidneyConstantComparator,
    GidneyConstantComparatorAdder,
    ControlledGidneyConstantComparator,
    ControlledCDKMComparator,
    ControlledCDKMAdder,
    MCXWithBorrowedBits,
)


def test_EfficientControlledModularAdder() -> None:

    random.seed(0)

    p = 2**255 - 19
    p = 115792089237316195423570985008687907853269984665640564039457584007908834671663

    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    # p = 2**255 + 2**254 + 3
    # p = 2**256 - 33354354778

    # p = 17
    #    qc = EfficientControlledModularAdder(p, padding=40, backends=BACKENDS)

    qc = EfficientControlledModularAdder(p, padding=40, backends=BACKENDS)
    auto_test(qc, nbr=1000, stop_on_first_exception=True)

    x, y = (
        102460838300123673682830523352172395490853289673244954067382771620347541264381,
        13331250937192521740740461656515512362416694992395609972074812387561293407282,
    )
    t = QartonBool(1), ModInt(x, p), ModInt(y, p)

    assert qc.simulate(t) == qc.dummy_classical_function(t)
