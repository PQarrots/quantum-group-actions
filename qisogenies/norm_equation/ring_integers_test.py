# type: ignore

import pytest
from qarton.tests import run_real

from .ring_integers import Conjugate, Multiply, Norm, RingInteger, Trace


def test_IdealElt() -> None:
    pytest.importorskip("sage.all")
    from sage.all import NumberField, var

    # test product formula
    p = 19  # 3 mod 4
    K = NumberField(name="pi", polynomial=var("x") ** 2 + p)
    pi = K.gens()[0]
    omega = (1 + pi) / 2

    a = 2 + omega
    b = 4 - 3 * omega
    c = a * b

    aelt = RingInteger(p, 6, 2, 1)
    belt = RingInteger(p, 6, 4, -3)
    celt = aelt * belt

    assert celt.a.v + celt.b.v * omega == c

    # test norm
    assert celt.norm() == c.norm()

    # test conjugate
    celt = celt.conjugate()
    assert celt.a.v + celt.b.v * omega == c.conjugate()


@run_real(Conjugate)
def test_Conjugate() -> None:
    p = 19
    w = 5
    qc = Conjugate(p, w)
    for i in range(1 << w):
        for j in range(1 << w):
            aelt = RingInteger(p, w, i, j)
            assert qc.dummy_classical_function(aelt) == qc.simulate(aelt)


@run_real(Multiply)
def test_Multiply() -> None:
    p = 11
    w = 3
    qc = Multiply(p, w)
    for i in range(1 << w):
        for j in range(1 << w):
            for k in range(1 << w):
                for t in range(1 << w):
                    aelt = RingInteger(p, w, i, j)
                    belt = RingInteger(p, w, k, t)
                    assert qc.dummy_classical_function((aelt, belt)) == qc.simulate(
                        (aelt, belt)
                    )


@run_real(Norm)
def test_Norm() -> None:
    p = 19
    w = 3

    qc = Norm(p, w)
    for i in range(1 << w):
        for j in range(1 << w):
            aelt = RingInteger(p, w, i, j)
            assert qc.dummy_classical_function(aelt) == qc.simulate(aelt)


@run_real(Trace)
def test_Trace() -> None:
    p = 19
    w = 3
    qc = Trace(p, w)
    for i in range(1 << w):
        for j in range(1 << w):
            aelt = RingInteger(p, w, i, j)
            assert qc.dummy_classical_function(aelt) == qc.simulate(aelt)
