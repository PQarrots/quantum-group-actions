from qarton.circuit import UInt
from qarton.signed_arithmetic import SInt
from qarton.tests import auto_test

from .mod4 import MinusXMod4, XMod4


def test_XMod4() -> None:
    qc = XMod4(4)
    auto_test(qc)


def test_MinusXMod4() -> None:
    qc = MinusXMod4(4)
    auto_test(qc)


def test_MinusXMod4_2() -> None:
    qc = MinusXMod4(394)
    assert qc.simulate(SInt(1, 394)) == qc.dummy_classical_function(SInt(1, 394))
    assert qc.simulate(SInt(1, 394)) == (SInt(1, 394), UInt(3))
