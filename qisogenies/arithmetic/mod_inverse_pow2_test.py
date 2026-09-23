from qarton.tests import auto_test

from .mod_inverse_pow2 import IPModInvPow2, ModInvPow2, OOPModInvPow2


def test_ModInvPow2() -> None:

    qc = ModInvPow2(4)
    auto_test(qc)

    qc = ModInvPow2(50)
    auto_test(qc, nbr=20)


def test_OOPModInvPow2() -> None:
    qc = OOPModInvPow2(50)
    auto_test(qc, nbr=20, stop_on_first_exception=True)


def test_IPInvMod2() -> None:

    qc = IPModInvPow2(50)
    auto_test(qc, nbr=20)


def test_large() -> None:

    qc = IPModInvPow2(500)
    auto_test(qc, nbr=20)
