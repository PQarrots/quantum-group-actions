from qarton.tests import auto_test

from .montgomery_neg import AffMontgomeryNegVar, ControlledAffMontgomeryNegVar


def test_AffMontgomeryNegVar() -> None:

    qc = AffMontgomeryNegVar(17)
    auto_test(qc, nbr=20)


def test_ControlledAffMontgomeryNegVar() -> None:

    qc = ControlledAffMontgomeryNegVar(17)
    auto_test(qc, nbr=20)
