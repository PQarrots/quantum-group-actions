from qarton.tests import auto_test

from .montgomery_curve import ECMontgomery
from .montgomery_prod import AffMontgomeryProd, AffMontgomeryProdVar


def test_AffProd() -> None:
    ec = ECMontgomery(17, 3, 1)

    qc = AffMontgomeryProd(ec, 3)
    auto_test(qc, nbr=20)


def test_AffProdVar() -> None:

    qc = AffMontgomeryProdVar(17, 4)
    auto_test(qc, nbr=20)
