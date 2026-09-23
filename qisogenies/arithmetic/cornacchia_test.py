import random
import time

from qarton.circuit import UInt
from qarton.tests import auto_test, run_real

from .cornacchia import (
    Cornacchia,
    CornacchiaIterations,
    CornacchiaTestOnly,
)
from .cornacchia_classical import cornacchia_iterations


@run_real(CornacchiaIterations)
def test_CornacchiaIterations() -> None:
    qc = CornacchiaIterations(10)
    random.seed(0)
    for _ in range(1 << 5):
        m = UInt(random.randrange(3, 1 << 10))
        a = UInt(random.randrange(1, m))
        cornacchia_iterations(a, m)
        assert qc.simulate((a, m))[2] == cornacchia_iterations(a, m)


def test_CornacchiaEven() -> None:
    qc = Cornacchia(50)
    t1 = time.time()
    auto_test(qc, nbr=20)
    print(time.time() - t1)


def test_CornacchiaEvenTestOnly() -> None:
    qc = CornacchiaTestOnly(50)
    auto_test(qc, nbr=20)
