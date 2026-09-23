from qarton.circuit import AndFanoutResourceReporter, PriorityBackends
from qarton.tests import auto_test

from qisogenies.arithmetic.efficient_mod_arithmetic import (
    EfficientControlledModularAdder,
    EfficientModularDouble,
)

from .montgomery_add import AffMontgomeryAdd, AffMontgomeryAddIPVar
from .montgomery_curve import ECMontgomery


def test_AffAdd() -> None:
    ec = ECMontgomery(17, 3, 1)

    qc = AffMontgomeryAdd(ec)
    auto_test(qc, nbr=20)


def test_AffOps() -> None:

    # we gain 0.5 in exponent using approximate mod arithmetic
    qc = AffMontgomeryAddIPVar(
        (1 << 255) - 19,
        backends=PriorityBackends(
            EfficientControlledModularAdder, EfficientModularDouble
        ),
    )
    print(qc.nbr_qubits())
    print(AndFanoutResourceReporter(qc).ccx_count_log2())

    qc = AffMontgomeryAddIPVar((1 << 255) - 19)
    print(qc.nbr_qubits())
    print(AndFanoutResourceReporter(qc).ccx_count_log2())
