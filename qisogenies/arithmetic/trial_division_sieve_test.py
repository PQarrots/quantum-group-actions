from qarton.tests import auto_test

from .trial_division_sieve import (
    TrialDivisionSieveNotTwo,
)


def test_TrialDivisionSieveNotTwo() -> None:
    qc = TrialDivisionSieveNotTwo(10, 5)
    auto_test(qc, nbr=20)
