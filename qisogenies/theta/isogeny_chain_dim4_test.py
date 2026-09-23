from random import randint
from typing import cast

from qarton.circuit import AndFanoutResourceReporter
from qarton.modular_arithmetic import ModInt
from qarton.tests import run_real
from sympy import nextprime as _nextprime  # type: ignore

from qisogenies.theta.theta_util import CoordsDim4

from ..montgomery import (
    ECMontgomery,
)
from .isogeny_chain_dim4 import (
    InputData,
    IsogenyChain,
)


def nextprime(n: int, ith: int = 1) -> int:
    """Typed wrapper around sympy's ``nextprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _nextprime(n, ith)
    if not isinstance(result, int):
        raise TypeError(f"sympy.nextprime returned {result!r}, expected an int")
    return result


@run_real(IsogenyChain)
def test_IsogenyChain_instantiation() -> None:
    # first test with random numbers as input with no real geometric meaning
    n_bits = 100
    p: int = nextprime((1 << n_bits), ith=randint(1, 10))

    # Ts = tuple(ThetaPointDim4(
    #     tuple(ModInt(randint(1, p-1), p) for _ in range(16))
    # ) for _ in range(4))

    # zero_idx = randint(0, 3)
    # sigmas = [choice([1, 3]) for _ in range(3)]
    # sigmas.insert(zero_idx, 0)
    # sigmas = tuple(UInt(x) for x in sigmas)

    # splitting_type = randint(0, 2)

    # inputs = (
    #     Ts, sigmas, splitting_type
    # )

    e = n_bits - 3
    A = randint(1, p - 1)
    E = ECMontgomery(p, A, 1)
    E_twist = ECMontgomery(p, (-A) % p, 1)
    prod_np = cast(CoordsDim4, tuple(ModInt(randint(1, p - 1), p) for _ in range(16)))
    inv_np = cast(CoordsDim4, tuple(ModInt(randint(1, p - 1), p) for _ in range(16)))
    inv_dual_np = cast(
        CoordsDim4, (tuple(ModInt(randint(1, p - 1), p) for _ in range(16)))
    )
    # dim1_linear_transf = tuple(ModInt(randint(1, p - 1), p) for _ in range(4))
    inp = InputData(E, E_twist, prod_np, inv_np, inv_dual_np)

    qc = IsogenyChain(e, inp)

    r = AndFanoutResourceReporter(qc)
    print("Nbr of qubits", r.nbr_bits())
    print("Nbr of CCX gates or equivalent (log2)", r.ccx_count_log2())
    print("Most costly sub-circuits:")

    print(qc.nbr_qubits())
    print(AndFanoutResourceReporter(qc).ccx_count_log2())


# test_IsogenyChain_instantiation()
