"""
Ciruits to check if an input integer is divisible by small primes. This is performed
via a sequence of Euclidean divisions by said primes.
"""

import functools

from qarton.arithmetic import EuclideanDividerConstant
from qarton.binary_operations import qc_mcx_neg
from qarton.circuit import (
    BitVector,
    Circuit,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_bool_reg,
)
from sympy import prime  # type: ignore

__all__ = ["TrialDivisionSieveNotTwo"]


@functools.cache
def prime_cached(i: int) -> int:
    return prime(i)  # type: ignore


@dummify
@memoize
class TrialDivisionSieveNotTwo(Circuit[UInt, tuple[UInt, BitVector]]):
    """
    Determine if an input integer is divisible by the k first primes (not 2).
    """

    def __init__(self, n: int, k: int) -> None:
        """
        :param n: Size of the integer.
        :type n: int
        :param k: Number of primes.
        :type k: int
        """
        super().__init__()
        lprimes: list[int] = [prime_cached(i) for i in range(2, k + 1)]  # type: ignore
        self.lprimes = lprimes

        xreg = self.add_input_output(UIntType(n))
        out = self.add_anc_output(len(lprimes))

        for i, pp in enumerate(lprimes):
            # yes this destroys xreg, BUT using the inverse will restore a register
            # with the same positions.
            sub_qc = EuclideanDividerConstant(pp, len(xreg))
            quo, rem = self.append(sub_qc, xreg)
            # use mcx on rem
            qc_mcx_neg(rem, qc_bool_reg(out[i]))

            (xreg,) = self.append(sub_qc.inverse(), quo, rem)
        self.remap(xreg, out)

    def dummy_classical_function(self, args: UInt) -> tuple[UInt, BitVector]:
        res = BitVector((args % pp == 0) for pp in self.lprimes)
        return args, res

    def dummy_classical_function_inverse(self, args: tuple[UInt, BitVector]) -> UInt:
        x, _ = args
        return x


def is_almostprime_but_not_two(n: int, k: int) -> bool:
    """
    Return True if the number is not divisible by any of the k first prime
    numbers (except 2). Return 0 otherwise.

    :param n: Input integer.
    :param k: Number of primes to test.
    """
    for i in range(2, k + 1):
        p: int = prime_cached(i)
        if n % p == 0:
            return False
    return True
