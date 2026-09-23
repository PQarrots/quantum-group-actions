"""
Circuit to compute a modular square root of -1, in the cases that
we need for Cornacchia.

"""

from functools import cache

from qarton.advanced_arithmetic import qc_jacobi_symbol
from qarton.arithmetic import (
    qc_cdecr_uint,
    qc_cincr_uint,
    qc_decr_uint,
    qc_div_uint,
    qc_div_uncompute_uint,
    qc_eq_uint,
    qc_incr_uint,
)
from qarton.binary_operations import (
    qc_copy,
    qc_copy_erase,
    qc_cshift_right,
    qc_cswap,
    qc_cxor,
    qc_xor,
)
from qarton.circuit import (
    BackendSpecifier,
    BitVector,
    BitVectorType,
    BoolType,
    Circuit,
    EmptyBackendSpecifier,
    UInt,
    UIntType,
    dummify,
    qc_bool_reg,
    qc_reg_cast,
)
from qarton.dispatch import qc_mul, qc_mul_erase
from qarton.mod_arithmetic_variable import (
    qc_add_modvar,
    qc_powadd_modvar,
)
from sympy import nextprime as _nextprime  # type: ignore

try:
    # Try to import gmpy2 to perform exponentiation, instead of python's pow, because
    # it's a bit faster for big integers.
    import gmpy2  # type: ignore

    def modexp(a: int, e: int, m: int) -> int:
        if m % 2 == 0:
            raise ValueError("Modular exponentiation with even modulus is disallowed")
        return int(gmpy2.powmod(a, e, m))  # type: ignore
except ImportError:

    def modexp(a: int, e: int, m: int) -> int:
        if m % 2 == 0:
            raise ValueError("Modular exponentiation with even modulus is disallowed")
        return pow(a, e, m)


__all__ = ["FindNonResidue", "MinusOneSqrtGeneral", "minusone_sqrt_general"]

PRIME_SIZE = 40
NUMBER_OF_PRIMES = 40
# start at 2**40 and take 40 primes


def nextprime(n: int, ith: int = 1) -> int:
    """Typed wrapper around sympy's ``nextprime``, whose stub return type is
    broader (``int | array[int] | None``) than what it actually returns for
    integer input."""
    result = _nextprime(n, ith)
    if not isinstance(result, int):
        raise TypeError(f"sympy.nextprime returned {result!r}, expected an int")
    return result


@cache
def first_primes_above() -> tuple[int, ...]:
    """
    Return the ``NUMBER_OF_PRIMES`` smallest primes strictly greater than
    ``1 << PRIME_SIZE``. The result is cached across calls.
    """
    primes: list[int] = []
    p = 1 << PRIME_SIZE
    for _ in range(NUMBER_OF_PRIMES):
        p = nextprime(p)
        primes.append(p)
    return tuple(primes)


def jacobi_symbol_iterations(a: int, m: int, n: int) -> int:
    """Function that reproduces exactly the output of Qarton's Jacobi symbol
    circuit. In particular, it fails in the same way as the circuit when the two
    inputs are not coprime (which is not handled).

    :param a: The first input
    :type a: int
    :param m: The second input
    :type m: int
    :param n: The bit-size of the numbers (determines the number of iterations)
    :type n: int
    :return: The Jacobi symbol (if the numbers are coprime) or an invalid value otherwise,
             only 1 or -1
    :rtype: int
    """
    t = 0  # the bit the circuit accumulates into its output register
    for _ in range(2 * n):
        odd = a & 1  # circuit's "b" control, captured before the swap
        if odd and a < m:
            a, m = m, a
            if a % 4 == 3 and m % 4 == 3:
                t ^= 1
        if odd:
            a -= m
        if a == 0:
            break
        a >>= 1
        v = m % 8
        if v == 3 or v == 5:
            t ^= 1
    return 1 if t == 0 else -1


def find_nonresidue(p: int, n: int) -> int:
    """
    Finds a nonresidue mod p. Only guaranteed to work if p is a prime.

    This function matches the output of ``FindNonResidue``, so it runs the
    ``jacobi_symbol_iterations`` internally instead of a true Jacobi symbol.
    """
    if n < 40:
        raise ValueError()
    if p <= 2:
        raise ValueError()

    a = 2
    l = first_primes_above()
    for a in l:
        ls = jacobi_symbol_iterations(a, p, n)
        if ls == -1:
            return a
    return 0


@dummify
class FindNonResidue(Circuit[UInt, tuple[UInt, UInt]]):
    """
    Find an integer which is not a residue mod p (p is an input which
    is odd and = 1 mod 4).

    The integer is selected from a built-in list of primes, which are
    big enough to ensure that with large probability, one of them will be
    coprime with the input p, even if p is not prime.
    """

    def __init__(self, n: int) -> None:
        """
        :param n: Number of bits of the input.
        :type n: int
        """
        if n < 50:
            raise Exception("n too small")
        super().__init__()
        self.n = n
        p_reg = self.add_input_output(UIntType(n))
        out_reg = self.add_anc_output(UIntType(n))

        trials = first_primes_above()
        k = len(trials)

        # -------------
        x_reg = self.add_anc(UIntType(n))

        # counter: increment when we have a non residue,
        # and write x_reg to output only if counter is 1
        # exactly
        ctr = self.add_anc(UIntType(1 + (k + 2).bit_length()))

        # outputs of the Jacobi symbol for all integers
        # it must be 1 for a nonresidue
        bits = self.add_anc(k)
        tmp = self.add_anc(BoolType())

        for i in range(k):
            # here ctr is always a multiple of 2
            qc_xor(UInt(trials[i]), x_reg)
            qc_jacobi_symbol(x_reg, p_reg, qc_bool_reg(bits[i]))
            qc_xor(UInt(trials[i]), x_reg)
            qc_cincr_uint(qc_bool_reg(bits[i]), ctr)
            # the only way for ctr to become 1 is if it was 0
            # when entering the loop and was increased
            qc_eq_uint(UInt(1), ctr, tmp)
            qc_cxor(tmp, UInt(trials[i]), out_reg)
            qc_eq_uint(UInt(1), ctr, tmp)
            qc_cincr_uint(qc_bool_reg(bits[i]), ctr)
            self.assert_anc(tmp)

        for i in reversed(range(k)):
            qc_cdecr_uint(qc_bool_reg(bits[i]), ctr)
            qc_cdecr_uint(qc_bool_reg(bits[i]), ctr)
            qc_xor(UInt(trials[i]), x_reg)
            qc_jacobi_symbol(x_reg, p_reg, qc_bool_reg(bits[i]))
            qc_xor(UInt(trials[i]), x_reg)

        self.assert_anc(x_reg, bits, ctr, tmp)

    def validate_input(self, args: UInt) -> bool:
        return args > 2 and args % 2 == 1

    def dummy_classical_function(self, args: UInt) -> tuple[UInt, UInt]:
        return args, UInt(find_nonresidue(args, self.n))

    def dummy_classical_function_inverse(self, args: tuple[UInt, UInt]) -> UInt:
        return args[0]


def minusone_sqrt_general(p: int, n: int) -> tuple[int, BitVector]:
    """
    Computes a square root of -1 modulo p if p is:

    * twice a number equal to 1 mod 4
    * equal to 1 mod 4 itself

    If the odd number (p or p/2) is not a prime, the result may not be correct, but
    it still finishes.
    """
    pp = p // 2 if p % 2 == 0 else p
    x = find_nonresidue(pp, n)
    y = modexp(x, (pp - 1) // 4, pp)
    res1 = y
    res2 = (pp + y * (1 + pp)) % p
    if p % 2 == 0:
        return res2, BitVector.from_int(res1, n)
    else:
        return res1, BitVector.from_int(res2, n)


@dummify
class MinusOneSqrtGeneral(Circuit[UInt, tuple[UInt, UInt, BitVector]]):
    """
    Compute a square-root of an integer p such that p = 2 mod 8 or = 1 mod 4.

    The circuit is only guaranteed to return the square root of -1 if
    p is twice a prime, or a prime. In other cases it may return any value.

    This circuit has exceptions in the following cases:

    - the output of FindNonResidue is 0 (we took enough trials to ensure that
      this happens with low probability)

    - the output of FindNonResidue is not coprime with the input p, leading
      to an error in the modular exponentiation. This should not happen except with
      negligible probability. Indeed, even if FindNonResidue returns a bogus output,
      it returns only one of many large random primes, which are coprime with p with
      large probability.

    """

    def __init__(
        self, n: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.n = n
        preg = self.add_input_output(UIntType(n))

        ppreg = qc_copy(preg)
        self.x(preg[0])  # if preg[0] is 0 we shift ppreg left
        qc_cshift_right(qc_bool_reg(preg[0]), ppreg, -1)
        self.x(preg[0])

        # in case p odd, ppreg contains p
        # in case p even, ppreg contains p / 2

        # step 1: find non-residue
        _, xreg = self.append(FindNonResidue(n), ppreg)
        # compute y = modexp(x, (pp - 1) // 4, pp)

        # register holding the exponent: we use a truncation, which works even
        # if p' is not = 1 mod 4.
        exp = self.add_anc(UIntType(n))
        qc_decr_uint(ppreg)
        qc_xor(ppreg[2:n], exp[0 : (n - 2)])
        qc_incr_uint(ppreg)

        # register holding the exponentiation result
        exp_result = self.add_anc(UIntType(n))
        # compute power
        qc_powadd_modvar(exp, xreg, exp_result, ppreg)

        # erase exp
        qc_decr_uint(ppreg)
        qc_xor(ppreg[2:n], exp[0 : (n - 2)])
        qc_incr_uint(ppreg)
        self.assert_anc(exp)

        # erase xreg
        self.append(FindNonResidue(n).inverse(), ppreg, xreg)
        # ------------------------

        # now in the even case we return (pp + exp_result * (1 + pp)) % p
        # in the odd case we return exp_result

        # so let's just compute both, and exchange between garbage / not
        # garbage depending on the value of p !!
        # exp_result we already have it, we compute the other one

        yreg = self.get_anc(UIntType(n))
        # multiply (p'+1) and exp_result
        qc_incr_uint(ppreg)
        prod = qc_mul(exp_result, ppreg)
        # reduce modulo p
        quo, rem = qc_div_uint(preg, prod)
        # put remainder into the yreg register
        qc_xor(rem, yreg)
        # uncompute
        prod = qc_div_uncompute_uint(preg, quo, rem)
        qc_mul_erase(exp_result, ppreg, prod)
        qc_decr_uint(ppreg)
        # Now yreg contains exp_result * (p'+1) mod p
        # add p' modulo p. Actually we do it in two parts, otherwise
        # the variable modular adder breaks down because ppreg = preg
        # in half the cases.
        qc_decr_uint(ppreg)
        qc_add_modvar(ppreg, yreg, preg)
        qc_add_modvar(UInt(1), yreg, preg)
        qc_incr_uint(ppreg)
        # now yreg is computed

        # now we can erase ppreg
        self.x(preg[0])
        qc_cshift_right(qc_bool_reg(preg[0]), ppreg, 1)
        self.x(preg[0])
        qc_copy_erase(preg, ppreg)
        # ---------

        qc_cswap(qc_bool_reg(preg[0]), yreg, exp_result)

        # this is the output
        self.add_output(yreg)
        # this is the garbage
        self.add_output(qc_reg_cast(exp_result, BitVectorType(n)))

    def validate_input(self, args: UInt) -> bool:
        return args > 3 and (args % 8 == 2 or args % 4 == 1)

    def dummy_classical_function(self, args: UInt) -> tuple[UInt, UInt, BitVector]:
        p = args
        out, garbage = minusone_sqrt_general(p, self.n)
        return p, UInt(out), garbage

    def dummy_classical_function_inverse(
        self, args: tuple[UInt, UInt, BitVector]
    ) -> UInt:
        p, _, _ = args
        return p
