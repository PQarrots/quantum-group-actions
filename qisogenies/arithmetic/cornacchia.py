"""
Quantum implementation of Cornacchia's algorithm, which matches the classical
functions defined in the ``cornacchia_classical`` module.
"""

from qarton.arithmetic import (
    EuclideanDivider,
    isqrt_output_size,
    qc_add_uint,
    qc_cdecr_uint,
    qc_cincr_uint,
    qc_csub_uint,
    qc_eq_uint,
    qc_lt_uint,
    qc_sqrt_erase_uint,
    qc_sqrt_uint,
)
from qarton.binary_operations import (
    qc_copy,
    qc_cshift_right,
    qc_cswap,
    qc_mcx,
    qc_mcx_neg,
    qc_shift_right,
)
from qarton.circuit import (
    BackendSpecifier,
    BitVector,
    BoolType,
    Circuit,
    EmptyBackendSpecifier,
    InPlaceCircuit,
    QartonBool,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_bool_reg,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.dispatch import qc_sqr, qc_sqr_erase, qc_sub

from .cornacchia_classical import cornacchia_classical_function
from .mod_sqrt import MinusOneSqrtGeneral

__all__ = [
    "Cornacchia",
    "CornacchiaIterations",
    "CornacchiaTestOnly",
]


@memoize
class _CornacchiaIterator(InPlaceCircuit):
    """
    Iterator for the CornacchiaIterations circuit. Because this circuit repeats the
    same operations many times, we define an iterator and use a Qarton *iterated
    operation* to define the circuit faster.

    The iterator operates on the following fields:

    * ``a`` (UIntType)
    * ``b`` (UIntType)
    * ``l`` (UIntType) (constant throughout the iterations)
    * ``shift`` (UIntType)
    * ``stop`` (UIntType)

    And garbage registers:

    * ``is_b_strictly_greater_than_a`` (BitVectorType)
    * ``is_twice_b_smaller_than_a`` (BitVectorType)
    * ``is_shift_zero`` (BitVectorType)
    """

    def __init__(self, n: int) -> None:
        """
        :param n: Size of the integers.
        :type n: int
        """
        super().__init__()

        it = 3 * n // 2
        a = self.add_input_output(UIntType(n))
        b = self.add_input_output(UIntType(n))
        l = self.add_input_output(UIntType(isqrt_output_size(n)))

        shift = self.add_input_output(UIntType(n.bit_length()))
        stop = self.add_input_output(UIntType(it.bit_length()))

        is_b_strictly_greater_than_a = self.add_input_output(it)
        is_twice_b_smaller_than_a = self.add_input_output(it)
        is_shift_zero = self.add_input_output(it)
        # ------------------

        # self.print(a, b, shift, msg="begin")

        # compute the three bits
        qc_lt_uint(a, b, qc_bool_reg(is_b_strictly_greater_than_a[0]))
        qc_mcx_neg(shift, qc_bool_reg(is_shift_zero[0]))  # 0 if shift is 0

        # if b > a and shift == 0, swap
        tmp = self.get_anc(BoolType())
        self.ccx(is_b_strictly_greater_than_a[0], is_shift_zero[0], tmp[0])
        qc_cswap(tmp, a, b)
        self.ccx(is_b_strictly_greater_than_a[0], is_shift_zero[0], tmp[0])
        self.assert_anc(tmp)
        self.test_anc(tmp)
        # ------------

        # tmp can be reused
        # stop += int(b <= l)
        qc_lt_uint(l, b, tmp)
        self.x(tmp[0])
        # increment stop
        qc_cincr_uint(tmp, stop)
        self.x(tmp[0])
        qc_lt_uint(l, b, tmp)

        self.test_anc(tmp)
        self.assert_anc(tmp)
        # --------------

        # ----- compute is_twice_b_smaller_than_a[i]
        # note the tiny trouble that b could overflow here, so we increase its size
        # by one bit
        bb = qc_reg_cast(b + tmp, UIntType(n + 1))
        qc_shift_right(bb, 1)
        # 2 * b <= a <=> not a < 2*b
        qc_lt_uint(a, bb, qc_bool_reg(is_twice_b_smaller_than_a[0]))
        self.x(is_twice_b_smaller_than_a[0])
        qc_shift_right(bb, -1)
        self.assert_anc(tmp)
        # ------------

        # --- if b > a and shift > 0
        self.x(is_shift_zero[0])
        self.ccx(is_b_strictly_greater_than_a[0], is_shift_zero[0], tmp[0])

        qc_cdecr_uint(tmp, shift)
        qc_cshift_right(tmp, b, -1)

        self.ccx(is_b_strictly_greater_than_a[0], is_shift_zero[0], tmp[0])
        self.x(is_shift_zero[0])

        self.test_anc(tmp)
        self.assert_anc(tmp)
        # ---

        is_stop_zero = self.get_anc(1)
        qc_mcx_neg(stop, is_stop_zero)

        # if 2 * b <= a and is_stop_zero
        self.ccx(is_twice_b_smaller_than_a[0], is_stop_zero[0], tmp[0])
        qc_cincr_uint(tmp, shift)
        qc_cshift_right(tmp, b, 1)
        self.ccx(is_twice_b_smaller_than_a[0], is_stop_zero[0], tmp[0])

        self.test_anc(tmp)
        self.assert_anc(tmp)
        # --------

        # if b <= a and a < 2 * b and is_stop_zero:
        self.x(is_b_strictly_greater_than_a[0])
        self.x(is_twice_b_smaller_than_a[0])
        qc_mcx(
            qc_reg_from_bits(
                is_b_strictly_greater_than_a[0],
                is_stop_zero[0],
                is_twice_b_smaller_than_a[0],
            ),
            tmp,
        )

        qc_csub_uint(tmp, b, a)
        qc_mcx(
            qc_reg_from_bits(
                is_b_strictly_greater_than_a[0],
                is_stop_zero[0],
                is_twice_b_smaller_than_a[0],
            ),
            tmp,
        )
        self.x(is_b_strictly_greater_than_a[0])
        self.x(is_twice_b_smaller_than_a[0])

        self.test_anc(tmp)
        self.assert_anc(tmp)

        # ---------

        qc_mcx_neg(stop, is_stop_zero)
        self.assert_anc(is_stop_zero)
        self.release_anc(is_stop_zero)

        qc_shift_right(is_b_strictly_greater_than_a, 1)
        qc_shift_right(is_twice_b_smaller_than_a, 1)
        qc_shift_right(is_shift_zero, 1)


@memoize
class CornacchiaIterations(
    Circuit[tuple[UInt, UInt], tuple[UInt, UInt, UInt, BitVector]]
):
    """
    Euclidean iterations in Cornacchia's algorithm. Not perfectly space-optimized.

    This circuit matches the function ``cornacchia_iterations`` from the module
    ``cornacchia_classical``.
    """

    def __init__(self, n: int) -> None:
        """
        :param n: Size of the integers.
        :type n: int
        """
        super().__init__()

        a = self.add_input_output(UIntType(n))
        mm = self.add_input_output(UIntType(n))

        m = qc_copy(mm)

        # l = square root of m
        l = qc_sqrt_uint(m)

        # compute b = m % a (remainder in euclidean division)
        a, quo, b = self.append(EuclideanDivider(n, n), a, m)

        # -----
        it = 3 * n // 2
        shift = self.get_anc(UIntType(n.bit_length()))
        stop = self.get_anc(UIntType(it.bit_length()))

        is_b_strictly_greater_than_a = self.get_anc(it)
        is_twice_b_smaller_than_a = self.get_anc(it)
        is_shift_zero = self.get_anc(it)

        # --------------
        # NOTE b is actually smaller than n bits, so space optimization might be possible

        self.append_iterated(
            _CornacchiaIterator(n),
            a,
            b,
            l,
            shift,
            stop,
            is_b_strictly_greater_than_a,
            is_twice_b_smaller_than_a,
            is_shift_zero,
            iterations=it,
        )

        # when we stop, b**2 is smaller than m, so the length of its register is given
        # by this:
        b_size = isqrt_output_size(n)
        btrunc = qc_reg_cast(b[:b_size], UIntType(b_size))
        self.add_output(btrunc)
        self.test_anc(b[b_size:])

        self.add_output(
            quo
            + shift
            + stop
            + is_b_strictly_greater_than_a
            + is_twice_b_smaller_than_a
            + is_shift_zero
            + l
        )


@dummify
@memoize
class Cornacchia(Circuit[UInt, tuple[UInt, QartonBool, UInt, UInt]]):
    """
    Cornacchia's algorithm.

    Accepts the following inputs:

    * z = 2 mod 8
    * z = 1 mod 4

    It is only guaranteed to work if z is twice a prime, or a prime.

    The size of the output registers (solution pair) is isqrt_output_size(n) where n
    is the size of the input register.
    """

    def __init__(
        self, n: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        """
        :param n: Size of the integers.
        :type n: int
        """
        super().__init__()
        self.n = n
        m = self.add_input_output(UIntType(n))

        # step 1: compute sqrt
        _, r0_reg, sqrt_garbage = self.append(
            MinusOneSqrtGeneral(n, backends=backends), m
        )

        # step 2: compute iterations
        _, _, b, garbage = self.append(CornacchiaIterations(n), r0_reg, m)
        # r0_reg, m preserved. b has size roughly n/2 .

        # ---------------------
        # now, if m - b**2 is a square, we have a solution
        bsquare = qc_sqr(b)
        bsquare_trunc = qc_reg_cast(bsquare[:n], UIntType(n))

        # step 3: check if we have a solution
        qc_sub(bsquare_trunc, m)
        # now m contains m - b**2, should be a square

        sol = qc_sqrt_uint(m)
        solsquare = qc_sqr(sol)
        # we can check if sol**2 == (m-b**2)

        out = self.add_anc_output(BoolType())
        # write 1 in output iff m == sol**2, i.e. (b, sol) is a solution
        qc_eq_uint(m, solsquare, out)

        # erase solsquare
        qc_sqr_erase(sol, solsquare)
        # restore m
        qc_add_uint(bsquare_trunc, m)
        # erase bsquare (this destroys the register).
        qc_sqr_erase(b, bsquare)

        # copy b and add it as output
        bb = qc_copy(b)
        self.add_output(bb)
        self.add_output(sol)

        assert len(bb) == isqrt_output_size(n)
        assert len(sol) == isqrt_output_size(n)

        # -------- erase Cornacchia iterations
        r0_reg, m = self.append(
            CornacchiaIterations(n).inverse(), r0_reg, m, b, garbage
        )
        # this destroys b and garbage
        self.append(
            MinusOneSqrtGeneral(n, backends=backends).inverse(), m, r0_reg, sqrt_garbage
        )

    def validate_input(self, args: UInt) -> bool:
        return (args % 8 == 2 or args % 4 == 1) and args > 2

    def dummy_classical_function(
        self, args: UInt
    ) -> tuple[UInt, QartonBool, UInt, UInt]:
        valid, b, sol, _ = cornacchia_classical_function(args, self.n)
        # print(args, args % 16, valid, b, sol)
        return args, QartonBool(valid), UInt(b), UInt(sol)

    def dummy_classical_function_inverse(
        self, args: tuple[UInt, QartonBool, UInt, UInt]
    ) -> UInt:
        return args[0]


@dummify
@memoize
class CornacchiaTestOnly(Circuit[UInt, tuple[UInt, QartonBool, BitVector]]):
    r"""
    Cornacchia's algorithm, but only checking if there is a solution, not writing it.
    Also returns garbage to avoid too much uncomputation.

    """

    def __init__(
        self, n: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        """
        :param n: Size of the integers.
        :type n: int
        """
        super().__init__()
        self.n = n
        m = self.add_input_output(UIntType(n))

        # step 1: compute sqrt
        _, r0_reg, sqrt_garbage = self.append(
            MinusOneSqrtGeneral(n, backends=backends), m
        )

        # step 2: compute iterations
        _, _, b, garbage = self.append(CornacchiaIterations(n), r0_reg, m)
        # r0_reg, m preserved. b has size roughly n/2 .

        # ---------------------
        # now, if m - b**2 is a square, we have a solution
        bsquare = qc_sqr(b)
        bsquare_trunc = qc_reg_cast(bsquare[:n], UIntType(n))

        # step 3: check if we have a solution
        qc_sub(bsquare_trunc, m)
        # now m contains m - b**2, should be a square

        sol = qc_sqrt_uint(m)
        # m, sol = self.append(ISqrt(n), m)
        solsquare = qc_sqr(sol)
        # we can check if sol**2 == (m-b**2)

        out = self.add_anc_output(BoolType())
        # write 1 in output iff m == sol**2, i.e. (b, sol) is a solution
        qc_eq_uint(m, solsquare, out)

        # erase solsquare
        qc_sqr_erase(sol, solsquare)
        # erase sol
        qc_sqrt_erase_uint(m, sol)

        # restore m
        qc_add_uint(bsquare_trunc, m)
        # erase bsquare (this destroys the register).
        qc_sqr_erase(b, bsquare)

        # -------- erase Cornacchia iterations. This destroys b
        r0_reg, m = self.append(
            CornacchiaIterations(n).inverse(), r0_reg, m, b, garbage
        )
        self.add_output(sqrt_garbage + r0_reg)

    def validate_input(self, args: UInt) -> bool:
        return (args % 8 == 2 or args % 4 == 1) and args > 2

    def dummy_classical_function(
        self, args: UInt
    ) -> tuple[UInt, QartonBool, BitVector]:
        valid, _, _, bv = cornacchia_classical_function(args, self.n)
        return args, QartonBool(valid), bv

    def dummy_classical_function_inverse(
        self, args: tuple[UInt, QartonBool, BitVector]
    ) -> UInt:
        return args[0]
