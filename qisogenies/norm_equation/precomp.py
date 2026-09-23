"""
Perform the precomputation step in Qlapoti.
"""

from math import floor, isqrt

from qarton.arithmetic import (
    qc_div_uint,
    qc_div_uncompute_uint,
    qc_load_uint,
    qc_truncate_uint,
)
from qarton.binary_operations import qc_copy, qc_unload
from qarton.circuit import (
    BackendSpecifier,
    Circuit,
    EmptyBackendSpecifier,
    ITupleType,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_make_ituple,
    qc_reg_cast,
)
from qarton.dispatch import (
    qc_add,
    qc_mul,
    qc_mul_erase,
    qc_sqr,
    qc_sqr_erase,
    qc_sqrt,
    qc_sqrt_erase,
    qc_sub,
)
from qarton.mod_arithmetic_variable import qc_invadd_modvar, qc_neg_modvar
from qarton.signed_arithmetic import qc_abs, qc_abs_uncompute

from .ring_integers import (
    RingInteger,
    RingIntegerType,
    qc_norm,
    qc_trace,
    qc_trace_erase,
)
from .util import InstanceData

__all__ = ["precomp_classical", "Precomp"]


def precomp_classical(
    d: InstanceData,
    algo_input: tuple[UInt, RingInteger],
) -> tuple[UInt, UInt, UInt, UInt]:
    """Precomputation function.

    :param d: General constant parameters.
    :type d: InstanceData
    :param algo_input: Input to the algorithm: (N, alpha)
    :type algo_input: tuple[int, RingInteger]
    :param test_bounds: If True, will check that the outputs meet their size bounds, defaults to False
    :type test_bounds: bool, optional
    :return: max_k, bound, tr_alpha_inv_neg, r
    :rtype: tuple[int, int, int, int]
    """
    # compute max_k, bound, tr_alpha_inv, r
    w, e, p = d.w, d.e, d.p
    N, ring_int = algo_input
    # a, b = ring_int.a.x, ring_int.b.x

    # precomputation starts here
    tr_alpha = ring_int.trace()
    # assert tr_alpha == 2 * a + b

    tr_alpha_inv_neg = N - pow(tr_alpha, -1, N)
    # assert ring_int.norm() % N == 0
    r = ring_int.norm() // N

    # assert r == ((p + 1) // 4 * b**2 + a * (a + b)) // N

    max_k = floor(isqrt(N * 2 ** (e - 2) // p))
    # truncate 2 bits (make max_k = 3 mod 4)
    # max_k = #4 * (max_k // 4) + 3
    bound = floor(isqrt(max_k**2 * ((1 << e) // N - max_k**2 // 4)))

    # size bounds
    assert (1 << (w // 2)) > N
    assert max_k < (1 << (w // 4))
    assert bound < (1 << (w // 2))
    assert tr_alpha_inv_neg < (1 << (w // 2))
    assert r < (1 << (w // 2 + 10))
    return UInt(max_k), UInt(bound), UInt(tr_alpha_inv_neg), UInt(r)


@memoize
@dummify
class Precomp(
    Circuit[
        tuple[UInt, RingInteger],
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
        ],
    ]
):
    """
    Precomputation step, before we can run the main loop in the norm equation
    algorithm.

    The input is:

    * N: UIntType(w//2). Value of the first ring integer (which is an integer).
        Positive, odd.
    * ring_int: RingIntegerType(p, w // 2). Small element alpha in the ring of integers.
        (Its second coordinate is actualy -1, so there is some space to gain here
        for further minimization).

    N and ring_int are both preserved as they are needed afterwards. The additional
    output is:

    * max_k: UIntType(w//4). Maximal value of k
    * bound: UIntType(w//2). Bound on v
    * tr_alpha_inv_neg: UIntType(w//2). Negation (modulo N) of the inverse of the trace of alpha
    * r: UIntType(w//2): norm of alpha divided by N.
    """

    def __init__(
        self,
        d: InstanceData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param d: Parameters.
        :type d: InstanceData
        """
        super().__init__()
        w, e, p = d.w, d.e, d.p
        self.d = d
        self.e = e
        self.w = w

        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )
        n_reg, ring_int = algo_input.a[0], algo_input.a[1]

        # n_reg = self.add_input_output(UIntType(w // 2))  # supposed to be odd
        # coordinates of the reduced interval are roughly w//2
        # ring_int = self.add_input_output(RingIntegerType(p, w // 2))

        # compute tr_alpha_inv_neg
        tr_alpha = qc_trace(ring_int)
        tr_alpha, sgn = qc_abs(tr_alpha)  # supposed to be positive
        assert len(tr_alpha) == (w // 2)
        # compute the inverse

        tr_alpha_inv = self.add_anc(UIntType(w // 2))
        qc_invadd_modvar(tr_alpha, tr_alpha_inv, n_reg)
        qc_neg_modvar(tr_alpha_inv, n_reg)  # ----> to output

        # uncompute the tr_alpha thing, we don't need it anymore
        tr_alpha = qc_abs_uncompute(tr_alpha, sgn)
        qc_trace_erase(ring_int, tr_alpha)
        # -----
        # now compute r
        r_reg = qc_norm(ring_int)
        # divide by N (it's divisible by N, so the remainder is 0)
        r_reg, rem = qc_div_uint(n_reg, r_reg)
        assert not n_reg._destroyed  # type: ignore
        self.test_anc(rem)
        self.assert_anc(rem)
        r_reg = qc_truncate_uint(r_reg, w // 2 + 10)  # in case it was too large
        # ----> r_reg to output

        # now compute:
        # max_k = int(floor(isqrt(N * 2 ** (e - 2) / p)))  # type: ignore
        # bound = int(floor(sqrt( max_k **2* ((1 << e) // N - max_k**2 // 4)) )  # type: ignore
        # max_k = 4 * (max_k // 4) + 3

        # compute N * 2 ** (e - 2) // p
        # = take N and shift it by e-2 bits
        padding = self.get_anc(e - 2)
        pow2eN = qc_reg_cast(padding + n_reg, UIntType(w // 2 + e - 2))
        # load p
        p_reg = qc_load_uint(UInt(p), self)
        # divide
        quo, rem = qc_div_uint(p_reg, pow2eN)
        # take square root
        max_k_copy = qc_sqrt(quo)
        # ensure that max_k is 3 mod 4. This is a little annoying.
        max_k = qc_reg_cast(
            self.get_anc(2) + qc_copy(max_k_copy[2:]), UIntType(len(max_k_copy))
        )
        self.x(max_k[0])
        self.x(max_k[1])
        # erase max_k_copy
        qc_sqrt_erase(quo, max_k_copy)
        pow2eN = qc_div_uncompute_uint(p_reg, quo, rem)
        n_reg = qc_reg_cast(pow2eN[(-w // 2) :], UIntType(w // 2))
        qc_unload(UInt(p), p_reg)
        self.assert_anc(pow2eN[: (e - 2)])

        max_k = qc_truncate_uint(max_k, w // 4)  # ------>  to output

        # now compute bound
        # compute (1 << e) / N - max_k**2 / 4 , always positive
        pow2e = self.get_anc(UIntType(e + 1))
        self.x(pow2e[e])
        assert not n_reg._destroyed  # type: ignore
        quo, rem = qc_div_uint(n_reg, pow2e)  # (1 << e) / N in quo
        # compute max_k**2
        maxk2 = qc_sqr(max_k)
        # divide by 4 (truncate two bits)
        maxk2div = qc_reg_cast(maxk2[2:], UIntType(len(maxk2) - 2))
        # subtract
        qc_sub(maxk2div, quo)
        # multiply by max_k**2
        tmp = qc_mul(quo, maxk2)
        # take square root -> this is the bound
        bound = qc_sqrt(tmp)  # ----------> to output
        bound = qc_truncate_uint(bound, w // 2)

        qc_mul_erase(quo, maxk2, tmp)  # tmp erased
        qc_add(maxk2div, quo)
        qc_sqr_erase(max_k, maxk2)  # maxk2 erased
        pow2e = qc_div_uncompute_uint(n_reg, quo, rem)
        self.x(pow2e[e])
        self.assert_anc(pow2e)

        restored_input = qc_make_ituple(n_reg, ring_int)
        new_output = qc_make_ituple(max_k, bound, tr_alpha_inv, r_reg)
        self.add_output(new_output)

        self.remap(restored_input, new_output)

        # self.add_output(max_k)
        # self.add_output(bound)
        # self.add_output(tr_alpha_inv)
        # self.add_output(r_reg)

    def validate_input(self, args: tuple[UInt, RingInteger]) -> bool:
        """
        Conditions for the input to be valid. We require N to be odd, and the trace to
        be positive (this should be ensured at the creation of the element )
        """
        n, ring_int = args
        return n % 2 == 1 and ring_int.trace() > 0

    def dummy_classical_function(
        self, args: tuple[UInt, RingInteger]
    ) -> tuple[tuple[UInt, RingInteger], tuple[UInt, UInt, UInt, UInt]]:
        max_k, bound, tr_alpha_inv, r = precomp_classical(self.d, args)
        return (
            args,
            (UInt(max_k), UInt(bound), UInt(tr_alpha_inv), UInt(r)),
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
        ],
    ) -> tuple[UInt, RingInteger]:
        tmp, _ = args
        return tmp
