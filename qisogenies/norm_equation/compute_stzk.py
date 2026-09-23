"""
The computation of (s,t,z,k) is the first step in the first test performed in the
search iterations.

This module contains two circuits:

* ComputeSTZK which is the main circuit
* ComputeSTZKStep2 which is a part of this circuit

In particular, ComputeSTZK takes as input the input of the algorithm (N, alpha) and
the output of the precomputation.

Note that the python 'dummy' functions defined here:

* compute_stzk_step2_dummy
* compute_stzk_dummy

use python types only (except RingInteger inputs), while the Qarton dummy functions use
Qarton types like SInt. We convert back and forth.

"""

from qarton.arithmetic import (
    qc_add_uint,
    qc_decr_uint,
    qc_div_uint,
    qc_div_uncompute_uint,
    qc_incr_uint,
    qc_lt_uint,
    qc_mul_erase_uint,
    qc_mul_uint,
    qc_sqr_erase_uint,
    qc_sqr_uint,
    qc_sub_uint,
    qc_truncate_uint,
)
from qarton.binary_operations import qc_copy, qc_mcx_neg
from qarton.circuit import (
    BackendSpecifier,
    BitVector,
    BoolType,
    Circuit,
    EmptyBackendSpecifier,
    ITupleType,
    PreCircuit,
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
    qc_neg,
    qc_sqr,
    qc_sqr_erase,
    qc_sub,
)
from qarton.mod_arithmetic_variable import qc_muladd_modvar
from qarton.signed_arithmetic import (
    SInt,
    qc_abs,
    qc_abs_uncompute,
    qc_expand_sint,
    qc_truncate_sint,
)

from .util import InstanceData

__all__ = [
    "compute_stzk_step2",
    "compute_stzk_classical",
    "ComputeSTZKStep2",
    "ComputeSTZK",
]


def compute_stzk_step2(
    d: InstanceData,
    args: tuple[tuple[UInt, UInt], tuple[UInt, UInt]],
) -> tuple[SInt, SInt, UInt, BitVector]:
    """Step 2 in the ``compute_stzk`` function. At this point we have computed
    the parameters ``k`` and ``bound``, as well as ``c`` and ``v``. This function handles
    the computation of the actual s,t,z. It also returns a few bits which should
    all be 1 if we are on track for a solution.

    :param args: (k, bound), (c, v)
    :type args: tuple[tuple[int, int], tuple[int, int]]
    :return: s,t,z and a bitvector of length 4.
    :rtype: tuple[int, int, int, BitVector]
    """
    w = d.w
    (k, bound), (c, v) = args
    # c, v will be destroyed in the quantum circuit
    assert k < (1 << (w // 4))
    assert bound < (1 << (w // 2))
    assert c < (1 << ((w // 2) + 20))
    assert v < (1 << (w // 2))

    test1 = v < bound

    xV = v // k  # is a good approximation of the original code
    # we should rather take - round(v/k) but it's easier to take v//K
    s, t = (xV * k - v, -xV)
    z = c - s**2 - t**2

    # sign goes to testing bit!!!
    test2 = z > 0
    z = abs(z)

    # bounds
    assert abs(s) < (1 << (d.w // 4 + 10))
    assert abs(t) < (1 << (d.w // 4 + 10))
    assert z < (1 << (d.w // 2 + 20))

    assert z >= 0
    return (
        SInt(s, d.w // 4 + 10),
        SInt(t, d.w // 4 + 10),
        UInt(z),
        BitVector(test1, test2),
    )


def compute_stzk_classical(
    d: InstanceData,
    n_tralpha: tuple[UInt, UInt],
    precomp_result: tuple[UInt, UInt, UInt, UInt],
    rchoice: UInt,
) -> tuple[SInt, SInt, UInt, UInt, BitVector]:
    """Compute the s, t, z, k parameters.

    :param n_tralpha: (N, tr_alpha)
            N: unsigned integer, size w/2
            tr_alpha: trace of alpha, size w/2
    :param precomp_result: Result of the precomputation, as:
            max_k: bound on k. Unsigned, size w/4
            bound: bound. Unsigned, size w/2
            tr_alpha_inv_neg: Unsigned, size w/2
            r: Unsigned, size w/2 + 10

    :param rchoice: Choice of i. Small integer.
    :type rchoice: int
    :param e: e, small constant integer
    :type e: int
    :return: s (signed, w/2), t (signed, w/2), z (unsigned, w/2 + 10), k (unsigned, w/4), BitVector of 4 Booleans
    :rtype: tuple[int, int, int, int, BitVector]
    """
    e = d.e
    w = d.w

    max_k, bound, tr_alpha_inv_neg, r = precomp_result
    assert max_k < (1 << (w // 4))
    assert bound < (1 << (w // 2))
    assert (tr_alpha_inv_neg) < (1 << (w // 2))
    assert tr_alpha_inv_neg >= 0
    assert r < (1 << (w // 2 + 10))

    N, tr_alpha = n_tralpha
    assert tr_alpha < (1 << (w // 2))
    assert (1 << (w // 2)) > N

    # computation of k, then s, t, z
    k = max_k - rchoice

    ksq = k**2
    c = (1 << e) - (1 + ksq) * r
    v = (tr_alpha_inv_neg * (c % N)) % N
    c += tr_alpha * v
    c = c // N

    s, t, z, bv = compute_stzk_step2(d, ((UInt(k), UInt(bound)), (UInt(c), UInt(v))))

    # final test
    add_test = BitVector(1)
    if N % 2 == 0:
        if k % 2 == 0 or s % 2 == r % 2 or t % 2 == r % 2:
            add_test[0] = 0

    return (s, t, z, UInt(k), bv + add_test)


@dummify
@memoize
class ComputeSTZKStep2(
    Circuit[
        tuple[tuple[UInt, UInt], tuple[UInt, UInt]],
        tuple[
            tuple[UInt, UInt],
            SInt,
            SInt,
            UInt,
            BitVector,
        ],
    ]
):
    """
    Circuit for a sub-component in the computation of s,t,z and k.

    It takes the following inputs: (k,bound) and (c,v).

    * k: IntType(w//4), preserved
    * bound: IntType(w//2), preserved

    * c: IntType(w//2 + 20), destroyed
    * v: IntType(w//2), destroyed

    In addition to (k, bound), it returns:

    * s (SIntType(w//4 + 10))
    * t: SIntType(w//4 + 10)
    * z: IntType(w//2 + 20)
    * a BitVector of length 3 with Booleans (the test is passed iff all 3 are 1).

    The value z that is returned is not always the "true" value of z. We ensure that
    the returned value is = 2 mod 8, but it's only the "true" value if all the
    returned bits are 1 (meaning that the conditions are satisfied).
    """

    def __init__(
        self,
        d: InstanceData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param w: Generic bit-size of the values.
        :type w: int
        """
        super().__init__()
        self.d = d
        w = d.w
        self.w = w

        preserved = self.add_input_output(
            ITupleType(UIntType(w // 4), UIntType(w // 2))
        )
        k_reg, bound_reg = preserved.a[0], preserved.a[1]

        destroyed = self.add_input(ITupleType(UIntType(w // 2 + 20), UIntType(w // 2)))
        c_reg, v_reg = destroyed.a[0], destroyed.a[1]
        # ----------------------

        # first test: is v < bound?
        test1 = self.get_anc(BoolType())
        qc_lt_uint(v_reg, bound_reg, test1)

        xV_copy, remv = qc_div_uint(k_reg, v_reg)  # we need to keep xV
        xV = qc_copy(xV_copy)
        v = qc_div_uncompute_uint(k_reg, xV_copy, remv)

        # now we still have v, which we will use, and xV, which we will also use.
        # make xV, v and k_reg into signed registers
        sg_anc = self.get_anc(BoolType())  # does not need to be restored
        xV_signed = qc_abs_uncompute(xV, sg_anc)  # should be of size w//2 still

        sg_anc = self.get_anc(BoolType())  # will need to be restored
        k_signed = qc_abs_uncompute(k_reg, sg_anc)

        sg_anc = self.get_anc(BoolType())
        v_signed = qc_abs_uncompute(v, sg_anc)

        # start s_reg at -v, add xV * k (that way we consume the v_signed register)
        s_reg = v_signed
        qc_neg(v_signed)
        tmp = qc_mul(xV_signed, k_signed)
        _tmp_size = len(tmp) - 1
        tmp = qc_truncate_sint(tmp, w // 2)
        qc_add(tmp, v_signed)  # s = xV * k - v
        tmp = qc_expand_sint(tmp, _tmp_size)
        qc_mul_erase(xV_signed, k_signed, tmp)
        # now tmp is properly destroyed

        # we restore the k_reg input, we know it's positive
        k_reg, k_sign = qc_abs(k_signed)
        self.assert_anc(k_sign)

        # truncate s_reg
        s_reg = qc_truncate_sint(s_reg, w // 4 + 10)
        self.add_output(s_reg)

        qc_neg(xV_signed)
        # t_reg (size w//2)
        t_reg = qc_truncate_sint(xV_signed, w // 4 + 10)
        self.add_output(t_reg)  # ---> output

        # make c_reg signed
        sg_anc = self.get_anc(BoolType())  # no need to uncompute
        z_reg = qc_abs_uncompute(c_reg, sg_anc)

        # subtract s**2 and t**2:
        tmp4 = qc_sqr(s_reg)
        qc_sub(tmp4, z_reg)
        qc_sqr_erase(s_reg, tmp4)
        tmp4 = qc_sqr(t_reg)
        qc_sub(tmp4, z_reg)
        qc_sqr_erase(t_reg, tmp4)

        # take abs value of z
        z_final, test2 = qc_abs(z_reg)
        # if z is negative, i.e. test2 = 1, it's bad. So we want test2 to be 0.
        self.x(test2[0])
        self.add_output(z_final)  # ----> output size w//2 + 20

        # output bits of conditions
        condbits = test1 + test2
        self.add_output(condbits)

        # remap everything
        preserved = qc_make_ituple(k_reg, bound_reg)
        self.remap(preserved, s_reg, t_reg, z_final, condbits)

    def dummy_classical_function(
        self, args: tuple[tuple[UInt, UInt], tuple[UInt, UInt]]
    ) -> tuple[tuple[UInt, UInt], SInt, SInt, UInt, BitVector]:
        (k, bound), (_, _) = args
        s, t, z, bv = compute_stzk_step2(self.d, args)

        return ((UInt(k), UInt(bound)), s, t, UInt(z), bv)


@dummify
@memoize
class ComputeSTZK(
    Circuit[
        tuple[
            tuple[UInt, UInt],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
        tuple[
            tuple[UInt, UInt],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            SInt,
            SInt,
            UInt,
            UInt,
            BitVector,
        ],
    ]
):
    """
    Input:

    * (N, tr_alpha) (preserved)
    * precomputation result (4 integers) (preserved). Note that this is tr_alpha_inv_neg_reg
            which is precomputed, and negated.
    * choice (preserved)

    Output:

    Same as inputs, in addition: s,t,z,k and a BitVector

    """

    def __init__(
        self,
        d: InstanceData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        e, w, x = d.e, d.w, d.x
        self.e = e
        self.d = d
        self.w = w

        n_tralpha_reg = self.add_input_output(
            ITupleType(UIntType(w // 2), UIntType(w // 2))
        )
        n_reg, tr_alpha_reg = n_tralpha_reg.a[0], n_tralpha_reg.a[1]

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )
        max_k_reg, bound_reg, tr_alpha_inv_neg_reg, r_reg = (
            precomp_result_reg.a[0],
            precomp_result_reg.a[1],
            precomp_result_reg.a[2],
            precomp_result_reg.a[3],
        )

        choice = self.add_input_output(UIntType(x))

        # computation of k: we copy max_k_reg and subtract the choice
        # (to optimize space we shouldn't copy it)
        k_reg = qc_copy(max_k_reg)
        qc_sub_uint(choice, k_reg)

        # k will be added to outputs later after having run step 2
        # next compute c = (1 << e) - (1 + ksq) * r
        # compute k^2
        ksq = qc_sqr_uint(k_reg)
        qc_incr_uint(ksq)
        # now ksq_reg contains k^2 + 1

        c_reg = self.get_anc(UIntType(w + 20))
        self.x(c_reg[e])  # now c_reg contains (1 << e)

        tmp1 = qc_mul_uint(ksq, r_reg)  # (1 + ksq) * r
        qc_sub_uint(tmp1, c_reg)  # (1 << e) - (1 + ksq) * r
        qc_mul_erase_uint(ksq, r_reg, tmp1)
        qc_decr_uint(ksq)
        qc_sqr_erase_uint(k_reg, ksq)
        # now c _regcontains (1 << e) - (1 + ksq) * r (always positive)

        # Computation of v
        # compute c % N temporarily
        quo, rem = qc_div_uint(n_reg, c_reg)  # currently in rem
        # deduce v
        v_reg = self.get_anc(UIntType(w // 2))
        qc_muladd_modvar(rem, tr_alpha_inv_neg_reg, v_reg, n_reg)
        # restore c
        c_reg = qc_div_uncompute_uint(n_reg, quo, rem)

        # add tr_alpha * v to c (all are positive)
        tralphav = qc_mul_uint(tr_alpha_reg, v_reg)
        qc_add_uint(tralphav, c_reg)
        qc_mul_erase_uint(tr_alpha_reg, v_reg, tralphav)  # tralphav is destroyed
        # next, divide c by N (c is always a multiple of N at this point apparently)

        quo, rem = qc_div_uint(n_reg, c_reg)
        self.test_anc(rem)
        self.assert_anc(rem)
        c_reg = quo
        # truncate
        c_reg = qc_truncate_uint(c_reg, w // 2 + 20)

        # step 2 takes:
        # * k: IntType(w//4), preserved
        # * bound: IntType(w//2), preserved
        # * c: IntType(w//2 + 20), destroyed
        # * v: IntType(w//2), destroyed
        # destroys c, v and returns the new outputs
        kbound = qc_reg_cast(
            k_reg + bound_reg, ITupleType(UIntType(w // 4), UIntType(w // 2))
        )
        cv = qc_reg_cast(
            c_reg + v_reg, ITupleType(UIntType(w // 2 + 20), UIntType(w // 2))
        )
        kbound, s, t, z, bv = self.append(
            PreCircuit(ComputeSTZKStep2, self.d), kbound, cv
        )

        # ---- finally there is a last test to do:

        tmp = self.add_anc(3)
        # write: (N % 2 == 0 and k % 2 == 0)
        self.x(n_reg[0])
        self.x(k_reg[0])
        self.ccx(n_reg[0], k_reg[0], tmp[0])
        self.x(k_reg[0])
        # write: (N % 2 ==0 and s % 2 == r % 2)
        self.x(s[0])
        self.cx(r_reg[0], s[0])
        self.ccx(n_reg[0], s[0], tmp[1])
        self.cx(r_reg[0], s[0])
        self.x(s[0])
        # write: (N % 2 == 0 and t % 2 == r % 2)
        self.x(t[0])
        self.cx(r_reg[0], t[0])
        self.ccx(n_reg[0], t[0], tmp[2])
        self.cx(r_reg[0], t[0])
        self.x(t[0])
        self.x(n_reg[0])

        final_cond = self.add_anc(BoolType())
        # flip final_cond if all bits are 0 (all conditions not satisfied)
        qc_mcx_neg(tmp, final_cond)
        # then erase

        self.x(n_reg[0])
        self.x(k_reg[0])
        self.ccx(n_reg[0], k_reg[0], tmp[0])
        self.x(k_reg[0])
        # write: (N % 2 ==0 and s % 2 == r % 2)
        self.x(s[0])
        self.cx(r_reg[0], s[0])
        self.ccx(n_reg[0], s[0], tmp[1])
        self.cx(r_reg[0], s[0])
        self.x(s[0])
        # write: (N % 2 == 0 and t % 2 == r % 2)
        self.x(t[0])
        self.cx(r_reg[0], t[0])
        self.ccx(n_reg[0], t[0], tmp[2])
        self.cx(r_reg[0], t[0])
        self.x(t[0])
        self.x(n_reg[0])
        self.release_anc(tmp)

        self.add_output(s)
        self.add_output(t)
        self.add_output(z)
        self.add_output(k_reg)
        self.add_output(bv + final_cond)

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[UInt, UInt],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ) -> tuple[
        tuple[UInt, UInt],
        tuple[UInt, UInt, UInt, UInt],
        UInt,
        SInt,
        SInt,
        UInt,
        UInt,
        BitVector,
    ]:
        n_tralpha, precomp_result, rchoice = args
        s, t, z, k, bv = compute_stzk_classical(
            self.d, n_tralpha, precomp_result, rchoice
        )

        return (
            n_tralpha,
            precomp_result,
            rchoice,
            s,
            t,
            UInt(z),
            UInt(k),
            bv,
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[UInt, UInt],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            SInt,
            SInt,
            UInt,
            UInt,
            BitVector,
        ],
    ) -> tuple[
        tuple[UInt, UInt],
        tuple[UInt, UInt, UInt, UInt],
        UInt,
    ]:
        n_tralpha, precomp_result, rchoice, _, _, _, _, _ = args
        return (n_tralpha, precomp_result, rchoice)
