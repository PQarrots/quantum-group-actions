"""
This module implements the *first testing layer* in the search for a k that yields
a solution.

We start from:

* the input of the algorithm (N, alpha)
* the output of the precomputation
* a random choice in a small integer interval

The first test performs many computations, leading to the following values which
are important for later:

* delta1, delta2 (ring elements)
* d1, d2 (integers)
* z (integer to send to Cornacchia)

We give two versions of the circuit:

* FirstCheck is the one that outputs all these values, alongside a BitVector of all
  the bits which should be 1 for the test to be passed

* FirstCheckTestOnly is the one that outputs only the bitvector and z. It is to be
  used in the quantum search iterates, where we don't need anything beyond z to perform
  the next tests. That makes the circuit more space-efficient.


"""

from qarton.arithmetic import (
    qc_add_uint,
    qc_sub_uint,
)
from qarton.binary_operations import (
    FormulaEvaluation,
    qc_copy,
    qc_copy_erase,
    qc_mcx,
    qc_or,
)
from qarton.circuit import (
    BackendSpecifier,
    BitVector,
    BoolType,
    Circuit,
    EmptyBackendSpecifier,
    ITupleType,
    PreCircuit,
    QartonBool,
    Register,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_make_ituple,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.dispatch import qc_add, qc_mul, qc_mul_erase, qc_sub
from qarton.signed_arithmetic import (
    SInt,
    qc_abs,
    qc_abs_uncompute,
    qc_truncate_sint,
)
from sympy import symbols  # type: ignore

from qisogenies.arithmetic.mod4 import (
    qc_minusmod4,
    qc_minusmod4_erase,
    qc_mod4,
    qc_mod4_erase,
)

from .compute_stzk import ComputeSTZK, compute_stzk_classical
from .ring_integers import (
    RingInteger,
    RingIntegerType,
    qc_conj,
    qc_make_ring_integer,
    qc_mul_ring_int,
    qc_trace,
    qc_trace_erase,
)
from .util import (
    InstanceData,
    qc_div_signed,
    qc_make_signed,
    qc_make_unsigned,
)

__all__ = [
    "fix_coeff",
    "qc_mod4_not_eq",
    "qc_mod4_not_eq_erase",
    "FixCoefft",
    "first_test",
    "FirstCheck",
    "FirstCheckTestOnly",
]


def fix_coeff(
    d: InstanceData, delta_1: RingInteger, delta_2: RingInteger, N: UInt, z: UInt
) -> tuple[BitVector, SInt, SInt, QartonBool]:
    """
    Pre check on the coefficients to avoid singular cases before cornacchia
    """
    w = d.w
    assert abs(delta_1.a.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(delta_1.b.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(delta_2.a.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(delta_2.b.v) < (1 << (w // 2 + w // 4 + 10))

    # s1 -> B1 + D1 = 2 mod 4
    # s2 -> B1 + B2 + D1 + D2 = 2 mod 4
    # s3 -> C1 - E1 = 2 mod 4
    # s4 -> C1 + C2 - E1 - E2 = 2 mod 4

    _delta_2_conj = delta_2.conjugate()
    tmp = delta_1 * _delta_2_conj
    (D1, D2) = (tmp.a.v // N, tmp.b.v // N)

    tmp = _delta_2_conj
    (xC1, xC2) = (tmp.a.v, tmp.b.v)

    tmp = delta_1.conjugate()
    (xE1, xE2) = (tmp.a.v, tmp.b.v)

    # first check z, it must be = 1 mod 8, = 5 mod 8 or = 2 mod 8
    z_good = z % 8 == 1 or z % 8 == 5 or z % 8 == 2

    # second initialise a bool to store which case we are in
    # when z % 8 == 2
    which_case_when_2mod8 = QartonBool(0)

    conds = BitVector(z_good, 0)

    if N % 2 == 0:
        conds[1] ^= 1

    elif z % 8 == 1:
        conds[1] ^= (D1 % 4 != 2) and ((D1 + D2) % 4 != 2)

    elif z % 8 == 5:
        conds[1] ^= (D1 % 4 != 0) and ((D1 + D2) % 4 != 0)
    else:
        # z%8 == 2 -> b2 odd
        # Two possibilities for (b1, b2) % 4
        # Case 1: (1, 1)
        case1 = (
            (N + D1) % 4 != 2
            and (N + D1 + D2) % 4 != 2
            and (xC1 - xE1) % 4 != 2
            and (xC1 + xC2 - xE1 - xE2) % 4 != 2
        )
        if case1:
            which_case_when_2mod8 = QartonBool(0)
        case2 = (
            (-N + D1) % 4 != 2
            and (-N + D1 + D2) % 4 != 2
            and (xC1 + xE1) % 4 != 2
            and (xC1 + xC2 + xE1 + xE2) % 4 != 2
        )
        if case2:
            which_case_when_2mod8 = QartonBool(1)
        conds[1] ^= case1 or case2

    assert abs(D1) < (1 << w)  # not by much
    assert abs(D2) < (1 << w)

    if all(conds) and False:
        print(d)
        print(delta_1)
        print(delta_2)
        print(N)
        print(z)
        print("----")
        print(conds)
        print(SInt(D1, w))
        print(SInt(D2, w))
        print(which_case_when_2mod8)

    return conds, SInt(D1, w), SInt(D2, w), which_case_when_2mod8


def _qc_not_eq_mod4(x: Register, value: int, out: Register) -> None:
    r"""XOR ``(x != value)`` into ``out``, where ``x`` is a 2-bit register holding a
    value mod 4, and ``value`` is 0, 1, 2 or 3. ``x`` is preserved. Self-inverse
    (calling it twice with the same arguments cancels ``out`` back to its original
    value), so it also serves as the erasure of its own output.
    """
    qc = x.parent_circuit
    flips = [i for i in range(2) if (value >> i) & 1]
    for i in flips:
        qc.x(x[i])
    qc_or(x[0], x[1], out)
    for i in flips:
        qc.x(x[i])


def qc_mod4_not_eq(x: Register, value: int) -> Register:
    """Allocates a fresh Boolean register holding ``x % 4 != value``."""
    qc = x.parent_circuit
    out = qc.get_anc(BoolType())
    _qc_not_eq_mod4(x, value, out)
    return out


def qc_mod4_not_eq_erase(x: Register, value: int, out: Register) -> None:
    """Erases a Boolean register obtained from ``qc_mod4_not_eq``."""
    _qc_not_eq_mod4(x, value, out)
    x.parent_circuit.release_anc(out)


@dummify
class FixCoefft(Circuit):
    """
    Given delta1 and delta2, computes the singularity conditions.
    """

    def __init__(self, d: InstanceData) -> None:
        super().__init__()
        self.d = d
        w = d.w
        delta_width = w // 2 + w // 4 + 10
        delta1 = self.add_input_output(RingIntegerType(d.p, delta_width))
        delta2 = self.add_input_output(RingIntegerType(d.p, delta_width))
        n_reg = self.add_input_output(UIntType(d.w // 2))
        z_reg = self.add_input_output(UIntType(d.w // 2 + 20))

        # bits used only to select the branch of fix_coeff we're in; they are read
        # (as controls), never modified, so n_reg/z_reg stay untouched through them
        n0 = n_reg[0]
        z0, z1, z2 = z_reg[0], z_reg[1], z_reg[2]

        qc_conj(delta2)
        tmp = qc_mul_ring_int(delta1, delta2)  # not necessarily positive
        tmpa, tmpb = tmp.a["a"], tmp.a["b"]

        n_reg = qc_make_signed(n_reg)  # N >= 0 always; needed signed for the division

        d1, rema, n_reg = qc_div_signed(tmpa, n_reg)  # ---> to output
        d2, remb, n_reg = qc_div_signed(tmpb, n_reg)  # ---> to output

        d1 = qc_truncate_sint(d1, w)
        d2 = qc_truncate_sint(d2, w)

        self.test_anc(rema, remb)
        self.assert_anc(rema, remb)
        # d1 and d2 are now constructed

        # bit-vector for condition bits
        conds = self.add_anc_output(2)

        self.add_output(d1)
        self.add_output(d2)

        # tiny special case when z = 2 mod 8
        which_case_when_2mod8 = self.add_anc_output(BoolType())

        # ----- now computing the conditions
        xc1, xc2 = delta2.a["a"], delta2.a["b"]
        qc_conj(delta1)
        xe1, xe2 = delta1.a["a"], delta1.a["b"]

        # ================================================================
        # z_good = (z % 8 in {1, 2, 5}), and the (mutually exclusive) branch
        # selectors e0..e3 matching the if/elif/elif/else chain of fix_coeff:
        #   e0: N even
        #   e1: N odd and z % 8 == 1
        #   e2: N odd and z % 8 == 5
        #   e3: N odd and z % 8 not in {1, 5}   (covers the "else" branch)
        # ================================================================
        y0, y1, y2 = symbols("x0 x1 x2", boolean=True)
        is1_z = y0 & ~y1 & ~y2
        is2_z = ~y0 & y1 & ~y2
        is5_z = y0 & ~y1 & y2
        z_good_formula = is1_z | is2_z | is5_z
        self.append(
            FormulaEvaluation(z_good_formula),
            qc_reg_from_bits([z0, z1, z2]),
            conds[0:1],
        )

        x0, x1, x2, x3 = symbols("x0 x1 x2 x3", boolean=True)
        is1_nz = x1 & ~x2 & ~x3
        is5_nz = x1 & ~x2 & x3
        e0_formula = ~x0
        e1_formula = x0 & is1_nz
        e2_formula = x0 & is5_nz
        e3_formula = x0 & ~is1_nz & ~is5_nz

        e0 = self.get_anc(BoolType())
        e1 = self.get_anc(BoolType())
        e2 = self.get_anc(BoolType())
        e3 = self.get_anc(BoolType())
        self.append(FormulaEvaluation(e0_formula), qc_reg_from_bits([n0]), e0)
        self.append(
            FormulaEvaluation(e1_formula), qc_reg_from_bits([n0, z0, z1, z2]), e1
        )
        self.append(
            FormulaEvaluation(e2_formula), qc_reg_from_bits([n0, z0, z1, z2]), e2
        )
        self.append(FormulaEvaluation(e3_formula), qc_reg_from_bits([n0, z0, z1]), e3)

        # ================================================================
        # condA: (D1 % 4 != 2) and ((D1 + D2) % 4 != 2)     -- z % 8 == 1 branch
        # condB: (D1 % 4 != 0) and ((D1 + D2) % 4 != 0)     -- z % 8 == 5 branch
        # (everything below is kept alive until the final combine step, then
        # erased in one block, in reverse order of creation)
        # ================================================================
        d1mod4_a = qc_mod4(d1)
        condA1 = qc_mod4_not_eq(d1mod4_a, 2)
        condB1 = qc_mod4_not_eq(d1mod4_a, 0)
        d2mod4_a = qc_mod4(d2)
        qc_add_uint(d2mod4_a, d1mod4_a)  # d1mod4_a = (D1 + D2) % 4
        condA2 = qc_mod4_not_eq(d1mod4_a, 2)
        condB2 = qc_mod4_not_eq(d1mod4_a, 0)

        condA = self.get_anc(BoolType())
        condB = self.get_anc(BoolType())
        self.ccx(condA1[0], condA2[0], condA[0])
        self.ccx(condB1[0], condB2[0], condB[0])

        # ================================================================
        # case1: (N+D1)%4!=2 and (N+D1+D2)%4!=2 and (xC1-xE1)%4!=2
        #        and (xC1+xC2-xE1-xE2)%4!=2                 -- z % 8 == 2, case 1
        # ================================================================
        nmod4_1 = qc_mod4(n_reg)
        d1mod4_1 = qc_mod4(d1)
        qc_add_uint(d1mod4_1, nmod4_1)  # nmod4_1 = (N + D1) % 4
        t1 = qc_mod4_not_eq(nmod4_1, 2)
        d2mod4_1 = qc_mod4(d2)
        qc_add_uint(d2mod4_1, nmod4_1)  # nmod4_1 = (N + D1 + D2) % 4
        t2 = qc_mod4_not_eq(nmod4_1, 2)

        xc1mod4_1 = qc_mod4(xc1)
        negxe1mod4_1 = qc_minusmod4(xe1)
        qc_add_uint(negxe1mod4_1, xc1mod4_1)  # xc1mod4_1 = (xC1 - xE1) % 4
        t3 = qc_mod4_not_eq(xc1mod4_1, 2)
        xc2mod4_1 = qc_mod4(xc2)
        qc_add_uint(xc2mod4_1, xc1mod4_1)
        negxe2mod4_1 = qc_minusmod4(xe2)
        qc_add_uint(negxe2mod4_1, xc1mod4_1)  # xc1mod4_1 = (xC1+xC2-xE1-xE2) % 4
        t4 = qc_mod4_not_eq(xc1mod4_1, 2)

        case1 = self.get_anc(BoolType())
        qc_mcx(qc_reg_from_bits([t1[0], t2[0], t3[0], t4[0]]), case1)

        # ================================================================
        # case2: (-N+D1)%4!=2 and (-N+D1+D2)%4!=2 and (xC1+xE1)%4!=2
        #        and (xC1+xC2+xE1+xE2)%4!=2                 -- z % 8 == 2, case 2
        # ================================================================
        negnmod4_2 = qc_minusmod4(n_reg)
        d1mod4_2 = qc_mod4(d1)
        qc_add_uint(d1mod4_2, negnmod4_2)  # negnmod4_2 = (-N + D1) % 4
        u1 = qc_mod4_not_eq(negnmod4_2, 2)
        d2mod4_2 = qc_mod4(d2)
        qc_add_uint(d2mod4_2, negnmod4_2)  # negnmod4_2 = (-N + D1 + D2) % 4
        u2 = qc_mod4_not_eq(negnmod4_2, 2)

        xc1mod4_2 = qc_mod4(xc1)
        xe1mod4_2 = qc_mod4(xe1)
        qc_add_uint(xe1mod4_2, xc1mod4_2)  # xc1mod4_2 = (xC1 + xE1) % 4
        u3 = qc_mod4_not_eq(xc1mod4_2, 2)
        xc2mod4_2 = qc_mod4(xc2)
        qc_add_uint(xc2mod4_2, xc1mod4_2)
        xe2mod4_2 = qc_mod4(xe2)
        qc_add_uint(xe2mod4_2, xc1mod4_2)  # xc1mod4_2 = (xC1+xC2+xE1+xE2) % 4
        u4 = qc_mod4_not_eq(xc1mod4_2, 2)

        case2 = self.get_anc(BoolType())
        qc_mcx(qc_reg_from_bits([u1[0], u2[0], u3[0], u4[0]]), case2)

        # ================================================================
        # combine: conds[1] = e0 or (e1 and condA) or (e2 and condB)
        #                      or (e3 and (case1 or case2))
        #          which_case_when_2mod8 = e3 and case2
        # (the e0..e3 are mutually exclusive, so "or" here behaves like xor)
        # ================================================================
        c0, c1, c2, c3, c4, c5, c6, c7 = symbols(
            "x0 x1 x2 x3 x4 x5 x6 x7", boolean=True
        )
        conds1_formula = c0 | (c1 & c4) | (c2 & c5) | (c3 & (c6 | c7))
        combine_bits = [
            e0[0],
            e1[0],
            e2[0],
            e3[0],
            condA[0],
            condB[0],
            case1[0],
            case2[0],
        ]
        self.append(
            FormulaEvaluation(conds1_formula),
            qc_reg_from_bits(combine_bits),
            conds[1:2],
        )

        wc0, wc1 = symbols("x0 x1", boolean=True)
        which_case_formula = wc0 & wc1
        self.append(
            FormulaEvaluation(which_case_formula),
            qc_reg_from_bits([e3[0], case2[0]]),
            which_case_when_2mod8,
        )

        # ================================================================
        # erase everything used only to build conds[1] / which_case_when_2mod8,
        # in reverse order of creation
        # ================================================================
        qc_mcx(qc_reg_from_bits([u1[0], u2[0], u3[0], u4[0]]), case2)
        self.release_anc(case2)

        qc_mod4_not_eq_erase(xc1mod4_2, 2, u4)
        qc_sub_uint(xe2mod4_2, xc1mod4_2)
        qc_mod4_erase(xe2, xe2mod4_2)
        qc_sub_uint(xc2mod4_2, xc1mod4_2)
        qc_mod4_erase(xc2, xc2mod4_2)
        qc_mod4_not_eq_erase(xc1mod4_2, 2, u3)
        qc_sub_uint(xe1mod4_2, xc1mod4_2)
        qc_mod4_erase(xe1, xe1mod4_2)
        qc_mod4_erase(xc1, xc1mod4_2)

        qc_mod4_not_eq_erase(negnmod4_2, 2, u2)
        qc_sub_uint(d2mod4_2, negnmod4_2)
        qc_mod4_erase(d2, d2mod4_2)
        qc_mod4_not_eq_erase(negnmod4_2, 2, u1)
        qc_sub_uint(d1mod4_2, negnmod4_2)
        qc_mod4_erase(d1, d1mod4_2)
        qc_minusmod4_erase(n_reg, negnmod4_2)

        qc_mcx(qc_reg_from_bits([t1[0], t2[0], t3[0], t4[0]]), case1)
        self.release_anc(case1)

        qc_mod4_not_eq_erase(xc1mod4_1, 2, t4)
        qc_sub_uint(negxe2mod4_1, xc1mod4_1)
        qc_minusmod4_erase(xe2, negxe2mod4_1)
        qc_sub_uint(xc2mod4_1, xc1mod4_1)
        qc_mod4_erase(xc2, xc2mod4_1)
        qc_mod4_not_eq_erase(xc1mod4_1, 2, t3)
        qc_sub_uint(negxe1mod4_1, xc1mod4_1)
        qc_minusmod4_erase(xe1, negxe1mod4_1)
        qc_mod4_erase(xc1, xc1mod4_1)

        qc_mod4_not_eq_erase(nmod4_1, 2, t2)
        qc_sub_uint(d2mod4_1, nmod4_1)
        qc_mod4_erase(d2, d2mod4_1)
        qc_mod4_not_eq_erase(nmod4_1, 2, t1)
        qc_sub_uint(d1mod4_1, nmod4_1)
        qc_mod4_erase(d1, d1mod4_1)
        qc_mod4_erase(n_reg, nmod4_1)

        self.ccx(condB1[0], condB2[0], condB[0])
        self.release_anc(condB)
        self.ccx(condA1[0], condA2[0], condA[0])
        self.release_anc(condA)

        qc_mod4_not_eq_erase(d1mod4_a, 0, condB2)
        qc_mod4_not_eq_erase(d1mod4_a, 2, condA2)
        qc_sub_uint(d2mod4_a, d1mod4_a)
        qc_mod4_erase(d2, d2mod4_a)
        qc_mod4_not_eq_erase(d1mod4_a, 0, condB1)
        qc_mod4_not_eq_erase(d1mod4_a, 2, condA1)
        qc_mod4_erase(d1, d1mod4_a)

        self.append(FormulaEvaluation(e3_formula), qc_reg_from_bits([n0, z0, z1]), e3)
        self.release_anc(e3)
        self.append(
            FormulaEvaluation(e2_formula), qc_reg_from_bits([n0, z0, z1, z2]), e2
        )
        self.release_anc(e2)
        self.append(
            FormulaEvaluation(e1_formula), qc_reg_from_bits([n0, z0, z1, z2]), e1
        )
        self.release_anc(e1)
        self.append(FormulaEvaluation(e0_formula), qc_reg_from_bits([n0]), e0)
        self.release_anc(e0)

        # restore delta1, delta2, n_reg to their original (preserved) forms
        qc_conj(delta1)
        qc_conj(delta2)
        n_reg = qc_make_unsigned(n_reg)

        self.remap(
            delta1,
            delta2,
            n_reg,
            z_reg,
            conds,
            d1,
            d2,
            which_case_when_2mod8,
        )

    def dummy_classical_function(
        self,
        args: tuple[RingInteger, RingInteger, UInt, UInt],
    ) -> tuple[RingInteger, RingInteger, UInt, UInt, BitVector, SInt, SInt, QartonBool]:
        delta_1, delta_2, N, z = args
        conds, D1, D2, which_case_when_2mod8 = fix_coeff(self.d, delta_1, delta_2, N, z)
        return (delta_1, delta_2, N, z, conds, D1, D2, which_case_when_2mod8)

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            RingInteger, RingInteger, UInt, UInt, BitVector, SInt, SInt, QartonBool
        ],
    ) -> tuple[RingInteger, RingInteger, UInt, UInt]:
        delta_1, delta_2, N, z, _, _, _, _ = args
        return delta_1, delta_2, N, z


def first_test(
    d: InstanceData,
    algo_input: tuple[UInt, RingInteger],
    precomp_result: tuple[UInt, UInt, UInt, UInt],
    rchoice: UInt,
    test_bounds: bool = False,
) -> tuple[RingInteger, RingInteger, SInt, SInt, UInt, BitVector, QartonBool]:
    """From the input of the algorithm and the precomputation result, perform the
    first test and returns delta1, delta2, D1, D2 (corresponding to the D1, D2 in qt-Pegasis),
    z and a BitVector whose bits indicate the tests which were passed / failed.

    :param d: Global parameters of the algorithm.
    :type d: InstanceData
    :param algo_input: Input of the algorithm (N, alpha)
    :type algo_input: tuple[int, RingInteger]
    :param precomp_result: Result of the precomputation
    :type precomp_result: tuple[int, int, int, int]
    :param rchoice: Random choice which determines the vlaue of k
    :type rchoice: int
    :return: (delta_1, delta_2, D1, D2, z, bvv) where bvv is the bitvector of test results.
    :rtype: tuple[RingInteger, RingInteger, int, int, int, BitVector]
    """
    w, p = d.w, d.p
    N, ring_int = algo_input

    assert abs(N) < (1 << (w // 2))
    assert abs(ring_int.a) < (1 << (w // 2))
    assert abs(ring_int.b) < (1 << (w // 2))

    n_tr_alpha = (N, ring_int.trace())

    (s, t, z, k, bv) = compute_stzk_classical(d, n_tr_alpha, precomp_result, rchoice)

    delta_1 = RingInteger(p, w, s.v * N + ring_int.a.v, ring_int.b.v)
    delta_2 = RingInteger(p, w, t.v * N + k * ring_int.a.v, k * ring_int.b.v)

    # correct size of delta_1, delta_2
    delta_1 = RingInteger(delta_1.p, w // 2 + w // 4 + 10, delta_1.a, delta_1.b)
    delta_2 = RingInteger(delta_2.p, w // 2 + w // 4 + 10, delta_2.a, delta_2.b)

    if z % 4 == 2:
        # Check to avoid products
        tt = delta_1 + delta_2
        newcond = (tt.a.v + tt.b.v) % 2  # if == 0 then we continue
    else:
        newcond = 1

    conds, D1, D2, which_case_when_2mod8 = fix_coeff(d, delta_1, delta_2, N, z)
    bvv = BitVector(bv, conds, newcond)

    # bounds
    assert abs(delta_1.a.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(delta_1.b.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(delta_2.a.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(delta_2.b.v) < (1 << (w // 2 + w // 4 + 10))
    assert abs(D1) < (1 << w)  # not by much
    assert abs(D2) < (1 << w)
    assert abs(z) < (1 << (w // 2 + 20))

    if all(bvv) and False:
        print(algo_input)
        print(precomp_result)
        print(rchoice)
        print("----")
        print(delta_1)
        print(delta_2)
        print(D1)
        print(D2)
        print(z)
        print(bvv)
        print(which_case_when_2mod8)

    # output what we need for the next computations
    return (delta_1, delta_2, D1, D2, z, bvv, which_case_when_2mod8)


@dummify
@memoize
class FirstCheck(
    Circuit[
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            RingInteger,
            RingInteger,
            SInt,
            SInt,
            UInt,
            BitVector,
            QartonBool,
        ],
    ]
):
    """
    Inputs:

    * the input of the algorithm: N (IntType(w//2)) and alpha (RingIntegerType(p, w//2))
    * the output of the precomputation: IntType(w // 4), IntType(w // 2), IntType(w // 2), IntType(w // 2 + 10)
    * the choice

    Additional outputs:

    * delta_1: RingIntegerType(w//2 + w//4 + 10)
    * delta_2: RingIntegerType(w//2 + w//4 + 10)
    * D1: SIntType(w)
    * D2: SIntType(w)
    * z: IntType(w // 2 + 20)
    * A small BitVector with the test results
    * which_case_when_2mod8: a QartonBool used only when z % 8 == 2
    """

    def __init__(
        self,
        d: InstanceData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        w, e, p, x = d.w, d.e, d.p, d.x
        self.d = d
        self.w = w
        self.e = e
        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )
        n_reg, ring_int = algo_input.a[0], algo_input.a[1]

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )

        choice = self.add_input_output(UIntType(x))
        # ------------

        tr_alpha = qc_trace(ring_int)
        tr_alpha, tr_alpha_sgn = qc_abs(tr_alpha)

        n_trace = qc_reg_cast(
            n_reg + tr_alpha, ITupleType(UIntType(w // 2), UIntType(w // 2))
        )
        _, _, _, s_reg, t_reg, z_reg, k_reg, bv_reg = self.append(
            PreCircuit(ComputeSTZK, d), n_trace, precomp_result_reg, choice
        )

        # compute s * N + ring_int.a.x
        # start by s * N, of size w
        # make n signed
        n_sgn = self.get_anc(BoolType())
        n_reg = qc_abs_uncompute(n_reg, n_sgn)
        k_sgn = self.get_anc(BoolType())
        k_reg = qc_abs_uncompute(k_reg, k_sgn)

        delta1_a = qc_mul(s_reg, n_reg)
        qc_add(ring_int.a["a"], delta1_a)
        delta1_b = qc_copy(ring_int.a["b"])

        delta1 = qc_make_ring_integer(delta1_a, delta1_b, p)
        assert isinstance(delta1.datatype, RingIntegerType)
        assert delta1.datatype.width == w // 2 + w // 4 + 10

        delta2_a = qc_mul(t_reg, n_reg)
        ka = qc_mul(k_reg, ring_int.a["a"])
        qc_add(ka, delta2_a)
        qc_mul_erase(k_reg, ring_int.a["a"], ka)
        delta2_b = qc_mul(k_reg, ring_int.a["b"])

        delta2 = qc_make_ring_integer(delta2_a, delta2_b, p)
        assert isinstance(delta2.datatype, RingIntegerType)
        assert delta2.datatype.width == w // 2 + w // 4 + 10

        k_reg, k_sgn = qc_abs(k_reg)
        self.test_anc(k_sgn)
        self.assert_anc(k_sgn)

        # erase s,t,...
        z_copy = qc_copy(z_reg)
        bv_copy = qc_copy(bv_reg)

        # reconstruct n_trace because we keep taking the sign on and off from n_reg
        n_reg, n_sgn = qc_abs(n_reg)
        self.test_anc(n_sgn)
        self.assert_anc(n_sgn)
        n_trace = qc_reg_cast(
            n_reg + tr_alpha, ITupleType(UIntType(w // 2), UIntType(w // 2))
        )
        self.append(
            PreCircuit(ComputeSTZK, d).inverse(),
            n_trace,
            precomp_result_reg,
            choice,
            s_reg,
            t_reg,
            z_reg,
            k_reg,
            bv_reg,
        )
        tr_alpha = qc_abs_uncompute(tr_alpha, tr_alpha_sgn)
        qc_trace_erase(ring_int, tr_alpha)

        # -----------------------------------

        # delta1, delta2, D1, D2, the mod-4 conditions and which_case_when_2mod8
        # are all delegated to FixCoefft, which preserves delta1, delta2, n_reg
        # and z_copy (its "z" input) unchanged
        delta1, delta2, n_reg, z_copy, conds, d1, d2, which_case_when_2mod8 = (
            self.append(PreCircuit(FixCoefft, d), delta1, delta2, n_reg, z_copy)
        )

        # ------------------------------------------------------------------
        # newcond: extra check on z % 4 == 2 (kept here since it isn't part of
        # fix_coeff / FixCoefft)
        # ------------------------------------------------------------------
        z0, z1 = z_copy[0], z_copy[1]
        zf0, zf1 = symbols("x0 x1", boolean=True)
        is_z4eq2_formula = ~zf0 & zf1
        flag_z4eq2 = self.get_anc(BoolType())
        self.append(
            FormulaEvaluation(is_z4eq2_formula),
            qc_reg_from_bits([z0, z1]),
            flag_z4eq2,
        )

        parity = self.get_anc(BoolType())
        self.cx(delta1.a["a"][0], parity[0])
        self.cx(delta1.a["b"][0], parity[0])
        self.cx(delta2.a["a"][0], parity[0])
        self.cx(delta2.a["b"][0], parity[0])

        newcond = self.get_anc(BoolType())
        nf0, nf1 = symbols("x0 x1", boolean=True)
        newcond_formula = ~nf0 | nf1
        self.append(
            FormulaEvaluation(newcond_formula),
            qc_reg_from_bits([flag_z4eq2[0], parity[0]]),
            newcond,
        )

        # erase flag_z4eq2 and parity, only used to build newcond
        self.cx(delta2.a["b"][0], parity[0])
        self.cx(delta2.a["a"][0], parity[0])
        self.cx(delta1.a["b"][0], parity[0])
        self.cx(delta1.a["a"][0], parity[0])
        self.release_anc(parity)

        self.append(
            FormulaEvaluation(is_z4eq2_formula),
            qc_reg_from_bits([z0, z1]),
            flag_z4eq2,
        )
        self.release_anc(flag_z4eq2)

        self.add_output(delta1)  # size w//2 + w//4 + 10
        self.add_output(delta2)  # size w//2 + w//4 + 10
        self.add_output(d1)  # size w
        self.add_output(d2)  # size w
        self.add_output(z_copy)  # size w//2 + 20

        o = bv_copy + conds + newcond
        self.add_output(o)  # length 3 + 2 + 1

        self.add_output(which_case_when_2mod8)

        algo_input = qc_make_ituple(n_reg, ring_int)
        self.remap(
            algo_input,
            precomp_result_reg,
            choice,
            delta1,
            delta2,
            d1,
            d2,
            z_copy,
            o,
            which_case_when_2mod8,
        )

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ) -> tuple[
        tuple[UInt, RingInteger],
        tuple[UInt, UInt, UInt, UInt],
        UInt,
        RingInteger,
        RingInteger,
        SInt,
        SInt,
        UInt,
        BitVector,
        QartonBool,
    ]:
        algo_input, precomp_result, rchoice = args
        (delta_1, delta_2, d1, d2, z, bvv, which_case_when_2mod8) = first_test(
            self.d, algo_input, precomp_result, rchoice
        )

        return (
            algo_input,
            precomp_result,
            rchoice,
            delta_1,
            delta_2,
            d1,
            d2,
            UInt(z),
            bvv,
            which_case_when_2mod8,
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            RingInteger,
            RingInteger,
            SInt,
            SInt,
            UInt,
            BitVector,
            QartonBool,
        ],
    ) -> tuple[
        tuple[UInt, RingInteger],
        tuple[UInt, UInt, UInt, UInt],
        UInt,
    ]:
        algo_input, precomp_result, rchoice, _, _, _, _, _, _, _ = args
        return algo_input, precomp_result, rchoice


@dummify
@memoize
class FirstCheckTestOnly(
    Circuit[
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            UInt,
            BitVector,
            QartonBool,
        ],
    ]
):
    """
    Same code as FirstTest, but we only return z, which is the only thing we need
    for later tests (and the BitVector of test results, and which_case_when_2mod8).

    """

    def __init__(
        self,
        d: InstanceData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        w, e, p, x = d.w, d.e, d.p, d.x
        self.d = d
        self.w = w
        self.e = e
        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )
        n_reg, ring_int = algo_input.a[0], algo_input.a[1]

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )

        choice = self.add_input_output(UIntType(x))
        # ------------

        tr_alpha = qc_trace(ring_int)
        tr_alpha, tr_alpha_sgn = qc_abs(tr_alpha)
        # print(self._memory_manager.used_bit_pool)

        n_trace = qc_reg_cast(
            n_reg + tr_alpha, ITupleType(UIntType(w // 2), UIntType(w // 2))
        )
        _, _, _, s_reg, t_reg, z_reg, k_reg, bv_reg = self.append(
            PreCircuit(ComputeSTZK, d), n_trace, precomp_result_reg, choice
        )

        # compute s * N + ring_int.a.x
        # start by s * N, of size w
        # make n signed
        n_sgn = self.get_anc(BoolType())
        n_reg = qc_abs_uncompute(n_reg, n_sgn)

        k_sgn = self.get_anc(BoolType())
        k_reg = qc_abs_uncompute(k_reg, k_sgn)

        # --- BEGIN: computation of delta1, delta2
        delta1_a = qc_mul(s_reg, n_reg)
        qc_add(ring_int.a["a"], delta1_a)
        delta1_b = qc_copy(ring_int.a["b"])

        delta1 = qc_make_ring_integer(delta1_a, delta1_b, p)
        assert isinstance(delta1.datatype, RingIntegerType)
        assert delta1.datatype.width == w // 2 + w // 4 + 10

        delta2_a = qc_mul(t_reg, n_reg)
        ka = qc_mul(k_reg, ring_int.a["a"])
        qc_add(ka, delta2_a)
        qc_mul_erase(k_reg, ring_int.a["a"], ka)
        delta2_b = qc_mul(k_reg, ring_int.a["b"])

        delta2 = qc_make_ring_integer(delta2_a, delta2_b, p)
        assert isinstance(delta2.datatype, RingIntegerType)
        assert delta2.datatype.width == w // 2 + w // 4 + 10
        # ---- END: computation of delta1, delta2

        # delta1, delta2, D1, D2, the mod-4 conditions and which_case_when_2mod8
        # are all delegated to FixCoefft, which preserves delta1, delta2 and z_reg
        # unchanged (it needs N unsigned)
        n_reg = qc_make_unsigned(n_reg)
        delta1, delta2, n_reg, z_reg, conds, d1, d2, which_case_when_2mod8 = (
            self.append(PreCircuit(FixCoefft, d), delta1, delta2, n_reg, z_reg)
        )

        # ------------------------------------------------------------------
        # newcond: extra check on z % 4 == 2 (kept here since it isn't part of
        # fix_coeff / FixCoefft); has to be computed now, before delta1, delta2
        # get erased below
        # ------------------------------------------------------------------
        z0, z1 = z_reg[0], z_reg[1]
        zf0, zf1 = symbols("x0 x1", boolean=True)
        is_z4eq2_formula = ~zf0 & zf1
        flag_z4eq2 = self.get_anc(BoolType())
        self.append(
            FormulaEvaluation(is_z4eq2_formula),
            qc_reg_from_bits([z0, z1]),
            flag_z4eq2,
        )

        parity = self.get_anc(BoolType())
        self.cx(delta1.a["a"][0], parity[0])
        self.cx(delta1.a["b"][0], parity[0])
        self.cx(delta2.a["a"][0], parity[0])
        self.cx(delta2.a["b"][0], parity[0])

        newcond = self.get_anc(BoolType())
        nf0, nf1 = symbols("x0 x1", boolean=True)
        newcond_formula = ~nf0 | nf1
        self.append(
            FormulaEvaluation(newcond_formula),
            qc_reg_from_bits([flag_z4eq2[0], parity[0]]),
            newcond,
        )

        # erase flag_z4eq2 and parity, only used to build newcond
        self.cx(delta2.a["b"][0], parity[0])
        self.cx(delta2.a["a"][0], parity[0])
        self.cx(delta1.a["b"][0], parity[0])
        self.cx(delta1.a["a"][0], parity[0])
        self.release_anc(parity)

        self.append(
            FormulaEvaluation(is_z4eq2_formula),
            qc_reg_from_bits([z0, z1]),
            flag_z4eq2,
        )
        self.release_anc(flag_z4eq2)

        # ------------------------------------------------------------------
        # unlike FirstCheck, delta1, delta2, D1 and D2 are not part of the
        # output here: keep only copies of conds and which_case_when_2mod8,
        # then run FixCoefft's inverse to erase delta1, delta2, D1 and D2 (and
        # restore n_reg, z_reg to what they were right before the FixCoefft call)
        # ------------------------------------------------------------------
        conds_copy = qc_copy(conds)
        which_case_copy = qc_copy(which_case_when_2mod8)

        delta1, delta2, n_reg, z_reg = self.append(
            PreCircuit(FixCoefft, d, inverse=True),
            delta1,
            delta2,
            n_reg,
            z_reg,
            conds,
            d1,
            d2,
            which_case_when_2mod8,
        )

        n_reg = qc_make_signed(n_reg)

        # --- BEGIN: uncomputation of delta1, delta2
        # there are some truncations to make, because we silently made expansions in
        # the forward code
        delta2_b, delta2_a = delta2.a["b"], delta2.a["a"]
        delta2_b = qc_truncate_sint(delta2_b, w // 2 + w // 4)
        qc_mul_erase(k_reg, ring_int.a["b"], delta2_b)  # delta2_b erased

        ka = qc_mul(k_reg, ring_int.a["a"])
        qc_sub(ka, delta2_a)
        qc_mul_erase(k_reg, ring_int.a["a"], ka)
        qc_mul_erase(t_reg, n_reg, delta2_a)

        delta1a, delta1b = delta1.a["a"], delta1.a["b"]
        delta1_b = qc_truncate_sint(delta1b, w // 2)
        qc_copy_erase(ring_int.a["b"], delta1_b)
        qc_sub(ring_int.a["a"], delta1a)
        qc_mul_erase(s_reg, n_reg, delta1a)

        # --------------------

        k_reg, k_sgn = qc_abs(k_reg)
        self.test_anc(k_sgn)
        self.assert_anc(k_sgn)

        # erase s,t,...
        z_copy = qc_copy(z_reg)
        bv_copy = qc_copy(bv_reg)

        # reconstruct (again!) the n_trace reg because we keep taking the sign on and
        # off from n_reg
        n_reg, n_sgn = qc_abs(n_reg)
        self.release_anc(n_sgn)
        assert isinstance(tr_alpha.datatype, UIntType)
        n_trace = qc_reg_cast(
            n_reg + tr_alpha, ITupleType(UIntType(w // 2), UIntType(w // 2))
        )

        self.append(
            PreCircuit(ComputeSTZK, d, inverse=True),
            n_trace,
            precomp_result_reg,
            choice,
            s_reg,
            t_reg,
            z_reg,
            k_reg,
            bv_reg,
        )

        tr_alpha = qc_abs_uncompute(tr_alpha, tr_alpha_sgn)
        qc_trace_erase(ring_int, tr_alpha)

        # -----------------------------------

        self.add_output(z_copy)  # size w//2 + 20
        o = bv_copy + conds_copy + newcond
        self.add_output(o)  # length 3 + 2 + 1

        self.add_output(which_case_copy)

        algo_input = qc_make_ituple(n_reg, ring_int)
        self.remap(algo_input, precomp_result_reg, choice, z_copy, o, which_case_copy)

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ) -> tuple[
        tuple[UInt, RingInteger],
        tuple[UInt, UInt, UInt, UInt],
        UInt,
        UInt,
        BitVector,
        QartonBool,
    ]:
        algo_input, precomp_result, rchoice = args
        (_, _, _, _, z, bvv, which_case_when_2mod8) = first_test(
            self.d, algo_input, precomp_result, rchoice
        )

        return (
            algo_input,
            precomp_result,
            rchoice,
            UInt(z),
            bvv,
            which_case_when_2mod8,
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            UInt,
            BitVector,
            QartonBool,
        ],
    ) -> tuple[
        tuple[UInt, RingInteger],
        tuple[UInt, UInt, UInt, UInt],
        UInt,
    ]:
        algo_input, precomp_result, rchoice, _, _, _ = args
        return algo_input, precomp_result, rchoice
