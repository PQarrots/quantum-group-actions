"""
Gluing.
"""

import itertools
from typing import Any, cast

from qarton.binary_operations import (
    qc_cswap,
    qc_cxor,
    qc_eq,
    qc_load,
    qc_or,
    qc_swap,
    qc_unload,
    qc_xor,
)
from qarton.circuit import (
    BitVectorType,
    BoolType,
    Circuit,
    InPlaceCircuit,
    PreCircuit,
    QartonBool,
    Register,
    TupleType,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_add_modint,
    qc_cmuladd_modint,
    qc_cmulsub_modint,
    qc_cneg_modint,
    qc_dbl_modint,
    qc_halve_modint,
    qc_load_mod_int,
    qc_muladd_modint,
    qc_mulsub_modint,
    qc_sqradd_modint,
    qc_sqrsub_modint,
    qc_sub_modint,
)

from qisogenies.montgomery import (
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariableType,
    ECMontgomery,
)

from .theta_dim4 import (
    ThetaPointDim4Type,
    qc_theta_hadamard,
    qc_theta_hadamard_bitflipped,
)
from .theta_util import (
    CoordsDim2Type,
    CoordsDim4,
    CoordsDim4Type,
    classical_hadamard,
    qc_coordwise_add,
    qc_coordwise_muladd,
    qc_coordwise_mulsub,
    qc_coordwise_sqradd,
    qc_coordwise_sub,
    qc_theta_hadamard_dim2,
)

__all__ = [
    "qc_coordwise_cneg",
    "qc_switch",
    "BarycentricType",
    "ProjPointType",
    "EllipticPointsToBarycentric",
    "qc_ellpoints_to_barycentric",
    "qc_ellpoints_to_barycentric_erase",
    "n2matrix_superglue4d_classical_function",
    "N2Matrix_Superglue4D",
    "qc_n2_matrix",
    "qc_n2_matrix_inv",
    "Superglue4DThetaPlusMinus",
    "Superglue4DEvaluation",
    "superglue4drabbitcodomain_classical_function",
    "Superglue4DRabbitCodomain",
]


def qc_coordwise_cneg(c: Register, tuple_reg: Register) -> None:
    dtype = tuple_reg.datatype
    assert isinstance(dtype, (TupleType, ThetaPointDim4Type, CoordsDim4Type))
    l = dtype.tuple_length
    # in dimension 4, it should be 16
    for m in range(l):
        qc_cneg_modint(c, tuple_reg.a[m])


def xor(i: int, j: int) -> int:
    if 1 ^ 1 == 0:
        return i ^ j
    else:  # safety in case the sage interpreter does something weird
        raise NotImplementedError


def qc_switch(
    conditions: tuple[Register, ...],
    options: tuple[Register, ...],
    dest: Register,
    default_clause: bool = False,
) -> None:
    if (
        any(not isinstance(condition.datatype, BoolType) for condition in conditions)
        or len(options) == 0
        or len(options) != len(conditions) + int(default_clause)
        or any(opt_reg.datatype != dest.datatype for opt_reg in options)
    ):
        raise TypeError()

    qc = conditions[0].parent_circuit
    n = len(conditions)
    if default_clause:
        n += 1
        conditions = *conditions, qc_load(QartonBool(1), qc, BoolType())
    assert len(conditions) == len(options) == n
    mask = qc.add_ancs(*([BoolType()] * n))

    # let mask have all bit 0 except for the bit of the FIRST true condition in conditions
    # mask = 0001001010
    qc_xor(conditions[0], mask[0])
    for i in range(1, n):
        qc_or(mask[i - 1], conditions[i], mask[i])
    # mask = 0001111111
    for i in range(n - 1, 0, -1):
        qc_xor(mask[i - 1], mask[i])
    # mask = 0001000000

    for i in range(n):
        qc_cxor(mask[i], options[i], dest)

    ############# uncompute ##############
    for i in range(1, n):
        qc_xor(mask[i - 1], mask[i])
    for i in range(n - 1, 0, -1):
        qc_or(mask[i - 1], conditions[i], mask[i])
    qc_xor(conditions[0], mask[0])
    for bit in mask:
        qc.assert_anc(bit)
        qc.release_anc(bit)
    if default_clause:
        qc_unload(QartonBool(1), conditions[-1])


ModInt3 = tuple[ModInt, ModInt, ModInt]
AffMontProductPoint = tuple[
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariable,
]


def BarycentricType(p: int) -> TupleType[ModInt]:
    return TupleType(3, ModIntType(p))


def ProjPointType(p: int) -> TupleType[ModInt]:
    return TupleType(3, ModIntType(p))


# def CoordsDim4Type(p: int) -> TupleType[ModInt]:
#    return TupleType(16, ModIntType(p))


def ellipticpoints_to_barycentric_dummy_classical_function(
    args: tuple[
        AffMontgomeryPointVariable,
        AffMontgomeryPointVariable,
        QartonBool,
        QartonBool,
    ],
    A: ModInt,
    p: int,
) -> tuple[
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariable,
    QartonBool,
    QartonBool,
    ModInt3,
    QartonBool,
]:
    P, Q, cP, cQ = args
    xP, yP, xQ, yQ = (ModInt(coord, p) for coord in (P.x, P.y, Q.x, Q.y))

    if cP:
        xP = -xP
    if cQ:
        xQ = -xQ

    # TODO when P is 0 we want yP to be 1, I guess?
    zP = ModInt(P.not_infty, p)
    zQ = ModInt(Q.not_infty, p)

    ## lines 1--2
    b0 = QartonBool(zP == 0)
    b1 = QartonBool(zQ == 0)

    ## lines 3--6
    t1 = zP * zQ
    t2 = xP * zQ
    t3 = zP * xQ
    t4 = yP * zQ
    t5 = zP * yQ
    b2 = QartonBool(t4 == t5)

    t6 = t4 * t5
    v = t6 * t1 * 2

    ## lines 9--11
    t7 = A * t1
    t2 += t3
    t7 += t2
    t3 += t3
    t2 -= t3

    # lines 11b--19
    t2 = t2**2
    b3 = QartonBool(t2 == 0)
    t4 = t4**2
    t5 = t5**2
    # lines 15--16: conditionally negate t4, t5 depending whether P, resp. Q, is on the twist
    if cP:
        t4 = -t4
    if cQ:
        t5 = -t5
    t4 += t5
    u = t4 * t1
    u -= t7 * t2
    w = t2 * t1

    t0 = (xP + zP) ** 2
    t1 = (xP - zP) ** 2
    t5 = t0 * t1 * 2
    t2 = t0 - t1
    A24 = (A + ModInt(2, p)) * ModInt(4, p) ** (-1)
    t0 = A24 * t2
    t0 += t1
    t6 = t0 * t2

    t4 = -t6 if (b3 and b2) else t6
    if cP:
        t4 = -t4

    if b3:
        u, v, w = t5, t4, t6
    if b0:
        u, v, w = xQ, ModInt(0, p), zQ
    if b1:
        u, v, w = xP, ModInt(0, p), zP
    if b0 and b1:
        u = ModInt(1, p)

    uvw = u, v, w

    p_eq_pm_q = QartonBool(b3 and (not b0) and (not b1))
    if p_eq_pm_q and cP == cQ:
        assert P.x == Q.x
        assert v in (w, -w)

    return *args, uvw, p_eq_pm_q


@dummify
@memoize
class EllipticPointsToBarycentric(
    Circuit[
        tuple[
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            QartonBool,
            QartonBool,
        ],
        tuple[
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            QartonBool,
            QartonBool,
            tuple[ModInt, ModInt, ModInt],
            QartonBool,
        ],
    ]
):
    def __init__(
        self,
        E: ECMontgomery,
    ) -> None:
        """
        Compute barycentric coordinates (u:v:w) of (P, Q)
        from Superglue https://eprint.iacr.org/2026/114 Algorithm 3

        :param E: Montgomery curve where P, Q live
        :type E: ECMontgomery

        COST: 11M + 3S + 7a
        """
        super().__init__()
        self.E = E
        p = E.q
        self.A = ModInt(E.A, p)
        self.p = p

        zero = qc_load_mod_int(ModInt(0, p), self)
        one = qc_load_mod_int(ModInt(1, p), self)

        ### i/o
        P = self.add_input_output(AffMontgomeryPointVariableType(p))
        Q = self.add_input_output(AffMontgomeryPointVariableType(p))
        c_is_p_on_twist = self.add_input_output(BoolType())
        c_is_q_on_twist = self.add_input_output(BoolType())
        uvw = self.add_anc_output(BarycentricType(p))
        P_eq_pm_Q = self.add_anc_output(BoolType())

        # ------------------------------------------

        xP, yP = P.a["x"], P.a["y"]
        xQ, yQ = Q.a["x"], Q.a["y"]
        qc_cneg_modint(c_is_p_on_twist, xP)
        qc_cneg_modint(c_is_q_on_twist, xQ)
        # TODO when P is 0 we want yP to be 1, I guess?
        zP, zQ = self.add_ancs(ModIntType(p), ModIntType(p))
        self.cx(P.a["not_infty"][0], zP[0])
        self.cx(Q.a["not_infty"][0], zQ[0])
        u, v, w = (uvw.a[i] for i in range(3))

        ### ancillas

        (
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t2sq,
            t4sq,
            t5sq,
        ) = self.add_ancs(*([ModIntType(p)] * 10))  # ancillas t_i for lines 1--19
        u_, v_, w_ = self.add_ancs(*([ModIntType(p)] * 3))
        b0, b1, b2, b3, b0b1, b2b3, b0orb1 = self.add_ancs(*([BoolType()] * 7))

        ## lines 1--2
        qc_eq(zP, zero, b0)  # b0 = (zP =? 0)
        qc_eq(zQ, zero, b1)  # b1 = (zQ =? 0)

        ## lines 3--6
        qc_muladd_modint(zP, zQ, t1)  # t1 = zP · zQ
        qc_muladd_modint(xP, zQ, t2)  # t2 = xP · zQ
        qc_muladd_modint(zP, xQ, t3)  # t3 = zP · xQ
        qc_muladd_modint(yP, zQ, t4)  # t4 = yP · zQ
        qc_muladd_modint(zP, yQ, t5)  # t5 = zP · yQ
        qc_eq(t4, t5, b2)  # b2 = (t4 =? t5)

        qc_muladd_modint(t4, t5, t6)  # t6 = t4 · t5
        # v = 2 · t6 · t1
        qc_muladd_modint(t6, t1, v_)
        qc_dbl_modint(v_)

        ## lines 9--11
        A = qc_load_mod_int(self.A, self)
        qc_muladd_modint(A, t1, t7)  # t7 = A · t1
        qc_add_modint(t3, t2)  # t2 += t3
        qc_add_modint(t2, t7)  # t7 += t2
        qc_dbl_modint(t3)  # t3 += t3
        qc_sub_modint(t3, t2)  # t2 -= t3

        # lines 11b--19
        qc_sqradd_modint(t2, t2sq)  # line 11a: t2 = t2^2. Save new t2 into t2sq
        qc_eq(zero, t2sq, b3)  # line 12: b3 = (t2 =? 0)
        qc_sqradd_modint(t4, t4sq)  # line 13b: t4 = t4^2. Save new t4 into t4sq
        qc_sqradd_modint(t5, t5sq)  # line 14: t5 = t5^2. Save new t5 into t5sq
        # lines 15--16: conditionally negate t4, t5 depending whether P, resp. Q, is on the twist
        qc_cneg_modint(c_is_p_on_twist, t4sq)
        qc_cneg_modint(c_is_q_on_twist, t5sq)
        # line 17a: t4(new) += t5(new)
        qc_add_modint(t5sq, t4sq)
        # line 18a: u += t4(new) · t1
        qc_muladd_modint(t4sq, t1, u_)
        # lines 13a and 18b: u -= t7 · t2(new)
        qc_mulsub_modint(t7, t2sq, u_)
        # line 19: w = t2(new) · t1
        qc_muladd_modint(t2sq, t1, w_)

        ###############
        tt0_, tt0, tt1_, tt1, tt4, tt5, tt6, tt0new = self.add_ancs(
            *([ModIntType(p)] * 8)
        )  # ancillas t_i for lines 20--30 # XXX change number of ancillas
        # line 20: t0 = (xP + zP)^2.
        qc_xor(xP, tt0_)
        qc_add_modint(zP, tt0_)
        qc_sqradd_modint(tt0_, tt0)
        # line 21: t1 = (xP - zP)^2.
        qc_xor(xP, tt1_)
        qc_sub_modint(zP, tt1_)
        qc_sqradd_modint(tt1_, tt1)
        # lines 22b, 23: t5 = 2 · t0 · t1.
        qc_muladd_modint(tt0, tt1, tt5)
        # line 22a: t2 = t0 - t1
        qc_sub_modint(tt1, tt0)
        tt2 = tt0
        # line 23b: t0 = ((A + 2)/4) · t2. Save in t0new
        A24 = self.A + ModInt(2, p) * ModInt(4, p) ** (-1)
        qc_muladd_modint(A24, tt2, tt0new)
        # line 24a: t0 += t1
        qc_add_modint(tt1, tt0new)
        # line 24b: t6 = t0 · t2.
        qc_muladd_modint(tt0new, tt2, tt6)

        # line 25a: t4 = (b3 & b2 ? t6 : -t6).
        self.ccx(b3[0], b2[0], b2b3[0])
        qc_xor(tt6, tt4)
        qc_cneg_modint(b2b3, tt4)
        qc_cneg_modint(c_is_p_on_twist, tt4)

        # line 25b: u = (b3 ? u : t5)
        # line 27b: u = (b0 ? u : xQ)
        # line 28b: u = (b1 ? u : xP)
        # line 29b: u = (b0 & b1 ? u : 1)
        # rephrased:
        # if b0b1:
        #     u = 1
        # elif b1:
        #     u = xP
        # elif b0:
        #     u = xQ
        # elif b3:
        #     u = t5
        # else:
        #     u = u
        self.ccx(b0[0], b1[0], b0b1[0])
        qc_switch((b0b1, b1, b0, b3), (one, xP, xQ, tt5, u_), u, default_clause=True)

        # line 26a: v = (b3 ? v : t4)
        # line 27a: v = (b0 or b1 ? v : 0)
        qc_or(b0, b1, b0orb1)
        qc_switch((b0orb1, b3), (zero, t4, v_), v, default_clause=True)

        # line 26b: w = (b3 ? w : t6)
        # line 28a: w = (b0 ? w : zQ)
        # line 29a: w = (b1 ? w : zP)
        qc_switch((b1, b0, b3), (zP, zQ, t6, w_), w, default_clause=True)

        # return final boolean:
        # b3 & (not b0) & (not b1)
        # = b3 & not (b0 or b1)
        self.x(b0orb1[0])
        self.ccx(b3[0], b0orb1[0], P_eq_pm_Q[0])

        ################ uncompute

        self.x(b0orb1[0])
        qc_or(b0, b1, b0orb1)
        self.ccx(b0[0], b1[0], b0b1[0])

        qc_cneg_modint(c_is_p_on_twist, tt4)
        qc_cneg_modint(b2b3, tt4)
        qc_xor(tt6, tt4)
        self.ccx(b3[0], b2[0], b2b3[0])

        qc_mulsub_modint(tt0new, tt2, tt6)
        qc_sub_modint(tt1, tt0new)
        qc_mulsub_modint(A24, tt2, tt0new)

        qc_add_modint(tt1, tt0)
        qc_mulsub_modint(tt0, tt1, tt5)
        qc_sqrsub_modint(tt1_, tt1)
        qc_add_modint(zP, tt1_)
        qc_xor(xP, tt1_)
        qc_sqrsub_modint(tt0_, tt0)
        qc_sub_modint(zP, tt0_)
        qc_xor(xP, tt0_)

        qc_mulsub_modint(t2sq, t1, w_)
        qc_muladd_modint(t7, t2sq, u_)
        qc_mulsub_modint(t4sq, t1, u_)
        qc_sub_modint(t5sq, t4sq)
        qc_cneg_modint(c_is_q_on_twist, t5sq)
        qc_cneg_modint(c_is_p_on_twist, t4sq)
        qc_sqrsub_modint(t5, t5sq)
        qc_sqrsub_modint(t4, t4sq)
        qc_eq(zero, t2sq, b3)
        qc_sqrsub_modint(t2, t2sq)

        qc_add_modint(t3, t2)
        qc_halve_modint(t3)
        qc_sub_modint(t2, t7)
        qc_sub_modint(t3, t2)
        qc_mulsub_modint(A, t1, t7)
        qc_unload(self.A, A)

        qc_halve_modint(v_)
        qc_mulsub_modint(t6, t1, v_)
        qc_mulsub_modint(t4, t5, t6)
        qc_eq(t4, t5, b2)
        qc_mulsub_modint(zP, yQ, t5)
        qc_mulsub_modint(yP, zQ, t4)
        qc_mulsub_modint(zP, xQ, t3)
        qc_mulsub_modint(xP, zQ, t2)
        qc_mulsub_modint(zP, zQ, t1)
        qc_eq(zero, zQ, b1)
        qc_eq(zero, zP, b0)

        qc_unload(ModInt(0, p), zero)
        qc_unload(ModInt(1, p), one)
        self.cx(P.a["not_infty"][0], zP[0])
        self.cx(Q.a["not_infty"][0], zQ[0])
        qc_cneg_modint(c_is_p_on_twist, xP)
        qc_cneg_modint(c_is_q_on_twist, xQ)

        for ancilla in (
            t1,
            t2,
            t3,
            t4,
            t5,
            t6,
            t7,
            t2sq,
            t4sq,
            t5sq,
            tt0_,
            tt0,
            tt1_,
            tt1,
            tt4,
            tt5,
            tt0new,
            tt6,
            b0,
            b1,
            b2,
            b3,
            b0b1,
            b2b3,
            b0orb1,
            u_,
            v_,
            w_,
            zP,
            zQ,
        ):
            self.assert_anc(ancilla)
            self.release_anc(ancilla)

    def dummy_classical_function(
        self,
        args: tuple[
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            QartonBool,
            QartonBool,
        ],
    ) -> tuple[
        AffMontgomeryPointVariable,
        AffMontgomeryPointVariable,
        QartonBool,
        QartonBool,
        ModInt3,
        QartonBool,
    ]:
        return ellipticpoints_to_barycentric_dummy_classical_function(
            args, self.A, self.p
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            QartonBool,
            QartonBool,
            ModInt3,
            QartonBool,
        ],
    ) -> tuple[
        AffMontgomeryPointVariable, AffMontgomeryPointVariable, QartonBool, QartonBool
    ]:
        return args[:-2]


def qc_ellpoints_to_barycentric(
    P: Register, Q: Register, cP: Register, cQ: Register, E: ECMontgomery
) -> tuple[Register, Register]:
    """
    Allocate and return two new registers: a tuple[ModInt, ModInt, ModInt] (uvw) and a boolean
    value (b).
    """
    qc = P.parent_circuit
    out = qc.append(PreCircuit(EllipticPointsToBarycentric, E), P, Q, cP, cQ)[-2:]
    assert len(out) == 2
    return out


def qc_ellpoints_to_barycentric_erase(
    P: Register,
    Q: Register,
    cP: Register,
    cQ: Register,
    uvw: Register,
    b: Register,
    E: ECMontgomery,
) -> None:
    """
    Erase ``qc_ellpoints_to_barycentric``.
    """
    qc = P.parent_circuit
    qc.append(
        PreCircuit(EllipticPointsToBarycentric, E, inverse=True), P, Q, cP, cQ, uvw, b
    )


def n2matrix_superglue4d_classical_function(
    args: tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]],
) -> tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]:
    x, sigmas = args
    mat_type = next(i for i in range(4) if sigmas[i] == 0)

    if mat_type == 0:
        nn2 = [
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0],
            [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
            [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0],
        ]
    elif mat_type == 1:
        nn2 = [
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1],
            [0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0],
        ]
    elif mat_type == 2:
        nn2 = [
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, -1],
            [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
            [0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0],
            [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0],
        ]
    elif mat_type == 3:
        nn2 = [
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0],
            [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1],
            [0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1],
            [0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0],
        ]
    else:
        raise Exception("Impossible")

    from .theta_util import mat_vec_product

    return CoordsDim4(mat_vec_product(nn2, x)), sigmas


@dummify
@memoize
class N2Matrix_Superglue4D(
    InPlaceCircuit[tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]]
):
    def __init__(self, p: int) -> None:
        """
        Circuit with input:
        - a point x with 16 coordinates, indexed as (x_{i,j} | i in [0..4], j in [0..3]) (i.e., x00 x01 x02 x03 x10 x11 x12 x13 ...)
        - the output of the norm equation s1, s2, s3, s4.
        If s_i == 0, then i is the type of N2 matrix as in Rabbits, Appendix A
        and output:
        - N2 * x (in place)

        How it works:
            We compute perm_type = (matrix_type + 2) mod 4 = c_0 + 2 * c_1.
            so the matrix type respectively 1, 2, 3, 4 corresponds to perm_type = 3, 0, 1, 2

            let M1, M2, M3, M4 denote the matrix of type respectively 1, 2, 3, 4;
            let P(l, k) be the permutation that swaps x_{l, j} with x_{k, j} for all j in [0..3].
                We have M2 = M2,
                        M3 = M2 * P(2, 3),
                        M4 = M2 * P(1, 2),
                        M1 = M2 * P(2, 3) * P(1, 2)
                so (1.) if c_1 = 1 apply P(1, 2) to x; then (2.) if c_0 = 1 apply P(2, 3) to x. In the end, just apply M2
        Regarding M2:
            denote M2 * x = (y0, y1, y2, y3) where yi = (y_{i, 0}, y_{i, 1}, y_{i, 2}, y_{i, 3}) for i in [0..3]
                   H = 2-dimensional (aka 4-by-4) Hadamard transform
            for all i in [0..3] we have:
                yi = H zi, where z_{i, j} = x_{2*i*j - i - j, j} FIXME doesn't seem right
        """
        super().__init__()
        self.p = p

        x = self.add_input_output(CoordsDim4Type(p))
        sigmas = self.add_input_output(TupleType(4, UIntType(2)))
        sigma1, _, sigma3, sigma4 = (sigmas.a[i] for i in range(4))

        # compute matrix type
        perm_type = self.add_anc(UIntType(2))
        # if sigma1 == 0: perm_type = 3 (little-endian: 11).
        self.x(sigma1[0])  # becomes 1 iff sigma1 is even
        self.cx(sigma1[0], perm_type[0])
        self.cx(sigma1[0], perm_type[1])
        # if sigma2 == 0: perm_type = 0 do nothing
        # if sigma3 == 0: perm_type = 1 (little-endian: 10).
        self.x(sigma3[0])  # becomes 1 iff sigma3 is even
        self.cx(sigma3[0], perm_type[0])
        # if sigma4 == 0: perm_type = 2 (little-endian: 01).
        self.x(sigma4[0])  # becomes 1 iff sigma4 is even
        self.cx(sigma4[0], perm_type[1])

        c_0 = qc_reg_from_bits(perm_type[0])
        c_1 = qc_reg_from_bits(perm_type[1])

        def idx(i: int, j: int) -> int:
            return 4 * i + j

        ### reduce to a type-1 matrix
        for j in range(4):
            qc_cswap(c_1, x.a[idx(1, j)], x.a[idx(2, j)])  # swap x_{1, j} and x_{2, j}
            qc_cswap(c_0, x.a[idx(2, j)], x.a[idx(3, j)])  # swap x_{2, j} and x_{3, j}

        ### apply matrix M1: first permute x into (z1, z2, z3, z4) as defined above, then apply dimension-2 Hadamard matrices
        for ij, kl in (
            ((1, 0), (3, 0)),
            ((2, 1), (3, 1)),
            ((1, 1), (2, 1)),
            ((0, 1), (1, 1)),
            ((0, 2), (2, 2)),
            ((0, 3), (1, 3)),
            ((1, 3), (2, 3)),
            ((2, 3), (3, 3)),
        ):
            qc_swap(x.a[idx(*ij)], x.a[idx(*kl)])
        for i in range(4):
            qc_theta_hadamard_dim2(
                qc_reg_cast(
                    x.a[idx(i, 0)] + x.a[idx(i, 1)] + x.a[idx(i, 2)] + x.a[idx(i, 3)],
                    CoordsDim2Type(p),
                )
            )

        ########## uncompute ############
        self.cx(sigma4[0], perm_type[1])
        self.x(sigma4[0])
        self.cx(sigma3[0], perm_type[0])
        self.x(sigma3[0])
        self.cx(sigma1[0], perm_type[1])
        self.cx(sigma1[0], perm_type[0])
        self.x(sigma1[0])
        self.assert_anc(perm_type)
        self.release_anc(perm_type)

    def validate_input(
        self, args: tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]
    ) -> bool:
        """
        Conditions for the input to be valid.
        Similar to NormeqOutputToHdKernel.
        """
        _, sigmas = args
        count_zeros = [s == 0 for s in sigmas].count(True)
        if count_zeros != 1:
            return False
        zero_idx = next(i for i in range(4) if sigmas[i] == 0)
        return {sigmas[(zero_idx + 1) % 4], sigmas[(zero_idx + 3) % 4]}.issubset({1, 3})
        # TODO any conditions on the element with index (zero_idx + 2) mod 4?

    def dummy_classical_function(
        self, args: tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]
    ) -> tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]:
        return n2matrix_superglue4d_classical_function(args)

    def dummy_classical_function_inverse(
        self, args: tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]
    ) -> tuple[CoordsDim4, tuple[UInt, UInt, UInt, UInt]]:
        y, sigmas = args
        mat_type = next(i for i in range(4) if sigmas[i] == 0)

        if mat_type == 0:
            Ninv = [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
                [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        elif mat_type == 1:
            Ninv = [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
                [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
            ]
        elif mat_type == 2:
            Ninv = [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
                [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        elif mat_type == 3:
            Ninv = [
                [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
                [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
            ]
        else:
            raise Exception("Impossible case")

        from .theta_util import mat_vec_product

        x = CoordsDim4(
            coord * ModInt(4, self.p) ** (-1) for coord in mat_vec_product(Ninv, y)
        )
        return x, sigmas


def qc_n2_matrix(w: Register, sigmas: Register) -> None:
    qc = w.parent_circuit
    dt = TupleType.get_from_register(w)
    et = dt.element_type
    if not isinstance(et, ModIntType):
        raise TypeError()
    p = et.p
    qc.append(PreCircuit(N2Matrix_Superglue4D, p), w, sigmas)


def qc_n2_matrix_inv(w: Register, sigmas: Register) -> None:
    qc = w.parent_circuit
    dt = TupleType.get_from_register(w)
    et = dt.element_type
    if not isinstance(et, ModIntType):
        raise TypeError()
    p = et.p
    qc.append(PreCircuit(N2Matrix_Superglue4D, p, inverse=True), w, sigmas)


def superglue4dthetaplusminus_dummy_classical_function(
    args: tuple[
        AffMontProductPoint,
        AffMontProductPoint,
        QartonBool,
        QartonBool,
        tuple[UInt, UInt, UInt, UInt],
    ],
    prod_np: CoordsDim4,
    A: ModInt,
    p: int,
) -> tuple[
    AffMontProductPoint,
    AffMontProductPoint,
    QartonBool,
    QartonBool,
    tuple[UInt, UInt, UInt, UInt],
    CoordsDim4,
]:
    P, T, cP, cT, sigmas = args

    uvw_1234 = [
        ellipticpoints_to_barycentric_dummy_classical_function(
            (P[m], T[m], cP, cT), A, p
        )[-2:]
        for m in range(4)
    ]

    c = int((-1) ** (cP ^ cT))

    ### compute u12, u34, v12, v34 from Rabbits Algorithm 3
    # uij = (ui uj + ΔP ΔT vi vj, wi uj, ui wj, wi wj)
    # vij = (uj vi + ui vj, wi vj, vi wj, 0)
    # c = 1 if (ΔP ΔT) = -1, and c = 0 if (ΔP ΔT) = 1
    # TODO replace eq. (11) with (17), (18) or (19) when needed: how to check exactly??

    uij_s: list[tuple[ModInt, ModInt, ModInt, ModInt]] = []
    vij_s: list[tuple[ModInt, ModInt, ModInt, ModInt]] = []
    for i, j in ((0, 1), (2, 3)):
        (ui, vi, wi), bi = uvw_1234[i]
        (uj, vj, wj), bj = uvw_1234[j]
        assert bool(bi) == ((P[i] in (T[i], -T[i])) and (cP == cT))
        assert bool(bj) == ((P[j] in (T[j], -T[j])) and (cP == cT))

        if (P[i] not in (T[i], -T[i])) and (
            P[j] not in (T[j], -T[j])
        ):  # TODO check: both points should be affine
            # eq (11)
            uij = (ui * uj + vi * vj * c, wi * uj, ui * wj, wi * wj)
            vij = (
                vi * uj + ui * vj,
                wi * vj,
                vi * wj,
                ModInt(0, p),
            )
        elif P[j] not in (T[j], -T[j]):
            # eq (17)
            uij = (
                ui * uj,
                wi * uj + (vi * vj) * c,
                ui * wj,
                wi * wj,
            )
            vij = (
                ui * vj,
                vi * uj + wi * vj,
                ModInt(0, p),
                vi * wj,
            )
        elif P[i] not in (T[i], -T[i]):
            # eq (18)
            uij = (
                ui * uj,
                wi * uj,
                ui * wj + vi * vj * c,
                wi * wj,
            )
            vij = (
                vi * uj,
                ModInt(0, p),
                ui * vj + vi * wj,
                wi * vj,
            )
        else:
            # eq (19)
            uij = (
                ui * uj,
                wi * uj,
                ui * wj,
                wi * wj + vi * vj * c,
            )
            vij = (
                ModInt(0, p),
                vi * uj,
                ui * vj,
                wi * vj + vi * wj,
            )

        uij_s.append(uij)
        vij_s.append(vij)
    u12, u34 = uij_s
    v12, v34 = vij_s

    def tens_prod(
        x: tuple[ModInt, ...], y: tuple[ModInt, ...]
    ) -> tuple[ModInt, ...]:
        nx = len(x)
        ny = len(y)
        return tuple(
            x[j] * y[i] for i, j in itertools.product(range(ny), range(nx))
        )

    # lines 7, 8
    u = tuple(
        a + b * c
        for a, b in zip(tens_prod(u12, u34), tens_prod(v12, v34), strict=False)
    )
    v = tuple(
        a + b
        for a, b in zip(tens_prod(u12, v34), tens_prod(v12, u34), strict=False)
    )

    # lines 9 -- 11
    w: CoordsDim4 = cast(
        CoordsDim4,
        tuple(
            a * b for a, b in zip(prod_np, classical_hadamard(u)[::-1], strict=True)
        ),
    )
    x: CoordsDim4 = cast(
        CoordsDim4,
        tuple(
            a * b for a, b in zip(prod_np, classical_hadamard(v)[::-1], strict=True)
        ),
    )
    # line 12
    w = n2matrix_superglue4d_classical_function((w, sigmas))[0]
    x = n2matrix_superglue4d_classical_function((x, sigmas))[0]
    # lines 13, 14
    theta_plus_minus = CoordsDim4(a**2 - b**2 * c for a, b in zip(w, x, strict=False))

    return (*args, theta_plus_minus)


@dummify
@memoize
class Superglue4DThetaPlusMinus(
    Circuit[
        tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            tuple[UInt, UInt, UInt, UInt],
        ],
        tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            tuple[UInt, UInt, UInt, UInt],
            CoordsDim4,
        ],
    ]
):
    def __init__(self, product_theta_nullpoint: CoordsDim4, E: ECMontgomery) -> None:
        super().__init__()
        self.E = E
        p = E.q
        self.p = p
        self.A = ModInt(E.A, p)
        ell_product_point_4d_type = TupleType(4, AffMontgomeryPointVariableType(p))
        self.prod_np = product_theta_nullpoint

        product_theta_nullpoint_reg = qc_load(
            product_theta_nullpoint, self, CoordsDim4Type(p)
        )

        # i/o
        pp = self.add_input_output(ell_product_point_4d_type)
        tt = self.add_input_output(ell_product_point_4d_type)
        c_is_pp_on_twist = self.add_input_output(BoolType())
        c_is_tt_on_twist = self.add_input_output(BoolType())
        sigmas = self.add_input_output(TupleType(4, UIntType(2)))

        out = self.add_anc_output(CoordsDim4Type(p))

        ### compute barycentric coordinates of (Pm, T(m)) for m in (1,2,3,4)
        uvw_1234 = [
            qc_ellpoints_to_barycentric(
                pp.a[m], tt.a[m], c_is_pp_on_twist, c_is_tt_on_twist, E
            )
            for m in range(4)
        ]

        c = c_is_tt_on_twist
        qc_xor(c_is_pp_on_twist, c)

        ### compute u12, u34, v12, v34 from Rabbits, equations (11), (17)--(19)
        # eq. 11, generic case:
        # uij = (ui uj + ΔP ΔT vi vj, wi uj, ui wj, wi wj)
        # vij = (uj vi + ui vj, wi vj, vi wj, 0)
        # c = 1 if (ΔP ΔT) = -1, and c = 0 if (ΔP ΔT) = 1
        u12, u34 = self.add_ancs(*([TupleType(4, ModIntType(p))] * 2))
        v12, v34 = self.add_ancs(*([TupleType(4, ModIntType(p))] * 2))

        uij_s = {(0, 1): u12, (2, 3): u34}
        vij_s = {(0, 1): v12, (2, 3): v34}

        bibj = self.add_anc(BitVectorType(1))
        for i, j in ((0, 1), (2, 3)):
            uvw_i, bi = uvw_1234[i]
            ui, vi, wi = (uvw_i.a[m] for m in range(3))
            uvw_j, bj = uvw_1234[j]
            uj, vj, wj = (uvw_j.a[m] for m in range(3))

            uij = uij_s[(i, j)]
            qc_muladd_modint(vi, vj, uij.a[0])
            qc_cneg_modint(c, uij.a[0])
            self.ccx(bi[0], bj[0], bibj[0])
            qc_cswap(bibj, uij.a[0], uij.a[3])
            self.ccx(bi[0], bj[0], bibj[0])
            qc_cswap(bi, uij.a[0], uij.a[1])
            qc_cswap(bj, uij.a[0], uij.a[2])

            qc_muladd_modint(ui, uj, uij.a[0])
            qc_muladd_modint(wi, uj, uij.a[1])
            qc_muladd_modint(ui, wj, uij.a[2])
            qc_muladd_modint(wi, wj, uij.a[3])

            vij = vij_s[(i, j)]
            qc_muladd_modint(ui, vj, vij.a[0])
            qc_muladd_modint(wi, vj, vij.a[1])

            qc_cswap(bj, vij.a[0], vij.a[2])
            qc_cswap(bj, vij.a[1], vij.a[3])

            qc_cmuladd_modint(bi, vi, uj, vij.a[1])
            qc_cmuladd_modint(bi, vi, wj, vij.a[3])
            self.x(bi[0])
            qc_cmuladd_modint(bi, vi, uj, vij.a[0])
            qc_cmuladd_modint(bi, vi, wj, vij.a[2])

        ### u_or_v12_34 = u_or_v12 ⊗ u_or_v34
        u12_u34, v12_v34, u12_v34, v12_u34 = self.add_ancs(*([CoordsDim4Type(p)] * 4))
        for u_or_v12, u_or_v34, u_or_v_12_34 in (
            (u12, u34, u12_u34),
            (v12, v34, v12_v34),
            (u12, v34, u12_v34),
            (v12, u34, v12_u34),
        ):
            for i, j in itertools.product(range(4), range(4)):
                qc_muladd_modint(
                    u_or_v12.a[j], u_or_v34.a[i], u_or_v_12_34.a[4 * i + j]
                )

        ### compute u, v from Lemma 4.4
        # TODO we can cneg just v12, if we first compute v then u: do cnegs on only 1/4 of the coordinates
        # u = u12 ⊗ u34 + ΔP ΔT v12 ⊗ v34
        qc_coordwise_cneg(c, v12_v34)
        u = v12_v34
        qc_coordwise_add(u12_u34, u)
        # v = u12 ⊗ v34 + v12 ⊗ u34
        v = u12_v34
        qc_coordwise_add(v12_u34, v)

        ### compute θ^A(P + T) · θ^A(P - T) as in Lemma 4.4
        w, x = self.add_ancs(*([CoordsDim4Type(p)] * 2))
        qc_theta_hadamard_bitflipped(u)  # u = Hbar(u)
        qc_theta_hadamard_bitflipped(v)  # v = Hbar(v)
        qc_coordwise_muladd(product_theta_nullpoint_reg, u, w)  # w = (⊗(am, bm)) · u
        qc_coordwise_muladd(product_theta_nullpoint_reg, v, x)  # x = (⊗(am, bm)) · v
        qc_n2_matrix(w, sigmas)
        qc_n2_matrix(x, sigmas)

        # θ^A(P + T) · θ^A(P - T) = w · w - ΔP ΔT x · x
        qc_coordwise_sqradd(x, out)
        self.x(c[0])
        qc_coordwise_cneg(c, out)
        qc_coordwise_sqradd(w, out)

        ############## unocmpute ###############
        self.x(c[0])
        qc_n2_matrix_inv(w, sigmas)
        qc_n2_matrix_inv(x, sigmas)
        # NOTE u and v are the same as v12_v34 and u12_v34 respectively
        qc_coordwise_mulsub(product_theta_nullpoint_reg, v, x)
        qc_coordwise_mulsub(product_theta_nullpoint_reg, u, w)
        qc_theta_hadamard_bitflipped(v, inverse=True)
        qc_theta_hadamard_bitflipped(u, inverse=True)
        qc_coordwise_sub(v12_u34, v)
        qc_coordwise_sub(u12_u34, u)
        qc_coordwise_cneg(c, v12_v34)

        qc_unload(product_theta_nullpoint, product_theta_nullpoint_reg)

        # undo tensor products
        for u_or_v12, u_or_v34, u_or_v_12_34 in (
            (u12, u34, u12_u34),
            (v12, v34, v12_v34),
            (u12, v34, u12_v34),
            (v12, u34, v12_u34),
        ):
            for i, j in itertools.product(range(4), range(4)):
                qc_mulsub_modint(
                    u_or_v12.a[j], u_or_v34.a[i], u_or_v_12_34.a[4 * i + j]
                )
        # undo u12, u34, v12, v34
        for i, j in ((0, 1), (2, 3)):
            uvw_i, bi = uvw_1234[i]
            ui, vi, wi = (uvw_i.a[m] for m in range(3))
            uvw_j, bj = uvw_1234[j]
            uj, vj, wj = (uvw_j.a[m] for m in range(3))

            uij = uij_s[(i, j)]
            vij = vij_s[(i, j)]

            qc_cmulsub_modint(bi, vi, wj, vij.a[2])
            qc_cmulsub_modint(bi, vi, uj, vij.a[0])
            self.x(bi[0])
            qc_cmulsub_modint(bi, vi, wj, vij.a[3])
            qc_cmulsub_modint(bi, vi, uj, vij.a[1])
            qc_cswap(bj, vij.a[1], vij.a[3])
            qc_cswap(bj, vij.a[0], vij.a[2])
            qc_mulsub_modint(wi, vj, vij.a[1])
            qc_mulsub_modint(ui, vj, vij.a[0])
            qc_mulsub_modint(wi, wj, uij.a[3])
            qc_mulsub_modint(ui, wj, uij.a[2])
            qc_mulsub_modint(wi, uj, uij.a[1])
            qc_mulsub_modint(ui, uj, uij.a[0])
            qc_cswap(bj, uij.a[0], uij.a[2])
            qc_cswap(bi, uij.a[0], uij.a[1])
            self.ccx(bi[0], bj[0], bibj[0])
            qc_cswap(bibj, uij.a[0], uij.a[3])
            self.ccx(bi[0], bj[0], bibj[0])
            qc_cneg_modint(c, uij.a[0])
            qc_mulsub_modint(vi, vj, uij.a[0])

        self.assert_anc(bibj)
        self.release_anc(bibj)

        qc_xor(c_is_pp_on_twist, c_is_tt_on_twist)

        for m in range(4):
            qc_ellpoints_to_barycentric_erase(
                pp.a[m],
                tt.a[m],
                c_is_pp_on_twist,
                c_is_tt_on_twist,
                uvw_1234[m][0],
                uvw_1234[m][1],
                E,
            )

        for reg in (
            u12,
            u34,
            v12,
            v34,
            u12_u34,
            u12_v34,
            v12_u34,
            v12_v34,
            w,
            x,
        ):
            self.assert_anc(reg)
            self.release_anc(reg)

    def dummy_classical_function(
        self,
        args: tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            tuple[UInt, UInt, UInt, UInt],
        ],
    ) -> tuple[
        AffMontProductPoint,
        AffMontProductPoint,
        QartonBool,
        QartonBool,
        tuple[UInt, UInt, UInt, UInt],
        CoordsDim4,
    ]:
        return superglue4dthetaplusminus_dummy_classical_function(
            args, self.prod_np, self.A, self.p
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            tuple[UInt, UInt, UInt, UInt],
            CoordsDim4,
        ],
    ) -> tuple[
        AffMontProductPoint,
        AffMontProductPoint,
        QartonBool,
        QartonBool,
        tuple[UInt, UInt, UInt, UInt],
    ]:
        return args[:-1]


def superglue4devaluation_dummy_classical_function(
    args: tuple[
        AffMontProductPoint,
        AffMontProductPoint,
        QartonBool,
        QartonBool,
        CoordsDim4,
        tuple[UInt, UInt, UInt, UInt],
    ],
    prod_np: CoordsDim4,
    A: ModInt,
    p: int,
) -> tuple[
    AffMontProductPoint,
    AffMontProductPoint,
    QartonBool,
    QartonBool,
    CoordsDim4,
    tuple[UInt, UInt, UInt, UInt],
    CoordsDim4,
]:
    P, T, cP, cT, inv_image_t, sigmas = args

    theta_plus_minus = superglue4dthetaplusminus_dummy_classical_function(
        (P, T, cP, cT, sigmas), prod_np, A, p
    )[-1]

    # lines 15, 16
    UB_fP = tuple(
        a * b
        for a, b in zip(
            classical_hadamard(theta_plus_minus), inv_image_t, strict=False
        )
    )
    # line 17
    theta_fP = classical_hadamard(UB_fP)

    return *args, CoordsDim4(theta_fP)


@dummify
@memoize
class Superglue4DEvaluation(
    Circuit[
        tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            CoordsDim4,
            tuple[UInt, UInt, UInt, UInt],
        ],
        tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            CoordsDim4,
            tuple[UInt, UInt, UInt, UInt],
            CoordsDim4,
        ],
    ]
):
    def __init__(self, product_theta_nullpoint: CoordsDim4, E: ECMontgomery) -> None:
        r"""
        Compute Superglue4D: eprint 2026-114 (Rabbits), Algorithm 2.

        The classical precomputed data contains:

        - starting curve E
        - a basis <P, Q> of the torsion E[2^{e+2}]
        - the product theta null point \otimes(am, bm)

        Input:

        - a point P to evaluate in A=E(Fp)^4 or in A^t=(E^t(Fp))^4
        - an auxiliary point T in A or in A^t
        - the inverse components of the image U^B(f(T))
        - change of theta coordinates

        Output:

        - the image theta point θ^B(f(P))
        """
        super().__init__()
        self.E = E
        p = E.q
        self.p = p
        self.A = ModInt(E.A, p)
        ell_product_point_4d_type = TupleType(4, AffMontgomeryPointVariableType(p))
        self.prod_np = product_theta_nullpoint

        # i/o
        pp = self.add_input_output(ell_product_point_4d_type)
        tt = self.add_input_output(ell_product_point_4d_type)
        c_is_pp_on_twist = self.add_input_output(BoolType())
        c_is_tt_on_twist = self.add_input_output(BoolType())
        inv_image_t = self.add_input_output(CoordsDim4Type(p))
        sigmas = self.add_input_output(TupleType(4, UIntType(2)))

        out = self.add_anc_output(CoordsDim4Type(p))

        self.circ_theta_ppt_theta_pmt = PreCircuit(
            Superglue4DThetaPlusMinus, product_theta_nullpoint, E
        )
        theta_ppt_theta_pmt = self.append(
            self.circ_theta_ppt_theta_pmt,
            pp,
            tt,
            c_is_pp_on_twist,
            c_is_tt_on_twist,
            sigmas,
        )[-1]

        ### last steps
        # U^B(f(P)) = H(θ^A(P + T) · θ^A(P - T))
        qc_theta_hadamard(theta_ppt_theta_pmt)
        # U^B(f(P)) ·= inv( U^B(f(T)) )
        qc_coordwise_muladd(theta_ppt_theta_pmt, inv_image_t, out)
        # return θ^B(f(P)) = H( U^B(f(P)) )
        qc_theta_hadamard(out)

        ############## uncompute #############
        qc_theta_hadamard(theta_ppt_theta_pmt, inverse=True)
        self.append(
            self.circ_theta_ppt_theta_pmt.inverse(),
            pp,
            tt,
            c_is_pp_on_twist,
            c_is_tt_on_twist,
            sigmas,
            theta_ppt_theta_pmt,
        )

    def dummy_classical_function(
        self,
        args: tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            CoordsDim4,
            tuple[UInt, UInt, UInt, UInt],
        ],
    ) -> tuple[
        AffMontProductPoint,
        AffMontProductPoint,
        QartonBool,
        QartonBool,
        CoordsDim4,
        tuple[UInt, UInt, UInt, UInt],
        CoordsDim4,
    ]:
        return superglue4devaluation_dummy_classical_function(
            args, self.prod_np, self.A, self.p
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            AffMontProductPoint,
            AffMontProductPoint,
            QartonBool,
            QartonBool,
            CoordsDim4,
            tuple[UInt, UInt, UInt, UInt],
            CoordsDim4,
        ],
    ) -> tuple[
        AffMontProductPoint,
        AffMontProductPoint,
        QartonBool,
        QartonBool,
        CoordsDim4,
        tuple[UInt, UInt, UInt, UInt],
    ]:
        return args[:-1]


def superglue4drabbitcodomain_classical_function(
    p: int,
    args: tuple[
        tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], UInt
    ],
) -> tuple[
    tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
    UInt,
    CoordsDim4,
]:
    g = 4
    HSK, rabbit_type = args
    rabbit_type = 15 - 2**rabbit_type
    paths = {
        7: (3, 11, 9, 5, 4, 6, 2, 10, 8, 0, 1, 13, 12, 14),
        11: (3, 7, 6, 10, 8, 9, 1, 5, 4, 0, 2, 14, 12, 13),
        13: (12, 14, 6, 5, 1, 9, 8, 10, 2, 0, 4, 7, 3, 11),
        14: (12, 13, 9, 10, 2, 6, 4, 5, 1, 0, 8, 11, 3, 7),
    }

    def path(i: int) -> int:
        return paths[rabbit_type][i]  # path = eta

    def edge(i: int) -> int:
        return path(i) ^ path(i + 1)  # edge(i) = s_i

    r1 = rabbit_type
    t1 = r1 ^ path(0)
    r2 = 15
    t2 = r2 ^ path(0)

    int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3, 3: 4, 12: 4}

    def pi(i: int, j: int) -> ModInt:
        return HSK[int_to_kernel_pt[j]][i]

    rho_left = [ModInt(1, p)] * (2**g - 2)
    rho_left[0] = pi(r1, t1) * pi(r2, t2)
    for i in range(1, 2**g - 2):
        rho_left[i] = rho_left[i - 1] * pi(path(i - 1), edge(i - 1))

    rho_right = [ModInt(1, p)] * (2**g - 2)
    rho_right[2**g - 3] = ModInt(1, p)
    for i in range(2**g - 4, -1, -1):
        rho_right[i] = rho_right[i + 1] * pi(path(i + 1), edge(i))

    xinv = [ModInt(1, p)] * (2**g)
    xinv[r1] = pi(path(0), t1) * pi(r2, t2) * rho_right[0]
    xinv[r2] = pi(r1, t1) * pi(path(0), t2) * rho_right[0]
    for i in range(2**g - 2):
        xinv[path(i)] = rho_left[i] * rho_right[i]

    return *args, CoordsDim4(xinv)


@dummify
@memoize
class Superglue4DRabbitCodomain(
    Circuit[
        tuple[tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], UInt],
        tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
            UInt,
            CoordsDim4,
        ],
    ]
):
    """
    See Rabbits, Algorithm 1 and Appendix C
    """

    def __init__(self, p: int) -> None:
        super().__init__()
        self.p = p
        coords_type = CoordsDim4Type(p)
        mod_int_type = ModIntType(p)

        hadsq_ker = self.add_input_output(TupleType(5, coords_type))
        c_rabbit_type = self.add_input_output(UIntType(2))
        # c_rabbit_type = i in [0, ..., 3] --> if c_rabbit_type = i, we use a rabbit of type 15 - 2**i as denoted in the Rabbits paper
        # i.e.: 0 -> type 14=1110,   1 -> type 13=1101,
        #       2 -> type 11=1011,   3 -> type  7=0111
        out = self.add_anc_output(coords_type)

        b0, b1 = (qc_reg_from_bits(c_rabbit_type[i]) for i in range(2))

        int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3, 3: 4, 12: 4}

        def _swaps_from_permutation(
            transpositions: list[tuple[int, int]],
        ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
            """
            Apply an automorphism of the hypercube on the coordinates of hadsq_ker
            e.g. if we apply a permutation swapping directions 0 and 1 on the hypercube
            we swap the order-0 and order-1 bit in all components:
            0 = 0000 -> 0 = 0000
            1 = 0001 -> 2 = 0010
            5 = 0101 -> 6 = 0110
            The points of hadsq_ker are indexed by a label: T1 -> 1, T2 -> 2, T3 -> 4, T4 -> 8, T1+2 -> 3, T3+4 -> 12
            and each of these points has 16 coordinates
            We apply the bit permutations on (label, coordinate index) and collect the corresponding swaps we need to make on (point, coordinate)
            """

            # we expect transpositions = [(0,1), (2,3)]
            # or = [(0, 2), (1, 3)]
            def swap_bits_ij(string: Any, i: Any, j: Any) -> Any:
                """
                take an integer and swap its i-th and j-th bit.
                """
                ith_bit = (string >> i) & 1
                jth_bit = (string >> j) & 1
                bits_remove = (ith_bit << i) ^ (jth_bit << j)
                bits_swapped = (ith_bit << j) ^ (jth_bit << i)
                return string ^ bits_remove ^ bits_swapped

            def swap_bits(string: Any) -> Any:
                for i, j in transpositions:
                    string = swap_bits_ij(string, i, j)
                return string

            swaps: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for ker_pt_idx in (1, 2, 4, 8, 3, 12):
                for component in range(16):
                    a, b = ker_pt_idx, component
                    c, d = swap_bits(ker_pt_idx), swap_bits(component)

                    a_, c_ = (int_to_kernel_pt[x] for x in (a, c))
                    if (
                        ((a_, b) != (c_, d))
                        and ((c_, d), (a_, b)) not in swaps
                        and ((a_, b), (c_, d)) not in swaps
                    ):
                        swaps.append(((a_, b), (c_, d)))
            return swaps

        swaps_1w2_3w4 = _swaps_from_permutation([(0, 1), (2, 3)])
        swaps_1w3_2w4 = _swaps_from_permutation([(0, 2), (1, 3)])

        for swap in swaps_1w2_3w4:
            (a, b), (c, d) = swap
            qc_cswap(
                b0,
                hadsq_ker.a[a].a[b],
                hadsq_ker.a[c].a[d],
            )
        for swap in swaps_1w3_2w4:
            (a, b), (c, d) = swap
            qc_cswap(
                b1,
                hadsq_ker.a[a].a[b],
                hadsq_ker.a[c].a[d],
            )

        # we've reduced to the rabbit of type 14
        def path(i: int) -> int:
            return [
                12,
                13,
                9,
                10,
                2,
                6,
                4,
                5,
                1,
                0,
                8,
                11,
                3,
                7,
            ][i]

        def edge(i: int) -> int:
            return xor(path(i), path(i + 1))

        r1 = 14
        r2 = 15
        t1 = xor(r1, path(0))
        t2 = xor(r2, path(0))

        def pi_mat(i: int, j: int) -> Register:
            return hadsq_ker.a[int_to_kernel_pt[j]].a[i]

        ## operations
        # x^{−1}_{η(i)} = ρ_L(i) · ρ_R(i)
        # x^{-1}_{r1} = π_{η(0), t1} * π_{r2, t2} * ρ_R(0)
        # x^{-1}_{r2} = π_{r1, t1} * π_{η(0), t2} * ρ_R(0)

        # TODO some of these coordinates are unused: check
        rho_left = self.add_anc(TupleType(14, mod_int_type))
        rho_right = self.add_anc(TupleType(14, mod_int_type))

        # compute ρ_L
        qc_muladd_modint(pi_mat(r1, t1), pi_mat(r2, t2), rho_left.a[0])
        for j in range(1, 13 + 1):
            qc_muladd_modint(
                rho_left.a[j - 1], pi_mat(path(j - 1), edge(j - 1)), rho_left.a[j]
            )

        # compute ρ_R
        # set rho_right[13] to 1
        self.x(rho_right.a[13][0])
        qc_xor(pi_mat(path(12 + 1), edge(12)), rho_right.a[12])
        for j in range(11, -1, -1):
            # ρ_R(2^g - 4 - i) = ρ_R(2^g - 4 - i) * π_{η(2^g - 4 - i), edge(2^g - 4 - 1)}
            qc_muladd_modint(
                rho_right.a[j + 1], pi_mat(path(j + 1), edge(j)), rho_right.a[j]
            )

        # final results:
        # x^{-1}_{r1} = π_{η(0), t1} * π_{r2, t2} * ρ_R(0)
        pi_eta0t1_pi_r2t2 = self.add_anc(mod_int_type)
        qc_muladd_modint(pi_mat(path(0), t1), pi_mat(r2, t2), pi_eta0t1_pi_r2t2)
        qc_muladd_modint(pi_eta0t1_pi_r2t2, rho_right.a[0], out.a[r1])

        # x^{-1}_{r2} = π_{r1, t1} * π_{η(0), t2} * ρ_R(0)
        pi_r1t1_pi_eta0t2 = self.add_anc(mod_int_type)
        qc_muladd_modint(pi_mat(r1, t1), pi_mat(path(0), t2), pi_r1t1_pi_eta0t2)
        qc_muladd_modint(pi_r1t1_pi_eta0t2, rho_right.a[0], out.a[r2])

        # x^{-1}_{η(i)} = ρ_L(i) · ρ_R(i)
        for i in range(14):
            qc_muladd_modint(rho_left.a[i], rho_right.a[i], out.a[path(i)])
        # swap components of out vector:
        # like above with swap_bits, but only on the components on out
        # first, if b0==1, swap 1st with 2nd bit, and 3rd with 4th
        for a, b in (
            (0b0001, 0b0010),
            (0b0100, 0b1000),
            (0b0101, 0b1010),
            (0b0110, 0b1001),
            (0b0111, 0b1011),
            (0b1101, 0b1110),
        ):
            qc_cswap(b0, out.a[a], out.a[b])
        # then, if b1==1, swap 1st with 3nd bit, and 2rd with 4th
        for a, b in (
            (0b0001, 0b0100),
            (0b0010, 0b1000),
            (0b0011, 0b1100),
            (0b0110, 0b1001),
            (0b0111, 0b1101),
            (0b1110, 0b1011),
        ):
            qc_cswap(b1, out.a[a], out.a[b])

        ############# uncompute ##################
        qc_mulsub_modint(pi_mat(r1, t1), pi_mat(path(0), t2), pi_r1t1_pi_eta0t2)
        qc_mulsub_modint(pi_mat(path(0), t1), pi_mat(r2, t2), pi_eta0t1_pi_r2t2)

        for j in range(0, 12):
            # ρ_R(2^g - 4 - i) = ρ_R(2^g - 4 - i) * π_{η(2^g - 4 - i), edge(2^g - 4 - 1)}
            qc_mulsub_modint(
                rho_right.a[j + 1], pi_mat(path(j + 1), edge(j)), rho_right.a[j]
            )
        qc_xor(pi_mat(path(12 + 1), edge(12)), rho_right.a[12])
        self.x(rho_right.a[13][0])

        for j in range(13, 0, -1):
            qc_mulsub_modint(
                rho_left.a[j - 1], pi_mat(path(j - 1), edge(j - 1)), rho_left.a[j]
            )
        qc_mulsub_modint(pi_mat(r1, t1), pi_mat(r2, t2), rho_left.a[0])

        for anc in (pi_eta0t1_pi_r2t2, pi_r1t1_pi_eta0t2, rho_left, rho_right):
            self.assert_anc(anc)
            self.release_anc(anc)

        for swap in swaps_1w2_3w4:
            (a, b), (c, d) = swap
            qc_cswap(
                b0,
                hadsq_ker.a[a].a[b],
                hadsq_ker.a[c].a[d],
            )
        for swap in swaps_1w3_2w4:
            (a, b), (c, d) = swap
            qc_cswap(
                b1,
                hadsq_ker.a[a].a[b],
                hadsq_ker.a[c].a[d],
            )

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], UInt
        ],
    ) -> tuple[
        tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
        UInt,
        CoordsDim4,
    ]:
        return superglue4drabbitcodomain_classical_function(self.p, args)

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
            UInt,
            CoordsDim4,
        ],
    ) -> tuple[tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], UInt]:
        return args[:-1]
