"""
Main circuit that computes a 4-dimensional isogeny chain.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

from qarton.binary_operations import (
    qc_cswap,
    qc_cxor,
    qc_load,
    qc_unload,
)
from qarton.circuit import (
    BackendSpecifier,
    BitVector,
    BitVectorType,
    BoolType,
    Circuit,
    DataType,
    EmptyBackendSpecifier,
    ITupleType,
    PreCircuit,
    QartonBool,
    Register,
    TupleType,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_concat,
    qc_empty_reg,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_muladd_modint,
    qc_mulsub_modint,
)

from qisogenies.montgomery.montgomery_curve import ECMontgomeryType
from qisogenies.theta.affpoint_dim4 import AffPointDim4, AffPointDim4Type

from ..montgomery import (
    AffMontgomeryPointType,
    AffMontgomeryPointVariableType,
    ECMontgomery,
)

# TEMPORARY: IsogenyChain is pointed at the working copy of
# IsogenyChainWithGarbage (with the k = 0 loop split) for testing.
# Revert to the IsogenyChainWithGarbage class defined below once validated.
from .isogeny_dim4 import (
    Theta2IsogenyDim4Generic_Codomain,
    Theta2IsogenyDim4Generic_Evaluation,
    Theta2IsogenyDim4Second_Codomain,
    theta2isogenydim4generic_codomain_classical_function,
    theta2isogenydim4generic_evaluation_classical_function,
    theta2isogenydim4second_codomain_classical_function,
)
from .normeq_to_hd import (
    NormeqOutputToRabbitType,
    normeqoutputtorabbittype_classical_function,
)
from .splitting_dim4 import (
    SplittingDim4,
    splittingdim4_classical_function,
)
from .superglue_dim4 import (
    N2Matrix_Superglue4D,
    Superglue4DEvaluation,
    Superglue4DRabbitCodomain,
    Superglue4DThetaPlusMinus,
    n2matrix_superglue4d_classical_function,
    superglue4devaluation_dummy_classical_function,
    superglue4drabbitcodomain_classical_function,
    superglue4dthetaplusminus_dummy_classical_function,
)
from .theta_dim4 import (
    ThetaArithmeticPrecomputation,
    ThetaPointDim4,
    ThetaPointDim4Type,
    ThetaStructureDim4,
    ThetaStructureDim4Type,
    qc_double_iter_prod_mont,
    qc_double_iter_prod_mont_var,
    qc_double_iter_theta,
    qc_theta_hadamard,
    qc_theta_hadamard_bitflipped,
)
from .theta_util import (
    CoordsDim4,
    CoordsDim4Type,
    MontgomeryToJInvariantDiv256,
    balanced_strategy,
    classical_hadamard,
    montgomerytojinvariantdiv256_classical_function,
    qc_coordwise_invadd,
    qc_coordwise_muladd,
    qc_coordwise_sqradd,
    qc_coordwise_sqrsub,
)

__all__ = [
    "qc_prodpoint_to_theta",
    "prodpoint_to_theta",
    "InputData",
    "IsogenyChainWithGarbage",
    "IsogenyChain",
]


def qc_prodpoint_to_theta(
    P: Register, sigmas: Register, prod_np: CoordsDim4, c_P_on_twist: Register
) -> Register:
    qc = P.parent_circuit
    dt = TupleType.get_from_register(P)
    et = dt.element_type
    assert isinstance(et, (AffMontgomeryPointType, AffMontgomeryPointVariableType))
    p = et.ec.q if isinstance(et, AffMontgomeryPointType) else et.q

    P_xz, prod_P_dim2, prod_P_dim4, out_coords = qc.add_ancs(
        TupleType(4, TupleType(2, ModIntType(p))),
        TupleType(2, TupleType(4, ModIntType(p))),
        CoordsDim4Type(p),
        CoordsDim4Type(p),
    )

    for i in range(4):
        qc_cxor(P.a[i].a["not_infty"], P.a[i].a["x"], P_xz.a[i].a[0])
        qc_cxor(P.a[i].a["not_infty"], ModInt(1, p), P_xz.a[i].a[1])
        qc.x(P.a[i].a["not_infty"][0])
        qc_cxor(P.a[i].a["not_infty"], ModInt(1, p), P_xz.a[i].a[0])

    # pairwise tensor products
    for i, j in itertools.product(range(2), repeat=2):
        qc_muladd_modint(P_xz.a[0].a[i], P_xz.a[1].a[j], prod_P_dim2.a[0].a[i + 2 * j])
        qc_muladd_modint(P_xz.a[2].a[i], P_xz.a[3].a[j], prod_P_dim2.a[1].a[i + 2 * j])

    # now pair the dim2 into dim4 via tensor product again
    for i, j in itertools.product(range(4), repeat=2):
        qc_muladd_modint(
            prod_P_dim2.a[0].a[i], prod_P_dim2.a[1].a[j], prod_P_dim4.a[i + 4 * j]
        )
    qc_theta_hadamard_bitflipped(prod_P_dim4)
    for i in range(8):
        qc_cswap(c_P_on_twist, prod_P_dim4.a[i], prod_P_dim4.a[15 - i])

    product_theta_nullpoint_reg = qc_load(prod_np, qc, CoordsDim4Type(p))
    qc_coordwise_muladd(prod_P_dim4, product_theta_nullpoint_reg, out_coords)
    qc.append(
        PreCircuit(N2Matrix_Superglue4D, p),
        out_coords,
        sigmas,
    )

    ################ uncompute ##############
    for i in range(8):
        qc_cswap(c_P_on_twist, prod_P_dim4.a[i], prod_P_dim4.a[15 - i])
    qc_theta_hadamard_bitflipped(prod_P_dim4, inverse=True)
    qc_unload(prod_np, product_theta_nullpoint_reg)
    for i, j in itertools.product(range(4), repeat=2):
        qc_mulsub_modint(
            prod_P_dim2.a[0].a[i], prod_P_dim2.a[1].a[j], prod_P_dim4.a[i + 4 * j]
        )
    for i, j in itertools.product(range(2), repeat=2):
        qc_mulsub_modint(P_xz.a[0].a[i], P_xz.a[1].a[j], prod_P_dim2.a[0].a[i + 2 * j])
        qc_mulsub_modint(P_xz.a[2].a[i], P_xz.a[3].a[j], prod_P_dim2.a[1].a[i + 2 * j])
    for i in range(4):
        qc_cxor(P.a[i].a["not_infty"], ModInt(1, p), P_xz.a[i].a[0])
        qc.x(P.a[i].a["not_infty"][0])
        qc_cxor(P.a[i].a["not_infty"], P.a[i].a["x"], P_xz.a[i].a[0])
        qc_cxor(P.a[i].a["not_infty"], ModInt(1, p), P_xz.a[i].a[1])

    qc.release_anc(P_xz, prod_P_dim2, prod_P_dim4)
    return out_coords


def prodpoint_to_theta(
    P: AffPointDim4,
    prod_np: CoordsDim4,
    sigmas: tuple[int, int, int, int],
    P_on_twist: QartonBool,
) -> ThetaPointDim4:
    p = P[0].ec.q

    P_xz = tuple(
        (ModInt(P[i].x, p), ModInt(1, p))
        if P[i].not_infty
        else (ModInt(1, p), ModInt(0, p))
        for i in range(4)
    )

    prod_P_xz = tuple(
        P_xz[0][i4] * P_xz[1][i3] * P_xz[2][i2] * P_xz[3][i1]
        for i1, i2, i3, i4 in itertools.product(range(2), repeat=4)
    )

    had_P = (
        classical_hadamard(prod_P_xz)[::-1]
        if not P_on_twist
        else classical_hadamard(prod_P_xz)
    )
    prod_P_theta = prod_np.mult(CoordsDim4(had_P))

    asigmas = cast(tuple[UInt, UInt, UInt, UInt], sigmas)  # typing stuff
    x = n2matrix_superglue4d_classical_function((prod_P_theta, asigmas))[0]

    # NOTE converting coords to ThetaPointDim4
    return ThetaPointDim4(x)


class InputData:
    def __init__(
        self,
        E: ECMontgomery,
        E_twist: ECMontgomery,
        prod_np: CoordsDim4,
        inv_np_E4: CoordsDim4,
        inv_dual_np_E4: CoordsDim4,
    ) -> None:
        self.E = E
        self.E_twist = E_twist
        self.prod_np = prod_np
        self.inv_np_E4 = inv_np_E4
        self.inv_dual_np_E4 = inv_dual_np_E4


@dummify
@memoize
class IsogenyChainWithGarbage(
    Circuit[
        tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            tuple[UInt, UInt, UInt, UInt],
        ],
        tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            tuple[UInt, UInt, UInt, UInt],
            ThetaStructureDim4,  # the output
            # all the "garbage" registers, packed into a single BitVector
            BitVector,
        ],
    ]
):
    def __init__(
        self, e: int, inp: InputData, strategy: tuple[int, ...] | None = None
    ) -> None:
        super().__init__()
        self.E = inp.E
        self.E_twist = inp.E_twist
        self.p = inp.E.q
        self.e = e
        p = self.p
        self.inv_np_E4 = inp.inv_np_E4
        self.inv_dual_np_E4 = inp.inv_dual_np_E4

        E = self.E
        E_twist = self.E_twist

        self.product_theta_nullpoint_E = inp.prod_np
        self.circ_theta_plus_minus = PreCircuit(
            Superglue4DThetaPlusMinus, self.product_theta_nullpoint_E, E
        )

        PointOnCurve4: DataType = AffPointDim4Type(E)
        PointOnTwist4: DataType = AffPointDim4Type(E_twist)

        PointOnVariableCurve4 = TupleType(4, AffMontgomeryPointVariableType(p))
        # i/o
        Ts = self.add_input_output(
            ITupleType(
                PointOnCurve4,
                PointOnCurve4,
                PointOnTwist4,
                PointOnTwist4,
                PointOnCurve4,
                PointOnTwist4,
            )
        )

        PointOnCurve4 = TupleType(4, AffMontgomeryPointType(E))
        PointOnTwist4 = TupleType(4, AffMontgomeryPointType(E_twist))

        T1, T2, T3, T4, T1p2, T3p4 = (Ts.a[i] for i in range(6))

        # fix typing
        T1 = qc_reg_cast(T1, PointOnCurve4)  # pyright: ignore[reportConstantRedefinition]
        T2 = qc_reg_cast(T2, PointOnCurve4)  # pyright: ignore[reportConstantRedefinition]
        T3 = qc_reg_cast(T3, PointOnTwist4)  # pyright: ignore[reportConstantRedefinition]
        T4 = qc_reg_cast(T4, PointOnTwist4)  # pyright: ignore[reportConstantRedefinition]
        T1p2 = qc_reg_cast(T1p2, PointOnCurve4)
        T3p4 = qc_reg_cast(T3p4, PointOnTwist4)

        sigmas = self.add_input_output(TupleType(4, UIntType(2)))

        # ------------------------------------------------
        garbage: list[Register] = []

        false, true = (qc_load(QartonBool(x), self, BoolType()) for x in (0, 1))

        rabbit_type = self.append(NormeqOutputToRabbitType(), sigmas)[-1]

        T5, Taux = T1p2, T3p4
        c_T5_on_twist = qc_load(QartonBool(0), self, BoolType())
        c_Taux_on_twist = qc_load(QartonBool(1), self, BoolType())

        qc_cswap(qc_reg_from_bits(rabbit_type[1]), c_T5_on_twist, c_Taux_on_twist)

        # There is a subtlety here: we have started from T5 and Taux
        # having fixed and incompatible types (they are on different curves)
        # But we control-swap them. So, we map them to variable types.
        # The curve they are on is stored in a quantum register.
        T5_curve_reg = qc_load(E, self, ECMontgomeryType(p))
        Taux_curve_reg = qc_load(E_twist, self, ECMontgomeryType(p))

        # casting them is possible, but we lose the information of the curve
        # which is in the type
        T5 = qc_reg_cast(T5, TupleType(4, AffMontgomeryPointVariableType(p)))  # pyright: ignore[reportConstantRedefinition]
        Taux = qc_reg_cast(Taux, TupleType(4, AffMontgomeryPointVariableType(p)))

        # now we can swap
        qc_cswap(qc_reg_from_bits(rabbit_type[1]), T5, Taux)
        # AND swap the curve information
        qc_cswap(qc_reg_from_bits(rabbit_type[1]), T5_curve_reg, Taux_curve_reg)

        # now we *can* use the curve information to compute the doubling
        T5_8 = qc_double_iter_prod_mont_var(T5, T5_curve_reg, e - 1)
        Taux_16 = qc_double_iter_prod_mont_var(Taux, Taux_curve_reg, e - 2)

        # now unload the curves, since we don't need them anymore
        # T5_8 and Taux_16 will remain variable types
        qc_cswap(qc_reg_from_bits(rabbit_type[1]), T5_curve_reg, Taux_curve_reg)
        qc_unload(E, T5_curve_reg)
        qc_unload(E_twist, Taux_curve_reg)

        garbage += [rabbit_type, c_T5_on_twist, c_Taux_on_twist, T5_8, Taux_16]

        n = e
        assert n > 1
        if strategy is None:
            self.strategy: tuple[int, ...] = balanced_strategy(n)
        else:
            self.strategy = strategy

        # Length of the chain
        n = self.e

        doublings = [0]
        kernel_elements = [[T1, T2, T3, T4]]
        strat_idx = 0

        # each of these is only (re)assigned when k == 0 (or k == 1), but read again
        # on later iterations / in later same-iteration branches checking k in a
        # separate if-statement; pre-bind them so pyright doesn't consider them
        # unbound there (they are always actually assigned by the time they're read)
        inv_np: Any = None
        inv_dual_np_dbl: Any = None
        fTaux: Any = None
        theta_str: Any = None
        inv_dual_np_codom: Register | None = None
        inv_image_Taux_dual: Register | None = None

        for k in range(n):
            curr_level = sum(doublings)
            ker = kernel_elements[-1]

            while curr_level != (n - 1 - k):
                doublings.append(self.strategy[strat_idx])
                curr_level += self.strategy[strat_idx]

                # Perform the doublings and update kernel elements
                if k == 0:
                    ker = [
                        qc_double_iter_prod_mont(T, self.strategy[strat_idx])
                        for T in ker
                    ]
                else:
                    ker = [qc_reg_cast(T, ThetaPointDim4Type(p)) for T in ker]
                    ker = [
                        qc_double_iter_theta(
                            T, inv_np, inv_dual_np_dbl, self.strategy[strat_idx]
                        )
                        for T in ker
                    ]
                kernel_elements.append(ker)
                garbage += ker

                # Update bookkeeping variable
                strat_idx += 1

            # Compute the codomain from the 8-torsion
            if k == 0:
                # For HSK_8, we need H((Taux_16 + T_i) · (Taux_16 - T_i))

                theta_plus_minuses = sum(
                    [
                        self.append(
                            self.circ_theta_plus_minus,
                            qc_reg_cast(Taux_16, PointOnVariableCurve4),
                            qc_reg_cast(Ti, PointOnVariableCurve4),
                            c_Taux_on_twist,
                            c_Ti_on_twist,
                            sigmas,
                        )[-1]
                        for Ti, c_Ti_on_twist in zip(
                            (*ker, T5_8),
                            (false, false, true, true, c_T5_on_twist),
                            strict=True,
                        )
                    ],
                    start=qc_empty_reg(self, BitVectorType(0)),
                )
                HSK_8 = qc_reg_cast(theta_plus_minuses, TupleType(5, CoordsDim4Type(p)))
                for i in range(5):
                    qc_theta_hadamard(HSK_8.a[i])

                inv_image_Taux_dual = self.append(
                    PreCircuit(Superglue4DRabbitCodomain, p), HSK_8, rabbit_type
                )[-1]

                Taux_16_theta = qc_prodpoint_to_theta(
                    Taux_16, sigmas, self.product_theta_nullpoint_E, c_Taux_on_twist
                )

                HS_Taux, theta_str = self.get_ancs(CoordsDim4Type(p), CoordsDim4Type(p))
                qc_coordwise_sqradd(Taux_16_theta, HS_Taux)
                qc_theta_hadamard(HS_Taux)
                qc_coordwise_muladd(inv_image_Taux_dual, HS_Taux, theta_str)
                qc_theta_hadamard(theta_str)

                # partial uncompute
                qc_theta_hadamard(HS_Taux, inverse=True)
                qc_coordwise_sqrsub(Taux_16_theta, HS_Taux)
                # self.assert_anc(HS_Taux)
                self.release_anc(HS_Taux)
                ##########################

                # ---- uncomputation of _theta_plus_minuses
                for i in range(5):
                    qc_theta_hadamard(HSK_8.a[i])

                _theta_plus_minuses = [HSK_8.a[i] for i in range(5)]
                for i, (Ti, c_Ti_on_twist) in enumerate(
                    zip(
                        (*ker, T5_8),
                        (false, false, true, true, c_T5_on_twist),
                        strict=True,
                    )
                ):
                    self.append(
                        self.circ_theta_plus_minus.inverse(),
                        qc_reg_cast(Taux_16, PointOnVariableCurve4),
                        qc_reg_cast(Ti, PointOnVariableCurve4),
                        c_Taux_on_twist,
                        c_Ti_on_twist,
                        sigmas,
                        _theta_plus_minuses[i],
                    )
                assert all(r._destroyed for r in _theta_plus_minuses)  # type: ignore
                # -----

                fTaux = self.add_anc(ThetaPointDim4Type(p))
                qc_coordwise_invadd(
                    inv_image_Taux_dual, qc_reg_cast(fTaux, CoordsDim4Type(p))
                )
                qc_theta_hadamard(fTaux)

                garbage += [inv_image_Taux_dual, fTaux, theta_str, Taux_16_theta]

            elif k == 1:
                HSK_8 = self.add_anc(TupleType(5, CoordsDim4Type(p)))  # pyright: ignore[reportConstantRedefinition]
                for i, T in enumerate((*ker, fTaux)):
                    qc_coordwise_sqradd(T, HSK_8.a[i])
                    qc_theta_hadamard(HSK_8.a[i])

                inv_dual_np_codom = self.append(
                    PreCircuit(Theta2IsogenyDim4Second_Codomain, p),
                    HSK_8,
                    c_Taux_on_twist,  # 1 iff Taux is T3p4
                )[-1]

                for i, T in enumerate((*ker, fTaux)):  # pyright: ignore[reportConstantRedefinition]
                    qc_theta_hadamard(HSK_8.a[i], inverse=True)
                    qc_coordwise_sqrsub(T, HSK_8.a[i])

                self.test_anc(HSK_8)
                self.release_anc(HSK_8)
                garbage += [inv_dual_np_codom]
            else:
                HSK_8 = self.add_anc(TupleType(4, CoordsDim4Type(p)))  # pyright: ignore[reportConstantRedefinition]
                for i, T in enumerate(ker):  # pyright: ignore[reportConstantRedefinition]
                    qc_coordwise_sqradd(T, HSK_8.a[i])
                    qc_theta_hadamard(HSK_8.a[i])

                inv_dual_np_codom = self.append(
                    PreCircuit(Theta2IsogenyDim4Generic_Codomain, p, (k <= 10)),
                    HSK_8,
                )[-1]

                for i, T in enumerate(ker):  # pyright: ignore[reportConstantRedefinition]
                    qc_theta_hadamard(HSK_8.a[i], inverse=True)
                    qc_coordwise_sqrsub(T, HSK_8.a[i])
                self.test_anc(HSK_8)
                self.release_anc(HSK_8)
                garbage += [inv_dual_np_codom]

            if k == 0:
                # lines equivalent to theta_str = ThetaStructureDim4(theta_np_codom)
                # need to compute theta_str.invert() for inv_np
                # need to compute invert(hadamard(square(theta_str))) for inv_dual_np_dbl
                inv_np, HS_np, inv_dual_np_dbl = self.add_ancs(
                    CoordsDim4Type(p), CoordsDim4Type(p), CoordsDim4Type(p)
                )
                qc_coordwise_invadd(theta_str, inv_np)
                qc_coordwise_sqradd(theta_str, HS_np)
                qc_theta_hadamard(HS_np)
                qc_coordwise_invadd(HS_np, inv_dual_np_dbl)

                # partial uncompute
                qc_theta_hadamard(HS_np, inverse=True)
                qc_coordwise_sqrsub(theta_str, HS_np)
                self.test_anc(HS_np)
                self.release_anc(HS_np)

            else:
                assert inv_dual_np_codom is not None
                inv_np, inv_dual_np_dbl = self.append(
                    PreCircuit(ThetaArithmeticPrecomputation, p),
                    inv_dual_np_codom,
                )[1:]

            garbage += [inv_np, inv_dual_np_dbl]

            # Remove elements from list
            kernel_elements.pop()
            doublings.pop()

            # Push through points for the next step
            gluing_eval = PreCircuit(
                Superglue4DEvaluation, self.product_theta_nullpoint_E, self.E
            )
            generic_eval = PreCircuit(Theta2IsogenyDim4Generic_Evaluation, p)
            self.gluing_eval = gluing_eval
            self.generic_eval = generic_eval
            if k == 0:
                assert inv_image_Taux_dual is not None  # for type checking
                kernel_elements = [
                    [
                        self.append(
                            gluing_eval,
                            qc_reg_cast(
                                T, TupleType(4, AffMontgomeryPointVariableType(p))
                            ),
                            qc_reg_cast(
                                Taux_16, TupleType(4, AffMontgomeryPointVariableType(p))
                            ),
                            T_on_twist,
                            c_Taux_on_twist,
                            inv_image_Taux_dual,
                            sigmas,
                        )[-1]
                        for T, T_on_twist in zip(
                            kernel[:4], (false, false, true, true), strict=True
                        )
                    ]
                    for kernel in kernel_elements
                ]
            else:
                assert inv_dual_np_codom is not None  # for type checking
                kernel_elements = [
                    [
                        self.append(
                            generic_eval,
                            qc_reg_cast(T, ThetaPointDim4Type(p)),
                            inv_dual_np_codom,
                        )[-1]
                        for T in kernel
                    ]
                    for kernel in kernel_elements
                ]
            for ker in kernel_elements:
                garbage += ker

        qc_unload(QartonBool(0), false)
        qc_unload(QartonBool(1), true)

        last_theta_str = self.add_anc(ThetaStructureDim4Type(p))
        qc_coordwise_invadd(inv_np, qc_reg_cast(last_theta_str, CoordsDim4Type(p)))
        self.add_output(last_theta_str)

        # pack all the garbage registers into a single BitVector output, instead of
        # a variable number of separately-typed ones; keep their datatypes around
        # (in the same order) so dummy_classical_function can mirror the packing
        self.garbage_types = [anc.datatype for anc in garbage]
        garbage_bits = qc_concat(*garbage)
        self.add_output(garbage_bits)

        assert kernel_elements == []

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            tuple[UInt, UInt, UInt, UInt],
        ],
    ) -> tuple[
        tuple[
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
        ],
        tuple[UInt, UInt, UInt, UInt],
        ThetaStructureDim4,
        BitVector,
    ]:
        Ts, sigmas = args
        T1, T2, T3, T4, T1p2, T3p4 = (AffPointDim4(T) for T in Ts)
        # p = f * 2**(e + 3) - 1
        # the Ti are in the 2**(e+2)-torsion.
        # the length of the chain is e.

        p = self.p
        garbage: list[Any] = []

        rabbit_type = normeqoutputtorabbittype_classical_function(sigmas)[1]

        Taux: AffPointDim4
        T5: AffPointDim4

        if rabbit_type >> 1:
            Taux, T5 = T1p2, T3p4  # pyright: ignore[reportConstantRedefinition]
        else:
            Taux, T5 = T3p4, T1p2  # pyright: ignore[reportConstantRedefinition]

        Taux_on_twist = QartonBool(Taux[0].ec == self.E_twist)
        T5_on_twist = QartonBool(not Taux_on_twist)
        true = QartonBool(1)
        false = QartonBool(0)

        garbage += [rabbit_type, T5_on_twist, Taux_on_twist]

        Taux = Taux * (2 ** (self.e - 2))  # 16-tors
        T5 = T5 * (2 ** (self.e - 1))  # pyright: ignore[reportConstantRedefinition] # 8-tors

        garbage += [T5, Taux]

        # Length of the chain
        n = self.e

        doublings = [0]
        kernel_elements: list[list[ThetaPointDim4 | AffPointDim4]] = [[T1, T2, T3, T4]]
        strat_idx = 0

        # mirrors the pre-binding in __init__ above, for the same reason
        fTaux: ThetaPointDim4 | None = None
        theta_np_codom: Any = None
        inv_dual_np_codom: CoordsDim4 | None = None
        inv_image_Taux_dual: CoordsDim4 | None = None
        theta_str: ThetaStructureDim4 | None = None

        for k in range(n):
            curr_level = sum(doublings)
            ker = kernel_elements[-1]

            while curr_level != (n - 1 - k):
                doublings.append(self.strategy[strat_idx])
                curr_level += self.strategy[strat_idx]

                # Perform the doublings and update kernel elements
                ker = [T * (1 << self.strategy[strat_idx]) for T in ker]
                kernel_elements.append(ker)
                garbage += ker

                # Update bookkeeping variable
                strat_idx += 1

            # Compute the codomain from the 8-torsion
            if k == 0:
                HSK_8 = tuple(
                    CoordsDim4(
                        classical_hadamard(
                            superglue4dthetaplusminus_dummy_classical_function(
                                (Taux, Ti, Taux_on_twist, Ti_on_twist, sigmas),  # type: ignore
                                self.product_theta_nullpoint_E,
                                ModInt(self.E.A, p),
                                p,
                            )[-1]
                        )
                    )
                    for Ti, Ti_on_twist in zip(
                        (*ker, T5), (false, false, true, true, T5_on_twist), strict=True
                    )
                )
                assert len(HSK_8) == 5  # for typing
                inv_image_Taux_dual = superglue4drabbitcodomain_classical_function(
                    p, (HSK_8, rabbit_type)
                )[-1]

                Taux_theta = prodpoint_to_theta(
                    Taux, self.product_theta_nullpoint_E, sigmas, Taux_on_twist
                )
                theta_np_codom = (
                    Taux_theta.coords.square()
                    .hadamard()
                    .mult(inv_image_Taux_dual)
                    .hadamard()
                )
                fTaux = ThetaPointDim4(inv_image_Taux_dual.invert().hadamard())
                garbage += [inv_image_Taux_dual, fTaux, theta_np_codom, Taux_theta]

            elif k == 1:
                assert all(isinstance(T, ThetaPointDim4) for T in ker)
                ker_theta = cast(list[ThetaPointDim4], ker)
                HSK_8 = tuple(  # pyright: ignore[reportConstantRedefinition]
                    T.coords.square().hadamard() for T in ker_theta
                )
                assert fTaux is not None
                HSK_Taux = fTaux.coords.square().hadamard()
                assert len(HSK_8) == 4  # for typing

                inv_dual_np_codom = theta2isogenydim4second_codomain_classical_function(
                    p, ((*HSK_8, HSK_Taux), Taux_on_twist)
                )[-1]

                garbage += [inv_dual_np_codom]
            else:
                assert all(isinstance(T, ThetaPointDim4) for T in ker)
                ker_theta = cast(list[ThetaPointDim4], ker)
                HSK_8 = tuple(  # pyright: ignore[reportConstantRedefinition]
                    T.coords.square().hadamard() for T in ker_theta
                )
                assert len(HSK_8) == 4
                inv_dual_np_codom = (
                    theta2isogenydim4generic_codomain_classical_function(
                        p, k <= 11, HSK_8
                    )[-1]
                )
                garbage += [inv_dual_np_codom]

            if k == 0:
                theta_str = ThetaStructureDim4(theta_np_codom)
            else:
                assert inv_dual_np_codom is not None  # for typing
                theta_str = ThetaStructureDim4.from_inv_dual_null_point(
                    inv_dual_np_codom
                )
            garbage += [
                theta_str.inverse_null_point,
                theta_str.inverse_dual_null_point_codomain,
            ]

            # Remove elements from list
            kernel_elements.pop()
            doublings.pop()

            # Push through points for the next step
            if k == 0:
                kernel_elements = [
                    [
                        ThetaPointDim4(
                            superglue4devaluation_dummy_classical_function(
                                (
                                    T,  # type: ignore
                                    Taux,  # type: ignore
                                    T_on_twist,  # type: ignore
                                    Taux_on_twist,  # type: ignore
                                    inv_image_Taux_dual,  # type: ignore
                                    sigmas,  # type: ignore
                                ),
                                self.product_theta_nullpoint_E,
                                ModInt(self.E.A, p),
                                p,
                            )[-1],
                            parent=theta_str,
                        )
                        for T, T_on_twist in zip(
                            kernel[:4],
                            (false, false, true, true),
                            strict=True,
                        )
                    ]
                    for kernel in kernel_elements
                ]

            else:
                kernel_elements = [
                    [
                        ThetaPointDim4(
                            theta2isogenydim4generic_evaluation_classical_function(
                                (T, inv_dual_np_codom)  # type: ignore
                            )[-1].coords,
                            parent=theta_str,
                        )
                        for T in kernel
                    ]
                    for kernel in kernel_elements
                ]
            for kernel in kernel_elements:
                garbage += kernel

        assert kernel_elements == []
        assert theta_str is not None

        # theta_str itself (not just its coords), matching the ThetaStructureDim4Type
        # register packed on the quantum side

        assert len(garbage) == len(self.garbage_types)

        garbage_bits = BitVector()
        for dtype, value in zip(self.garbage_types, garbage, strict=True):
            try:
                garbage_bits += dtype.to_register(value)
            except Exception as e:
                print("Incorrect value to datatype mapping!")
                print(dtype)
                print(value)
                raise e

        return *args, theta_str, garbage_bits

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            tuple[UInt, UInt, UInt, UInt],
            ThetaStructureDim4,
            BitVector,
        ],
    ) -> tuple[
        tuple[
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
        ],
        tuple[UInt, UInt, UInt, UInt],
    ]:
        return args[:-2]


@dummify
@memoize
class IsogenyChain(
    Circuit[
        tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            UInt,
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
        tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            UInt,
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            ModInt,
        ],
    ]
):
    """
    Circuit to compute the 4-dimensional isogeny chain.
    """

    def __init__(
        self,
        e: int,
        inp: InputData,
        strategy: tuple[int, ...] | None = None,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()

        E = inp.E
        E_twist = inp.E_twist
        p = E.q
        self.p = p
        self.input_data = inp

        PointOnCurve4 = AffPointDim4Type(E)
        PointOnTwist4 = AffPointDim4Type(E_twist)

        # PointOnCurve4 = TupleType(4, AffMontgomeryPointType(E))
        # PointOnTwist4 = TupleType(4, AffMontgomeryPointType(E_twist))

        # i/o
        Ts = self.add_input_output(
            ITupleType(
                PointOnCurve4,
                PointOnCurve4,
                PointOnTwist4,
                PointOnTwist4,
                PointOnCurve4,
                PointOnTwist4,
            )
        )
        # T1, T2, T3, T4, T1p2, T3p4 = self.add_ancs(*([ThetaPointDim4Type(p)] * 6))
        norm_mod2 = self.add_input_output(UIntType(1))
        sigmas = self.add_input_output(TupleType(4, UIntType(2)))
        splitting_type = self.add_input_output(UIntType(2))

        self.garbage_circ = PreCircuit(IsogenyChainWithGarbage, e, inp, strategy)
        theta_str, garbage_bits = self.append(self.garbage_circ, Ts, sigmas)[-2:]

        self.x(norm_mod2[0])
        norm_is_even = qc_reg_cast(norm_mod2, BoolType())

        self.splitting_circ = PreCircuit(SplittingDim4, p)
        A_codom = self.append(
            self.splitting_circ, theta_str, splitting_type, norm_is_even
        )[-1]
        J_codom = self.append(PreCircuit(MontgomeryToJInvariantDiv256, p), A_codom)[-1]
        self.add_output(J_codom)

        ################# uncompute a huuuge quantity of things #################
        self.append(
            self.splitting_circ.inverse(),
            theta_str,
            splitting_type,
            norm_is_even,
            A_codom,
        )
        self.x(norm_mod2[0])
        self.append(self.garbage_circ.inverse(), Ts, sigmas, theta_str, garbage_bits)

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            UInt,
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ) -> tuple[
        tuple[
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
        ],
        UInt,
        tuple[UInt, UInt, UInt, UInt],
        UInt,
        ModInt,
    ]:
        Ts, norm_mod2, sigmas, splitting_type = args

        p = self.p
        theta_str = self.garbage_circ.dummy_classical_function((Ts, sigmas))[-2]

        A_codomain = splittingdim4_classical_function(
            p, (theta_str, splitting_type, QartonBool(1 - norm_mod2))
        )[-1]
        J_codomain = montgomerytojinvariantdiv256_classical_function(p, A_codomain)[-1]

        return *args, J_codomain

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
            UInt,
            tuple[UInt, UInt, UInt, UInt],
            UInt,
            ModInt,
        ],
    ) -> tuple[
        tuple[
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
        ],
        UInt,
        tuple[UInt, UInt, UInt, UInt],
        UInt,
    ]:
        return args[:-1]
