"""
*Pebbled* version of the dimension-4 isogeny chain circuit.

We build the same circuit, but rewrite it as a sequence of operations which transform
the kernel elements and current theta structure. Having a sequence of operations, we
rely on Qarton's built-in ``PebblingLine`` class, which creates a pebbling
based on Gidney's ``sweep-and-clean`` spooky pebbling strategy.

Most of the classes in this module are just definitions of the successive steps. Some
of these steps only modify part of the registers, but we still have to keep the complete
input and output.

"""

from __future__ import annotations

from typing import Any, cast

from qarton.algorithms import PebblingLineGeneric
from qarton.binary_operations import (
    qc_cswap,
    qc_load,
    qc_swap,
    qc_unload,
    qc_xor,
)
from qarton.circuit import (
    BackendSpecifier,
    BitVectorType,
    BoolType,
    Circuit,
    DataType,
    EmptyBackendSpecifier,
    InPlaceCircuit,
    ITupleType,
    PartialCircuit,
    PreCircuit,
    QartonBool,
    Register,
    TupleType,
    UInt,
    UIntType,
    memoize,
    qc_concat,
    qc_empty_reg,
    qc_make_ituple,
    qc_make_tuple,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
)

from qisogenies.montgomery.montgomery_curve import ECMontgomeryType
from qisogenies.norm_equation.util import FAST_BACKENDS
from qisogenies.theta.affpoint_dim4 import AffPointDim4, AffPointDim4Type
from qisogenies.theta.isogeny_chain_dim4 import (
    InputData,
    prodpoint_to_theta,
    qc_prodpoint_to_theta,
)
from qisogenies.theta.splitting_dim4 import (
    SplittingDim4,
    splittingdim4_classical_function,
)

from ..montgomery import (
    AffMontgomeryPointType,
    AffMontgomeryPointVariableType,
)
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
from .superglue_dim4 import (
    Superglue4DEvaluation,
    Superglue4DRabbitCodomain,
    Superglue4DThetaPlusMinus,
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
    "IsogenyChainWithPebbling",
]


def step_type(n: int, p: int) -> DataType:
    # kernel, then inv_np, inv_dual_np_dbl, inv_codom
    return ITupleType(
        TupleType(n, TupleType(4, ThetaPointDim4Type(p))),
        CoordsDim4Type(p),
        CoordsDim4Type(p),
        CoordsDim4Type(p),
    )


class _Step1(InPlaceCircuit):
    def __init__(
        self,
        nbr_of_kernel_elements: int,
        p: int,
        strategy: tuple[int, ...],
        strat_idx: int,
        new_elements: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        input_reg = self.add_input_output(step_type(nbr_of_kernel_elements, p))
        output_reg = self.add_input_output(
            step_type(nbr_of_kernel_elements + new_elements, p)
        )

        ker = input_reg.a[0].a[-1]  # last kernel elt
        klen = TupleType.get_from_register(ker).tuple_length
        # start by copying everything to output_reg (except the new elements, still 0)

        qc_xor(input_reg.a[1:4], output_reg.a[1:4])
        qc_xor(
            input_reg.a[0],
            output_reg.a[0].a[0:nbr_of_kernel_elements],
        )
        # then take the last kernel elt and double

        inv_np = input_reg.a[1]
        inv_dual_np_dbl = input_reg.a[2]
        for i in range(new_elements):
            newker = qc_make_tuple(
                *list(
                    qc_double_iter_theta(
                        ker.a[j], inv_np, inv_dual_np_dbl, strategy[strat_idx]
                    )
                    for j in range(klen)
                )
            )
            strat_idx += 1
            qc_swap(newker, output_reg.a[0].a[nbr_of_kernel_elements + i])
            self.test_anc(newker)
            self.release_anc(newker)
            # this becomes the new element that we double
            ker = output_reg.a[0].a[nbr_of_kernel_elements + i]
            klen = TupleType.get_from_register(ker).tuple_length


class _Step2B(InPlaceCircuit):
    """
    Update of inv_dual_np_codom using kernel.
    """

    def __init__(
        self,
        nbr_of_kernel_elements: int,
        p: int,
        kleq10: bool,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        input_reg = self.add_input_output(step_type(nbr_of_kernel_elements, p))
        output_reg = self.add_input_output(step_type(nbr_of_kernel_elements, p))

        ker = input_reg.a[0].a[-1]  # tuple type of ThetaPointDim4

        HSK_8 = self.add_anc(TupleType(4, CoordsDim4Type(p)))  # pyright: ignore[reportConstantRedefinition]
        for i in range(4):  # pyright: ignore[reportConstantRedefinition]
            qc_coordwise_sqradd(ker.a[i], HSK_8.a[i])
            qc_theta_hadamard(HSK_8.a[i])

        _, inv_dual_np_codom = self.append(
            PreCircuit(Theta2IsogenyDim4Generic_Codomain, p, kleq10),
            HSK_8,
        )

        for i in range(4):  # pyright: ignore[reportConstantRedefinition]
            qc_theta_hadamard(HSK_8.a[i], inverse=True)
            qc_coordwise_sqrsub(ker.a[i], HSK_8.a[i])
        self.test_anc(HSK_8)
        self.release_anc(HSK_8)
        qc_swap(inv_dual_np_codom, output_reg.a[3])

        self.test_anc(inv_dual_np_codom)
        self.release_anc(inv_dual_np_codom)

        # now we need to copy all other registers
        qc_xor(input_reg.a[0:3], output_reg.a[0:3])


class _Step3(InPlaceCircuit):
    """
    Update of inv_dual_np_codom using kernel.
    """

    def __init__(
        self,
        nbr_of_kernel_elements: int,
        p: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        input_reg = self.add_input_output(step_type(nbr_of_kernel_elements, p))
        output_reg = self.add_input_output(step_type(nbr_of_kernel_elements - 1, p))

        inv_dual_np_codom = input_reg.a[3]

        _, new_inv_np, new_inv_dual_np_dbl = self.append(
            PreCircuit(ThetaArithmeticPrecomputation, p),
            inv_dual_np_codom,
        )
        qc_swap(new_inv_np, output_reg.a[1])
        qc_swap(new_inv_dual_np_dbl, output_reg.a[2])
        qc_xor(inv_dual_np_codom, output_reg.a[3])
        self.test_anc(new_inv_np, new_inv_dual_np_dbl)
        self.release_anc(new_inv_np, new_inv_dual_np_dbl)

        generic_eval = PreCircuit(Theta2IsogenyDim4Generic_Evaluation, p)

        kernel = input_reg.a[0]
        new_kernel = output_reg.a[0]
        # evaluate the kernel elements
        for i in range(nbr_of_kernel_elements - 1):
            for j in range(4):
                _, _, tmp = self.append(
                    generic_eval,
                    kernel.a[i].a[j],
                    inv_dual_np_codom,
                )
                # tmp is the new point
                qc_swap(tmp, new_kernel.a[i].a[j])
                self.test_anc(tmp)
                self.release_anc(tmp)
        # full output kernel should now be created


class _LastStep(InPlaceCircuit):
    def __init__(
        self,
        nbr_of_kernel_elements: int,
        p: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        input_reg = self.add_input_output(step_type(nbr_of_kernel_elements, p))
        last_theta_str = self.add_input_output(ThetaStructureDim4Type(p))
        inv_np = input_reg.a[1]
        qc_coordwise_invadd(inv_np, qc_reg_cast(last_theta_str, CoordsDim4Type(p)))


def nbr_of_kernel_elements_at_beginning(strategy: tuple[int, ...], n: int) -> int:

    doublings = [0]
    nbr = 1
    curr_level = sum(doublings)
    strat_idx = 0

    while curr_level != (n - 1):
        doublings.append(strategy[strat_idx])
        curr_level += strategy[strat_idx]
        nbr += 1
        strat_idx += 1
    nbr -= 1
    doublings.pop()

    curr_level = sum(doublings)

    while curr_level != (n - 2):
        doublings.append(strategy[strat_idx])
        curr_level += strategy[strat_idx]
        nbr += 1
        strat_idx += 1
    nbr -= 1
    doublings.pop()

    return nbr


class IsogenyChainFirstStepWithGarbage(Circuit):
    """
    Take as input 6 points and the splitting type, and perform the
    first computations in the isogeny chain (iterations k = 0 and k = 1). Produces
    garbage which is computed-uncomputed.
    """

    def __init__(
        self,
        e: int,
        inp: InputData,
        strategy: tuple[int, ...] | None = None,
        backends: BackendSpecifier = EmptyBackendSpecifier,
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

        # ==================================================================
        # k = 0: handled separately from the loop below (it differs from
        # every other step in almost every part of its body), all other
        # steps k = 1, ..., n - 1 are handled by the loop.
        # ==================================================================
        curr_level = sum(doublings)
        ker = kernel_elements[-1]

        while curr_level != (n - 1):
            doublings.append(self.strategy[strat_idx])
            curr_level += self.strategy[strat_idx]

            # Perform the doublings and update kernel elements
            ker = [qc_double_iter_prod_mont(T, self.strategy[strat_idx]) for T in ker]
            kernel_elements.append(ker)
            garbage += ker

            # Update bookkeeping variable
            strat_idx += 1

        # Compute the codomain from the 8-torsion
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
        qc_coordwise_invadd(inv_image_Taux_dual, qc_reg_cast(fTaux, CoordsDim4Type(p)))
        qc_theta_hadamard(fTaux)

        garbage += [inv_image_Taux_dual, fTaux, theta_str, Taux_16_theta]

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

        # Remove elements from list
        kernel_elements.pop()
        doublings.pop()

        # Push through points for the next step
        gluing_eval = PreCircuit(
            Superglue4DEvaluation, self.product_theta_nullpoint_E, self.E
        )

        kernel_elements = [
            [
                self.append(
                    gluing_eval,
                    qc_reg_cast(T, TupleType(4, AffMontgomeryPointVariableType(p))),
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

        #
        current_garbage = qc_make_ituple(
            qc_make_tuple(*(qc_make_tuple(*k) for k in kernel_elements)),
            inv_np,
            inv_dual_np_dbl,
        )

        garbage.append(current_garbage)

        # now k = 1 -------------------------------------

        curr_level = sum(doublings)
        ker = kernel_elements[-1]

        while curr_level != (n - 2):
            doublings.append(self.strategy[strat_idx])
            curr_level += self.strategy[strat_idx]
            ker = [qc_reg_cast(T, ThetaPointDim4Type(p)) for T in ker]
            ker = [
                qc_double_iter_theta(
                    T, inv_np, inv_dual_np_dbl, self.strategy[strat_idx]
                )
                for T in ker
            ]
            kernel_elements.append(ker)
            garbage += ker
            strat_idx += 1

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
        # garbage += [inv_dual_np_codom]

        assert inv_dual_np_codom is not None
        inv_np, inv_dual_np_dbl = self.append(
            PreCircuit(ThetaArithmeticPrecomputation, p),
            inv_dual_np_codom,
        )[1:]

        # garbage += [inv_np, inv_dual_np_dbl]

        # Remove elements from list
        kernel_elements.pop()
        doublings.pop()

        # Push through points for the next step
        generic_eval = PreCircuit(Theta2IsogenyDim4Generic_Evaluation, p)

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
        # ---------------------- end of the first iterations

        # ---------------------------------------------
        # type fixing
        kernel_elements = [
            [qc_reg_cast(_tmp, ThetaPointDim4Type(p)) for _tmp in _ttmp]
            for _ttmp in kernel_elements
        ]

        # -----------------------------------------------------------------------------
        # ---------------------------------------------------------------

        # nbr_of_kernel_elements = len(kernel_elements)
        # inv_dual_np_codom = self.get_anc(CoordsDim4Type(p))  # just sitting there
        complete_state = qc_make_ituple(
            qc_make_tuple(*(qc_make_tuple(*k) for k in kernel_elements)),
            inv_np,
            inv_dual_np_dbl,
            inv_dual_np_codom,
        )

        qc_unload(QartonBool(0), false)
        qc_unload(QartonBool(1), true)

        # garbage.append(complete_state)
        self.add_output(complete_state)
        self.add_output(qc_concat(*garbage))


class IsogenyChainFirstStep(InPlaceCircuit):
    def __init__(
        self,
        e: int,
        inp: InputData,
        strategy: tuple[int, ...] | None = None,
        backends: BackendSpecifier = EmptyBackendSpecifier,
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

        # i/o
        ts_sigmas = self.add_input_output(
            ITupleType(
                ITupleType(
                    PointOnCurve4,
                    PointOnCurve4,
                    PointOnTwist4,
                    PointOnTwist4,
                    PointOnCurve4,
                    PointOnTwist4,
                ),
                TupleType(4, UIntType(2)),
            )
        )

        n = e
        assert n > 1
        if strategy is None:
            self.strategy: tuple[int, ...] = balanced_strategy(n)
        else:
            self.strategy = strategy

        qc = PreCircuit(IsogenyChainFirstStepWithGarbage, e, inp, self.strategy)
        nbr_of_kernel_elts = nbr_of_kernel_elements_at_beginning(self.strategy, e)

        complete_state = self.add_input_output(step_type(nbr_of_kernel_elts, p))

        _, _, comp, garbage = self.append(qc, ts_sigmas.a[0], ts_sigmas.a[1])
        qc_xor(comp, complete_state)
        self.append(qc.inverse(), ts_sigmas.a[0], ts_sigmas.a[1], comp, garbage)

        # should now be good


class IsogenyChainLastStep(InPlaceCircuit):
    def __init__(
        self, p: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()

        # last step take: the last theta structure, norm_mod2, the splitting type,
        # the new output register in which result is written

        norm_mod2 = self.add_input_output(UIntType(1))
        splitting_type = self.add_input_output(UIntType(2))

        theta_str = self.add_input_output(ThetaStructureDim4Type(p))
        result = self.add_input_output(ModIntType(p))
        # ------------------------------

        self.x(norm_mod2[0])
        norm_is_even = qc_reg_cast(norm_mod2, BoolType())

        A_codom = self.append(
            PreCircuit(SplittingDim4, p), theta_str, splitting_type, norm_is_even
        )[-1]
        _, J_codom = self.append(PreCircuit(MontgomeryToJInvariantDiv256, p), A_codom)
        qc_swap(J_codom, result)
        self.release_anc(J_codom)

        self.append(
            PreCircuit(SplittingDim4, p, inverse=True),
            theta_str,
            splitting_type,
            norm_is_even,
            A_codom,
        )
        self.x(norm_mod2[0])
        # destroys A_codom so should be good


@memoize
class IsogenyChainWithPebbling(
    PebblingLineGeneric,
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
    ],
):
    def __init__(
        self,
        e: int,
        inp: InputData,
        strategy: tuple[int, ...] | None = None,
        backends: BackendSpecifier = EmptyBackendSpecifier,
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

        norm_mod2 = self.add_input_output(UIntType(1))
        sigmas = self.add_input_output(TupleType(4, UIntType(2)))
        splitting_type = self.add_input_output(UIntType(2))

        result = self.add_anc_output(ModIntType(p))

        ts_sigmas = qc_make_ituple(Ts, sigmas)

        # ----------------------------------------------

        n = e
        assert n > 1
        if strategy is None:
            self.strategy: tuple[int, ...] = balanced_strategy(n)
        else:
            self.strategy = strategy

        # Length of the chain
        n = self.e

        # ==================================================================
        # replaying the schedule for k = 0 and k = 1
        doublings = [0]
        nbr_of_kernel_elements = 1
        curr_level = sum(doublings)
        strat_idx = 0

        while curr_level != (n - 1):
            doublings.append(self.strategy[strat_idx])
            curr_level += self.strategy[strat_idx]
            nbr_of_kernel_elements += 1
            strat_idx += 1
        nbr_of_kernel_elements -= 1
        doublings.pop()

        curr_level = sum(doublings)

        while curr_level != (n - 2):
            doublings.append(self.strategy[strat_idx])
            curr_level += self.strategy[strat_idx]
            nbr_of_kernel_elements += 1
            strat_idx += 1
        nbr_of_kernel_elements -= 1
        doublings.pop()

        # ----------------------------------------------

        # first step
        _pc = IsogenyChainFirstStep(e, inp, strategy, backends=FAST_BACKENDS)
        self.add_first_step(ts_sigmas, _pc)

        for k in range(2, n):
            curr_level = sum(doublings)

            # ------------------------------
            # play the strategy genuinely

            _strat_idx = strat_idx  # to give to the sub-circuit
            _nbr_of_kernel_elements = nbr_of_kernel_elements
            new_elements = 0
            while curr_level != (n - 1 - k):
                doublings.append(self.strategy[strat_idx])
                curr_level += self.strategy[strat_idx]
                new_elements += 1
                strat_idx += 1

            nbr_of_kernel_elements += new_elements

            self.add_step(
                _Step1(
                    _nbr_of_kernel_elements,
                    p,
                    self.strategy,
                    _strat_idx,
                    new_elements,
                    backends=FAST_BACKENDS,
                )
            )

            self.add_step(
                _Step2B(nbr_of_kernel_elements, p, (k <= 10), backends=FAST_BACKENDS)
            )

            self.add_step(_Step3(nbr_of_kernel_elements, p, backends=FAST_BACKENDS))

            nbr_of_kernel_elements -= 1
            doublings.pop()

        # -------------------------------

        self.add_step(_LastStep(nbr_of_kernel_elements, p, backends=FAST_BACKENDS))

        _qc = IsogenyChainLastStep(p, backends=FAST_BACKENDS)
        self.add_last_step(PartialCircuit(_qc, norm_mod2, splitting_type, None, result))

        self.pebble_root()

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
        T1, T2, T3, T4, T1p2, T3p4 = (AffPointDim4(T) for T in Ts)
        # p = f * 2**(e + 3) - 1
        # the Ti are in the 2**(e+2)-torsion.
        # the length of the chain is e.

        p = self.p

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

        Taux = Taux * (2 ** (self.e - 2))  # 16-tors
        T5 = T5 * (2 ** (self.e - 1))  # pyright: ignore[reportConstantRedefinition] # 8-tors

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

            if k == 0:
                theta_str = ThetaStructureDim4(theta_np_codom)
            else:
                assert inv_dual_np_codom is not None  # for typing
                theta_str = ThetaStructureDim4.from_inv_dual_null_point(
                    inv_dual_np_codom
                )

            # Remove elements from list
            kernel_elements.pop()
            doublings.pop()

            # Push through points for the next step
            if k == 0:
                assert inv_image_Taux_dual is not None
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

        assert kernel_elements == []
        assert theta_str is not None

        A_codomain = splittingdim4_classical_function(
            p, (theta_str, splitting_type, QartonBool(1 - norm_mod2))
        )[-1]
        J_codomain = montgomerytojinvariantdiv256_classical_function(p, A_codomain)[-1]

        return *args, J_codomain
