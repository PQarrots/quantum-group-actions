"""
Circuits to process the output of the norm equation.
"""

from qarton.binary_operations import (
    qc_copy,
    qc_copy_erase,
    qc_cxor,
    qc_load,
    qc_unload,
    qc_xor,
)
from qarton.circuit import (
    BackendSpecifier,
    Circuit,
    DataType,
    ITupleType,
    PreCircuit,
    TupleType,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_bool_reg,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_muladd_modint,
    qc_mulsub_modint,
)

from qisogenies.arithmetic.mod_inverse_pow2 import IPModInvPow2
from qisogenies.montgomery import (
    AffMontgomeryPoint as AffPoint,
)
from qisogenies.montgomery import (
    AffMontgomeryPointType as AffPointType,
)
from qisogenies.montgomery import (
    qc_add_affmontgomerypoint as qc_add_affpoint,
)
from qisogenies.montgomery import (
    qc_powadd_affmontgomerypoint as qc_powadd_affpoint,
)
from qisogenies.montgomery import (
    qc_powsub_affmontgomerypoint as qc_powsub_affpoint,
)
from qisogenies.norm_equation.util import FAST_BACKENDS
from qisogenies.theta.affpoint_dim4 import AffPointDim4, AffPointDim4Type

__all__ = [
    "NormeqOutputToHdKernel",
    "normeqoutputtorabbittype_classical_function",
    "NormeqOutputToRabbitType",
]


@memoize
@dummify
class NormeqOutputToHdKernel(
    Circuit[
        tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
        tuple[
            tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
        ],
    ]
):
    """
    Postcomputation step 2/2, after we run the norm equation algorithm.

    The input is the output of PostComp.
    All inputs are preserved.

    The output is the kernel of the 4D isogeny we want to compute
    (a 4D isogeny embedding the 1D isogeny corresponding to the ideal action we're evaluating)
    in the form of a 4-tuple (T1, ..., T4),
    where each point Ti is a 4-tuple of elliptic curve points.
    More precisely, T1, T1 lie on E^4 and T3, T4 on the twist (E^T)^4)
    """

    def __init__(
        self,
        e: int,
        P_input: AffPoint,
        Q_input: AffPoint,
        TP_input: AffPoint,
        TQ_input: AffPoint,
        backends: BackendSpecifier = FAST_BACKENDS,
    ) -> None:
        """
        :param d: Parameters.
        :type d: InstanceData
        """
        super().__init__()
        self.e = e
        self.tors_basis_with_two_tors_multiples = P_input, Q_input, TP_input, TQ_input
        E = P_input.ec
        E_twist = Q_input.ec

        minus2eQ_input = -Q_input * (1 << e)

        algo_input = self.add_input_output(TupleType(5, ModIntType(1 << (e + 2))))
        N1, sigma1, sigma2, sigma3, sigma4 = (algo_input.a[i] for i in range(5))

        PointOnCurve4: DataType = AffPointDim4Type(E)
        PointOnTwist4: DataType = AffPointDim4Type(E_twist)

        algo_output = self.add_anc_output(
            ITupleType(
                PointOnCurve4,  # T1
                PointOnCurve4,  # T2
                PointOnTwist4,  # T3
                PointOnTwist4,  # T4
                PointOnCurve4,  # T1+2
                PointOnTwist4,  # T3+4
            )
        )

        PointOnCurve4 = TupleType(4, AffPointType(E))
        PointOnTwist4 = TupleType(4, AffPointType(E_twist))

        T1, T2, T3, T4, T1p2, T3p4 = (algo_output.a[i] for i in range(6))
        # we fix the typing
        T1 = qc_reg_cast(T1, PointOnCurve4)  # pyright: ignore[reportConstantRedefinition]
        T2 = qc_reg_cast(T2, PointOnCurve4)  # pyright: ignore[reportConstantRedefinition]
        T3 = qc_reg_cast(T3, PointOnTwist4)  # pyright: ignore[reportConstantRedefinition]
        T4 = qc_reg_cast(T4, PointOnTwist4)  # pyright: ignore[reportConstantRedefinition]
        T1p2 = qc_reg_cast(T1p2, PointOnCurve4)
        T3p4 = qc_reg_cast(T3p4, PointOnTwist4)

        as1, as2, as3, as4, as5, as6 = self.add_ancs(
            ModIntType(1 << (e + 2)),
            ModIntType(1 << (e + 2)),
            ModIntType(1 << (e + 2)),
            ModIntType(1 << (e + 2)),
            1,
            1,
        )

        # alpha = N1**(-1) mod 2**(e + 2)
        alpha = qc_copy(N1)
        self.append(PreCircuit(IPModInvPow2, len(alpha)), alpha)

        # sigma5 = sigma1 + sigma2 % 2
        sigma5 = qc_copy(qc_bool_reg(sigma1[0]))
        qc_xor(qc_reg_from_bits(sigma2[0]), sigma5)
        # sigma6 = sigma3 + sigma4 % 2
        sigma6 = qc_copy(qc_bool_reg(sigma3[0]))
        qc_xor(qc_reg_from_bits(sigma4[0]), sigma6)

        for sigma_i, alphasigma_i in zip(
            (sigma1, sigma2, sigma3, sigma4), (as1, as2, as3, as4), strict=True
        ):
            qc_muladd_modint(alpha, sigma_i, alphasigma_i)
        self.ccx(
            alpha[0], sigma5[0], as5[0]
        )  # for sigma5 and as5, the subscript [0] is just a shortcut to transform sigma5 into bit...
        self.ccx(alpha[0], sigma6[0], as6[0])

        P, TP = (qc_load(R, self, AffPointType(E)) for R in (P_input, TP_input))
        Q, TQ = (qc_load(R, self, AffPointType(E_twist)) for R in (Q_input, TQ_input))
        minus2eQ = qc_load(minus2eQ_input, self, AffPointType(E_twist))

        # T1 = (N1 P, 0, s3 P + s6 TP, -s1 P + s5 TP)
        # T2 = (0, N1 P, s2 P + s5 TP,  s4 P - s6 TP )
        qc_powadd_affpoint(N1, P, T1.a[0])
        qc_xor(T1.a[0], T2.a[1])

        qc_cxor(sigma6, TP, T1.a[2])
        qc_cxor(sigma6, TP, T2.a[3])
        qc_cxor(sigma5, TP, T1.a[3])
        qc_cxor(sigma5, TP, T2.a[2])

        qc_powadd_affpoint(sigma3, P, T1.a[2])
        qc_powadd_affpoint(sigma2, P, T2.a[2])
        qc_powadd_affpoint(sigma4, P, T2.a[3])

        qc_powsub_affpoint(sigma1, P, T1.a[3])

        # T3 = (1 - alpha 2^e Q, 0, as4 Q + as6 TQ, -as2 Q + as5 TQ)
        # T4 = (0, 1 - alpha 2^e Q, as1 Q + as5 TQ,  as3 Q - as6 TQ)
        qc_xor(Q, T3.a[0])
        qc_powadd_affpoint(
            qc_reg_cast(qc_reg_from_bits(alpha[:2]), UIntType(2)), minus2eQ, T3.a[0]
        )
        qc_xor(T3.a[0], T4.a[1])

        qc_cxor(as5, TQ, T3.a[3])
        qc_cxor(as5, TQ, T4.a[2])  # TODO could be a xor from the value computed above
        qc_cxor(as6, TQ, T3.a[2])
        qc_cxor(as6, TQ, T4.a[3])  # TODO could be a xor from the value computed above

        qc_powadd_affpoint(as4, Q, T3.a[2])
        qc_powadd_affpoint(as1, Q, T4.a[2])
        qc_powadd_affpoint(as3, Q, T4.a[3])

        qc_powsub_affpoint(as2, Q, T3.a[3])

        qc_xor(T1, T1p2)
        qc_xor(T3, T3p4)
        for i in range(4):
            qc_add_affpoint(T2.a[i], T1p2.a[i])
            qc_add_affpoint(T4.a[i], T3p4.a[i])

        ############# Uncompute ############
        for pt, reg in zip(
            (P_input, TP_input, Q_input, TQ_input, minus2eQ_input),
            (P, TP, Q, TQ, minus2eQ),
            strict=True,
        ):
            qc_unload(pt, reg)

        self.ccx(alpha[0], sigma5[0], as5[0])
        self.ccx(alpha[0], sigma6[0], as6[0])

        for sigma_i, alphasigma_i in zip(
            (sigma1, sigma2, sigma3, sigma4), (as1, as2, as3, as4), strict=True
        ):
            qc_mulsub_modint(alpha, sigma_i, alphasigma_i)

        self.append(PreCircuit(IPModInvPow2, len(alpha)), alpha)
        # qc_inv_modint(alpha)
        qc_copy_erase(N1, alpha)

        qc_xor(qc_reg_from_bits(sigma4[0]), sigma6)
        qc_copy_erase(qc_bool_reg(sigma3[0]), sigma6)
        qc_xor(qc_reg_from_bits(sigma2[0]), sigma5)
        qc_copy_erase(qc_bool_reg(sigma1[0]), sigma5)

        for anc in (as1, as2, as3, as4, as5, as6):
            self.assert_anc(anc)
            self.release_anc(anc)

    def validate_input(
        self, args: tuple[ModInt, ModInt, ModInt, ModInt, ModInt]
    ) -> bool:
        """
        Conditions for the input to be valid. Currently not implemented
        (always returns True).
        """
        return True

    def dummy_classical_function(
        self,
        args: tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
    ) -> tuple[
        tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
        tuple[
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
            AffPointDim4,
        ],
    ]:
        e = self.e
        P, Q, TP, TQ = self.tors_basis_with_two_tors_multiples

        # N1, sigma1, sigma2, sigma3, sigma4 = args
        # sigma5 = (sigma1 - sigma2) % 2
        # sigma6 = (sigma4 - sigma3) % 2

        # alpha = ModInt(N1, 2**(d.e + 2))**(-1)
        # beta  = ModInt(N2, 2**(d.e + 2))**(-1)

        N1, s1, s2, s3, s4 = args
        alpha = N1 ** (-1)
        s5 = (s1.v + s2.v) % 2
        s6 = (s3.v + s4.v) % 2

        as1, as2, as3, as4, as5, as6 = (alpha * s for s in (s1, s2, s3, s4, s5, s6))
        qcoeff = -alpha * (1 << e) + ModInt(1, alpha.p)

        N1, s1, s2, s3, s4, alpha, as1, as2, as3, as4, as5, as6, qcoeff = (  # type: ignore
            n.v
            for n in (N1, s1, s2, s3, s4, alpha, as1, as2, as3, as4, as5, as6, qcoeff)
        )

        T1 = AffPointDim4(P * N1, P * 0, P * s3 + TP * s6, -P * s1 + TP * s5)
        T2 = AffPointDim4(P * 0, P * N1, P * s2 + TP * s5, P * s4 - TP * s6)
        T3 = AffPointDim4(Q * qcoeff, Q * 0, Q * as4 + TQ * as6, -Q * as2 + TQ * as5)
        T4 = AffPointDim4(Q * 0, Q * qcoeff, Q * as1 + TQ * as5, Q * as3 - TQ * as6)

        T1p2 = AffPointDim4(x + y for x, y in zip(T1, T2, strict=True))
        T3p4 = AffPointDim4(x + y for x, y in zip(T3, T4, strict=True))

        return (args, (T1, T2, T3, T4, T1p2, T3p4))

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
            tuple[
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
                AffPointDim4,
            ],
        ],
    ) -> tuple[ModInt, ModInt, ModInt, ModInt, ModInt]:
        inp, _ = args
        return inp


##############################


def normeqoutputtorabbittype_classical_function(
    args: tuple[UInt, UInt, UInt, UInt],
) -> tuple[tuple[UInt, UInt, UInt, UInt], UInt]:
    sigma1, sigma2, sigma3, sigma4 = (s % 4 for s in args)

    if ((sigma1 == 0) and ((sigma2 * sigma4) % 4 == 1)) or (
        (sigma3 == 0) and ((sigma2 * sigma4) % 4 == 3)
    ):
        rabbit_type = UInt(0)  # 14
    elif ((sigma1 == 0) and ((sigma2 * sigma4) % 4 == 3)) or (
        (sigma3 == 0) and ((sigma2 * sigma4) % 4 == 1)
    ):
        rabbit_type = UInt(3)  # 7
    elif ((sigma2 == 0) and ((sigma1 * sigma3) % 4 == 1)) or (
        (sigma4 == 0) and ((sigma1 * sigma3) % 4 == 3)
    ):
        rabbit_type = UInt(2)  # 11
    elif ((sigma2 == 0) and ((sigma1 * sigma3) % 4 == 3)) or (
        (sigma4 == 0) and ((sigma1 * sigma3) % 4 == 1)
    ):
        rabbit_type = UInt(1)  # 13
    else:
        raise Exception("Impossible")

    return args, rabbit_type


@memoize
@dummify
class NormeqOutputToRabbitType(
    Circuit[tuple[UInt, UInt, UInt, UInt], tuple[tuple[UInt, UInt, UInt, UInt], UInt]]
):
    """
    The input is the output of PostComp.
    All inputs are preserved.

    The output is the rabbit type, see [rabbits paper] eq. (16).
    NOTE rabbit_type = i <--> the rabbit type in the rabbits paper is 15 - 2^i.
    i.e. 0 <--> 14, 1 <--> 13, 2 <--> 11, 3 <--> 7
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        sigmas = self.add_input_output(TupleType(4, UIntType(2)))
        sigma1, sigma2, sigma3, sigma4 = (sigmas.a[i] for i in range(4))

        rabbit_type = self.add_anc_output(UIntType(2))

        # is sigma_i nonzero?
        s1_nz, s2_nz, s3_nz, s4_nz = (si[0] for si in (sigma1, sigma2, sigma3, sigma4))

        # if sigma1 == 0: s2 * s4 == 1 --> rabbit type 0. s2 * s4 == 3 --> rabbit type 3
        # if sigma1 == 0, then s2 and s4 are supposed to be odd. that means, they're either 1 or 3 (mod 4).
        # We can check bit#1 of s2 and s4: if they're equal (xor to 0) then the product is 1, otherwise it's 3
        self.cx(sigma2[1], sigma4[1])
        s24_is_3 = sigma4[1]
        self.ccx(s1_nz, s24_is_3, rabbit_type[0])
        self.ccx(s1_nz, s24_is_3, rabbit_type[1])

        # if sigma3 == 0: s2 * s4 == 3 --> rabbit type 0. s2 * s4 == 1 --> rabbit type 3
        self.x(s24_is_3)  # opposite behaviour
        self.ccx(s3_nz, s24_is_3, rabbit_type[0])
        self.ccx(s3_nz, s24_is_3, rabbit_type[1])
        self.x(s24_is_3)  # change value back
        self.cx(sigma2[1], sigma4[1])  # restore value

        # if sigma2 == 0: s1 * s3 == 1 --> rabbit type 2. s1 * s3 == 3 --> rabbit type 1
        self.cx(sigma1[1], sigma3[1])
        s13_is_3 = sigma3[1]
        self.ccx(s2_nz, s13_is_3, rabbit_type[0])
        self.x(s13_is_3)
        self.ccx(s2_nz, s13_is_3, rabbit_type[1])

        # if sigma4 == 0: s1 * s3 == 3 --> rabbit type 2. s1 * s3 == 1 --> rabbit type 1
        self.ccx(s4_nz, s13_is_3, rabbit_type[0])
        self.x(s13_is_3)
        self.ccx(s4_nz, s13_is_3, rabbit_type[1])
        self.cx(sigma1[1], sigma3[1])  # restore value

    def validate_input(self, args: tuple[UInt, UInt, UInt, UInt]) -> bool:
        """
        Conditions for the input to be valid.
        Similar to NormeqOutputToHdKernel.
        """
        count_zeros = [s == 0 for s in args].count(True)
        if count_zeros != 1:
            return False
        zero_idx = next(i for i in range(4) if args[i] == 0)
        if not {args[(zero_idx + 1) % 4], args[(zero_idx + 3) % 4]}.issubset({1, 3}):  # noqa: SIM103
            return False
        # TODO any conditions on the element with index (zero_idx + 2) mod 4?
        return True

    def dummy_classical_function(
        self, args: tuple[UInt, UInt, UInt, UInt]
    ) -> tuple[tuple[UInt, UInt, UInt, UInt], UInt]:
        return normeqoutputtorabbittype_classical_function(args)

    def dummy_classical_function_inverse(
        self, args: tuple[tuple[UInt, UInt, UInt, UInt], UInt]
    ) -> tuple[UInt, UInt, UInt, UInt]:
        inp, _ = args
        return inp
