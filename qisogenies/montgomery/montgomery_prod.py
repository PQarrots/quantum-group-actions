"""
Product of points on Montgomery curves, adapted from Qarton's code for Weierstrass curves.

"""

from qarton.advanced_arithmetic import (
    fibodec_indices_length,
    qc_fibodec,
    qc_fibodec_uncompute,
)
from qarton.binary_operations import (
    qc_shift_right,
)
from qarton.circuit import (
    BackendSpecifier,
    EmptyBackendSpecifier,
    InPlaceCircuit,
    PreCircuit,
    Register,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_bool_reg,
)
from qarton.dispatch import (
    define,
    qc_powadd,
    qc_powadd_var,
    qc_powsub,
    qc_powsub_var,
)

from .montgomery_add import (
    qc_add_affmontgomerypoint,
    qc_add_affmontgomerypoint_var,
    qc_cadd_affmontgomerypoint,
    qc_cadd_affmontgomerypoint_var,
)
from .montgomery_curve import (
    AffMontgomeryPoint,
    AffMontgomeryPointType,
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariableType,
    ECMontgomery,
    ECMontgomeryType,
)

__all__ = [
    "AffMontgomeryProd",
    "qc_powadd_affmontgomerypoint",
    "qc_powsub_affmontgomerypoint",
    "AffMontgomeryProdVar",
    "qc_powadd_affmontgomerypoint_var",
    "qc_powsub_affmontgomerypoint_var",
]


@memoize
class _AffProdIterate(InPlaceCircuit):
    def __init__(
        self,
        ec: ECMontgomery,
        n: int,
        write_in_output: bool = True,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()

        zsize = fibodec_indices_length(n)

        ccomp = self.add_input_output(AffMontgomeryPointType(ec))
        dcomp = self.add_input_output(AffMontgomeryPointType(ec))
        ecomp = self.add_input_output(AffMontgomeryPointType(ec))

        expdec = self.add_input_output(zsize)

        qc_add_affmontgomerypoint(ccomp, dcomp)
        self.swap_reg(ccomp, dcomp)

        if write_in_output:
            # controlled IP addition if xdec[i] is 1, but only if we're currently writing
            # the output
            qc_cadd_affmontgomerypoint(qc_bool_reg(expdec[0]), ccomp, ecomp)
        qc_shift_right(expdec, -1)


@dummify
@memoize
class AffMontgomeryProd(
    InPlaceCircuit[tuple[UInt, AffMontgomeryPoint, AffMontgomeryPoint]]
):
    """
    Input: k, x, y
    Sums result in y.

    If exponent is 0, we have nothing to sum in the output.
    """

    def __init__(
        self,
        ec: ECMontgomery,
        n: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()

        kreg = self.add_input_output(UIntType(n))
        pp = self.add_input_output(AffMontgomeryPointType(ec))  # point that we multiply
        qq = self.add_input_output(
            AffMontgomeryPointType(ec)
        )  # register where we add the result

        # step 1: compute decomposition of exponent
        expdec = qc_fibodec(kreg)

        # next, initialize:
        # - register c to O
        # - register d to P
        # - register e (output) to O

        dcomp = pp  # will need to be restored
        ccomp = self.add_anc(AffMontgomeryPointType(ec))
        ecomp = qq  # self.add_anc(AffMontgomeryPointType(ec))

        self.append_iterated(
            PreCircuit(_AffProdIterate, ec, n),
            ccomp,
            dcomp,
            ecomp,
            expdec,
            iterations=len(expdec),
        )

        self.append_iterated(
            PreCircuit(_AffProdIterate, ec, n, False, inverse=True),
            ccomp,
            dcomp,
            ecomp,
            expdec,
            iterations=len(expdec),
        )

        self.assert_anc(ccomp)
        kreg = qc_fibodec_uncompute(expdec)  # kreg should be restored hopefully
        self.remap(kreg, pp, qq)

    def dummy_classical_function(
        self, args: tuple[UInt, AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[UInt, AffMontgomeryPoint, AffMontgomeryPoint]:
        k, pp, qq = args
        return k, pp, qq + (pp * k)

    def dummy_classical_function_inverse(
        self, args: tuple[UInt, AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[UInt, AffMontgomeryPoint, AffMontgomeryPoint]:
        k, pp, qq = args
        return k, pp, qq - (pp * k)


@define(qc_powadd, UIntType, AffMontgomeryPointType, AffMontgomeryPointType)
def qc_powadd_affmontgomerypoint(e: Register, x: Register, y: Register) -> None:
    ec = AffMontgomeryPointType.get_from_register(x).ec
    e.parent_circuit.append(PreCircuit(AffMontgomeryProd, ec, len(e)), e, x, y)


@define(qc_powsub, UIntType, AffMontgomeryPointType, AffMontgomeryPointType)
def qc_powsub_affmontgomerypoint(e: Register, x: Register, y: Register) -> None:
    ec = AffMontgomeryPointType.get_from_register(x).ec
    e.parent_circuit.append(
        PreCircuit(AffMontgomeryProd, ec, len(e), inverse=True), e, x, y
    )


@memoize
class _AffProdIterateVar(InPlaceCircuit):
    def __init__(
        self,
        q: int,
        n: int,
        write_in_output: bool = True,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()

        zsize = fibodec_indices_length(n)

        ccomp = self.add_input_output(AffMontgomeryPointVariableType(q))
        dcomp = self.add_input_output(AffMontgomeryPointVariableType(q))
        ecomp = self.add_input_output(AffMontgomeryPointVariableType(q))

        expdec = self.add_input_output(zsize)
        ec = self.add_input_output(ECMontgomeryType(q))

        qc_add_affmontgomerypoint_var(ccomp, dcomp, ec)
        self.swap_reg(ccomp, dcomp)

        if write_in_output:
            # controlled IP addition if xdec[i] is 1, but only if we're currently writing
            # the output
            qc_cadd_affmontgomerypoint_var(qc_bool_reg(expdec[0]), ccomp, ecomp, ec)
        qc_shift_right(expdec, -1)


@dummify
@memoize
class AffMontgomeryProdVar(
    InPlaceCircuit[
        tuple[
            UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ]
    ]
):
    """
    Input: k, x, y
    Sums result in y.

    If exponent is 0, we have nothing to sum in the output.
    """

    def __init__(
        self,
        q: int,
        n: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()

        kreg = self.add_input_output(UIntType(n))
        pp = self.add_input_output(
            AffMontgomeryPointVariableType(q)
        )  # point that we multiply
        qq = self.add_input_output(
            AffMontgomeryPointVariableType(q)
        )  # register where we add the result
        ec = self.add_input_output(ECMontgomeryType(q))

        # step 1: compute decomposition of exponent
        # copy_expreg = qc_copy(kreg)
        expdec = qc_fibodec(kreg)

        # next, initialize:
        # - register c to O
        # - register d to P
        # - register e (output) to O

        dcomp = pp  # will need to be restored
        ccomp = self.add_anc(AffMontgomeryPointVariableType(q))
        ecomp = qq  # self.add_anc(AffMontgomeryPointType(ec))

        self.append_iterated(
            PreCircuit(_AffProdIterateVar, q, n),
            ccomp,
            dcomp,
            ecomp,
            expdec,
            ec,
            iterations=len(expdec),
        )

        self.append_iterated(
            PreCircuit(_AffProdIterateVar, q, n, False, inverse=True),
            ccomp,
            dcomp,
            ecomp,
            expdec,
            ec,
            iterations=len(expdec),
        )

        self.assert_anc(ccomp)
        kreg = qc_fibodec_uncompute(expdec)  # kreg should be restored hopefully
        self.remap(kreg, pp, qq, ec)

    def validate_input(
        self,
        args: tuple[
            UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> bool:
        """
        Input is valid if the two points are indeed on the curve.
        """
        _, pp, qq, ec = args
        return pp.is_on_curve(ec) and qq.is_on_curve(ec)

    def validate_output(
        self,
        args: tuple[
            UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> bool:
        return self.validate_input(args)

    def dummy_classical_function(
        self,
        args: tuple[
            UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> tuple[
        UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
    ]:
        k, pp, qq, ec = args
        _pp = pp.to_AffMontgomeryPoint(ec)
        _qq = qq.to_AffMontgomeryPoint(ec)
        return (
            k,
            pp,
            AffMontgomeryPointVariable.from_AffMontgomeryPoint(_qq + (_pp * k)),
            ec,
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> tuple[
        UInt, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
    ]:
        k, pp, qq, ec = args
        _pp = pp.to_AffMontgomeryPoint(ec)
        _qq = qq.to_AffMontgomeryPoint(ec)
        return (
            k,
            pp,
            AffMontgomeryPointVariable.from_AffMontgomeryPoint(_qq - (_pp * k)),
            ec,
        )


@define(
    qc_powadd_var,
    UIntType,
    AffMontgomeryPointVariableType,
    AffMontgomeryPointVariableType,
    ECMontgomeryType,
)
def qc_powadd_affmontgomerypoint_var(
    e: Register, x: Register, y: Register, ec: Register
) -> None:
    q = ECMontgomeryType.get_from_register(ec).q
    e.parent_circuit.append(PreCircuit(AffMontgomeryProdVar, q, len(e)), e, x, y, ec)


@define(
    qc_powsub_var,
    UIntType,
    AffMontgomeryPointVariableType,
    AffMontgomeryPointVariableType,
    ECMontgomeryType,
)
def qc_powsub_affmontgomerypoint_var(
    e: Register, x: Register, y: Register, ec: Register
) -> None:
    q = ECMontgomeryType.get_from_register(ec).q
    e.parent_circuit.append(
        PreCircuit(AffMontgomeryProdVar, q, len(e), inverse=True), e, x, y, ec
    )
