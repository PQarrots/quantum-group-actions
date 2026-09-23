"""
Negation of affine points on Montgomery curves, adapted from Qarton's code for Weierstrass curves.

"""

from qarton.circuit import (
    BoolType,
    InvolutoryCircuit,
    PreCircuit,
    QartonBool,
    Register,
    dummify,
    memoize,
)
from qarton.dispatch import (
    define,
    qc_cneg,
    qc_cneg_var,
    qc_neg,
    qc_neg_var,
)
from qarton.modular_arithmetic import (
    qc_cneg_modint,
    qc_neg_modint,
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
    "AffMontgomeryNeg",
    "AffMontgomeryNegVar",
    "qc_cneg_affmontgomerypoint",
    "qc_neg_affmontgomerypoint",
    "ControlledAffMontgomeryNeg",
    "ControlledAffMontgomeryNegVar",
    "qc_neg_affmontgomerypoint_var",
    "qc_cneg_affmontgomerypoint_var",
]


@dummify
@memoize
class AffMontgomeryNeg(InvolutoryCircuit[AffMontgomeryPoint]):
    """
    Negation of the point: negate the y coordinate modulo p.
    The point at infinity (where x and y contain both the value 0) is unchanged.
    """

    def __init__(self, ec: ECMontgomery) -> None:
        super().__init__()
        self.ec = ec

        pp = self.add_input_output(AffMontgomeryPointType(ec))
        ppy = pp.a["y"]
        qc_neg_modint(ppy)

    def dummy_classical_function(self, args: AffMontgomeryPoint) -> AffMontgomeryPoint:
        return -args


@define(qc_neg, AffMontgomeryPointType)
def qc_neg_affmontgomerypoint(pp: Register) -> None:
    """Negate the point (in-place).

    :param pp: Point (AffMontgomeryPointType), preserved.
    :type pp: Register
    """
    ec = AffMontgomeryPointType.get_from_register(pp).ec
    pp.parent_circuit.append(PreCircuit(AffMontgomeryNeg, ec), pp)


@dummify
@memoize
class ControlledAffMontgomeryNeg(
    InvolutoryCircuit[tuple[QartonBool, AffMontgomeryPoint]]
):
    """
    Negation of the point: negate the y coordinate modulo p.
    The point at infinity (where x and y contain both the value 0) is unchanged.
    """

    def __init__(self, ec: ECMontgomery) -> None:
        super().__init__()
        self.ec = ec
        c = self.add_input_output(BoolType())
        pp = self.add_input_output(AffMontgomeryPointType(ec))
        ppy = pp.a["y"]
        qc_cneg_modint(c, ppy)

    def dummy_classical_function(
        self, args: tuple[QartonBool, AffMontgomeryPoint]
    ) -> tuple[QartonBool, AffMontgomeryPoint]:
        b, p = args
        return b, -p if b else p


@define(qc_cneg, BoolType, AffMontgomeryPointType)
def qc_cneg_affmontgomerypoint(c: Register, pp: Register) -> None:
    """Negate the point (in-place).

    :param c: control (BoolType), preserved
    :type c: Register
    :param pp: Point (AffMontgomeryPointType), preserved.
    :type pp: Register
    :raises TypeError: If invalid type.
    """
    ec = AffMontgomeryPointType.get_from_register(pp).ec
    pp.parent_circuit.append(PreCircuit(ControlledAffMontgomeryNeg, ec), c, pp)


@dummify
@memoize
class AffMontgomeryNegVar(InvolutoryCircuit[AffMontgomeryPointVariable]):
    """
    Negation of the point: negate the y coordinate modulo p.
    The point at infinity (where x and y contain both the value 0) is unchanged.
    """

    def __init__(self, q: int) -> None:
        super().__init__()
        self.q = q

        pp = self.add_input_output(AffMontgomeryPointVariableType(q))
        ppy = pp.a["y"]
        qc_neg_modint(ppy)  # operation indep of ECMontgomery

    def dummy_classical_function(
        self, args: AffMontgomeryPointVariable
    ) -> AffMontgomeryPointVariable:
        return -args


@define(qc_neg_var, AffMontgomeryPointVariableType, ECMontgomeryType)
def qc_neg_affmontgomerypoint_var(pp: Register, ec: Register) -> None:
    q = AffMontgomeryPointVariableType.get_from_register(pp).q
    pp.parent_circuit.append(PreCircuit(AffMontgomeryNegVar, q), pp)


@dummify
@memoize
class ControlledAffMontgomeryNegVar(
    InvolutoryCircuit[tuple[QartonBool, AffMontgomeryPointVariable]]
):
    def __init__(self, q: int) -> None:
        super().__init__()
        self.q = q
        c = self.add_input_output(BoolType())
        pp = self.add_input_output(AffMontgomeryPointVariableType(q))
        ppy = pp.a["y"]
        qc_cneg_modint(c, ppy)

    def dummy_classical_function(
        self, args: tuple[QartonBool, AffMontgomeryPointVariable]
    ) -> tuple[QartonBool, AffMontgomeryPointVariable]:
        b, p = args
        return b, -p if b else p


@define(qc_cneg_var, BoolType, AffMontgomeryPointVariableType, ECMontgomeryType)
def qc_cneg_affmontgomerypoint_var(c: Register, pp: Register, ec: Register) -> None:
    q = AffMontgomeryPointVariableType.get_from_register(pp).q
    pp.parent_circuit.append(PreCircuit(ControlledAffMontgomeryNegVar, q), c, pp)
