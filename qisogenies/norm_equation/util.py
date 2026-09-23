"""
Some functions for nice printing of tables with circuit counts (taken from another project).
"""

from dataclasses import dataclass
from math import ceil, log2

import tabulate
from qarton.advanced_arithmetic import DummyFiboDec
from qarton.arithmetic import (
    FastConstantMulAdd,
    FastControlledConstantMulAdd,
    FastControlledMulAdd,
    FastControlledSquareAdd,
    FastMulAdd,
    FastSquareAdd,
    qc_div_uint,
    qc_div_uncompute_uint,
)
from qarton.binary_operations import qc_or
from qarton.circuit import (
    AndFanoutResourceReporter,
    BoolType,
    Circuit,
    EfficientlySimulatedHadamardMeasure,
    EfficientlySimulatedPhaseCorrection,
    PriorityBackends,
    Register,
    UIntType,
)
from qarton.signed_arithmetic import SIntType, qc_abs, qc_abs_uncompute

from qisogenies.arithmetic.efficient_mod_arithmetic import (
    SpecialControlledModularAdder,
    SpecialModularDouble,
)

__all__ = [
    "InstanceData",
    "IterationsData",
    "FAST_BACKENDS",
    "circuit_lines",
    "circuit_table",
    "qc_make_signed",
    "qc_make_unsigned",
    "qc_div_signed",
    "qc_div_signed_erase",
    "qc_not2",
]


def _round(i: int | float, prec: int | None = None) -> str:
    return "{:.{}f}".format(i, prec) if prec is not None else str(ceil(i))


def _format_latex(i: int | None) -> str:
    return f"{round(i, None) if i is not None else ' '}"


def _format_str(i: int | None) -> str:
    return f"{round(i, None) if i is not None else ' '}"


def _format_latex_pow(i: float | None) -> str:
    return f"""$2^{_round(i, 2) if i is not None else " "}$"""


def _format_str_pow(i: float | None) -> str:
    return f"""2^{_round(i, 2) if i is not None else " "}"""


def _log2(t: float | int) -> float | None:
    if t <= 0:
        return None
    else:
        return log2(t)


def circuit_lines(
    qc: Circuit, name: str, as_latex: bool = True, with_cx: bool = False
) -> tuple[list[str | int | None], list[str | int | None]]:
    # headers = ["Qubits", "CCX + And", "CX + CZ", "Fanout", "CX + CZ + Fanout", "CCX depth"]

    _r = AndFanoutResourceReporter(qc)
    tmp: list[int]

    if with_cx:
        tmp = [
            _r.ccx_count(),
            _r.cx_count(),
            _r.fanout_count(),
            _r.cx_count() + _r.fanout_count(),
            _r.ccx_depth(),
            _r.measure_depth(),
        ]
    else:
        tmp = [
            _r.ccx_count(),
            _r.ccx_depth(),
            _r.measure_depth(),
        ]

    line1: list[str | int | None] = []
    line2: list[str | int | None] = []
    if as_latex:
        line1 += [name, qc.nbr_qubits()]
        line1 += [_format_latex(t) for t in tmp]
        line2 += [None, None]
        line2 += [_format_latex_pow(_log2(t)) for t in tmp]
    else:
        line1 += [name, qc.nbr_qubits()]
        line1 += [_format_str(t) for t in tmp]
        line2 += [None, None]
        line2 += [_format_str_pow(_log2(t)) for t in tmp]

    return (line1, line2)


def circuit_table(
    *dd: dict[str, Circuit], as_latex: bool = True, with_cx: bool = False
) -> None:
    if with_cx:
        headers = [
            "Qubits",
            "CCX + And",
            "CX + CZ",
            "Fanout",
            "CX + CZ + Fanout",
            "CCX depth",
            "Measurement depth",
        ]
    else:
        headers = [
            "Qubits",
            "CCX + And",
            "CCX depth",
            "Measurement depth",
        ]
    data: list[list[int | str | None]] = []
    for d in dd:
        for n in d:
            data += list(circuit_lines(d[n], n, as_latex, with_cx))

    print(
        tabulate.tabulate(
            data,
            headers=headers,
            tablefmt=("latex_raw" if as_latex else "fancy_grid"),
            disable_numparse=True,
        )
    )


@dataclass(frozen=True)
class InstanceData:
    """
    Data class which determines all the parameters needed to construct the circuits:
    * w: base size of the numbers (sizes of registers are taken in fractions of w)
    * x: size of the search register
    * e: exponent from qt-Pegasis
    * p: prime from qt-Pegasis
    """

    w: int
    x: int
    e: int
    p: int
    nb_primes: int


@dataclass(frozen=True)
class IterationsData:
    """
    Data class which contains the number of iterations to make at each level in
    the nested quantum search.
    """

    level1: int
    level2: int


FAST_BACKENDS = PriorityBackends(
    FastControlledMulAdd,
    FastControlledSquareAdd,
    FastMulAdd,
    FastSquareAdd,
    FastControlledConstantMulAdd,
    FastConstantMulAdd,
    SpecialModularDouble,
    SpecialControlledModularAdder,
    DummyFiboDec,
    EfficientlySimulatedHadamardMeasure,
    EfficientlySimulatedPhaseCorrection,
)
"""
Special set of backends used in our circuits:

* We use "fast" constructions for modular operations modulo a power of 2, they are 
  sub-optimal both in space and gate count, but faster to construct (and non-dominating
  anyway in our results).

* We use the "dummy" version of the FiboDec, which has a rather slow construction
  otherwise for large values.

* We use "special" constructions for modular double and controlled adder modulo
  a prime of the form 2^e f - 1, which is the type of prime used in qt-Pegasis.

* We use efficiently-simulated circuits for Hadamard-measure and phase-correction, which
  are used in the "pebbled" isogeny chain circuit.  

"""


def qc_make_signed(x: Register) -> Register:
    """
    Convert a UInt into a SInt, in a circuit. The initial register is destroyed.
    """
    assert isinstance(x.datatype, UIntType)
    qc = x.parent_circuit
    s = qc.get_anc(1)
    return qc_abs_uncompute(x, s)


def qc_make_unsigned(x: Register) -> Register:
    """
    Convert a SInt, of which we are 100% sure that it's positive, into a UInt.
    The initial register is destroyed.
    """
    assert isinstance(x.datatype, SIntType)
    qc = x.parent_circuit
    xx, s = qc_abs(x)
    qc.release_anc(s)
    return xx


def qc_div_signed(x: Register, y: Register) -> tuple[Register, Register, Register]:
    """
    Function to perform an Euclidean division when the two registers are signed.

    Does not take into account the sign of y. Simply flips quo with the sign
    of x. Rem remains unsigned. Also returns y.
    """
    if not isinstance(x.datatype, SIntType):
        raise TypeError()
    if not isinstance(y.datatype, SIntType):
        raise TypeError()
    xabs, xsgn = qc_abs(x)
    yabs, ysgn = qc_abs(y)
    quo, rem = qc_div_uint(yabs, xabs)
    # will restore the register
    return (
        qc_abs_uncompute(quo, xsgn),
        rem,
        qc_abs_uncompute(yabs, ysgn),
    )


def qc_div_signed_erase(
    quo: Register, rem: Register, y: Register
) -> tuple[Register, Register]:
    """Erases the result of ``qc_div_signed``."""
    if not isinstance(quo.datatype, SIntType):
        raise TypeError()
    if not isinstance(y.datatype, SIntType):
        raise TypeError()
    # xabs, xsgn = qc_abs(x)
    yabs, ysgn = qc_abs(y)
    quo, sgn = qc_abs(quo)
    x = qc_div_uncompute_uint(yabs, quo, rem)
    return qc_abs_uncompute(x, sgn), qc_abs_uncompute(yabs, ysgn)


def qc_not2(x: Register) -> Register:
    """Returns a Boolean register that indicates if the input (IntType) is not equal
    to 2 modulo 4.

    :param x: Input integer (IntType).
    :type x: Register
    :return: Boolean output that contains 1 iff the input is not equal to 2 modulo 4.
    :rtype: Register
    """
    qc = x.parent_circuit
    cond = qc.get_anc(BoolType())
    qc.x(x[1])
    qc_or(x[0], x[1], cond)
    qc.x(x[1])
    return cond
