r"""
Circuits to compute and uncompute :math:`x \bmod{4}` when the input x is a signed integer.
"""

from qarton.circuit import (
    Circuit,
    Register,
    UInt,
    UIntType,
    dummify,
    memoize,
)
from qarton.signed_arithmetic import SInt, SIntType

__all__ = [
    "qc_mod4",
    "qc_mod4_erase",
    "qc_minusmod4",
    "qc_minusmod4_erase",
    "XMod4",
    "MinusXMod4",
]


@memoize
@dummify
class XMod4(Circuit[SInt, tuple[SInt, UInt]]):
    r"""Computes :math:`x \bmod{4}` (out of place) when the input x is a signed integers."""

    def __init__(self, w: int) -> None:
        """
        :param w: Width of input.
        :type w: int
        """
        super().__init__()
        # If x0 ... x(k-1) xs is a signed integer register
        # if xs = 0 (positive) then x % 4 = (x0 x1) and (-x) % 4 = (neg x0 neg x1) + 1
        # if xs = 1 (negative) then x % 4 = (neg x0 neg x1) + 1 and -x % 4 = x0 x1
        self.w = w
        xreg = self.add_input_output(SIntType(w))
        out = self.add_anc_output(UIntType(2))
        # out[0] == 1 iff xreg[0] == 1
        self.cx(xreg[0], out[0])
        # out[1] == 1 iff xreg[1] == 1
        self.cx(xreg[1], out[1])

    def dummy_classical_function(self, args: SInt) -> tuple[SInt, UInt]:
        return args, UInt(args % 4)

    def dummy_classical_function_inverse(self, args: tuple[SInt, UInt]) -> SInt:
        x, _ = args
        return x


def qc_mod4(x: Register) -> Register:
    r"""Computes :math:`x \bmod{4}` (out of place) when the input x is a signed integers.

    :param x: Input (SIntType), preserved.
    :type x: Register
    :return: x mod 4 (IntType), new register.
    :rtype: Register
    :raises TypeError: If register type does not match.
    """
    qc = x.parent_circuit
    if not isinstance(x.datatype, SIntType):
        raise TypeError()
    return qc.append(XMod4(x.datatype.width), x)[1]


def qc_mod4_erase(x: Register, out: Register) -> None:
    r"""Erases :math:`x \bmod{4}`.

    :param x: Input (SIntType), preserved.
    :type x: Register
    :param out: x mod 4 (IntType), destroyed.
    :type out: Register
    :raises TypeError: If register type does not match.
    """
    qc = x.parent_circuit
    if not isinstance(x.datatype, SIntType):
        raise TypeError()
    qc.append(XMod4(x.datatype.width).inverse(), x, out)


@memoize
@dummify
class MinusXMod4(Circuit[SInt, tuple[SInt, UInt]]):
    r"""Computes :math:`(-x) \bmod{4}` (out of place) when the input x is a signed integers."""

    def __init__(self, w: int) -> None:
        """
        :param w: Width of the input.
        :type w: int
        """
        super().__init__()
        self.w = w
        # If x0 ... x(k-1) xs is a signed integer register
        # if xs = 0 (positive) then x % 4 = (x0 x1) and (-x) % 4 = neg x0 neg x1
        # if xs = 1 (negative) then x % 4 = neg x0 neg x1 and -x % 4 = x0 x1

        xreg = self.add_input_output(SIntType(w))
        out = self.add_anc_output(UIntType(2))
        # out[0] == 1 iff xreg[0] == 1
        self.cx(xreg[0], out[0])
        # out[1] == 1 iff xreg[0] ^ xreg[1] == 1
        self.cx(xreg[0], out[1])
        self.cx(xreg[1], out[1])

    def dummy_classical_function(self, args: SInt) -> tuple[SInt, UInt]:
        return args, UInt((-args) % 4)

    def dummy_classical_function_inverse(self, args: tuple[SInt, UInt]) -> SInt:
        x, _ = args
        return x


def qc_minusmod4(x: Register) -> Register:
    r"""Computes :math:`(-x) \bmod{4}` (out of place) when the input x is a signed integers.

    Example::

        >>> from qarton.signed_arithmetic import SIntType, SInt
        >>> qc = Circuit()
        >>> r = qc.add_input_output(SIntType(5))
        >>> o = qc_minusmod4(r)
        >>> qc.add_output(o)
        >>> qc.simulate(SInt(7, 5))
        (SInt(7, 5), 1)

    :param x: Input (SIntType), preserved.
    :type x: Register
    :return: (-x) mod 4 (IntType), new register.
    :rtype: Register
    :raises TypeError: If register type does not match.
    """
    qc = x.parent_circuit
    if not isinstance(x.datatype, SIntType):
        raise TypeError()
    return qc.append(MinusXMod4(x.datatype.width), x)[1]


def qc_minusmod4_erase(x: Register, out: Register) -> None:
    r"""Erases :math:`(-x) \bmod{4}`.

    :param x: Input (SIntType), preserved.
    :type x: Register
    :param out: (-x) mod 4 (IntType), destroyed.
    :type out: Register
    :raises TypeError: If register type does not match.
    """
    qc = x.parent_circuit
    if not isinstance(x.datatype, SIntType):
        raise TypeError()
    qc.append(MinusXMod4(x.datatype.width).inverse(), x, out)
