r"""
Non-exact modular arithmetic circuits:

* working for a generic prime, and:
* working specifically for a prime of the form :math:`f 2^e - 1` as used in qt-Pegasis.

References
----------

.. [Sch26] ``Optimized Point Addition Circuits for Elliptic Curve Discrete Logarithms'',
   André Schrottenloher, https://eprint.iacr.org/2026/1128,
   https://gitlab.inria.fr/capsule/qarton-projects/ec-point-addition

"""

from __future__ import annotations

from qarton.arithmetic import (
    qc_cadd_uint,
    qc_cincr_uint,
    qc_clt_uint,
    qc_csub_uint,
    qc_lt_uint,
)
from qarton.binary_operations import qc_cxor, qc_mcx, qc_mcx_neg, qc_shift_right
from qarton.circuit import (
    BackendSpecifier,
    BoolType,
    CircuitParameterInvalid,
    EmptyBackendSpecifier,
    InPlaceCircuit,
    OnCallOutputType,
    PreCircuit,
    QartonBool,
    Register,
    UInt,
    UIntType,
    define_on_call,
    memoize,
    qc_reg_cast,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_cadd_modint,
    qc_csub_modint,
    qc_dbl_modint,
    qc_halve_modint,
)

__all__ = [
    "EfficientControlledModularAdder",
    "EfficientModularDouble",
    "SpecialControlledModularAdder",
    "SpecialModularDouble",
]

PADDING = 80
"""
Padding for non-exact operations. Probability of failure will be around
2^(-PADDING) for each operation, so this leaves a comfortable margin.
"""


@memoize
class EfficientControlledModularAdder(
    InPlaceCircuit[tuple[QartonBool, ModInt, ModInt]]
):
    """
    Non-exact controlled modular adder, for any prime (simplifies the comparisons).
    """

    def __init__(
        self,
        p: int,
        padding: int = 50,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param p: Modulus.
        """
        super().__init__()
        bit_size = p.bit_length()
        if p.bit_count() == 1:
            raise CircuitParameterInvalid(self, p, "Should not be a power of 2")

        self.p = UInt(p)
        p_high_bits = UInt(p >> (bit_size - padding))
        control = self.add_input_output(BoolType())

        x_reg = self.add_input_output(ModIntType(p))
        y_reg = self.add_input_output(ModIntType(p))

        anc_x = self.add_anc(1)
        anc_y = self.add_anc(1)

        qc_cadd_uint(
            control,
            qc_reg_cast(x_reg + anc_x, datatype=UIntType(bit_size + 1)),
            qc_reg_cast(y_reg + anc_y, datatype=UIntType(bit_size + 1)),
        )

        self.assert_anc(anc_x)
        self.release_anc(anc_x)

        anc = self.get_anc(1)

        # we're checking if y_reg >= p
        qc_lt_uint(
            qc_reg_cast(y_reg[-padding:] + anc_y, datatype=UIntType(padding + 1)),
            p_high_bits,
            anc,
        )
        self.x(anc[0])

        qc_csub_uint(
            anc, self.p, qc_reg_cast(y_reg + anc_y, datatype=UIntType(bit_size + 1))
        )
        # y is not bigger than p, and anc_y is 0
        self.test_anc(anc_y)
        self.assert_anc(anc_y)
        self.release_anc(anc_y)
        # self.print(y_reg)

        # self.print(x_reg, msg="yo")

        # self.print(x_reg, y_reg, msg="test")
        # anc <=> there is a subtraction of p
        # there was a subtraction of p iff y < x in the result
        # qc_clt_uint(control, y_reg, x_reg, anc)
        qc_clt_uint(
            control,
            qc_reg_cast(y_reg[-padding:], datatype=UIntType(padding)),
            qc_reg_cast(x_reg[-padding:], datatype=UIntType(padding)),
            anc,
        )

        # self.print(anc_y, anc)
        self.assert_anc(anc)

    def dummy_classical_function(
        self, args: tuple[QartonBool, ModInt, ModInt]
    ) -> tuple[QartonBool, ModInt, ModInt]:
        c, x, y = args
        return (c, x, x + y if c else y)

    def dummy_classical_function_inverse(
        self, args: tuple[QartonBool, ModInt, ModInt]
    ) -> tuple[QartonBool, ModInt, ModInt]:
        c, x, y = args
        return (c, x, y - x if c else y)

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_cadd_modint)
        def _(c: Register, x: Register | ModInt, y: Register) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            p = ModIntType.get_from_register(x).p
            return PreCircuit(cls, p, PADDING, inverse=False)

        @define_on_call(cls, qc_csub_modint)
        def _(c: Register, x: Register | ModInt, y: Register) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            p = ModIntType.get_from_register(x).p
            return PreCircuit(cls, p, PADDING, inverse=True)


@memoize
class EfficientModularDouble(InPlaceCircuit[ModInt]):
    """
    Non-exact modular doubling, for any prime (simplifies the comparisons).
    """

    def __init__(
        self,
        p: int,
        padding: int = 50,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param p: Modulus.
        """
        super().__init__()
        if p % 2 == 0:
            raise CircuitParameterInvalid(self, p, "must be odd")
        bit_size = p.bit_length()
        self.p = UInt(p)
        p_high_bits = UInt(p >> (bit_size - padding))

        x_reg = self.add_input_output(ModIntType(p))
        anc1 = self.add_anc(1)
        anc2 = self.add_anc(1)

        # shift
        qc_shift_right(x_reg + anc1, 1)

        qc_lt_uint(
            qc_reg_cast(x_reg[-padding:] + anc1, datatype=UIntType(padding + 1)),
            p_high_bits,
            anc2,
        )
        self.x(anc2[0])

        qc_csub_uint(
            anc2, self.p, qc_reg_cast(x_reg + anc1, datatype=UIntType(bit_size + 1))
        )

        # erase anc2 with first bit of x_reg (it's 1 iff there was a reduction)
        self.cx(x_reg[0], anc2[0])
        self.assert_anc(anc1, anc2)

    def dummy_classical_function(self, args: ModInt) -> ModInt:
        return ModInt((2 * args.v) % self.p, self.p)

    def dummy_classical_function_inverse(self, args: ModInt) -> ModInt:
        return ModInt((pow(2, -1, self.p) * args.v) % self.p, self.p)

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_dbl_modint)
        def _(x: Register) -> OnCallOutputType:
            p = ModIntType.get_from_register(x).p
            return PreCircuit(cls, p, PADDING, inverse=False)

        @define_on_call(cls, qc_halve_modint)
        def _(x: Register) -> OnCallOutputType:
            p = ModIntType.get_from_register(x).p
            return PreCircuit(cls, p, PADDING, inverse=True)


def pseudo_mersenne_factor(p: int) -> tuple[int, int]:
    """
    Given an odd ``p``, returns ``(e, f)`` such that ``p = f * 2**e - 1`` with
    ``f`` odd (``e`` is the 2-adic valuation of ``p + 1``).
    """
    e = 0
    val = p + 1
    while val % 2 == 0:
        val //= 2
        e += 1
    return e, val


@memoize
class SpecialModularDouble(InPlaceCircuit[ModInt]):
    r"""
    Modular doubling for a prime of the special form :math:`p = f \cdot 2^e - 1`
    with :math:`f` small.

    The circuit is non-exact: it fails on random inputs with some probability.

    Doubling :math:`x` gives :math:`y = 2x`, spread over ``bit_size + 1`` bits after
    the shift (``bit_size = e + h`` where :math:`h` is the bit-length of the "high"
    part). Writing :math:`y = y_{high} \cdot 2^e + y_{low}`, reduction mod :math:`p`
    amounts to conditionally applying :math:`y - p = (y + 1) - f \cdot 2^e`, i.e.
    incrementing the low ``e`` bits by 1 and subtracting ``f`` from the high part
    (bits ``e`` and above), instead of subtracting the whole (generic, bit_size-wide)
    constant ``p``.

    The low-part increment is non-exact: the carry is only propagated until some
    point. The high-part subtraction is exact.
    """

    def __init__(
        self,
        p: int,
        padding: int = PADDING,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param p: Modulus, must be of the form ``f * 2**e - 1`` with ``f`` fitting
            on at most ``padding`` bits.
        """
        super().__init__()
        if p % 2 == 0:
            raise CircuitParameterInvalid(self, p, "must be odd")

        e, f = pseudo_mersenne_factor(p)
        if e <= padding:
            raise CircuitParameterInvalid(
                self,
                p,
                f"e = {e} is not large enough (should have >= {padding} bits)",
            )

        bit_size = p.bit_length()
        h = bit_size - e

        self.p = UInt(p)
        self.f = UInt(f)
        self.e = e

        x_reg = self.add_input_output(ModIntType(p))
        anc1 = self.add_anc(1)
        anc2 = self.add_anc(1)

        # shift (double)
        qc_shift_right(x_reg + anc1, 1)

        # low_reg = qc_reg_cast(x_reg[:e], datatype=UIntType(e))
        high_reg = qc_reg_cast(x_reg[e:] + anc1, datatype=UIntType(h + 1))

        # compare the high-order bits (positions e onward) to f: this is an exact
        # (not approximate) comparison, since the high part is small
        qc_lt_uint(high_reg, self.f, anc2)
        self.x(anc2[0])

        # increment the low-order bits by 1
        low_reg = qc_reg_cast(x_reg[:padding], datatype=UIntType(padding))
        qc_cincr_uint(anc2, low_reg)

        # subtract f from the high-order bits (positions e to the end, no padding
        # needed since this register is already small)
        qc_csub_uint(anc2, self.f, high_reg)

        # erase anc2 with first bit of x_reg (it's 1 iff there was a reduction);
        # after subtracting, the highest-order bit (anc1) is always 0
        self.cx(x_reg[0], anc2[0])
        self.assert_anc(anc1, anc2)

    def dummy_classical_function(self, args: ModInt) -> ModInt:
        return ModInt((2 * args.v) % self.p, self.p)

    def dummy_classical_function_inverse(self, args: ModInt) -> ModInt:
        return ModInt((pow(2, -1, self.p) * args.v) % self.p, self.p)

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_dbl_modint)
        def _(x: Register) -> OnCallOutputType:
            p = ModIntType.get_from_register(x).p
            return PreCircuit(cls, p, PADDING, inverse=False)

        @define_on_call(cls, qc_halve_modint)
        def _(x: Register) -> OnCallOutputType:
            p = ModIntType.get_from_register(x).p
            return PreCircuit(cls, p, PADDING, inverse=True)


@memoize
class SpecialControlledModularAdder(InPlaceCircuit[tuple[QartonBool, ModInt, ModInt]]):
    r"""
    In-place modular addition for a prime of the form :math:`p = f \cdot 2^e - 1`
    with :math:`f` small.

    After adding ``x`` into ``y`` (controlled), we conditionally subtract ``p`` using
    the same increment-low/subtract-high-by-f trick as :class:`SpecialModularDouble`
    -- gated by comparing the high-order bits of ``y`` (positions ``e`` and above) to
    ``f``.

    That comparison alone misses one case: if ``x + y = p`` exactly, then the
    high-order bits equal ``f - 1`` (so the comparison says "no reduction needed"),
    but the low-order bits are all 1 -- the sum should reduce to 0, not stay at
    ``p``. This is handled separately: an MCX on the low bits detects this "all 1"
    pattern and, depending on the result, XORs ``p`` into the register (which zeroes
    it out exactly in this case, since ``y = p`` at that point). Checking that the low
    bits became all 0 afterward erases the control bit again.

    This circuit supports:

    - random inputs (x,y)
    - random inputs (x,y) such that x + y = p
    - random inputs (0,y)
    - random inputs (x,0)
    """

    def __init__(
        self,
        p: int,
        padding: int = PADDING,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param p: Modulus, must be of the form ``f * 2**e - 1`` with ``e >= padding``.
        """
        super().__init__()
        e, f = pseudo_mersenne_factor(p)
        if e <= padding:
            raise CircuitParameterInvalid(
                self,
                p,
                f"e = {e} is not large enough (should have >= {padding} bits)",
            )

        bit_size = p.bit_length()
        h = bit_size - e

        self.p = UInt(p)
        self.f = UInt(f)
        self.e = e

        control = self.add_input_output(BoolType())

        x_reg = self.add_input_output(ModIntType(p))
        y_reg = self.add_input_output(ModIntType(p))

        anc_x = self.add_anc(1)
        anc_y = self.add_anc(1)

        # ----------------------

        # add x into y (controlled)
        qc_cadd_uint(
            control,
            qc_reg_cast(x_reg + anc_x, datatype=UIntType(bit_size + 1)),
            qc_reg_cast(y_reg + anc_y, datatype=UIntType(bit_size + 1)),
        )
        self.release_anc(anc_x)

        anc2 = self.get_anc(1)
        low_reg = qc_reg_cast(y_reg[:padding], datatype=UIntType(padding))
        high_reg = qc_reg_cast(y_reg[e:] + anc_y, datatype=UIntType(h + 1))

        # check if the high-order bits of y are bigger than f
        qc_lt_uint(high_reg, self.f, anc2)
        self.x(anc2[0])

        # if so, subtract f on the high bits and add 1 to the low bits
        qc_cincr_uint(anc2, low_reg)
        qc_csub_uint(anc2, self.f, high_reg)

        # corner case x + y = p: the high bits are not bigger than f, but the low
        # bits are all 1. MCX on the low bits detects this, and we XOR p into the
        # register (which zeroes it out exactly when x + y = p).
        anc3 = self.get_anc(1)
        qc_mcx(low_reg, anc3)
        qc_cxor(anc3, self.p, qc_reg_cast(y_reg, datatype=UIntType(bit_size)))

        # fold the corner case into anc2: it now reflects "a reduction happened",
        # either via the main branch or via this corner case
        self.cx(anc3[0], anc2[0])

        # erase anc3 by checking that the low bits are now all 0
        # breaks if x + y = 0, then the low bits will always be 0
        qc_mcx_neg(low_reg, anc3)
        self.assert_anc(anc3)

        # y is not bigger than p, and anc_y is 0
        self.assert_anc(anc_y)

        self.release_anc(anc_y)

        # anc2 <=> there was a reduction; there was a reduction iff y < x in the result
        qc_clt_uint(
            control,
            qc_reg_cast(y_reg[-padding:], datatype=UIntType(padding)),
            qc_reg_cast(x_reg[-padding:], datatype=UIntType(padding)),
            anc2,
        )
        self.assert_anc(anc2)

    def dummy_classical_function(
        self, args: tuple[QartonBool, ModInt, ModInt]
    ) -> tuple[QartonBool, ModInt, ModInt]:
        c, x, y = args
        return (c, x, x + y if c else y)

    def dummy_classical_function_inverse(
        self, args: tuple[QartonBool, ModInt, ModInt]
    ) -> tuple[QartonBool, ModInt, ModInt]:
        c, x, y = args
        return (c, x, y - x if c else y)
