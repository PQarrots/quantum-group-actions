"""
Compute the inverse modulo a power of 2.

The algorithm that we use is a reversible variant of Stein's extended GCD, in the
most general version. We assume that the input is odd, so that it will be inversible
modulo 2. Note that Kaliski's algorithm, and most of the binary GCD variants found in
the literature, and their quantum implementations, assume that we invert modulo an
odd prime, so they cannot be used here.

"""

from qarton.arithmetic import qc_add_uint, qc_csub_uint
from qarton.binary_operations import (
    qc_cshift_right,
    qc_cswap,
    qc_shift_right,
    qc_xor,
)
from qarton.circuit import (
    Bit,
    BitVector,
    BoolType,
    Circuit,
    InPlaceCircuit,
    InvolutoryCircuit,
    PreCircuit,
    QartonBool,
    Register,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_bool_reg,
    qc_reg_cast,
)
from qarton.signed_arithmetic import (
    SInt,
    SIntType,
    qc_abs,
    qc_abs_uncompute,
    qc_cadd_sint,
    qc_csub_sint,
)

__all__ = ["OOPModInvPow2", "IPModInvPow2", "ModInvPow2"]


def qc_ccx_oop(a: Bit, b: Bit) -> Register:
    qc = a.parent_circuit
    t = qc.get_anc(1)
    qc.ccx(a, b, t[0])
    return t


def qc_ccx_oop_erase(a: Bit, b: Bit, t: Register) -> None:
    qc = a.parent_circuit
    qc.ccx(a, b, t[0])
    qc.test_anc(t)
    qc.release_anc(t)


@memoize
class CShiftSInt(InPlaceCircuit[tuple[QartonBool, SInt]]):
    """
    Controlled-shift (i.e., multiplication by 2 in place) of a signed integer.
    It reduces to a controlled-shift of the absolute value.
    """

    def __init__(self, w: int, offset: int) -> None:
        super().__init__()
        self.offset = offset
        c = self.add_input_output(BoolType())
        xreg = self.add_input_output(SIntType(w))
        xabs, xsgn = qc_abs(xreg)
        qc_cshift_right(c, xabs, offset)
        xnew = qc_abs_uncompute(xabs, xsgn)
        self.remap(c, xnew)


def qc_cshift_sint(c: Register, s: Register, offset: int) -> None:
    """
    Controlled shift of a signed integer in place.
    """
    qc = c.parent_circuit
    w = SIntType.get_from_register(s).width
    qc.append(CShiftSInt(w, offset), c, s)


def binary_xgcd(a: int, n: int) -> tuple[int, BitVector]:
    """Binary extended GCD to compute the inverse of a modulo :math:`2^n`. This is the
    classical function which our quantum circuit implements.

    The algorithm is originally from Knuth (TAOCP) and was modified for our specific
    case, and simplified for the reversible setting.

    :param a: Integer to invert (must be odd)
    :type a: int
    :param n: Power of 2
    :type n: int
    :return: The inverse of a, and the garbage produced when running the algorithm.
    :rtype: tuple[int, BitVector]
    """
    b = 1 << n
    assert a > 0
    assert a % 2 == 1
    assert b > 0

    u, v = a, b

    aa, bb = 1, 0
    cc, dd = 0, 1

    swapvec = BitVector.from_int(0, n)
    vbitvec = BitVector.from_int(0, n)
    ccddcondvec = BitVector.from_int(0, n)

    for i in range(n):
        # exactly n iterations is sufficient
        swap = ((v % 2 == 1) and (u % 2 == 0)) or (
            (v % 2 == 1) and u % 2 == 1 and u > v
        )
        swapvec[i] = swap
        if swap:
            u, v = v, u
            aa, cc = cc, aa
            bb, dd = dd, bb

        # store swap
        # store v & 1
        # store (cc & 1) != 0 or (dd & 1) != 0
        if (v & 1) == 0:
            vbitvec[i] = 1
            v >>= 1
            if not ((cc & 1) == 0 and (dd & 1) == 0):
                ccddcondvec[i] = 1
                cc = cc + b
                dd = dd - a
            cc >>= 1
            dd >>= 1
        else:
            v -= u
            cc -= aa
            dd -= bb
        if swap:
            u, v = v, u
            aa, cc = cc, aa
            bb, dd = dd, bb

    x = cc
    assert x >= 0  # always true here (apparently)
    assert x == pow(a, -1, b)

    garbage = UIntType(n + 1).to_register(UInt(u))
    garbage += UIntType(n + 1).to_register(UInt(v))
    garbage += SIntType(n + 2).to_register(SInt(aa, n + 2))
    garbage += SIntType(n + 2).to_register(SInt(bb, n + 2))
    # garbage += UIntType(2).to_register(UInt(x // b))
    garbage += SIntType(n + 2).to_register(SInt(dd, n + 2))
    garbage += swapvec
    garbage += vbitvec
    garbage += ccddcondvec
    return x, garbage


@memoize
class _ModInvPow2Iterator(InPlaceCircuit):
    """
    Iterator for the ModInvPow2 circuit. Because this circuit repeats the same
    operations n times, we define an iterator and use a Qarton *iterated operation* to
    build the circuit faster.

    The iterator operates on the following fields (constant size across iterations):

    * ``u``, ``v`` (UIntType(n + 1))
    * ``areg_sint`` (SIntType(n + 2)): the (signed) value being inverted, constant
      throughout the iterations
    * ``aa``, ``bb``, ``cc``, ``dd`` (SIntType(n + 2))

    And garbage registers (BitVectorType of size n, one bit produced by each
    iteration):

    * ``swap``
    * ``vbit``
    * ``ccddcond``
    """

    def __init__(self, n: int) -> None:
        """
        :param n: Bit-size (i.e., the power of 2).
        :type n: int
        """
        super().__init__()
        bit_size = n + 2  # padding required for the 4 integers aa, bb, cc, dd

        u = self.add_input_output(UIntType(n + 1))
        v = self.add_input_output(UIntType(n + 1))
        areg_sint = self.add_input_output(SIntType(n + 2))
        aa = self.add_input_output(SIntType(bit_size))
        bb = self.add_input_output(SIntType(bit_size))
        cc = self.add_input_output(SIntType(bit_size))
        dd = self.add_input_output(SIntType(bit_size))

        swap = self.add_input_output(n)
        vbit = self.add_input_output(n)
        ccddcond = self.add_input_output(n)
        # ------------------

        # swap = ((v % 2 == 1) and (u % 2 == 0)) or (
        #                (v % 2 == 1) and u % 2 == 1 and u > v
        #            )
        self.x(u[0])
        self.ccx(v[0], u[0], swap[0])
        self.x(u[0])

        self.cx(v[0], vbit[0])
        self.x(vbit[0])  # condition is that it's 0

        # ccddcond = not( cc & 1 == 0 and dd & 1 == 0)
        self.x(cc[0])
        self.x(dd[0])
        self.ccx(cc[0], dd[0], ccddcond[0])
        self.x(cc[0])
        self.x(dd[0])
        self.x(ccddcond[0])

        # control-swap
        qc_cswap(qc_bool_reg(swap[0]), u, v)
        qc_cswap(qc_bool_reg(swap[0]), aa, cc)
        qc_cswap(qc_bool_reg(swap[0]), dd, bb)

        # controlled on vbit[0]
        qc_cshift_right(qc_bool_reg(vbit[0]), v, -1)

        # controlled on vbit[0] and ccddcond[0]
        tmp = qc_ccx_oop(vbit[0], ccddcond[0])
        qc_cadd_sint(tmp, SInt((1 << n), bit_size), cc)  # cc = cc + b
        qc_csub_sint(tmp, areg_sint, dd)  # dd = dd - a
        qc_ccx_oop_erase(vbit[0], ccddcond[0], tmp)

        # controlled on vbit[0]
        qc_cshift_sint(qc_bool_reg(vbit[0]), cc, -1)
        qc_cshift_sint(qc_bool_reg(vbit[0]), dd, -1)

        self.x(vbit[0])
        qc_csub_uint(qc_bool_reg(vbit[0]), u, v)  # v -= u
        qc_csub_sint(qc_bool_reg(vbit[0]), bb, dd)  # dd -= bb
        qc_csub_sint(qc_bool_reg(vbit[0]), aa, cc)  # cc -= aa
        self.x(vbit[0])

        qc_cswap(qc_bool_reg(swap[0]), u, v)
        qc_cswap(qc_bool_reg(swap[0]), aa, cc)
        qc_cswap(qc_bool_reg(swap[0]), dd, bb)

        # roll the accumulators, so that the next iteration writes into slot 0, and
        # the bit written at iteration i ends up (after all n iterations) at index i,
        # matching binary_xgcd's direct vec[i] = ... indexing
        qc_shift_right(swap, -1)
        qc_shift_right(vbit, -1)
        qc_shift_right(ccddcond, -1)


@dummify
@memoize
class ModInvPow2(Circuit[UInt, tuple[UInt, UInt, BitVector]]):
    """
    Inverts an odd integer modulo a power of 2. Which power is given by the size of the
    register.

    The input register is preserved. The circuit produces two output registers: the
    inverse, and a register of garbage stored during the iterations of the XGCD.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        areg = self.add_input_output(UIntType(n))

        # a, bb = 1, 0
        # cc, dd = 0, 1
        # u,v  = a,b # b = 2**n
        u, v = self.add_anc(UIntType(n + 1)), self.add_anc(UIntType(n + 1))
        qc_xor(areg, u[:n])

        areg_padded = qc_reg_cast(areg + self.add_anc(2), UIntType(n + 2))
        sgn = self.add_anc(1)
        areg_sint = qc_abs_uncompute(areg_padded, sgn)

        bit_size = n + 2  # padding required for the 4 integers aa, bb, cc, dd
        # initialize aa, bb, cc, dd
        aa, bb, cc, dd = (
            self.add_anc(SIntType(bit_size)),
            self.add_anc(SIntType(bit_size)),
            self.add_anc(SIntType(bit_size)),
            self.add_anc(SIntType(bit_size)),
        )
        self.x(aa[0])
        self.x(dd[0])

        self.x(v[-1])

        swap = self.add_anc(n)
        vbit = self.add_anc(n)
        ccddcond = self.add_anc(n)

        self.append_iterated(
            _ModInvPow2Iterator(n),
            u,
            v,
            areg_sint,
            aa,
            bb,
            cc,
            dd,
            swap,
            vbit,
            ccddcond,
            iterations=n,
        )

        # cc should be always >= 0 and contains inverse of areg
        # first take abs
        ccabs, ccsgn = qc_abs(cc)
        self.test_anc(ccsgn)
        self.release_anc(ccsgn)
        # truncate ccabs
        ccuint = qc_reg_cast(ccabs[:n], UIntType(n))
        ccpad = ccabs[n:]
        self.assert_anc(ccpad)
        garbage = u + v + aa + bb + dd + swap + vbit + ccddcond
        self.add_output(ccuint)
        self.add_output(garbage)

        # need to restore areg
        areg_padded, sgn = qc_abs(areg_sint)
        areg = qc_reg_cast(areg_padded[:n], UIntType(n))
        self.assert_anc(sgn)
        self.assert_anc(areg_padded[n:])
        self.remap(areg, ccuint, garbage)

    def dummy_classical_function(self, args: UInt) -> tuple[UInt, UInt, BitVector]:
        x, g = binary_xgcd(args, self.n)
        return args, UInt(x), g

    def validate_input(self, args: UInt) -> bool:
        return args % 2 == 1

    def dummy_classical_function_inverse(
        self, args: tuple[UInt, UInt, BitVector]
    ) -> UInt:
        x, _, _ = args
        return x


@memoize
@dummify
class OOPModInvPow2(InPlaceCircuit[tuple[UInt, UInt]]):
    """
    Implementation of the inverse modulo a power of 2 (works for odd integers).
    Out of place.

    Example::

        >>> qc = OOPModInvPow2(5)
        >>> qc.simulate((UInt(9), UInt(0)))
        (9, 25)

    """

    def __init__(self, n: int) -> None:
        """
        :param n: Bit-size (i.e., the power of 2).
        :type n: int
        """
        super().__init__()
        xreg = self.add_input_output(UIntType(n))
        yreg = self.add_input_output(UIntType(n))
        self.n = n
        xreg, o, g = self.append(ModInvPow2(n), xreg)
        qc_add_uint(o, yreg)
        self.append(ModInvPow2(n).inverse(), xreg, o, g)

    def validate_input(self, args: tuple[UInt, UInt]) -> bool:
        """
        Input is valid only if it's odd.
        """
        x, _ = args
        return x % 2 == 1

    def validate_output(self, args: tuple[UInt, UInt]) -> bool:
        return super().validate_input(args)

    def dummy_classical_function(self, args: tuple[UInt, UInt]) -> tuple[UInt, UInt]:
        x, y = args
        return x, UInt((y + pow(x, -1, 1 << self.n)) % (1 << self.n))

    def dummy_classical_function_inverse(
        self, args: tuple[UInt, UInt]
    ) -> tuple[UInt, UInt]:
        x, y = args
        return x, UInt((y - pow(x, -1, 1 << self.n)) % (1 << self.n))


@dummify
@memoize
class IPModInvPow2(InvolutoryCircuit[UInt]):
    """
    In-place modular inverse modulo a power of 2. Not defined for even integers.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        x_reg = self.add_input_output(UIntType(n))
        y_reg = self.add_anc(UIntType(n))

        # y <- x^{-1}
        self.append(PreCircuit(OOPModInvPow2, n), x_reg, y_reg)

        # x <- x + y^{-1} == 0
        self.append(PreCircuit(OOPModInvPow2, n, inverse=True), y_reg, x_reg)
        self.swap_reg(x_reg, y_reg)
        self.assert_anc(y_reg)

    def validate_input(self, args: UInt) -> bool:
        return args % 2 == 1

    def dummy_classical_function(self, args: UInt) -> UInt:
        return UInt(pow(args, -1, 1 << self.n))
