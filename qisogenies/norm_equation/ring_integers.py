r"""
Elements of the ring of integers Z[omega], and basic operations on them. The coordinates
are signed integers. We use omega = (1 + sqrt(-p))/2  where p > 0 and p = 3 mod 4 is a
prime. The number field is K = Q(sqrt(-p)) and the discriminant is -p.
"""

from __future__ import annotations

import random
from typing import Any

from qarton.binary_operations import qc_copy
from qarton.circuit import (
    BackendSpecifier,
    BitVector,
    Circuit,
    DataType,
    EmptyBackendSpecifier,
    InvolutoryCircuit,
    PreCircuit,
    QartonCustomObject,
    Register,
    RegisterIndexError,
    RegisterKey,
    UInt,
    dummify,
    memoize,
    qc_reg_cast,
)
from qarton.dispatch import (
    qc_add,
    qc_mul,
    qc_mul_erase,
    qc_neg,
    qc_sub,
)
from qarton.signed_arithmetic import (
    SInt,
    SIntType,
    qc_abs,
    qc_expand_sint,
    qc_mul_erase_sint,
    qc_mul_sint,
    qc_truncate_sint,
)

__all__ = [
    "RingInteger",
    "RingIntegerType",
    "qc_make_ring_integer",
    "qc_truncate_sint_ring_integer",
    "Conjugate",
    "qc_conj",
    "Multiply",
    "qc_mul_ring_int",
    "qc_mul_erase_ring_int",
    "Norm",
    "qc_norm",
    "qc_norm_erase",
    "Trace",
    "qc_trace",
    "qc_trace_erase",
]

# as a datatype: like a tuple, but with the additional info of the prime.


class RingInteger(QartonCustomObject):
    """
    Element in Z[omega], represented as: (a,b) where the element is a + omega b.
    Here omega = (1 + sqrt(-p))/2. omegabar = 1 - omega.

    Ideal element can have any size.
    """

    __slots__ = "a", "b", "p", "w"

    def __init__(self, p: int, w: int, a: int | SInt, b: int | SInt) -> None:
        super().__init__()
        if p % 4 != 3:
            raise ValueError("Invalid p, should be = 3 mod 4")
        if w > (1 << 32):
            raise ValueError(
                "w is a width, this one is too big, most certainly the order of arguments was mixed up"
            )
        # if p > (1 << w):
        #    raise ValueError("p too large for width, this will be an issue")
        self.p = p
        self.a = SInt(a.v if isinstance(a, SInt) else a, w)
        self.b = SInt(b.v if isinstance(b, SInt) else b, w)
        self.w = w

    @staticmethod
    def random(p: int, w: int) -> RingInteger:
        return RingInteger(
            p,
            w,
            random.randrange(-(1 << w), 1 << w),
            random.randrange(-(1 << w), 1 << w),
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, RingInteger):
            return False
        return self.p == o.p and self.a == o.a and self.b == o.b

    def __hash__(self) -> int:
        return hash((self.p, self.a, self.b))

    def conjugate(self) -> RingInteger:
        """
        Conjugate of a + omega b is a + (omega bar)b = a + (1-omega) b = (a+b) - omega b
        """
        return RingInteger(self.p, self.w, (self.a + self.b).v, (-self.b).v)

    def __add__(self, other: RingInteger) -> RingInteger:
        """
        Addition maintains or expands the range w.
        """
        # assert self.p == other.p
        assert self.w >= other.w
        return RingInteger(self.p, self.w, (self.a + other.a).v, (self.b + other.b).v)

    def norm(self) -> UInt:
        """
        Norm produces a positive integer of size 2w + size(p) (roughly).
        """
        return UInt(
            (self.p + 1) // 4 * self.b.v * self.b.v + self.a.v * (self.a.v + self.b.v)
        )

    def trace(self) -> UInt:
        """
        Trace produces a signed integer of size w.
        """
        return UInt(SInt((self.a + self.a + self.b).v, self.w).v)

    def __mul__(self, other: RingInteger) -> RingInteger:
        """
        Multiplication produces a ring integer element with coordinates of size 2w + size(p) (roughly).

        (a + omega b) (c + omega d) = ac + omega^2 (bd) + omega (ad + bc)
        omega^2 = omega - (p+1)/4
        (a + omega b) (c + omega d) = (ac - (p+1)/4 bd) + omega (ad + bc + bd)
        """
        const = (self.p + 1) // 4
        return RingInteger(
            self.p,
            2 * self.w + const.bit_length(),
            -(self.p + 1) // 4 * self.b.v * other.b.v + self.a.v * other.a.v,
            self.a.v * other.b.v + self.b.v * (other.a.v + other.b.v),
        )

    def __floordiv__(self, other: UInt | SInt | int) -> RingInteger:
        """
        Divide the coordinates of this element by the other integer.
        """
        newv = other.v if isinstance(other, SInt) else other
        return RingInteger(self.p, self.w, self.a.v // newv, self.b.v // newv)

    def __str__(self) -> str:
        return str((self.a, self.b))

    def __repr__(self) -> str:
        return str(self)

    def to_pair(self) -> tuple[Any, Any]:
        from sage.all import ZZ  # type: ignore

        # tuple of a' + b' sqrt(-p) (matches the sage representation)
        return (self.a.v + ZZ(self.b.v) / 2, ZZ(self.b.v) / 2)  # type: ignore


class RingIntegerType(DataType[RingInteger]):
    """
    Datatype for an element of the ring of integers, represented as a + b omega where
    omega = (1 + sqrt(-p))/2, a and b are fixed-size signed integers.
    """

    __slots__ = "_internal_type", "p", "width"

    def __init__(self, p: int, w: int) -> None:
        """
        :param p: prime
        :type p: int
        :param w: Size of coordinates (as signed integers).
        :type w: int
        """
        super().__init__()
        if p % 4 != 3:
            raise ValueError("Invalid p, should be =3 mod 4")
        self.width = w
        self.p = p
        self._internal_type = SIntType(w)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, RingIntegerType):
            return o.width == self.width and o.p == self.p
        return False

    def to_register(self, o: RingInteger) -> BitVector:
        i1 = self._internal_type.to_register(o.a)
        i2 = self._internal_type.to_register(o.b)
        return i1 + i2

    def from_register(self, l: BitVector) -> RingInteger:
        # signed int is on w + 1 bits
        l1 = l[: (self.width + 1)]
        l2 = l[(self.width + 1) :]
        return RingInteger(
            self.p,
            self.width,
            self._internal_type.from_register(l1).v,
            self._internal_type.from_register(l2).v,
        )

    def register_length(self) -> int:
        return 2 * (self.width + 1)

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        if key == "a" or key == 0:
            return qc_reg_cast(r[: (self.width + 1)], datatype=self._internal_type)
        elif key == "b" or key == 1:
            return qc_reg_cast(r[(self.width + 1) :], datatype=self._internal_type)
        else:
            raise RegisterIndexError(key)

    def default_value(self) -> RingInteger:
        return RingInteger(self.p, self.width, 0, 0)


def qc_make_ring_integer(x: Register, y: Register, p: int) -> Register:
    """Cast (x,y) into a ring integer. The registers x and y are not destroyed. There
    may be an expansion if they have different sizes.

    :param x: Register
    :type x: Register
    :param y: Register
    :type y: Register
    :param p: Prime
    :type p: int
    :return: Register (RingIntegerType)
    :rtype: Register
    """
    if not isinstance(x.datatype, SIntType):
        raise TypeError()
    if not isinstance(y.datatype, SIntType):
        raise TypeError()
    # take the largest size of both
    w = max(x.datatype.width, y.datatype.width)
    xx = qc_expand_sint(x, w) if x.datatype.width < w else x
    yy = qc_expand_sint(y, w) if y.datatype.width < w else y
    return qc_reg_cast(xx + yy, RingIntegerType(p, w))


def qc_truncate_sint_ring_integer(x: Register, w: int) -> Register:
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    p = x.datatype.p
    a = qc_truncate_sint(x.a["a"], w)
    b = qc_truncate_sint(x.a["b"], w)
    return qc_reg_cast(a + b, RingIntegerType(p, w))


@dummify
@memoize
class Conjugate(InvolutoryCircuit[RingInteger]):
    """
    Takes the conjugate of an element in the ring of integers, in-place.
    """

    def __init__(
        self,
        p: int,
        w: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        r = self.add_input_output(RingIntegerType(p, w))
        self.p = p
        self.w = w

        # self.a, self.b = self.a + self.b, -self.b
        # add b to a
        qc_add(r.a["b"], r.a["a"])

        # negate b
        qc_neg(r.a["b"])

    def dummy_classical_function(self, args: RingInteger) -> RingInteger:
        return args.conjugate()


def qc_conj(r: Register) -> None:
    """In-place conjugate of an element in the ring of integers. Self-inverse operation.

    :param r: Input register (RingIntegerType)
    :type r: Register
    """
    if not isinstance(r.datatype, RingIntegerType):
        raise TypeError()
    p, w = r.datatype.p, r.datatype.width
    qc = r.parent_circuit
    qc.append(PreCircuit(Conjugate, p, w), r)


@memoize
@dummify
class Multiply(
    Circuit[
        tuple[RingInteger, RingInteger], tuple[RingInteger, RingInteger, RingInteger]
    ]
):
    """
    Circuit that multiplies two ring elements.
    """

    def __init__(
        self,
        p: int,
        w: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        xreg = self.add_input_output(RingIntegerType(p, w))
        yreg = self.add_input_output(RingIntegerType(p, w))
        ya, yb = yreg.a["a"], yreg.a["b"]

        # compute: (p+1)// 4 * xreg.b * yreg.b
        # (ac - (p+1)/4 bd) + omega (ad + bc + bd)
        _cst = (p + 1) // 4
        _scst = SInt(_cst, w=_cst.bit_length())
        tmp = qc_mul_sint(_scst, xreg.a["b"])
        outa = qc_mul_sint(tmp, yreg.a["b"])
        qc_neg(outa)
        qc_mul_erase_sint(_scst, xreg.a["b"], tmp)

        tmp = qc_mul(xreg.a["a"], ya)
        qc_add(tmp, outa)
        qc_mul_erase_sint(xreg.a["a"], ya, tmp)

        # out a is now the a coordinate
        # create the second coordinate with same size
        assert isinstance(outa.datatype, SIntType)
        outb = self.get_anc(SIntType(width=outa.datatype.width))

        tmp = qc_mul_sint(xreg.a["a"], yb)
        qc_add(tmp, outb)
        qc_mul_erase_sint(xreg.a["a"], yb, tmp)

        # y.b <- y.a + y.b
        # expand by 1 bit to avoid overfull
        expanded = qc_expand_sint(yb, w + 1)
        qc_add(ya, expanded)

        # self.print(expanded)
        tmp = qc_mul_sint(xreg.a["b"], expanded)
        qc_add(tmp, outb)
        qc_mul_erase_sint(xreg.a["b"], expanded, tmp)
        # restore
        qc_sub(ya, expanded)
        yb = qc_truncate_sint(expanded, w)

        output = qc_reg_cast(outa + outb, RingIntegerType(p, w=outa.datatype.width))
        self.add_output(output)
        yreg = qc_make_ring_integer(ya, yb, p)
        self.remap(xreg, yreg, output)

    def dummy_classical_function(
        self, args: tuple[RingInteger, RingInteger]
    ) -> tuple[RingInteger, RingInteger, RingInteger]:
        x, y = args
        return x, y, x * y

    def dummy_classical_function_inverse(
        self, args: tuple[RingInteger, RingInteger, RingInteger]
    ) -> tuple[RingInteger, RingInteger]:
        x, y, _ = args
        return x, y


def qc_mul_ring_int(x: Register, y: Register) -> Register:
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    qc = x.parent_circuit
    p, w = x.datatype.p, x.datatype.width
    return qc.append(PreCircuit(Multiply, p, w), x, y)[2]


def qc_mul_erase_ring_int(x: Register, y: Register, xy: Register) -> None:
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    qc = x.parent_circuit
    p, w = x.datatype.p, x.datatype.width
    qc.append(PreCircuit(Multiply, p, w, inverse=True), x, y, xy)


@memoize
@dummify
class Norm(Circuit[RingInteger, tuple[RingInteger, UInt]]):
    """
    Circuit that computes the norm of a ring element (out of place).
    """

    def __init__(
        self,
        p: int,
        w: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        xreg = self.add_input_output(RingIntegerType(p, w))
        xa, xb = xreg.a["a"], xreg.a["b"]

        # (p+1)//4 * (b**2) + a * (a+b)
        _cst = (p + 1) // 4
        _scst = SInt(_cst, w=_cst.bit_length())
        tmp = qc_mul(_scst, xb)
        out = qc_mul(tmp, xb)
        qc_mul_erase(_scst, xb, tmp)

        # self.print(xreg)
        # expand by 1 bit to avoid overfull
        expanded = qc_expand_sint(xb, w + 1)

        # self.b <- self.a + self.b
        qc_add(xa, expanded)
        tmp = qc_mul(xa, expanded)
        qc_add(tmp, out)
        qc_mul_erase(xa, expanded, tmp)
        # restore
        qc_sub(xa, expanded)
        xb = qc_truncate_sint(expanded, w)

        # assert positive
        abs_val, sgn = qc_abs(out)
        self.test_anc(sgn)
        self.assert_anc(sgn)
        self.add_output(abs_val)  # int type, of appropriate size
        self.remap(qc_make_ring_integer(xa, xb, p), abs_val)

    def dummy_classical_function(self, args: RingInteger) -> tuple[RingInteger, UInt]:
        return args, UInt(args.norm())

    def dummy_classical_function_inverse(
        self, args: tuple[RingInteger, UInt]
    ) -> RingInteger:
        x, _ = args
        return x


def qc_norm(x: Register) -> Register:
    """Computes the norm of a ring element.

    :param x: Input (RingIntegerType). Preserved.
    :type x: Register
    :raises TypeError: if wrong type.
    :return: New register that holds the norm (IntType).
    :rtype: Register
    """
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    qc = x.parent_circuit
    p, w = x.datatype.p, x.datatype.width
    return qc.append(PreCircuit(Norm, p, w), x)[1]


def qc_norm_erase(x: Register, nr: Register) -> None:
    """Erases the output of ``qc_norm``.

    :param x: Input (RingIntegerType). Preserved.
    :type x: Register
    :param nr: Register that contains the norm. Destroyed.
    :type nr: Register
    :raises TypeError: If invalid type.
    """
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    qc = x.parent_circuit
    p, w = x.datatype.p, x.datatype.width
    qc.append(PreCircuit(Norm, p, w, inverse=True), x, nr)


@memoize
@dummify
class Trace(Circuit[RingInteger, tuple[RingInteger, SInt]]):
    """
    Computes the trace of a ring element.
    """

    def __init__(
        self,
        p: int,
        w: int,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        self.w = w
        xreg = self.add_input_output(RingIntegerType(p, w))
        out = qc_copy(xreg.a["a"])
        qc_add(xreg.a["a"], out)
        qc_add(xreg.a["b"], out)
        self.add_output(out)  # signed integer type

    def dummy_classical_function(self, args: RingInteger) -> tuple[RingInteger, SInt]:
        return args, SInt(args.trace(), self.w)

    def dummy_classical_function_inverse(
        self, args: tuple[RingInteger, SInt]
    ) -> RingInteger:
        x, _ = args
        return x


def qc_trace(x: Register) -> Register:
    """Computes the trace of a ring element (out of place).

    :param x: Input (RingIntegerType). Preserved.
    :type x: Register
    :raises TypeError: If invalid type
    :return: The trace (SIntType), new register.
    :rtype: Register
    """
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    qc = x.parent_circuit
    p, w = x.datatype.p, x.datatype.width
    return qc.append(PreCircuit(Trace, p, w), x)[1]


def qc_trace_erase(x: Register, nr: Register) -> None:
    """Erases the output of ``qc_trace``.

    :param x: Input (RingIntegerType). Preserve.
    :type x: Register
    :param nr: Register containing the trace (SIntType). Destroyed.
    :type nr: Register
    :raises TypeError: If wrong type.
    """
    if not isinstance(x.datatype, RingIntegerType):
        raise TypeError()
    qc = x.parent_circuit
    p, w = x.datatype.p, x.datatype.width
    qc.append(PreCircuit(Trace, p, w, inverse=True), x, nr)
