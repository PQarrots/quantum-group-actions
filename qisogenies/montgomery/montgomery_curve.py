"""
Datatypes for Montgomery curves, adapted from Qarton's code for Weierstrass curves.

"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import cast

from qarton.circuit import (
    AutoEncodableObject,
    BitVector,
    BoolType,
    DataType,
    DataTypeConversionError,
    EncodableObject,
    EncodingData,
    QartonBool,
    QartonCustomObject,
    QartonError,
    Register,
    RegisterIndexError,
    RegisterKey,
    UInt,
    UIntType,
    qc_reg_cast,
)
from qarton.modular_arithmetic import ModInt, ModIntType
from qarton.signed_arithmetic import SInt
from sympy import sqrt_mod  # type: ignore

__all__ = [
    "ECMontgomery",
    "AffMontgomeryPoint",
    "AffMontgomeryPointType",
    "ECMontgomeryType",
    "AffMontgomeryPointVariableType",
    "AffMontgomeryPointVariable",
]


def _modinv(a: int, p: int) -> int:
    # Computes the modular inverse of a mod p
    return pow(a, -1, p)


class ECMontgomery(QartonCustomObject, EncodableObject):
    r"""
    Elliptic curve on a prime field, Montgomery form equation: B y^2 = x^3 + Ax^2 + x

    for A, B such that B(A^2 - 4) \neq 0.
    """

    __slots__ = "_a", "_b", "_q"

    def __init__(self, q: int, a: int, b: int) -> None:
        if (b * (a**2 - 4) % q) == 0:
            raise ValueError("Invalid parameters a,b")
        self._q = q
        self._a = a
        self._b = b
        self._hash = hash((self._q, self._a, self._b))

    @classmethod
    def from_data(cls, data: tuple[EncodingData, ...]) -> ECMontgomery:
        # type refinement assertions
        assert type(data) is tuple
        assert len(data) == 3
        assert type(data[0]) is int
        assert type(data[1]) is int
        assert type(data[2]) is int
        return ECMontgomery(data[0], data[1], data[2])

    def to_data(self) -> tuple[EncodingData, ...]:
        return (self._q, self._a, self._b)

    @property
    def A(self) -> int:
        return self._a

    @property
    def B(self) -> int:
        return self._b

    @property
    def q(self) -> int:
        return self._q

    def __hash__(self) -> int:
        return self._hash


class ECMontgomeryType(DataType[ECMontgomery], AutoEncodableObject):
    """Datatype for an elliptic curve in Montgomery form B y^2 = x^3 + Ax^2 + x over a prime
    field, given by the coefficients A, B. The prime q remains a fixed parameter of the datatype."""

    def __init__(self, q: int) -> None:
        super().__init__()
        if q < 3:
            raise ValueError()
        self.q = q
        self._dtype = ModIntType(q)
        self._dlen = self._dtype.register_length()

    def to_register(self, o: ECMontgomery) -> BitVector:
        return self._dtype.to_register(ModInt(o.A, self.q)) + self._dtype.to_register(
            ModInt(o.B, self.q)
        )

    def from_register(self, l: BitVector) -> ECMontgomery:
        a = self._dtype.from_register(l[0 : self._dlen])
        b = self._dtype.from_register(l[self._dlen : (2 * self._dlen)])
        return ECMontgomery(self.q, int(a), int(b))

    def register_length(self) -> int:
        return 2 * self._dlen

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        if key == "A":
            return qc_reg_cast(r[0 : self._dlen], datatype=self._dtype)
        elif key == "B":
            return qc_reg_cast(r[self._dlen : (2 * self._dlen)], datatype=self._dtype)
        else:
            raise RegisterIndexError(key)

    def default_value(self) -> ECMontgomery:
        # should be the default curve
        raise NotImplementedError()

    def iterate_values(self) -> Iterator[ECMontgomery]:
        raise NotImplementedError()

    def random_value(self) -> ECMontgomery:
        """Return a random elliptic curve equation, non singular.

        The prime being fixed, we take A and B uniformly at random.
        Then we check if the values are valid, and repeat.

        :return: A random ECMontgomery object.
        :rtype: ECMontgomery
        """

        while True:
            a = random.randrange(self.q)
            b = random.randrange(self.q)

            tmp = (b * (a**2 - 4)) % self.q
            if tmp != 0:
                return ECMontgomery(self.q, a, b)


class AffMontgomeryPoint(QartonCustomObject, EncodableObject):
    """
    Affine point on an elliptic curve in Montgomery form B y^2 = x^3 + Ax^2 + x.

    It does not test that the point indeed belongs to the curve. Coordinates are
    not mutable.

    At the moment, coordinates are stored as UInt, not as ModInt. But when operating
    on affine point registers, we still extract the x / y coordinates as ModIntType.
    """

    x: UInt
    y: UInt
    not_infty: QartonBool
    ec: ECMontgomery

    __slots__ = "ec", "not_infty", "x", "y"

    def __init__(self, x: int, y: int, ec: ECMontgomery) -> None:
        self.x = UInt(x % ec.q)
        self.y = UInt(y % ec.q)
        self.not_infty = QartonBool(1)
        self.ec = ec
        if (self.x**3 + ec.A * self.x**2 + self.x) % ec.q != (self.y**2 * ec.B) % ec.q:
            raise QartonError("Point not on curve")

    @classmethod
    def from_data(cls, data: tuple[EncodingData, ...]) -> AffMontgomeryPoint:
        assert len(data) == 4
        x, y, not_infty, ec = data
        assert type(x) is int
        assert type(y) is int
        assert type(not_infty) is bool
        assert isinstance(ec, ECMontgomery)
        if not not_infty:
            return AffMontgomeryPoint.infinity(ec)
        else:
            return AffMontgomeryPoint(x, y, ec)

    def to_data(self) -> tuple[EncodingData, ...]:
        return int(self.x), int(self.y), bool(self.not_infty), self.ec

    @classmethod
    def infinity(cls, ec: ECMontgomery) -> AffMontgomeryPoint:
        p = cls.__new__(cls)
        p.ec = ec
        p.x = UInt(0)
        p.y = UInt(0)
        p.not_infty = QartonBool(0)
        return p

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.ec))

    def __neg__(self) -> AffMontgomeryPoint:
        if not self.not_infty:
            return AffMontgomeryPoint.infinity(self.ec)
        else:
            return AffMontgomeryPoint(self.x, -self.y, self.ec)

    def __sub__(self, other: AffMontgomeryPoint) -> AffMontgomeryPoint:
        return self + (-other)

    def __add__(self, other: AffMontgomeryPoint) -> AffMontgomeryPoint:
        if self.ec != other.ec:
            raise ValueError("Should have the same base curve")

        x1, y1 = self.x, self.y
        x2, y2 = other.x, other.y
        p = self.ec.q

        if not other.not_infty and not self.not_infty:
            # both points are infinity
            return AffMontgomeryPoint.infinity(self.ec)

        if not other.not_infty:
            # one point is infinity
            return AffMontgomeryPoint(self.x, self.y, self.ec)

        if not self.not_infty:
            # the other point is infinity
            return AffMontgomeryPoint(other.x, other.y, other.ec)

        if x1 == x2 and (y1 + y2) % p == 0:
            # P + (-P) = 0
            return AffMontgomeryPoint.infinity(self.ec)

        a, b = self.ec.A, self.ec.B
        if (x1, y1) == (x2, y2):
            # doubling case
            # s = (3x1^2 + 2a x1 + 1) / (2y1 b)
            s = ((3 * x1 * x1 + 2 * a * x1 + 1) * _modinv(2 * y1 * b, p)) % p
        else:
            # General addition
            # s = (y2-y1) / (x2-x1)
            s = ((y2 - y1) * _modinv(x2 - x1, p)) % p

        x3 = (b * s * s - a - x1 - x2) % p
        y3 = (s * (x1 - x3) - y1) % p

        return AffMontgomeryPoint(x3, y3, self.ec)

    def __mul__(self, k: int | ModInt | SInt) -> AffMontgomeryPoint:
        if isinstance(k, ModInt):
            return self * k.v
        elif isinstance(k, SInt):
            sgn = k.sgn()
            if sgn == 1:
                return -self * k.v
            else:
                return self * k.v
        elif k < 0:
            return -self * (-k)
        elif k == 0:
            return AffMontgomeryPoint.infinity(self.ec)
        elif k == 1:
            return self
        else:
            pt = self
            for k_i in bin(k)[3:]:
                pt = pt + pt
                if k_i == "1":
                    pt = pt + self
            return pt

    def __str__(self) -> str:
        if self.not_infty:
            return "AffMontgomeryPoint" + str((self.x, self.y))
        else:
            return "AffMontgomeryPoint(Infty)"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AffMontgomeryPoint)
            and self.not_infty == other.not_infty
            and self.x == other.x
            and self.y == other.y
            and self.ec == other.ec
        )

    def __repr__(self) -> str:
        return str(self)

    @staticmethod
    def random(ec: ECMontgomery) -> AffMontgomeryPoint:
        """
        Sample a random point on the curve (it does not return the point at infinity though).
        """
        while True:
            x = random.randrange(ec.q)

            rhs = ((x**3 + ec.A * x**2 + x) * _modinv(ec.B, ec.q)) % ec.q
            roots = sqrt_mod(rhs, ec.q, all_roots=True)
            if roots:
                return AffMontgomeryPoint(
                    x=x, y=int(random.choice(cast(list[int], roots))), ec=ec
                )


def _elliptic_curve_AffMontgomeryPoints(
    ec: ECMontgomery,
) -> Iterator[AffMontgomeryPoint]:
    yield AffMontgomeryPoint.infinity(ec)

    for x in range(ec.q):
        rhs = ((x**3 + ec.A * x**2 + x) * _modinv(ec.B, ec.q)) % ec.q
        ys = sqrt_mod(rhs, ec.q, all_roots=True)
        if ys is None:
            continue
        assert type(ys) is list
        for y in ys:  # type: ignore
            assert type(y) is int  # type: ignore
            yield AffMontgomeryPoint(x, y, ec)


class AffMontgomeryPointType(DataType[AffMontgomeryPoint], AutoEncodableObject):
    """
    DataType for an affine point on an elliptic curve.
    """

    def __init__(self, ec: ECMontgomery) -> None:
        """
        :param ec: Base curve.
        :type d: ECMontgomery
        """
        super().__init__()
        self.ec = ec
        self._q = self.ec.q
        self._dlen = self._q.bit_length()
        self._dtype = UIntType(self._dlen)
        self._coordtype = ModIntType(self._q)
        self._btype = BoolType()

    def to_register(self, o: AffMontgomeryPoint) -> BitVector:
        return (
            self._dtype.to_register(o.x)
            + self._dtype.to_register(o.y)
            + self._btype.to_register(o.not_infty)
        )

    def from_register(self, l: BitVector) -> AffMontgomeryPoint:
        not_infty = self._btype.from_register(l[2 * self._dlen :])  # -> qartonbool
        x = self._dtype.from_register(l[0 : self._dlen])
        y = self._dtype.from_register(l[self._dlen : (2 * self._dlen)])
        if not not_infty:
            if x != 0 or y != 0:
                raise DataTypeConversionError(
                    self, l, "If not_infty is set to 0, x and y should be 0"
                )
            return AffMontgomeryPoint.infinity(self.ec)
        else:
            return AffMontgomeryPoint(x=x, y=y, ec=self.ec)

    def register_length(self) -> int:
        return 2 * self._dlen + 1

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        """Redefinition of get_sub_register to access the following fields: "x", "y",
        "not_infty".

        :param r: Input register (AffMontgomeryPointType)
        :type r: Register
        :param key: Register key. Only recognized: "x", "y", "not_infty"
        :type key: RegisterKey
        :raises RegisterIndexError: If wrong key.
        :return: Sub-register.
        :rtype: Register
        """
        if key == "x":
            return qc_reg_cast(r[0 : self._dlen], datatype=self._coordtype)
        elif key == "y":
            return qc_reg_cast(
                r[self._dlen : (2 * self._dlen)], datatype=self._coordtype
            )
        elif key == "not_infty":
            return qc_reg_cast(r[2 * self._dlen :], datatype=self._btype)
        else:
            raise RegisterIndexError(key)

    def default_value(self) -> AffMontgomeryPoint:
        """Default value is the point at infinity, which corresponds to an all-0 string.

        :return: The point at infinity.
        :rtype: AffMontgomeryPoint
        """
        return AffMontgomeryPoint.infinity(self.ec)

    def iterate_values(self) -> Iterator[AffMontgomeryPoint]:
        """Iterate over all valid affine points.

        :yield: AffMontgomeryPoint
        :rtype: Iterator[AffMontgomeryPoint]
        """
        yield from _elliptic_curve_AffMontgomeryPoints(self.ec)

    def random_value(self) -> AffMontgomeryPoint:
        return AffMontgomeryPoint.random(self.ec)


class AffMontgomeryPointVariable(QartonCustomObject):
    __slots__ = "not_infty", "q", "x", "y"

    def __init__(self, x: int, y: int, not_infty: bool | QartonBool, q: int) -> None:
        self.x = UInt(x)
        self.y = UInt(y)
        self.not_infty = QartonBool(not_infty)
        self.q = q

    def is_on_curve(self, ec: ECMontgomery) -> bool:
        if not self.not_infty:
            return True  # infty point always on curve
        else:
            return (self.x**3 + ec.A * self.x**2 + self.x) % self.q == (
                ec.B * self.y**2
            ) % self.q

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.not_infty, self.q))

    def to_AffMontgomeryPoint(self, ec: ECMontgomery) -> AffMontgomeryPoint:
        if not self.not_infty:
            return AffMontgomeryPoint.infinity(ec)
        else:
            return AffMontgomeryPoint(self.x, self.y, ec)

    def __neg__(self) -> AffMontgomeryPointVariable:
        return AffMontgomeryPointVariable(
            self.x, (-self.y) % self.q, self.not_infty, self.q
        )

    @staticmethod
    def from_AffMontgomeryPoint(a: AffMontgomeryPoint) -> AffMontgomeryPointVariable:
        return AffMontgomeryPointVariable(a.x, a.y, a.not_infty, a.ec.q)

    def __str__(self) -> str:
        return (
            f"AffMontgomeryPointVariable({self.x},{self.y},{self.not_infty},{self.q})"
        )

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AffMontgomeryPointVariable):
            return False
        return (
            self.x == value.x
            and self.y == value.y
            and self.not_infty == value.not_infty
            and self.q == value.q
        )


def random_bool(p: float) -> QartonBool:
    return QartonBool(int(random.random() > p))


class AffMontgomeryPointVariableType(
    DataType[AffMontgomeryPointVariable], AutoEncodableObject
):
    """
    DataType for an affine point on a Montgomery elliptic curve, with variable curve (so it's just
    precisely a triple of x,y, not_infty, with constant prime q built-in in the type).
    """

    def __init__(self, q: int) -> None:
        super().__init__()
        self.q = q
        self._dlen = self.q.bit_length()
        self._dtype = UIntType(self._dlen)
        self._coordtype = ModIntType(self.q)
        self._btype = BoolType()

    def to_register(self, o: AffMontgomeryPointVariable) -> BitVector:
        return (
            self._dtype.to_register(o.x)
            + self._dtype.to_register(o.y)
            + self._btype.to_register(o.not_infty)
        )

    def from_register(self, l: BitVector) -> AffMontgomeryPointVariable:
        not_infty = self._btype.from_register(l[2 * self._dlen :])
        x = self._dtype.from_register(l[0 : self._dlen])
        y = self._dtype.from_register(l[self._dlen : (2 * self._dlen)])
        if not not_infty:
            if x != 0 or y != 0:
                raise DataTypeConversionError(
                    self, l, "If not_infty is set to 0, x and y should be 0"
                )
        return AffMontgomeryPointVariable(x=x, y=y, not_infty=not_infty, q=self.q)

    def register_length(self) -> int:
        return 2 * self._dlen + 1

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        """Redefinition of get_sub_register to access the following fields: "x", "y",
        "not_infty".

        :param r: Input register (AffMontgomeryPointType)
        :type r: Register
        :param key: Register key. Only recognized: "x", "y", "not_infty"
        :type key: RegisterKey
        :raises RegisterIndexError: If wrong key.
        :return: Sub-register.
        :rtype: Register
        """
        if key == "x":
            return qc_reg_cast(r[0 : self._dlen], datatype=self._coordtype)
        elif key == "y":
            return qc_reg_cast(
                r[self._dlen : (2 * self._dlen)], datatype=self._coordtype
            )
        elif key == "not_infty":
            return qc_reg_cast(r[2 * self._dlen :], datatype=self._btype)
        else:
            raise RegisterIndexError(key)

    def default_value(self) -> AffMontgomeryPointVariable:
        raise NotImplementedError()

    def iterate_values(self) -> Iterator[AffMontgomeryPointVariable]:
        raise NotImplementedError()

    def random_value(self) -> AffMontgomeryPointVariable:
        b = random_bool(1 / self.q)
        if b:  # not infty
            return AffMontgomeryPointVariable(
                int(self._coordtype.random_value()),
                int(self._coordtype.random_value()),
                b,
                self.q,
            )
        else:
            # infty: full 0
            return AffMontgomeryPointVariable(0, 0, b, self.q)
