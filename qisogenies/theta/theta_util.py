"""
Various useful functions and circuits for this package.
"""

from collections.abc import Iterable
from typing import Any, Self

from qarton.binary_operations import (
    qc_eq,
    qc_load,
    qc_mcx_neg,
    qc_swap,
    qc_unload,
    qc_xor,
)
from qarton.circuit import (
    BitVector,
    BitVectorType,
    BoolType,
    Circuit,
    CircuitParameterInvalid,
    DataType,
    InPlaceCircuit,
    InvolutoryCircuit,
    PreCircuit,
    QartonBool,
    Register,
    RegisterKey,
    TupleType,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_empty_reg,
    qc_reg_cast,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_add_modint,
    qc_dbl_modint,
    qc_invadd_modint,
    qc_invsub_modint,
    qc_muladd_modint,
    qc_mulsub_modint,
    qc_powadd_modint,
    qc_powsub_modint,
    qc_sqradd_modint,
    qc_sqrsub_modint,
    qc_sub_modint,
)

__all__ = [
    "CoordsDim4",
    "CoordsDim2",
    "CoordsDim4Type",
    "CoordsDim2Type",
    "coordwise",
    "qc_coordwise_add",
    "qc_coordwise_sub",
    "qc_coordwise_muladd",
    "qc_coordwise_mulsub",
    "qc_coordwise_sqradd",
    "qc_coordwise_sqrsub",
    "qc_coordwise_invadd",
    "qc_coordwise_invsub",
    "classical_hadamard",
    "ThetaHadamardBitflipped",
    "ThetaHadamard",
    "ThetaHadamardDim2",
    "qc_theta_hadamard_dim2",
    "BatchedInversionAdd",
    "qc_batched_inversion_add",
    "qc_batched_inversion_sub",
    "issquare_classical_function",
    "IsSquare",
    "is_square_modp",
    "qc_is_square",
    "qc_is_square_uncompute",
    "is_2_power",
    "mat_vec_product",
    "IsAnyCoordZero",
    "qc_is_any_coord_zero",
    "proj_equal",
    "balanced_strategy",
    "montgomerytojinvariantdiv256_classical_function",
    "MontgomeryToJInvariantDiv256",
]


class _FixedLengthModIntTuple(tuple[ModInt, ...]):
    """
    Base class for a tuple of `ModInt` with a fixed, enforced length.

    Behaves like a plain tuple (indexing, iteration, equality, hashing), but the
    constructor accepts a single iterable -- exactly like `tuple(...)` -- and
    raises a `ValueError` if it doesn't contain exactly `_LENGTH` elements.

    Also allows  the methods square, invert and mult which are used in both
    dimension 2 and 4 coordinates.
    """

    _LENGTH: int

    def __new__(cls, coords: Iterable[ModInt] = ()) -> Self:
        values = tuple(coords)
        if len(values) != cls._LENGTH:
            raise ValueError(
                f"{cls.__name__} needs exactly {cls._LENGTH} coordinates, "
                f"got {len(values)}"
            )
        return super().__new__(cls, values)

    def square(self) -> Self:
        """
        Return a new point where all the coordinates are squared.
        """
        return type(self)(xi**2 for xi in self)

    def invert(self) -> Self:
        """
        Return a new point where all coordinates are inverse.

        This is extended to the pseudo-inverse: 0 is inverted as 0.
        """
        return type(self)(xi ** (-1) if xi.v != 0 else xi for xi in self)

    def mult(self, other: Self) -> Self:
        """
        Return a new point where all coordinates are multiples of the coordinates of
        self, and other.
        """
        return type(self)(xi * yi for xi, yi in zip(self, other, strict=True))

    def hadamard(self) -> Self:
        """
        Return a new point resulting from applying a Hadamard transformation on this one.
        """
        return type(self)(classical_hadamard(self))


class CoordsDim4(_FixedLengthModIntTuple):
    """The 16 coordinates of a dimension-4 theta point."""

    _LENGTH = 16


class CoordsDim2(_FixedLengthModIntTuple):
    """The 4 coordinates of a dimension-2 theta point."""

    _LENGTH = 4


class CoordsDim4Type(DataType[CoordsDim4]):
    def __init__(self, p: int) -> None:
        """
        Datatype for dimension-4 coordinate - just a wrapper around a 16-element
        tuple type.

        :param p: Base prime
        :type p: int
        """
        super().__init__()
        self._p = p
        self._dlen = self._p.bit_length()
        self._dtype = ModIntType(self._p)
        self._coords_type = TupleType(16, self._dtype)

    @property
    def p(self) -> int:
        return self._p

    @property
    def tuple_length(self) -> int:
        return 16

    @property
    def element_type(self) -> DataType:
        return self._dtype

    def to_register(self, o: CoordsDim4) -> BitVector:
        return self._coords_type.to_register(o)

    def from_register(self, l: BitVector) -> CoordsDim4:
        coords = self._coords_type.from_register(l)
        return CoordsDim4(coords)

    def register_length(self) -> int:
        return self._coords_type.register_length()

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        return self._coords_type.get_sub_register(r, key)

    def random_value(self) -> CoordsDim4:
        coords = self._coords_type.random_value()
        return CoordsDim4(coords)

    def convert_class(self, t: type[DataType]) -> DataType:
        if t == TupleType:
            return self._coords_type
        return super().convert_class(t)

    def matches(self, other: DataType[Any]) -> bool:
        return other == self._coords_type or super().matches(other)


class CoordsDim2Type(DataType[CoordsDim2]):
    def __init__(self, p: int) -> None:
        """
        Datatype for dimension-2 coordinate - just a wrapper around a 4-element
        tuple type.

        :param p: Base prime
        :type p: int
        """
        super().__init__()
        self._p = p
        self._dlen = self._p.bit_length()
        self._dtype = ModIntType(self._p)
        self._coords_type = TupleType(4, self._dtype)

    @property
    def p(self) -> int:
        return self._p

    @property
    def tuple_length(self) -> int:
        return 4

    @property
    def element_type(self) -> DataType:
        return self._dtype

    def to_register(self, o: CoordsDim2) -> BitVector:
        return self._coords_type.to_register(o)

    def from_register(self, l: BitVector) -> CoordsDim2:
        coords = self._coords_type.from_register(l)
        return CoordsDim2(coords)

    def register_length(self) -> int:
        return self._coords_type.register_length()

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        return self._coords_type.get_sub_register(r, key)

    def random_value(self) -> CoordsDim2:
        coords = self._coords_type.random_value()
        return CoordsDim2(coords)

    def convert_class(self, t: type[DataType]) -> DataType:
        if t == TupleType:
            return self._coords_type
        return super().convert_class(t)

    def matches(self, other: DataType[Any]) -> bool:
        return other == self._coords_type or super().matches(other)


############# coordinate-wise operations #################


def coordwise(qc_arithmetic_fun: Any, length: int = 16) -> Any:
    def coordwise_apply_fun(*tuple_regs: Register) -> None:
        for m in range(length):
            qc_arithmetic_fun(*[tup.a[m] for tup in tuple_regs])

    return coordwise_apply_fun


qc_coordwise_add = coordwise(qc_add_modint)
"""
Performs coordinate-wise addition on 16-tuples x, y.
(x_i, y_i) -> (x_i, y_i + x_i)

:param x:
:type x: Register of type TupleType(16, ModIntType(p))
:param y:None
:type y: Register of type TupleType(16, ModIntType(p))
"""
qc_coordwise_sub = coordwise(qc_sub_modint)
"""
Performs coordinate-wise subtraction on 16-tuples x, y.
(x_i, y_i) -> (x_i, y_i - x_i)

:param x:
:type x: Register of type TupleType(16, ModIntType(p))
:param y:
:type y: Register of type TupleType(16, ModIntType(p))
"""
qc_coordwise_muladd = coordwise(qc_muladd_modint)
"""
Performs coordinate-wise out-of-place multiplication followed by addition on 16-tuples x, y, z.
(x_i, y_i, z_i) -> (x_i, y_i, z_i + x_i * y_i)

:param x:
:type x: Register of type TupleType(16, ModIntType(p))
:param y:
:type y: Register of type TupleType(16, ModIntType(p))
:param z:
:type z: Register of type TupleType(16, ModIntType(p))
"""
qc_coordwise_mulsub = coordwise(qc_mulsub_modint)
"""
Performs coordinate-wise out-of-place multiplication followed by subtraction on 16-tuples x, y, z.
(x_i, y_i, z_i) -> (x_i, y_i, z_i - x_i * y_i)

:param x:
:type x: Register of type TupleType(16, ModIntType(p))
:param y:
:type y: Register of type TupleType(16, ModIntType(p))
:param z:
:type z: Register of type TupleType(16, ModIntType(p))
"""
qc_coordwise_sqradd = coordwise(qc_sqradd_modint)
"""
Performs coordinate-wise out-of-place squaring followed by addition on 16-tuples x, y.
(x_i, y_i) -> (x_i, y_i + x_i**2)

:param x:
:type x: Register of type TupleType(16, ModIntType(p))
:param y:
:type y: Register of type TupleType(16, ModIntType(p))
"""
qc_coordwise_sqrsub = coordwise(qc_sqrsub_modint)
"""
Performs coordinate-wise out-of-place squaring followed by subtraction on 16-tuples x, y.
(x_i, y_i) -> (x_i, y_i - x_i**2)

:param x:
:type x: Register of type TupleType(16, ModIntType(p))
:param y:
:type y: Register of type TupleType(16, ModIntType(p))
"""

############# hadamard #######################


def _qc_hadamard_rec_swapped(coords_rec: Register) -> None:
    """
    Helper function for the Hadamard in-place circuits.
    Takes 2^n coordinates, computes the bitflipped-Hadamard transform
    by recursively computing it on tuples of half the length then merging

    :param coords_rec: tuple of 2^n coordinates
    :type coords_rec: Register of type TupleType(16, ModIntType(p))
    """
    qc = coords_rec.parent_circuit
    l = TupleType.get_from_register(coords_rec).tuple_length
    _d = TupleType.get_from_register(coords_rec).element_type
    assert isinstance(_d, ModIntType)
    p = _d.p

    assert l in [2, 4, 8, 16], f"{l}"  # sanity check
    if l == 2:
        a, b = coords_rec.a[0], coords_rec.a[1]
        qc_sub_modint(b, a)
        qc_dbl_modint(b)
        qc_add_modint(a, b)  # (a, b) <- (a-b, a+b)
        return

    mid = l // 2
    left = qc_reg_cast(
        sum((coords_rec.a[i] for i in range(mid)), qc_empty_reg(qc, BitVectorType(0))),
        TupleType(mid, ModIntType(p)),
    )
    right = qc_reg_cast(
        sum(
            (coords_rec.a[i] for i in range(mid, 2 * mid)),
            qc_empty_reg(qc, BitVectorType(0)),
        ),
        TupleType(mid, ModIntType(p)),
    )
    _qc_hadamard_rec_swapped(left)
    _qc_hadamard_rec_swapped(right)

    for j in range(mid - 1, -1, -1):
        a, b = left.a[j], right.a[j]
        qc_sub_modint(b, a)
        qc_dbl_modint(b)
        qc_add_modint(a, b)  # (a, b) <- (a-b, a+b)


def classical_hadamard(coords: tuple[ModInt, ...]) -> tuple[ModInt, ...]:
    """
    Helper function for the Hadamard in-place circuits.
    Takes 2^n coordinates, computes the bitflipped-Hadamard transform
    by recursively computing it on tuples of half the length then merging.

    :param coords_rec: tuple of 2^n coordinates
    :type coords_rec: tuple[ModInt, ...]
    """

    def _hadamard_rec(coords_rec: tuple[ModInt, ...]) -> tuple[ModInt, ...]:
        if len(coords_rec) == 1:
            return coords_rec

        mid = len(coords_rec) // 2
        left = coords_rec[:mid]
        right = coords_rec[mid:]

        left = _hadamard_rec(left)
        right = _hadamard_rec(right)

        return tuple(x + y for x, y in zip(left, right, strict=True)) + tuple(
            x - y for x, y in zip(left, right, strict=True)
        )

    return _hadamard_rec(coords)


@dummify
@memoize
class ThetaHadamardBitflipped(InPlaceCircuit[CoordsDim4]):
    """
    In-place circuit that computes the (bit-flipped) Hadamard map of x in the same register.
    Input: x = (x[i] | i in range(16))
    Output: y = ~Had(x), with y[j] = sum((-1)**<bitflip(j)|i> * x[i] for i in range(16)).
    """

    def __init__(
        self,
        p: int,
    ) -> None:
        super().__init__()

        if p % 2 == 0:
            raise CircuitParameterInvalid(self, p, "Needs p to be odd")
        self.p = p
        coords = self.add_input_output(CoordsDim4Type(p))

        _qc_hadamard_rec_swapped(coords)

    def dummy_classical_function(self, args: CoordsDim4) -> CoordsDim4:
        return CoordsDim4(args.hadamard()[::-1])

    def dummy_classical_function_inverse(self, args: CoordsDim4) -> CoordsDim4:
        # Hadamard is a matrix transformation H with H*H = 16 Id.
        # the function here is S*H, with S the bitflip swap of the coordinates.
        # So H^(-1) = H * S/ 16.
        coords = self.dummy_classical_function(CoordsDim4(args[::-1]))[::-1]  # H
        # a, b, c, d = (x * ModInt(pow(16, -1, self.p) % self.p, self.p)
        acoords = (x * ModInt(16, self.p) ** (-1) for x in coords)  # / 16
        return CoordsDim4(acoords)


@dummify
@memoize
class ThetaHadamard(InPlaceCircuit[CoordsDim4]):
    """
    In-place circuit that computes the Hadamard map of x in the same register.
    Input: x = (x[i] | i in range(16))
    Output: y = Had(x), with y[j] = sum((-1)**<j|i> * x[i] for i in range(16)).
    NOTE Had(Had(x)) = 16 * x
    """

    def __init__(
        self,
        p: int,
    ) -> None:
        super().__init__()

        if p % 2 == 0:
            raise CircuitParameterInvalid(self, p, "Needs p to be odd")
        self.p = p
        coords = self.add_input_output(CoordsDim4Type(p))

        self.append(PreCircuit(ThetaHadamardBitflipped, p), coords)
        for j in range(8):
            qc_swap(coords.a[j], coords.a[15 - j])

    def dummy_classical_function(self, args: CoordsDim4) -> CoordsDim4:
        return CoordsDim4(args.hadamard())

    def dummy_classical_function_inverse(self, args: CoordsDim4) -> CoordsDim4:
        # Hadamard is a matrix transformation H with H*H = 16 Id.
        # So H^(-1) = H / 16.
        coords = args.hadamard()  # H
        # a, b, c, d = (x * ModInt(pow(16, -1, self.p) % self.p, self.p)
        acoords = (x * ModInt(16, self.p) ** (-1) for x in coords)  # / 16
        return CoordsDim4(acoords)


@dummify
@memoize
class ThetaHadamardDim2(InPlaceCircuit[CoordsDim2]):
    """
    In-place circuit that computes the Hadamard map of x = (a, b, c, d) in the same register.
    Input: a, b, c, d
    Output: (a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d).
    NOTE Had(Had(x)) = 4 * (a, b, c, d)
    """

    def __init__(self, p: int) -> None:
        super().__init__()

        if p % 2 == 0:
            raise CircuitParameterInvalid(self, p, "Needs p to be odd")
        self.p = p
        coords = self.add_input_output(CoordsDim2Type(p))

        _qc_hadamard_rec_swapped(coords)
        for j in range(2):
            qc_swap(coords.a[j], coords.a[3 - j])

    def dummy_classical_function(self, args: CoordsDim2) -> CoordsDim2:
        return CoordsDim2(args.hadamard())

    def dummy_classical_function_inverse(self, args: CoordsDim2) -> CoordsDim2:
        # Hadamard is a matrix transformation H with H*H = 4 Id.
        # So H^(-1) = H / 4.
        coords = args.hadamard()  # H
        # a, b, c, d = (x * ModInt(pow(4, -1, self.p) % self.p, self.p)
        acoords = (x * ModInt(4, self.p) ** (-1) for x in coords)  # / 4

        return CoordsDim2(acoords)


def qc_theta_hadamard_dim2(x_reg: Register, inverse: bool = False) -> None:
    """Compute the Hadamard linear transformation of x_reg modulo p, in place.

    :param x_reg: Input register (CoordsDim2 type or TupleType of length 4)
    """
    qc = x_reg.parent_circuit
    tup_type = TupleType.get_from_register(x_reg)
    mod_type = tup_type.element_type
    assert isinstance(mod_type, ModIntType)
    p = mod_type.p
    if not inverse:
        qc.append(PreCircuit(ThetaHadamardDim2, p), x_reg)
    else:
        qc.append(PreCircuit(ThetaHadamardDim2, p, inverse=True), x_reg)


################ inversions #########################


@dummify
@memoize
class BatchedInversionAdd(
    InPlaceCircuit[tuple[tuple[ModInt, ...], tuple[ModInt, ...]]]
):
    def __init__(self, p: int, length: int = 16) -> None:
        """
        x, y are tuples of ModInts, of length n
        (x, y) -> (x, y + x**(-1))

        :param p: base prime
        :type p: int
        :param length: length of the tuples: how many elements are we inverting
        :type length: int
        """
        super().__init__()
        self.p = p
        self.length = length

        mod_int_type = ModIntType(p)

        x = self.add_input_output(TupleType(length, mod_int_type))
        y = self.add_input_output(TupleType(length, mod_int_type))

        prefix_products = self.add_anc(TupleType(length, mod_int_type))
        prefix_inverses = self.add_anc(TupleType(length, mod_int_type))

        # compute the products of tuple_reg.a[:i]
        qc_xor(x.a[0], prefix_products.a[0])
        for i in range(1, length):
            qc_muladd_modint(prefix_products.a[i - 1], x.a[i], prefix_products.a[i])

        # invert the last one (product of all the elements of the tuple)
        qc_invadd_modint(prefix_products.a[length - 1], prefix_inverses.a[length - 1])

        # go backwards with the prefix inverses: (prod(x.a[:i]))**(-1)
        for i in range(length - 2, -1, -1):
            qc_muladd_modint(prefix_inverses.a[i + 1], x.a[i + 1], prefix_inverses.a[i])

        # xi**(-1) = (x0 ... xi) * (x0 ... x_{i+1})**(-1)
        qc_add_modint(prefix_inverses.a[0], y.a[0])
        for i in range(1, length):
            qc_muladd_modint(prefix_products.a[i - 1], prefix_inverses.a[i], y.a[i])

        ### uncompute ancillas
        for i in range(length - 1):
            qc_mulsub_modint(prefix_inverses.a[i + 1], x.a[i + 1], prefix_inverses.a[i])
        qc_invsub_modint(prefix_products.a[length - 1], prefix_inverses.a[length - 1])
        for i in range(length - 1, 0, -1):
            qc_mulsub_modint(prefix_products.a[i - 1], x.a[i], prefix_products.a[i])
        qc_xor(x.a[0], prefix_products.a[0])

        for anc in (prefix_products, prefix_inverses):
            self.assert_anc(anc)
            self.release_anc(anc)

    def dummy_classical_function(
        self, args: tuple[tuple[ModInt, ...], tuple[ModInt, ...]]
    ) -> tuple[tuple[ModInt, ...], tuple[ModInt, ...]]:
        x, y = args
        return x, tuple(yi + xi ** (-1) for xi, yi in zip(x, y, strict=True))

    def dummy_classical_function_inverse(
        self, args: tuple[tuple[ModInt, ...], tuple[ModInt, ...]]
    ) -> tuple[tuple[ModInt, ...], tuple[ModInt, ...]]:
        x, y = args
        return x, tuple(yi - xi ** (-1) for xi, yi in zip(x, y, strict=True))


def qc_batched_inversion_add(tuple_reg: Register, out_reg: Register) -> None:
    t1 = TupleType.get_from_register(tuple_reg)
    t2 = TupleType.get_from_register(out_reg)
    if t1.tuple_length == 0 or t1.tuple_length != t2.tuple_length:
        raise TypeError("Wrong types")
    _d = t1.element_type
    assert isinstance(_d, ModIntType)
    p = _d.p
    l = t1.tuple_length

    qc = tuple_reg.parent_circuit
    qc.append(PreCircuit(BatchedInversionAdd, p, l), tuple_reg, out_reg)


def qc_batched_inversion_sub(tuple_reg: Register, out_reg: Register) -> None:
    t1 = TupleType.get_from_register(tuple_reg)
    t2 = TupleType.get_from_register(out_reg)
    if t1.tuple_length == 0 or t1.tuple_length != t2.tuple_length:
        raise TypeError("Wrong types")
    _d = t1.element_type
    assert isinstance(_d, ModIntType)
    p = _d.p
    l = t1.tuple_length

    qc = tuple_reg.parent_circuit
    qc.append(PreCircuit(BatchedInversionAdd, p, l, inverse=True), tuple_reg, out_reg)


qc_coordwise_invadd = qc_batched_inversion_add
qc_coordwise_invsub = qc_batched_inversion_sub


############### quadratic residuosity test ##############
# do Euler's test
# We could also use the Jacobi symbol circuit here, but this is
# non-dominating anyway.


def issquare_classical_function(p: int, x: ModInt) -> tuple[ModInt, QartonBool]:
    if x == 0:
        y = QartonBool(True)
    y = QartonBool(True) if p == 2 else QartonBool(x ** (p >> 1) == ModInt(1, p))
    return x, y


@dummify
@memoize
class IsSquare(Circuit[ModInt, tuple[ModInt, QartonBool]]):
    def __init__(self, p: int) -> None:
        super().__init__()
        self.p = p

        x = self.add_input_output(ModIntType(p))
        is_square = self.add_anc_output(BoolType())

        if p == 2:
            self.x(is_square[0])
        else:
            exp = qc_load(UInt(p >> 1), self, UIntType((p >> 1).bit_length()))
            result = self.add_anc(ModIntType(p))
            qc_powadd_modint(exp, x, result)

            qc_eq(ModInt(1, p), result, is_square)
            qc_powsub_modint(exp, x, result)

            self.assert_anc(result)
            self.release_anc(result)
            qc_unload(UInt(p >> 1), exp)

    def dummy_classical_function(self, args: ModInt) -> tuple[ModInt, QartonBool]:
        return issquare_classical_function(self.p, args)

    def dummy_classical_function_inverse(
        self, args: tuple[ModInt, QartonBool]
    ) -> ModInt:
        return args[0]


def is_square_modp(n: ModInt) -> QartonBool:
    return issquare_classical_function(n.p, n)[1]


def qc_is_square(x: Register) -> Register:
    if not isinstance(x.datatype, ModIntType):
        raise TypeError()
    qc = x.parent_circuit
    p = x.datatype.p
    return qc.append(PreCircuit(IsSquare, p), x)[1]


def qc_is_square_uncompute(x: Register, result: Register) -> None:
    if not isinstance(x.datatype, ModIntType) or not isinstance(
        result.datatype, BoolType
    ):
        raise TypeError()
    qc = x.parent_circuit
    p = x.datatype.p
    qc.append(PreCircuit(IsSquare, p, inverse=True), x, result)


################ misc ##################


def is_2_power(n: int) -> bool:
    return n == 1 << (n.bit_length() - 1)


def mat_vec_product(M: list[list[int]], x: tuple[ModInt, ...]) -> tuple[ModInt, ...]:
    n = len(x)
    m = len(M)
    assert n == len(M[0])
    p = x[0].p
    y = [
        ModInt(0, p),
    ] * m
    for i in range(m):
        for j in range(n):
            if M[i][j] == 0:
                pass
            elif M[i][j] == 1:
                y[i] += x[j]
            elif M[i][j] == -1:
                y[i] -= x[j]
            else:
                raise ValueError("currently only supported M with entries 0, +1, -1")
    return tuple(y)


@dummify
@memoize
class IsAnyCoordZero(InvolutoryCircuit[tuple[tuple[ModInt, ...], QartonBool]]):
    """
    Involutory circuit to check, given a vector x, whether (x[i] for i in which_coords) contains a zero.
    Xor the boolean result into the output register c.

    Input:
    - x a register of datatype TupleType(n, ModInt(p))
    - c a register of datatype BoolType()
    """

    def __init__(self, p: int, length: int, which_coords: tuple[int, ...]) -> None:
        super().__init__()
        self.p = p
        self.length = length
        self.which_coords = which_coords
        n = len(self.which_coords)

        x = self.add_input_output(TupleType(length, ModIntType(p)))
        c = self.add_input_output(BoolType())

        zeros = self.add_anc(TupleType(n, BoolType()))

        for z_idx, j in enumerate(which_coords):
            qc_mcx_neg(x.a[j], zeros.a[z_idx])
            # qc_eq(ModInt(0, p), x.a[j], zeros.a[z_idx])
        # zeros[i] = (x[which_coords[i]] == 0)

        qc_mcx_neg(zeros, c)  # c = c ^ (zeros == 0)
        self.x(c[0])  # c = c ^ (any(zeros[i] != 0))

        ######## uncompute
        for z_idx, j in enumerate(which_coords):
            qc_mcx_neg(x.a[j], zeros.a[z_idx])
            # qc_eq(ModInt(0, p), x.a[j], zeros.a[z_idx])
        self.assert_anc(zeros)
        self.release_anc(zeros)

    def dummy_classical_function(
        self, args: tuple[tuple[ModInt, ...], QartonBool]
    ) -> tuple[tuple[ModInt, ...], QartonBool]:
        x, c = args
        return x, QartonBool(c ^ any(x[i] == 0 for i in self.which_coords))


def qc_is_any_coord_zero(
    x_reg: Register, c_reg: Register, which_coords: Iterable[int] = range(16)
) -> None:
    qc = x_reg.parent_circuit
    which_coords = tuple(which_coords)
    t = TupleType.get_from_register(x_reg)
    length = t.tuple_length
    _d = t.element_type
    assert isinstance(_d, ModIntType)
    p = _d.p
    qc.append(IsAnyCoordZero(p, length, which_coords), x_reg, c_reg)


def proj_equal(vec1: Any, vec2: Any) -> Any:
    if all(x == 0 for x in vec1):
        return all(x == 0 for x in vec2)
    nz = next(i for i in range(len(vec2)) if vec2[i] != 0)
    factor = vec1[nz] * (vec2[nz]) ** (-1)
    # print(f"{factor = }")
    return all(x == y * factor for x, y in zip(vec1, vec2, strict=True))


def balanced_strategy(n: int) -> tuple[int, ...]:
    if n < 1:
        raise ValueError(f"{n} must be positive")
    if n == 1:
        return tuple()
    return (
        (n + 1) // 2,
        *balanced_strategy(n - (n + 1) // 2),
        *balanced_strategy((n + 1) // 2),
    )


def montgomerytojinvariantdiv256_classical_function(
    p: int, A: ModInt
) -> tuple[ModInt, ModInt]:
    return A, (A**2 - ModInt(3, p)) ** 3 * (A**2 - ModInt(4, p)) ** (-1)


@dummify
@memoize
class MontgomeryToJInvariantDiv256(Circuit[ModInt, tuple[ModInt, ModInt]]):
    def __init__(self, p: int) -> None:
        super().__init__()
        self.p = p

        A_coeff = self.add_input_output(ModIntType(p))
        J_inv = self.add_anc_output(ModIntType(p))
        A2m3, A2m3_squared, A2m3_cubed, A2m4, A2m4_inverse = self.add_ancs(
            *([ModIntType(p)] * 5)
        )

        # compute A^2 - 3, A^2 - 4
        qc_sqradd_modint(A_coeff, A2m3)
        qc_xor(A2m3, A2m4)
        qc_sub_modint(ModInt(3, p), A2m3)
        qc_sub_modint(ModInt(4, p), A2m4)

        # compute (A^2 - 3)^3, (A^2 - 4)^(-1)
        qc_sqradd_modint(A2m3, A2m3_squared)
        qc_muladd_modint(A2m3, A2m3_squared, A2m3_cubed)
        qc_invadd_modint(A2m4, A2m4_inverse)

        # output J/256 = (A^2 - 3)^3 / (A^2 - 4)
        qc_muladd_modint(A2m3_cubed, A2m4_inverse, J_inv)

        ########## uncompute
        qc_invsub_modint(A2m4, A2m4_inverse)
        qc_mulsub_modint(A2m3, A2m3_squared, A2m3_cubed)
        qc_sqrsub_modint(A2m3, A2m3_squared)
        qc_add_modint(ModInt(4, p), A2m4)
        qc_add_modint(ModInt(3, p), A2m3)
        qc_xor(A2m3, A2m4)
        qc_sqrsub_modint(A_coeff, A2m3)
        self.release_anc(A2m3, A2m3_squared, A2m3_cubed, A2m4, A2m4_inverse)

    def dummy_classical_function(self, args: ModInt) -> tuple[ModInt, ModInt]:
        return montgomerytojinvariantdiv256_classical_function(self.p, args)

    def dummy_classical_function_inverse(self, args: tuple[ModInt, ModInt]) -> ModInt:
        return args[0]
