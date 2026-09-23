"""
Operations and structures in Theta coordinates.
"""

from __future__ import annotations

from collections.abc import Iterator

from qarton.binary_operations import (
    qc_copy,
    qc_copy_erase,
    qc_xor,
)
from qarton.circuit import (
    BitVector,
    Circuit,
    DataType,
    PreCircuit,
    QartonCustomObject,
    Register,
    RegisterKey,
    TupleType,
    dummify,
    memoize,
    qc_reg_cast,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
)

from qisogenies.montgomery import (
    qc_add_affmontgomerypoint,
    qc_sub_affmontgomerypoint,
)
from qisogenies.montgomery.montgomery_add import (
    qc_add_affmontgomerypoint_var,
    qc_sub_affmontgomerypoint_var,
)
from qisogenies.montgomery.montgomery_curve import (
    AffMontgomeryPointType,
    AffMontgomeryPointVariableType,
)

from .theta_util import (
    CoordsDim4,
    CoordsDim4Type,
    ThetaHadamard,
    ThetaHadamardBitflipped,
    qc_coordwise_invadd,
    qc_coordwise_invsub,
    qc_coordwise_muladd,
    qc_coordwise_mulsub,
    qc_coordwise_sqradd,
    qc_coordwise_sqrsub,
)

__all__ = [
    "ThetaStructureDim4",
    "ThetaStructureDim4Type",
    "ThetaPointDim4",
    "ThetaPointDim4Type",
    "qc_theta_hadamard_bitflipped",
    "qc_theta_hadamard",
    "ThetaDouble",
    "qc_double_iter_theta",
    "qc_double_iter_prod_mont",
    "qc_double_iter_prod_mont_var",
    "ThetaArithmeticPrecomputation",
]


class ThetaStructureDim4(QartonCustomObject):
    """
    Class for a 4-dimensional theta structure over a prime field GF(p).
    """

    def __init__(self, null_point_coords: tuple[ModInt, ...] | CoordsDim4) -> None:
        if len(null_point_coords) != 16:
            raise ValueError(f"expected 16 coordinates, got {len(null_point_coords)}")
        null_point_coords = CoordsDim4(null_point_coords)
        p = null_point_coords[0].p
        if p % 2 == 0:
            raise ValueError("even base characteristic p not supported")

        self._np_coords = null_point_coords
        self._p = null_point_coords[0].p

        self._inv_np = null_point_coords.invert()

        self._inv_dual_np_codomain = null_point_coords.square().hadamard().invert()

    @classmethod
    def from_inv_dual_null_point(
        cls, inv_dual_np: CoordsDim4 | tuple[ModInt, ...]
    ) -> ThetaStructureDim4:
        np = CoordsDim4(inv_dual_np).invert().hadamard()
        return ThetaStructureDim4(null_point_coords=np)

    @property
    def p(self) -> int:
        return self._p

    @property
    def coords(self) -> CoordsDim4:
        return self._np_coords

    @property
    def null_point(self) -> CoordsDim4:
        return self._np_coords

    @property
    def inverse_null_point(self) -> CoordsDim4:
        return CoordsDim4(self._inv_np)

    @property
    def inverse_dual_null_point_codomain(self) -> CoordsDim4:
        return self._inv_dual_np_codomain

    def __len__(self) -> int:
        return len(self._np_coords)

    def __hash__(self) -> int:
        return hash(self._np_coords)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThetaStructureDim4):
            return False
        return self._np_coords == other._np_coords

    def __repr__(self) -> str:
        return str(self._np_coords)

    def __str__(self) -> str:
        return str(self._np_coords)


class ThetaStructureDim4Type(DataType[ThetaStructureDim4]):
    """
    Datatype for a 4-dimensional level-2 theta structure over GF(p)
    given by the 16 projective coordinates of its theta null point.

    The prime is a fixed parameter of the datatype.
    """

    def __init__(self, p: int) -> None:
        super().__init__()
        self._p = p
        self._dtype = ModIntType(p)
        self._dlen = self._dtype.register_length()
        self._coords_type = TupleType(16, self._dtype)
        self._length = 16  # for compatiblity with TupleType

    def to_register(self, o: ThetaStructureDim4) -> BitVector:
        return self._coords_type.to_register(o.null_point)

    def from_register(self, l: BitVector) -> ThetaStructureDim4:
        null_point = self._coords_type.from_register(l)
        return ThetaStructureDim4(null_point)

    def register_length(self) -> int:
        return self._coords_type.register_length()

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        return self._coords_type.get_sub_register(r, key)


class ThetaPointDim4(QartonCustomObject):
    _parent: ThetaStructureDim4 | None
    coords: CoordsDim4
    p: int
    __slots__ = "_parent", "coords", "p"

    def __init__(
        self, coords: CoordsDim4, parent: ThetaStructureDim4 | None = None
    ) -> None:
        if len(coords) != 16:
            raise ValueError(f"expected 16 coordinates, got {len(coords)}")

        self.coords = coords
        self._parent = parent
        self.p = coords[0].p

    def set_parent(self, parent: ThetaStructureDim4) -> None:
        self._parent = parent

    @property
    def parent(self) -> ThetaStructureDim4:
        if self._parent is None:
            raise AttributeError("parent not yet set. Use method set_parent")
        return self._parent

    def __len__(self) -> int:
        return len(self.coords)

    def __iter__(self) -> Iterator[ModInt]:
        return iter(self.coords)

    def __getitem__(self, key: int) -> ModInt:
        return self.coords[key]

    def __hash__(self) -> int:
        return hash(self.coords)

    def double(self) -> ThetaPointDim4:
        """
        Compute P -> [2] P.
        """

        # TODO make sure arithmetic precomputation is done on the parent when the parent is created
        P_coords = self.coords

        U_chi_P = P_coords.square().hadamard().square()
        U_chi_P = U_chi_P.mult(self.parent.inverse_dual_null_point_codomain)
        theta_2P = (U_chi_P).hadamard()
        theta_2P = theta_2P.mult(self.parent.inverse_null_point)

        return ThetaPointDim4(theta_2P, self.parent)

    def double_iter(self, e: int) -> ThetaPointDim4:
        """
        Computes P -> [2**e] P  by applying e times doubling to P.

        :param e: exponent, required >= 0
        :type e: int

        :rtype: ThetaPointDim4
        """
        if e < 0:
            raise ValueError(f"exponent e must be nonnegative, got {e}")
        pp = self
        for _ in range(e):
            pp = pp.double()
        return pp

    def __mul__(self, m: int) -> ThetaPointDim4:
        m = abs(m)
        if m == 0:
            return ThetaPointDim4(CoordsDim4(self.parent.null_point), self.parent)
        if m != 1 << (m.bit_length() - 1):  # if m is not a power of 2
            raise NotImplementedError(
                "Can only multiply by 0 or powers of 2, since differential addition is not available"
            )
        else:
            return self.double_iter(m.bit_length() - 1)

    def __rmul__(self, m: int) -> ThetaPointDim4:
        return self * m

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThetaPointDim4):
            return False
        return self.coords == other.coords

    def __str__(self) -> str:
        return f"Theta point {self.coords}"

    def __repr__(self) -> str:
        return str(self)


class ThetaPointDim4Type(DataType[ThetaPointDim4]):
    def __init__(self, p: int) -> None:
        """
        Datatype for an affine theta point on a variable theta structure over GF(p)
        (where the prime p is built-in in the type)

        :param p: Base prime
        :type p: int
        """
        super().__init__()
        self._p = p
        self._dlen = self._p.bit_length()
        self._dtype = ModIntType(self._p)
        self._coords_type = CoordsDim4Type(self._p)

    @property
    def p(self) -> int:
        return self._p

    @property
    def tuple_length(self) -> int:
        return 16

    @property
    def element_type(self) -> DataType:
        return self._dtype

    def to_register(self, o: ThetaPointDim4) -> BitVector:
        return self._coords_type.to_register(o.coords)

    def from_register(self, l: BitVector) -> ThetaPointDim4:
        coords = self._coords_type.from_register(l)
        return ThetaPointDim4(CoordsDim4(coords))

    def register_length(self) -> int:
        return self._coords_type.register_length()

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        return self._coords_type.get_sub_register(r, key)

    def random_value(self) -> ThetaPointDim4:
        coords = self._coords_type.random_value()
        return ThetaPointDim4(CoordsDim4(coords))

    def convert_class(self, t: type[DataType]) -> DataType:
        if t == TupleType:
            return self._coords_type
        return super().convert_class(t)


def qc_theta_hadamard_bitflipped(x_reg: Register, inverse: bool = False) -> None:
    """Compute the bit-flipped Hadamard linear transformation of x_reg modulo p, in place.

    :param x_reg: Input register (TupleType(16, ModIntType(p))).
    """
    qc = x_reg.parent_circuit
    assert isinstance(x_reg.datatype, (ThetaPointDim4Type, CoordsDim4Type))
    if isinstance(x_reg.datatype, ThetaPointDim4Type):
        x_reg = qc_reg_cast(x_reg, x_reg.datatype._coords_type)  # type: ignore

    dt = TupleType.get_from_register(x_reg)
    et = dt.element_type
    if not isinstance(et, ModIntType):
        raise TypeError()
    p = et.p
    if not inverse:
        qc.append(PreCircuit(ThetaHadamardBitflipped, p), x_reg)
    else:
        qc.append(PreCircuit(ThetaHadamardBitflipped, p, inverse=True), x_reg)


def qc_theta_hadamard(x_reg: Register, inverse: bool = False) -> None:
    """Compute the 4-dimensional (16-by-16) Hadamard linear transformation of x_reg modulo p, in place.

    :param x_reg: Input register (TupleType(16, ModIntType(p))).
    """
    qc = x_reg.parent_circuit
    if isinstance(x_reg.datatype, ThetaPointDim4Type):
        x_reg = qc_reg_cast(x_reg, x_reg.datatype._coords_type)  # type: ignore
    dt = TupleType.get_from_register(x_reg)
    et = dt.element_type
    if not isinstance(et, ModIntType):
        raise TypeError()
    p = et.p
    if not inverse:
        qc.append(PreCircuit(ThetaHadamard, p), x_reg)
    else:
        qc.append(PreCircuit(ThetaHadamard, p, inverse=True), x_reg)


@dummify
@memoize
class ThetaDouble(
    Circuit[
        tuple[ThetaPointDim4, CoordsDim4, CoordsDim4],
        tuple[ThetaPointDim4, CoordsDim4, CoordsDim4, ThetaPointDim4],
    ]
):
    def __init__(self, p: int) -> None:
        super().__init__()
        self.p = p
        point_type = ThetaPointDim4Type(p)
        coords_type = CoordsDim4Type(p)

        pp = self.add_input_output(point_type)
        inv_np = self.add_input_output(coords_type)
        inv_dual_np = self.add_input_output(coords_type)
        out = self.add_anc_output(point_type)

        had_sq, sq_had_sq, had_shs_uinv = (self.add_anc(coords_type) for _ in range(3))

        qc_coordwise_sqradd(pp, had_sq)
        qc_theta_hadamard(had_sq)

        qc_coordwise_sqradd(had_sq, sq_had_sq)

        qc_coordwise_muladd(sq_had_sq, inv_dual_np, had_shs_uinv)
        qc_theta_hadamard(had_shs_uinv)
        qc_coordwise_muladd(had_shs_uinv, inv_np, out)

        ## uncompute the intermediate registers: do everything above but in reverse
        qc_theta_hadamard(had_shs_uinv, inverse=True)
        qc_coordwise_mulsub(sq_had_sq, inv_dual_np, had_shs_uinv)
        qc_coordwise_sqrsub(had_sq, sq_had_sq)
        qc_theta_hadamard(had_sq, inverse=True)
        qc_coordwise_sqrsub(pp, had_sq)

        for reg in (had_sq, sq_had_sq, had_shs_uinv):
            self.assert_anc(reg)
            self.release_anc(reg)

    def dummy_classical_function(
        self, args: tuple[ThetaPointDim4, CoordsDim4, CoordsDim4]
    ) -> tuple[ThetaPointDim4, CoordsDim4, CoordsDim4, ThetaPointDim4]:
        P, inv_np, inv_dual_np = args
        theta_structure = ThetaStructureDim4(inv_np.invert())
        P.set_parent(theta_structure)

        return P, inv_np, inv_dual_np, P.double()

    def dummy_classical_function_inverse(
        self, args: tuple[ThetaPointDim4, CoordsDim4, CoordsDim4, ThetaPointDim4]
    ) -> tuple[ThetaPointDim4, CoordsDim4, CoordsDim4]:
        return args[:-1]


def qc_double_iter_theta(
    pp: Register, inv_np: Register, inv_dual_np: Register, e: int
) -> Register:
    qc = pp.parent_circuit
    dt = ThetaPointDim4Type.get_from_register(pp)
    p = dt.p
    double_circ = PreCircuit(ThetaDouble, p)
    doubles = [pp]
    for _ in range(e):
        pp = qc.append(double_circ, pp, inv_np, inv_dual_np)[-1]
        doubles.append(pp)
    ret = doubles[-1]
    for i in range(e - 1, 0, -1):
        qc.append(
            double_circ.inverse(), doubles[i - 1], inv_np, inv_dual_np, doubles[i]
        )
    return ret


def qc_double_iter_prod_mont(PP: Register, e: int) -> Register:
    """
    Multiply the point PP by (2^e).

    PP must be a tuple of AffMontgomeryPoint for a fixed curve (not variable).
    """
    qc = PP.parent_circuit
    dt = TupleType.get_from_register(PP)
    et = dt.element_type
    assert isinstance(et, AffMontgomeryPointType)

    (out,) = qc.add_ancs(PP.datatype)

    for j in range(dt.tuple_length):
        doubles = [PP.a[j]]
        for _ in range(e):
            pp = doubles[-1]
            twoP = qc_copy(pp)
            qc_add_affmontgomerypoint(pp, twoP)
            doubles.append(twoP)
        qc_xor(doubles[-1], out.a[j])

        # uncompute
        for i in range(e, 0, -1):
            pp, twoP = doubles[i - 1], doubles[i]
            qc_sub_affmontgomerypoint(pp, twoP)
            qc_copy_erase(pp, twoP)
    return out


def qc_double_iter_prod_mont_var(PP: Register, ec: Register, e: int) -> Register:
    """
    Multiply the point PP by (2^e), variable curve.

    PP must be a tuple of AffMontgomeryPointVar for a variable curve (given as input).
    """

    qc = PP.parent_circuit
    dt = TupleType.get_from_register(PP)
    et = dt.element_type
    assert isinstance(et, AffMontgomeryPointVariableType)

    (out,) = qc.add_ancs(PP.datatype)

    for j in range(dt.tuple_length):
        doubles = [PP.a[j]]
        for _ in range(e):
            pp = doubles[-1]
            twoP = qc_copy(pp)
            qc_add_affmontgomerypoint_var(pp, twoP, ec)
            doubles.append(twoP)
        qc_xor(doubles[-1], out.a[j])

        # uncompute
        for i in range(e, 0, -1):
            pp, twoP = doubles[i - 1], doubles[i]
            qc_sub_affmontgomerypoint_var(pp, twoP, ec)
            qc_copy_erase(pp, twoP)
    return out


@dummify
@memoize
class ThetaArithmeticPrecomputation(
    Circuit[CoordsDim4, tuple[CoordsDim4, CoordsDim4, CoordsDim4]]
):
    """
    Given the inverse coordinates of the dual null point
    of a theta structure A over GF(p),
    compute the inverse coordinates of the null point
    and the inverse coordinates of the dual null point of B

    where phi : A -> B is the 2-isogeny with kernel K2
    given by the canonical symplectic decomposition A[2] = K1 ⊕ K2. (????)
    """

    def __init__(self, p: int) -> None:
        """
        Input: inverse of the dual theta null point

        :param p: prime characteristic
        :type p: int
        """
        super().__init__()
        self.p = p
        point_type = ThetaPointDim4Type(p)
        coords_type = point_type._coords_type  # type: ignore

        inv_dual_null_point = self.add_input_output(coords_type)
        inv_np = self.add_anc_output(coords_type)
        inv_dual_np_codomain = self.add_anc_output(coords_type)

        dual_null_point, squared_null_point = self.add_ancs(*([coords_type] * 2))
        qc_coordwise_invadd(inv_dual_null_point, dual_null_point)

        qc_theta_hadamard(dual_null_point)
        null_point = dual_null_point

        qc_coordwise_sqradd(null_point, squared_null_point)
        qc_theta_hadamard(squared_null_point)

        coords_type_twice = TupleType(
            coords_type.tuple_length * 2, coords_type.element_type
        )
        qc_coordwise_invadd(
            qc_reg_cast(null_point + squared_null_point, coords_type_twice),
            qc_reg_cast(inv_np + inv_dual_np_codomain, coords_type_twice),
        )

        ## uncompute
        qc_theta_hadamard(squared_null_point, inverse=True)
        qc_coordwise_sqrsub(null_point, squared_null_point)
        qc_theta_hadamard(dual_null_point, inverse=True)
        qc_coordwise_invsub(inv_dual_null_point, null_point)

        for anc in (dual_null_point, squared_null_point):
            self.assert_anc(anc)
            self.release_anc(anc)

    def dummy_classical_function(
        self, args: CoordsDim4
    ) -> tuple[CoordsDim4, CoordsDim4, CoordsDim4]:
        inv_dual_np = args
        assert isinstance(inv_dual_np, CoordsDim4)
        np = inv_dual_np.invert().hadamard()
        # print(f"{np = }")
        inv_np = np.invert()
        # print(f"{inv_np = }")
        inv_dual_np_codomain = np.square().hadamard().invert()

        return (inv_dual_np, inv_np, inv_dual_np_codomain)

    def dummy_classical_function_inverse(
        self, args: tuple[CoordsDim4, CoordsDim4, CoordsDim4]
    ) -> CoordsDim4:
        return args[0]
