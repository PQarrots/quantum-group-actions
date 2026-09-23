"""
Custom object and datatype for 4-dimensional affine points.

"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from qarton.circuit import (
    BitVector,
    DataType,
    QartonCustomObject,
    Register,
    RegisterKey,
    TupleType,
)

from qisogenies.montgomery import (
    AffMontgomeryPoint,
    AffMontgomeryPointType,
    ECMontgomery,
)

__all__ = ["AffPointDim4", "AffPointDim4Type"]


class AffPointDim4(QartonCustomObject):
    def __init__(
        self, *args: AffMontgomeryPoint | Iterable[AffMontgomeryPoint]
    ) -> None:
        l: list[AffMontgomeryPoint] = []
        for a in args:
            if isinstance(a, AffMontgomeryPoint):
                l.append(a)
            else:
                l += list(a)
        if len(l) != 4:
            raise ValueError("Should have 4 inputs")
        self._pts = tuple(l)

    def __str__(self) -> str:
        return f"AffPointDim4({self._pts!s})"

    def __repr__(self) -> str:
        return f"AffPointDim4({self._pts!s})"

    def __len__(self) -> int:
        return 4

    def __hash__(self) -> int:
        return hash(self._pts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffPointDim4):
            return False
        return self._pts == other._pts

    def __getitem__(self, key: int) -> AffMontgomeryPoint:
        return self._pts[key]

    def __iter__(self) -> Iterator[AffMontgomeryPoint]:
        yield from self._pts

    def __add__(self, other: AffPointDim4) -> AffPointDim4:
        return AffPointDim4(P + Q for P, Q in zip(self, other, strict=True))

    def __sub__(self, other: AffPointDim4) -> AffPointDim4:
        return AffPointDim4(P - Q for P, Q in zip(self, other, strict=True))

    def __mul__(self, n: int) -> AffPointDim4:
        return AffPointDim4(P * n for P in self)


class AffPointDim4Type(DataType[AffPointDim4]):
    """
    Datatype for a tuple of 4 affine Montgomery points on the same curve.
    """

    def __init__(self, ec: ECMontgomery) -> None:
        super().__init__()
        self.ec = ec
        self._pt_type = AffMontgomeryPointType(ec)
        self._pts_type = TupleType(4, self._pt_type)

    @property
    def tuple_length(self) -> int:
        return 4

    def to_register(self, o: AffPointDim4) -> BitVector:
        return self._pts_type.to_register(tuple(o))

    def from_register(self, l: BitVector) -> AffPointDim4:
        return AffPointDim4(self._pts_type.from_register(l))

    def register_length(self) -> int:
        return self._pts_type.register_length()

    def get_sub_register(self, r: Register, key: RegisterKey) -> Register:
        return self._pts_type.get_sub_register(r, key)

    def default_value(self) -> AffPointDim4:
        return AffPointDim4(self._pts_type.default_value())

    def random_value(self) -> AffPointDim4:
        return AffPointDim4(self._pts_type.random_value())

    def convert_class(self, t: type[DataType]) -> DataType:
        if t == TupleType:
            return self._pts_type
        return super().convert_class(t)

    def matches(self, other: DataType[Any]) -> bool:
        if self._pts_type.matches(other):
            return True
        return super().matches(other)
