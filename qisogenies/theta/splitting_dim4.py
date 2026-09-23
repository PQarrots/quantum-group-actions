"""
Computing the splitting.
"""

from typing import Any

from qarton.binary_operations import (
    qc_cswap,
    qc_cxor,
    qc_mcx_neg,
    qc_swap,
)
from qarton.circuit import (
    BoolType,
    Circuit,
    QartonBool,
    Register,
    TupleType,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_add_modint,
    qc_cneg_modint,
    qc_dbl_modint,
    qc_halve_modint,
    qc_muladd_modint,
    qc_mulsub_modint,
    qc_neg_modint,
    qc_sqradd_modint,
    qc_sqrsub_modint,
    qc_sub_modint,
)

from .theta_dim4 import (
    ThetaStructureDim4,
    ThetaStructureDim4Type,
)
from .theta_util import (
    is_square_modp,
    mat_vec_product,
    qc_batched_inversion_add,
    qc_batched_inversion_sub,
    qc_is_square,
    qc_is_square_uncompute,
)

__all__ = [
    "is_product",
    "qc_diffsum",
    "qc_diffsum_inv",
    "qc_isnonzero",
    "splittingdim4_classical_function",
    "SplittingDim4",
]


def is_product(null_point: Any, theta1: Any, theta2: Any) -> Any:
    null_point_dim2: list[Any] = []
    for i in range(4):
        null_point_dim2.append(null_point[4 * i])

    null_point_bis = [None for _ in range(16)]
    for i0 in range(2):
        for i1 in range(2):
            for i3 in range(4):
                null_point_bis[i0 + 2 * i1 + 4 * i3] = (
                    theta1[i0] * theta2[i1] * null_point_dim2[i3]
                )

    res = True
    for i in range(16):
        for j in range(i):
            res = res and (
                null_point[i] * null_point_bis[j] == null_point[j] * null_point_bis[i]
            )

    return res


def qc_diffsum(a: Register, b: Register, swap: bool = False) -> None:
    if any(not isinstance(reg.datatype, ModIntType) for reg in (a, b)):
        raise TypeError()
    qc_sub_modint(b, a)
    qc_dbl_modint(b)
    qc_add_modint(a, b)  # (a, b) <- (a-b, a+b)
    if swap:
        qc_swap(a, b)


def qc_diffsum_inv(a: Register, b: Register, swap: bool = False) -> None:
    if any(not isinstance(reg.datatype, ModIntType) for reg in (a, b)):
        raise TypeError()
    if swap:
        qc_swap(a, b)
    qc_sub_modint(a, b)
    qc_halve_modint(b)
    qc_add_modint(b, a)


def qc_isnonzero(a: Register, c: Register) -> None:
    if not isinstance(a.datatype, ModIntType) or len(c) != 1:
        raise TypeError()
    qc = a.parent_circuit
    qc_mcx_neg(a, c)
    qc.x(c[0])


def splittingdim4_classical_function(
    p: int, args: tuple[ThetaStructureDim4, UInt, QartonBool]
) -> tuple[ThetaStructureDim4, UInt, QartonBool, ModInt]:
    theta_np, splitting_type, norm_is_even = args

    if splitting_type == 0:
        nn = [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ]
    elif splitting_type == 1:
        nn = [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
        ]
    elif splitting_type == 2:
        nn = [
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1, -1],
        ]
    else:
        raise Exception("Impossible case")

    prod_np = mat_vec_product(nn, theta_np.coords)

    i0 = 0
    for i in range(4):
        if prod_np[4 * i] != 0:
            i0 = i
    a1, b1 = prod_np[4 * i0], prod_np[4 * i0 + 1]
    if norm_is_even:
        a1, b1 = prod_np[4 * i0], prod_np[4 * i0 + 2]

    # check = False
    # # for debugging purposes:
    # if check:
    #     a2, b2 = prod_np[4 * i0], prod_np[4 * i0 + 2]
    #     assert is_product(prod_np, (a1, b1), (a2, b2))

    def np_to_montgomery(a: ModInt, b: ModInt) -> ModInt:
        return -(a**4 + b**4) * (a**4 - b**4) ** (-1) * 2

    aa = np_to_montgomery(a1, b1)
    Abis = np_to_montgomery(a1 + b1, a1 - b1)

    if is_square_modp(aa + ModInt(2, p)):
        aa = Abis

    return *args, aa


@dummify
@memoize
class SplittingDim4(
    Circuit[
        tuple[ThetaStructureDim4, UInt, QartonBool],
        tuple[ThetaStructureDim4, UInt, QartonBool, ModInt],
    ]
):
    def __init__(self, p: int) -> None:
        super().__init__()
        self.p = p

        # i/o
        codom_np = self.add_input_output(ThetaStructureDim4Type(p))
        splitting_type = self.add_input_output(UIntType(2))
        norm_is_even = self.add_input_output(BoolType())

        case1 = qc_reg_from_bits(splitting_type[0])

        out_curve = self.add_anc_output(ModIntType(p))

        # apply matrix case 0. Note: we only care about components 4*i + 0, 4*i + 1
        for i in range(8):
            qc_diffsum(codom_np.a[2 * i], codom_np.a[2 * i + 1])
            if i in (0, 1, 4, 5):
                qc_swap(codom_np.a[2 * i], codom_np.a[2 * i + 1])
        for i in range(4):
            qc_cneg_modint(case1, codom_np.a[4 * i + 3])
        for i in (0, 1):
            qc_add_modint(codom_np.a[4 * i + 2], codom_np.a[4 * i + 0])
            qc_add_modint(codom_np.a[4 * i + 3], codom_np.a[4 * i + 1])
        for i in (2, 3):
            qc_sub_modint(codom_np.a[4 * i + 2], codom_np.a[4 * i + 0])
            qc_sub_modint(codom_np.a[4 * i + 3], codom_np.a[4 * i + 1])

        # more swaps to select the correct components from the product theta null point:
        # make sure the first component of codom_np is nonzero.
        # The first two components (if the ideal norm was odd) or 1st and 3rd (if the ideal norm was even)
        #   will be the theta null point of the output curve!
        c4i_isnonzero = self.add_anc(TupleType(4, BoolType()))
        for i in range(4):
            qc_isnonzero(codom_np.a[4 * i], c4i_isnonzero.a[i])
            qc_cswap(
                norm_is_even, codom_np.a[4 * i + 1], codom_np.a[4 * i + 2]
            )  # swap codomain curves Ea and Eabar if acting with an ideal of even norm
        for i in range(1, 4):
            qc_cswap(c4i_isnonzero.a[i], codom_np.a[0], codom_np.a[4 * i + 0])
            qc_cswap(c4i_isnonzero.a[i], codom_np.a[1], codom_np.a[4 * i + 1])

        # theta_to_montgomery, either from (a, b) or from (a+b, a-b)
        asq, bsq, a4, b4, csq, dsq, c4, d4, amb4_inv, cmd4_inv, A, Abis = self.add_ancs(
            *([ModIntType(p)] * 12)
        )
        a, b = codom_np.a[0], codom_np.a[1]
        qc_sqradd_modint(a, asq)
        qc_sqradd_modint(b, bsq)
        qc_sqradd_modint(asq, a4)
        qc_sqradd_modint(bsq, b4)
        qc_diffsum(a4, b4)  # now they're a4 - b4, a4 + b4

        qc_diffsum(a, b, swap=True)  # replace (a, b) with (a+b, a-b)
        qc_sqradd_modint(a, csq)
        qc_sqradd_modint(b, dsq)
        qc_sqradd_modint(csq, c4)
        qc_sqradd_modint(dsq, d4)
        qc_diffsum(c4, d4)

        qc_batched_inversion_add(
            qc_reg_cast(a4 + c4, TupleType(2, ModIntType(p))),
            qc_reg_cast(amb4_inv + cmd4_inv, TupleType(2, ModIntType(p))),
        )
        qc_muladd_modint(amb4_inv, b4, A)
        qc_muladd_modint(cmd4_inv, d4, Abis)

        qc_dbl_modint(A)
        qc_neg_modint(A)

        qc_dbl_modint(Abis)
        qc_neg_modint(Abis)

        # check squareness
        qc_add_modint(ModInt(2, p), A)
        Ap2_is_square = qc_is_square(A)
        qc_sub_modint(ModInt(2, p), A)

        qc_cxor(Ap2_is_square, Abis, out_curve)
        self.x(Ap2_is_square[0])
        qc_cxor(Ap2_is_square, A, out_curve)

        ############# uncompute
        self.x(Ap2_is_square[0])
        qc_add_modint(ModInt(2, p), A)
        qc_is_square_uncompute(A, Ap2_is_square)
        qc_sub_modint(ModInt(2, p), A)
        qc_neg_modint(Abis)
        qc_halve_modint(Abis)
        qc_neg_modint(A)
        qc_halve_modint(A)
        qc_mulsub_modint(cmd4_inv, d4, Abis)
        qc_mulsub_modint(amb4_inv, b4, A)
        qc_batched_inversion_sub(
            qc_reg_cast(a4 + c4, TupleType(2, ModIntType(p))),
            qc_reg_cast(amb4_inv + cmd4_inv, TupleType(2, ModIntType(p))),
        )
        qc_diffsum_inv(c4, d4)
        qc_sqrsub_modint(dsq, d4)
        qc_sqrsub_modint(csq, c4)
        qc_sqrsub_modint(b, dsq)
        qc_sqrsub_modint(a, csq)
        qc_diffsum_inv(a, b, swap=True)
        qc_diffsum_inv(a4, b4)
        qc_sqrsub_modint(bsq, b4)
        qc_sqrsub_modint(asq, a4)
        qc_sqrsub_modint(b, bsq)
        qc_sqrsub_modint(a, asq)

        for anc in (asq, bsq, a4, b4, csq, dsq, c4, d4, amb4_inv, cmd4_inv, A, Abis):
            self.assert_anc(anc)
            self.release_anc(anc)

        for i in (3, 2, 1):
            qc_cswap(c4i_isnonzero.a[i], codom_np.a[0], codom_np.a[4 * i + 0])
            qc_cswap(c4i_isnonzero.a[i], codom_np.a[1], codom_np.a[4 * i + 1])
        for i in range(4):
            qc_cswap(
                norm_is_even, codom_np.a[4 * i + 1], codom_np.a[4 * i + 2]
            )  # swap codomain curves Ea and Eabar if acting with an ideal of even norm
            qc_isnonzero(codom_np.a[4 * i], c4i_isnonzero.a[i])
        self.assert_anc(c4i_isnonzero)
        self.release_anc(c4i_isnonzero)
        for i in (0, 1):
            qc_sub_modint(codom_np.a[4 * i + 2], codom_np.a[4 * i + 0])
            qc_sub_modint(codom_np.a[4 * i + 3], codom_np.a[4 * i + 1])
        for i in (2, 3):
            qc_add_modint(codom_np.a[4 * i + 2], codom_np.a[4 * i + 0])
            qc_add_modint(codom_np.a[4 * i + 3], codom_np.a[4 * i + 1])
        for i in range(4):
            qc_cneg_modint(case1, codom_np.a[4 * i + 3])
        for i in range(8):
            if i in (0, 1, 4, 5):
                qc_swap(codom_np.a[2 * i], codom_np.a[2 * i + 1])
            qc_diffsum_inv(codom_np.a[2 * i], codom_np.a[2 * i + 1])

    def dummy_classical_function(
        self, args: tuple[ThetaStructureDim4, UInt, QartonBool]
    ) -> tuple[ThetaStructureDim4, UInt, QartonBool, ModInt]:
        return splittingdim4_classical_function(self.p, args)

    def dummy_classical_function_inverse(
        self, args: tuple[ThetaStructureDim4, UInt, QartonBool, ModInt]
    ) -> tuple[ThetaStructureDim4, UInt, QartonBool]:
        return args[:-1]
