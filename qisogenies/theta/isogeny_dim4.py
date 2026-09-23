"""
Various circuits for dimension-4 isogenies.
"""

import itertools
from collections.abc import Callable
from typing import Any

from qarton.binary_operations import (
    qc_cswap,
    qc_cxor,
    qc_or,
    qc_xor,
)
from qarton.circuit import (
    BitVectorType,
    BoolType,
    Circuit,
    QartonBool,
    Register,
    TupleType,
    dummify,
    memoize,
    qc_empty_reg,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_add_modint,
    qc_cadd_modint,
    qc_cmuladd_modint,
    qc_muladd_modint,
    qc_mulsub_modint,
    qc_sub_modint,
)

from .theta_dim4 import (
    ThetaPointDim4,
    ThetaPointDim4Type,
    qc_theta_hadamard,
)
from .theta_util import (
    CoordsDim4,
    qc_coordwise_muladd,
    qc_coordwise_sqradd,
    qc_coordwise_sqrsub,
    qc_is_any_coord_zero,
)

__all__ = [
    "qc_aut_swaps",
    "qc_aut_swaps_uncompute",
    "theta2isogenydim4generic_codomain_classical_function",
    "Theta2IsogenyDim4Generic_Codomain",
    "Theta2IsogenyDim4Generic_Evaluation",
    "theta2isogenydim4second_codomain_classical_function",
    "Theta2IsogenyDim4Second_Codomain",
]


def xor(i: int, j: int) -> int:
    """
    XORs two integers. We use this function as safety measure since we use both python
    and Sagemath.
    """
    if 1 ^ 1 == 0:
        return i ^ j
    else:  # safety in case the sage interpreter does something weird
        raise NotImplementedError


def swap_bits(bitstring: int, i: int, j: int) -> int:
    """
    Swap the bits at position i and j in a bit-string represented as an integer.
    Return the modified bit-string.
    """
    ith_bit = (bitstring >> i) & 1
    jth_bit = (bitstring >> j) & 1
    bits_remove = (ith_bit << i) ^ (jth_bit << j)
    bits_swapped = (ith_bit << j) ^ (jth_bit << i)
    return bitstring ^ bits_remove ^ bits_swapped


def _swaps_from_permutation_coord_only(i: int, j: int) -> list[tuple[int, int]]:
    """
    Apply an automorphism of the hypercube on the coordinates of hadsq_ker
    e.g. if we apply a permutation swapping directions 0 and 1 on the hypercube
    we swap the order-0 and order-1 bit in all components:
    0 = 0000 -> 0 = 0000
    1 = 0001 -> 2 = 0010
    5 = 0101 -> 6 = 0110
    We apply the bit permutations on the coordinate index and collect the corresponding swaps we need to make
    """
    swaps: list[tuple[int, int]] = []
    for comp in range(16):
        b, d = comp, swap_bits(comp, i, j)
        if (b != d) and ((d, b) not in swaps):
            swaps.append((b, d))
    return swaps


def _swaps_from_permutation_pt_coord(
    i: int, j: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Apply an automorphism of the hypercube on the coordinates of hadsq_ker
    e.g. if we apply a permutation swapping directions 0 and 1 on the hypercube
    we swap the order-0 and order-1 bit in all components:
    0 = 0000 -> 0 = 0000
    1 = 0001 -> 2 = 0010
    5 = 0101 -> 6 = 0110
    The points of hadsq_ker are indexed by a label: T1 -> 1, T2 -> 2, T3 -> 4, T4 -> 8.
    and each of these points has 16 coordinates
    We apply the bit permutations on (label, coordinate index) and collect the corresponding swaps we need to make on (point, coordinate)
    """
    swaps: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for ker_pt_idx in (1, 2, 4, 8):
        for comp in range(16):
            a, b = ker_pt_idx, comp
            c, d = swap_bits(ker_pt_idx, i, j), swap_bits(comp, i, j)
            if ((a, b) != (c, d)) and ((c, d), (a, b)) not in swaps:
                swaps.append(((a, b), (c, d)))
    return swaps


def qc_aut_swaps(hadsq_ker: Register, cs: Register, out: Register) -> None:
    """
    Make sure that there's no zero in the components of kernel point 0:
    if there's a zero, swap kernel pt 0 with 1,
    repeat the check: conditionally swap 0 with 2,
    keep conditionally swapping: (0, 3), (1, 2), (1, 3), (2, 3)

    NOTE if it's true that only applying permutation automorphisms to the hypercube
    is sufficient to make generic isogeny codomain computations well-defined,
    then the above algorithm finds a good one and applies it
    TODO be a bit more formal

    :param hadsq_ker: TODO
    :type hadsq_ker: Register
    :param cs: TODO
    :type cs: Register

    NOTE Cost: 6 * 24 cswaps, +
    (let p.bit_length() = n) 2 * (3*16 + 2*8 + 1*4) n-Toffoli, +
    3 16-Toffoli + 2 8-Toffoli + 1 4-Toffoli
    """
    ranges = {0: range(16), 1: range(1, 16, 2), 2: range(2, 16, 4)}
    int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3}

    for c_idx, (i, j) in enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))):
        cbit = qc_reg_from_bits(cs[c_idx])
        qc_is_any_coord_zero(hadsq_ker.a[i], cbit, ranges[i])
        for swap in _swaps_from_permutation_pt_coord(i, j):
            (a, b), (c, d) = swap
            qc_cswap(
                cbit,
                hadsq_ker.a[int_to_kernel_pt[a]].a[b],
                hadsq_ker.a[int_to_kernel_pt[c]].a[d],
            )


def qc_aut_swaps_uncompute(hadsq_ker: Register, cs: Register, out: Register) -> None:
    """
    Inverse circuit of qc_aut_swaps.
    """
    ranges = {0: range(16), 1: range(1, 16, 2), 2: range(2, 16, 4)}
    int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3}

    for c_idx, (i, j) in tuple(
        enumerate(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    )[::-1]:
        cbit = qc_reg_from_bits(cs[c_idx])
        for swap in _swaps_from_permutation_pt_coord(i, j)[::-1]:
            (a, b), (c, d) = swap
            qc_cswap(
                cbit,
                hadsq_ker.a[int_to_kernel_pt[a]].a[b],
                hadsq_ker.a[int_to_kernel_pt[c]].a[d],
            )
        for _swap in _swaps_from_permutation_coord_only(i, j)[::-1]:
            b, d = _swap
            qc_cswap(
                cbit,
                out.a[b],
                out.a[d],
            )

        qc_is_any_coord_zero(hadsq_ker.a[i], cbit, ranges[i])


def theta2isogenydim4generic_codomain_classical_function(
    p: int,
    find_aut: bool,
    args: tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
) -> tuple[tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], CoordsDim4]:
    HSK = args
    g = 4

    int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3}

    def pi(i: int, j: int) -> ModInt:
        return HSK[int_to_kernel_pt[j]][i]

    # look for path
    # TODO replace the swaps approach with dumb testing for all of the 24 hypercube auts
    if find_aut:
        base_paths = [
            (0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8),  #  if gray path
        ]
        sufficient_permutations = [
            [0, 1, 2, 3],
            [0, 1, 3, 2],
            [0, 2, 1, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2],
            [0, 3, 2, 1],
            [1, 0, 2, 3],
            [1, 0, 3, 2],
            [1, 2, 0, 3],
            [1, 2, 3, 0],
            [1, 3, 0, 2],
            [1, 3, 2, 0],
            [2, 0, 1, 3],
            [2, 0, 3, 1],
            [2, 1, 0, 3],
            [2, 1, 3, 0],
            [2, 3, 0, 1],
            [2, 3, 1, 0],
            [3, 0, 1, 2],
            [3, 0, 2, 1],
            [3, 1, 0, 2],
            [3, 1, 2, 0],
            [3, 2, 0, 1],
            [3, 2, 1, 0],
        ]

        def gen_path(
            sigma: Any, base_path: tuple[int, ...] | list[int]
        ) -> Callable[[int], int]:
            def position(i: int) -> int:
                base_pos = base_path[i]
                i1 = (base_pos & 0b0001) >> 0
                i2 = (base_pos & 0b0010) >> 1
                i3 = (base_pos & 0b0100) >> 2
                i4 = (base_pos & 0b1000) >> 3

                ret = sum(
                    digit_i << sigma[i] for i, digit_i in enumerate((i1, i2, i3, i4))
                )

                return ret  # type: ignore

            return position

        paths = [
            gen_path(sigma, base_path)
            for base_path, sigma in itertools.product(
                base_paths, sufficient_permutations
            )
        ]

        def _aedge(i: int, _path: Any) -> int:
            return _path(i) ^ _path(i + 1)  # type: ignore

        int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3}

        try:
            _path = next(
                _path
                for _path in paths
                if not any(
                    (pi(_path(n), _aedge(n, _path)) == 0)
                    or (pi(_path(n + 1), _aedge(n, _path)) == 0)
                    for n in range(2**g - 1)
                )
            )
        except StopIteration as e:
            print(
                "tried all 'sufficient' paths in generic isogeny: no good one found :("
            )
            raise ValueError from e
        _apath = tuple(_path(i) for i in range(16))
        # path found! now apply Rabbits, algorithm 1
    else:
        _apath = tuple(i ^ (i >> 1) for i in range(16))  # gray_path

    def _edge(i: int, path: tuple[int, ...] | list[int]) -> int:
        return path[i] ^ path[i + 1]

    assert not any(
        (pi(_apath[n], _edge(n, _apath)) == 0)
        or (pi(_apath[n + 1], _edge(n, _apath)) == 0)
        for n in range(2**g - 1)
    )

    # path found! now apply Rabbits, algorithm 1
    def path(i: int) -> int:
        return _apath[i]  # type: ignore

    def edge(i: int) -> int:
        return path(i) ^ path(i + 1)

    # edge(i) = s_j in the notations from the Rabbits paper

    rho_left = [ModInt(1, p)] * 2**g
    rho_right = [ModInt(1, p)] * 2**g

    rho_left[1] = pi(path(0), edge(0))
    rho_right[2**g - 2] = pi(path(2**g - 1), edge(2**g - 2))

    for i in range(2, 2**g):
        rho_left[i] = rho_left[i - 1] * pi(path(i - 1), edge(i - 1))
        rho_right[2**g - 1 - i] = rho_right[2**g - i] * pi(
            path(2**g - i), edge(2**g - 1 - i)
        )

    x_inv = [ModInt(1, p)] * 2**g
    for i in range(2**g):
        x_inv[path(i)] = rho_left[i] * rho_right[i]
    # x_inv = tuple(x_inv)

    return HSK, CoordsDim4(x_inv)


@dummify
@memoize
class Theta2IsogenyDim4Generic_Codomain(
    Circuit[
        tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
        tuple[tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], CoordsDim4],
    ]
):
    def __init__(self, p: int, find_aut: bool = False) -> None:
        """
        For a generic 2-isogeny φ: A -> B at a point P in A
        compute the inverse dual theta null point of the codomain B

        Input:
        - H(T · T) for T in K_8 = (T1, T2, T3, T4), where:
            (T1, T2, T3, T4) are 4 (ℤ/8)-independent elements
            and 4*K_8 generate ker φ, a maximal isotropic subgroup of A[2]
        NOTE π_{i, j} = H(Tj · Tj)[i] (the i-th component of the hadamard squared Tj)

        Output:
        - the inverse dual theta null point inv_dual_np_codomain
        """
        super().__init__()
        self.p = p
        self.find_aut = find_aut
        theta_point_type = ThetaPointDim4Type(p)
        coords_type = theta_point_type._coords_type  # type: ignore

        ## define useful types and constants used later
        hadsq_ker = self.add_input_output(TupleType(4, coords_type))
        out = self.add_anc_output(coords_type)

        j_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3}

        def pi_mat(i: int, j: int) -> Register:
            return hadsq_ker.a[j_to_kernel_pt[j]].a[i]

        # find_aut never changes below, but each of these is only (re)bound inside
        # one of several separate `if find_aut:` blocks; pre-bind them here (unused
        # when find_aut is False) so later `if find_aut:` blocks don't look unbound
        n_paths: Any = None
        path_has_zero: Any = None
        path_selector: Any = None
        path_coords: Any = None
        base_paths: Any = None
        sufficient_permutations: Any = None
        gen_path: Any = None  # type: ignore
        path: Any = None
        edge: Any = None  # type: ignore

        if find_aut:
            n_paths = 24
            path_has_zero = self.add_anc(TupleType(n_paths, BoolType()))
            path_selector = self.add_anc(TupleType(n_paths, BoolType()))
            path_coords = self.add_anc(TupleType(30, ModIntType(p)))

            ############### path-management utilities
            base_paths = [
                (0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8),  #  if gray path
            ]
            sufficient_permutations = [
                [0, 1, 2, 3],
                [0, 1, 3, 2],
                [0, 2, 1, 3],
                [0, 2, 3, 1],
                [0, 3, 1, 2],
                [0, 3, 2, 1],
                [1, 0, 2, 3],
                [1, 0, 3, 2],
                [1, 2, 0, 3],
                [1, 2, 3, 0],
                [1, 3, 0, 2],
                [1, 3, 2, 0],
                [2, 0, 1, 3],
                [2, 0, 3, 1],
                [2, 1, 0, 3],
                [2, 1, 3, 0],
                [2, 3, 0, 1],
                [2, 3, 1, 0],
                [3, 0, 1, 2],
                [3, 0, 2, 1],
                [3, 1, 0, 2],
                [3, 1, 2, 0],
                [3, 2, 0, 1],
                [3, 2, 1, 0],
            ]

            def gen_path(
                sigma: Any, base_path: tuple[int, ...] | list[int]
            ) -> Callable[[int], int]:
                def position(i: int) -> int:
                    base_pos = base_path[i]
                    i1 = (base_pos & 0b0001) >> 0
                    i2 = (base_pos & 0b0010) >> 1
                    i3 = (base_pos & 0b0100) >> 2
                    i4 = (base_pos & 0b1000) >> 3

                    ret = sum(
                        digit_i << sigma[i]
                        for i, digit_i in enumerate((i1, i2, i3, i4))
                    )

                    return ret  # type: ignore

                return position

            self.base_paths = base_paths
            self.sufficient_permutations = sufficient_permutations
            self.gen_path = gen_path

            for path_idx, (base_path, sigma) in enumerate(
                itertools.product(base_paths, sufficient_permutations)
            ):
                path = gen_path(sigma, base_path)
                path_register = qc_empty_reg(self, BitVectorType(0))
                for i in range(15):
                    _edge = path(i + 1) ^ path(i)
                    path_register += pi_mat(path(i), _edge)
                    path_register += pi_mat(path(i + 1), _edge)

                # check if the path has a zero component (invalid path)
                qc_is_any_coord_zero(
                    qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                    path_has_zero.a[path_idx],
                    range(30),
                )

            # ASSUMPTION: there should be at least a valid path (i.e. a path_has_zero.a[i] = 0) for 0 <= i <= 23

            ## find the index i of the first valid path
            self.x_reg(path_has_zero)
            qc_xor(path_has_zero.a[0], path_selector.a[0])
            for i in range(1, n_paths):
                qc_or(path_selector.a[i - 1], path_has_zero.a[i], path_selector.a[i])
            # path_selector = 0001111111
            for i in range(n_paths - 1, 0, -1):
                qc_xor(path_selector.a[i - 1], path_selector.a[i])

            # now only one bit in path_selector is 1. Do a conditional addition on all of the paths.
            for path_idx, (base_path, sigma) in enumerate(
                itertools.product(base_paths, sufficient_permutations)
            ):
                path = gen_path(sigma, base_path)
                path_register = qc_empty_reg(self, BitVectorType(0))
                for i in range(15):
                    _edge = xor(path(i + 1), path(i))
                    path_register += pi_mat(path(i), _edge)
                    path_register += pi_mat(path(i + 1), _edge)

                qc_cxor(
                    path_selector.a[path_idx],
                    qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                    path_coords,
                )

        ## https://eprint.iacr.org/2026/114.pdf algorithm 1
        # using the Gray Hamiltonian path
        ## operations
        # x^{−1}_{η(i)} = ρ_L(i) * ρ_R(i)
        rho_left = self.add_anc(coords_type)
        rho_right = self.add_anc(coords_type)

        if find_aut:
            # qc_add_modint(pi_mat(path(0), edge(0)), rho_left.a[1])
            qc_add_modint(path_coords.a[0], rho_left.a[1])
            # qc_add_modint(pi_mat(path(16 - 1), edge(16 - 2)), rho_right(16 - 2))
            qc_add_modint(path_coords.a[2 * (16 - 2) + 1], rho_right.a[16 - 2])

            for i in range(2, 16):
                # ρ_L(i) = ρ_L(i - 1) * π_{η(i), edge(i)}
                qc_muladd_modint(
                    rho_left.a[i - 1],
                    # pi_mat(path(i - 1), edge(i - 1)),
                    path_coords.a[2 * (i - 1)],
                    rho_left.a[i],
                )
                # ρ_R(2^g - 1 - i) = ρ_R(2^g - i) * π_{η(2^g - i), edge(2^g - 1)}
                qc_muladd_modint(
                    rho_right.a[16 - i],
                    # pi_mat(path(16 - i), edge(16 - 1 - i)),
                    path_coords.a[2 * (16 - 1 - i) + 1],
                    rho_right.a[16 - 1 - i],
                )

            for path_idx, (base_path, sigma) in enumerate(
                itertools.product(base_paths, sufficient_permutations)
            ):
                path = gen_path(sigma, base_path)

                qc_cadd_modint(
                    path_selector.a[path_idx], rho_right.a[0], out.a[path(0)]
                )
                qc_cadd_modint(
                    path_selector.a[path_idx], rho_left.a[16 - 1], out.a[path(16 - 1)]
                )
                for i in range(1, 16 - 1):
                    qc_cmuladd_modint(
                        path_selector.a[path_idx],
                        rho_left.a[i],
                        rho_right.a[i],
                        out.a[path(i)],
                    )

        else:

            def binary_to_gray_pos(i: int) -> int:
                return xor(i, i >> 1)

            path = binary_to_gray_pos  # TODO change here in case of different paths

            def edge(i: int) -> int:
                return xor(path(i + 1), path(i))

            qc_add_modint(pi_mat(path(0), edge(0)), rho_left.a[1])
            qc_add_modint(pi_mat(path(16 - 1), edge(16 - 2)), rho_right.a[16 - 2])

            for i in range(2, 16):
                # ρ_L(i) = ρ_L(i - 1) * π_{η(i), edge(i)}
                qc_muladd_modint(
                    rho_left.a[i - 1], pi_mat(path(i - 1), edge(i - 1)), rho_left.a[i]
                )
                # ρ_R(2^g - 1 - i) = ρ_R(2^g - i) * π_{η(2^g - i), edge(2^g - 1)}
                qc_muladd_modint(
                    rho_right.a[16 - i],
                    pi_mat(path(16 - i), edge(16 - 1 - i)),
                    rho_right.a[16 - 1 - i],
                )

            qc_add_modint(rho_right.a[0], out.a[path(0)])
            qc_add_modint(rho_left.a[16 - 1], out.a[path(16 - 1)])
            for i in range(1, 16 - 1):
                qc_muladd_modint(rho_left.a[i], rho_right.a[i], out.a[path(i)])

        ########## uncompute and release ancillas
        if find_aut:
            # undo HIIP
            for i in range(15, 1, -1):
                qc_mulsub_modint(
                    rho_right.a[16 - i],
                    # pi_mat(path(16 - i), edge(16 - 1 - i)),
                    path_coords.a[2 * (16 - 1 - i) + 1],
                    rho_right.a[16 - 1 - i],
                )
                qc_mulsub_modint(
                    rho_left.a[i - 1],
                    # pi_mat(path(i - 1), edge(i - 1)),
                    path_coords.a[2 * (i - 1)],
                    rho_left.a[i],
                )
            qc_sub_modint(path_coords.a[2 * (16 - 2) + 1], rho_right.a[16 - 2])
            qc_sub_modint(path_coords.a[0], rho_left.a[1])

            self.x(rho_left.a[0][0])
            self.x(rho_right.a[16 - 1][0])
            self.release_anc(rho_left, rho_right)

            # undo copy of path in the right position
            for path_idx, (base_path, sigma) in enumerate(
                itertools.product(base_paths, sufficient_permutations)
            ):
                path = gen_path(sigma, base_path)
                path_register = qc_empty_reg(self, BitVectorType(0))
                for i in range(15):
                    _edge = xor(path(i + 1), path(i))
                    path_register += pi_mat(path(i), _edge)
                    path_register += pi_mat(path(i + 1), _edge)

                qc_cxor(
                    path_selector.a[path_idx],
                    qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                    path_coords,
                )

            # undo path selector
            for i in range(1, n_paths):
                qc_xor(path_selector.a[i - 1], path_selector.a[i])
            for i in range(n_paths - 1, 0, -1):
                qc_or(path_selector.a[i - 1], path_has_zero.a[i], path_selector.a[i])
            qc_xor(path_has_zero.a[0], path_selector.a[0])
            self.x_reg(path_has_zero)

            for path_idx, (base_path, sigma) in enumerate(
                itertools.product(base_paths, sufficient_permutations)
            ):
                path = gen_path(sigma, base_path)
                path_register = qc_empty_reg(self, BitVectorType(0))
                for i in range(15):
                    _edge = path(i + 1) ^ path(i)
                    path_register += pi_mat(path(i), _edge)
                    path_register += pi_mat(path(i + 1), _edge)

                qc_is_any_coord_zero(
                    qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                    path_has_zero.a[path_idx],
                    range(30),
                )
            self.release_anc(path_coords, path_has_zero, path_selector)

        else:
            for i in range(15, 1, -1):
                # ρ_L(i) = ρ_L(i - 1) * π_{η(i), edge(i)}
                qc_mulsub_modint(
                    rho_left.a[i - 1], pi_mat(path(i - 1), edge(i - 1)), rho_left.a[i]
                )
                # ρ_R(2^g - 1 - i) = ρ_R(2^g - i) * π_{η(2^g - i), edge(2^g - 1)}
                qc_mulsub_modint(
                    rho_right.a[16 - i],
                    pi_mat(path(16 - i), edge(16 - 1 - i)),
                    rho_right.a[16 - 1 - i],
                )
            qc_sub_modint(pi_mat(path(0), edge(0)), rho_left.a[1])
            qc_sub_modint(pi_mat(path(16 - 1), edge(16 - 2)), rho_right.a[16 - 2])

            for anc in (rho_left, rho_right):
                self.assert_anc(anc)
                self.release_anc(anc)

    def dummy_classical_function(
        self,
        args: tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
    ) -> tuple[tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], CoordsDim4]:
        return theta2isogenydim4generic_codomain_classical_function(
            self.p, self.find_aut, args
        )

    def dummy_classical_function_inverse(
        self,
        args: tuple[tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], CoordsDim4],
    ) -> tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4]:
        return args[0]


def theta2isogenydim4generic_evaluation_classical_function(
    args: tuple[ThetaPointDim4, CoordsDim4],
) -> tuple[ThetaPointDim4, CoordsDim4, ThetaPointDim4]:
    P, inv_dual_np_codomain = args
    HSP = P.coords.square().hadamard()
    phiP_coords = HSP.mult(inv_dual_np_codomain)
    return *args, ThetaPointDim4(phiP_coords.hadamard())


@dummify
@memoize
class Theta2IsogenyDim4Generic_Evaluation(
    Circuit[
        tuple[ThetaPointDim4, CoordsDim4],
        tuple[ThetaPointDim4, CoordsDim4, ThetaPointDim4],
    ]
):
    def __init__(self, p: int) -> None:
        """
        Evaluate a generic isogeny phi: A -> B at a point P in A
        given the inverse dual theta null point: inv_dual_np_codomain = coordwise_inverse(U^B(0_B))
        (P, inv_dual_np_codomain) -> (P, inv_dual_np_codomain, phi(P))

        NOTE following the qt-Pegasis-Fp code, isogeny_dim4.c, gen_isog_eval
        https://github.com/Pierrick-Dartois/qt-pegasis-Fp/blob/3b7fd3ec830e5d8b60bde7911242ee6a70df279d/C_Code/src/dim4/isogeny_dim4.c#L289

        Assumptions:
        - the kernel of the isogeny phi is in position K2, with A[2] = K1 x K2

        input:
        - point P to evaluate
        - inv_dual_np_codomain: coordinate-wise inverse of the dual theta null point of the codomain

        output:
        - phi(P)
        """
        super().__init__()
        self.p = p
        theta_point_type = ThetaPointDim4Type(p)
        coords_type = theta_point_type._coords_type  # type: ignore

        ## set up inputs and outputs

        P = self.add_input_output(theta_point_type)
        inv_dual_np_codomain = self.add_input_output(coords_type)

        phiP = self.add_anc_output(theta_point_type)

        ## operations
        # had_sq_P = H(P · P)
        had_sq_P = self.add_anc(coords_type)
        qc_coordwise_sqradd(P, had_sq_P)
        qc_theta_hadamard(had_sq_P)
        # phi(P) = had_sq_P · inv_dual_np_codomain
        qc_coordwise_muladd(had_sq_P, inv_dual_np_codomain, phiP)
        qc_theta_hadamard(phiP)

        ########### uncompute ############
        qc_theta_hadamard(had_sq_P, inverse=True)
        qc_coordwise_sqrsub(P, had_sq_P)

        ## release ancillas
        self.assert_anc(had_sq_P)
        self.release_anc(had_sq_P)

    def dummy_classical_function(
        self, args: tuple[ThetaPointDim4, CoordsDim4]
    ) -> tuple[ThetaPointDim4, CoordsDim4, ThetaPointDim4]:
        return theta2isogenydim4generic_evaluation_classical_function(args)

    def dummy_classical_function_inverse(
        self, args: tuple[ThetaPointDim4, CoordsDim4, ThetaPointDim4]
    ) -> tuple[ThetaPointDim4, CoordsDim4]:
        return args[:-1]


def theta2isogenydim4second_codomain_classical_function(
    p: int,
    args: tuple[
        tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
        QartonBool,
    ],
) -> tuple[
    tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
    QartonBool,
    CoordsDim4,
]:
    HSK, Taux_is_T3p4 = args
    g = 4

    sufficient_permutations = [
        (2, 1, 3, 0),
        (3, 2, 0, 1),
        (1, 2, 3, 0),
        (3, 2, 1, 0),
        (3, 1, 2, 0),
        (3, 1, 0, 2),
    ]

    base_paths = [
        (0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8),  #  if gray path
        (6, 5, 4, 0, 1, 3, 2, 10, 11, 9, 8, 12, 13, 14, 15, 7),  #  Taux = T1+T2
    ]

    def gen_path(
        sigma1: Any, sigma2: Any, base_path: Any, swapped: bool = False
    ) -> Callable[[int], Any]:
        def position(i: int) -> Any:
            # when S = {e1, e2, e3, e4, e1 + e2}
            # see Rabbits, Lemma E.1 (ii)
            base_pos = base_path[i]
            i2_i1 = base_pos & 0b11
            i2_i1 = sigma2[i2_i1]

            i4_i3 = (base_pos & 0b1100) >> 2
            if sigma1 and (i4_i3 in (0b01, 0b10)):
                i4_i3 ^= 0b11  # swap
            ret = (i4_i3 << 2) ^ i2_i1
            if not swapped:
                return ret
            else:
                return swap_bits(swap_bits(ret, 0, 2), 1, 3)

        return position

    paths = [
        gen_path(sigma1, sigma2, base_path, swapped=bool(Taux_is_T3p4))
        for base_path, sigma1, sigma2 in itertools.product(
            base_paths, range(2), sufficient_permutations
        )
    ]

    def edge(i: int, path: Any) -> int:
        return path(i) ^ path(i + 1)  # type: ignore

    int_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3, 3: 4, 12: 4}

    def pi(i: int, j: int) -> ModInt:
        return HSK[int_to_kernel_pt[j]][i]

    try:
        path = next(
            path
            for path in paths
            if not any(
                (pi(path(n), edge(n, path)) == 0)
                or (pi(path(n + 1), edge(n, path)) == 0)
                for n in range(2**g - 1)
            )
        )
    except StopIteration as e:
        print(
            "tried all 'sufficient' hamiltonian paths in the second isogeny: no good one found :("
        )
        raise ValueError from e
        path = paths[-1]

    # path found! now apply Rabbits, algorithm 1
    def _edge(i: int) -> int:
        return path(i) ^ path(i + 1)  # type: ignore

    # edge(i) = s_j in the notations from the Rabbits paper

    rho_left = [ModInt(1, p)] * 2**g
    rho_right = [ModInt(1, p)] * 2**g

    rho_left[1] = pi(path(0), _edge(0))
    rho_right[2**g - 2] = pi(path(2**g - 1), _edge(2**g - 2))

    for i in range(2, 2**g):
        rho_left[i] = rho_left[i - 1] * pi(path(i - 1), _edge(i - 1))
        rho_right[2**g - 1 - i] = rho_right[2**g - i] * pi(
            path(2**g - i), _edge(2**g - 1 - i)
        )

    x_inv = [ModInt(1, p)] * 2**g
    for i in range(2**g):
        x_inv[path(i)] = rho_left[i] * rho_right[i]
    # x_inv = tuple(x_inv)

    return *args, CoordsDim4(x_inv)


@dummify
@memoize
class Theta2IsogenyDim4Second_Codomain(
    Circuit[
        tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
            QartonBool,
        ],
        tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
            QartonBool,
            CoordsDim4,
        ],
    ]
):
    def __init__(
        self,
        p: int,
    ) -> None:
        """
        For the second 2-isogeny φ: A -> B of a chain in qt-Pegasis,
        compute the inverse dual theta null point of the codomain B

        Input:
        - H(T · T) for T in K_8 = (T1, T2, T3, T4, T5), where:
            (T1, T2, T3, T4) are 4 (ℤ/8)-independent elements
            and T5 is either T1 + T2 or T3 + T4 (depending on the output of the norm equation)
            and 4*K_8 generate ker φ, a maximal isotropic subgroup of A[2]
        NOTE π_{i, j} = H(Tj · Tj)[i] (the i-th component of the hadamard squared Tj)

        Output:
        - the inverse dual theta null point inv_dual_np_codomain
        """
        super().__init__()
        self.p = p
        theta_point_type = ThetaPointDim4Type(p)
        coords_type = theta_point_type._coords_type  # type: ignore

        ## define useful types and constants used later
        hadsq_ker = self.add_input_output(TupleType(5, coords_type))
        c_Taux_is_34 = self.add_input_output(BoolType())
        out = self.add_anc_output(coords_type)

        path_has_zero = self.add_anc(TupleType(24, BoolType()))
        path_selector = self.add_anc(TupleType(24, BoolType()))
        path_coords = self.add_anc(TupleType(30, ModIntType(p)))

        ############### path-management utilities
        base_paths = [
            (0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8),  #  if gray path
            (6, 5, 4, 0, 1, 3, 2, 10, 11, 9, 8, 12, 13, 14, 15, 7),  #  if mask = 3,
        ]
        sufficient_permutations = [
            (2, 1, 3, 0),
            (3, 2, 0, 1),
            (1, 2, 3, 0),
            (3, 2, 1, 0),
            (3, 1, 2, 0),
            (3, 1, 0, 2),
        ]

        def gen_path(
            sigma1: Any, sigma2: Any, base_path: Any, swapped: bool = False
        ) -> Callable[[int], Any]:
            def position(i: int) -> Any:
                # when S = {e1, e2, e3, e4, e1 + e2}
                # see Rabbits, Lemma E.1 (ii)
                base_pos = base_path[i]
                i2_i1 = base_pos & 0b11
                i2_i1 = sigma2[i2_i1]

                i4_i3 = (base_pos & 0b1100) >> 2
                if sigma1 and (i4_i3 in (0b01, 0b10)):
                    i4_i3 ^= 0b11  # swap
                ret = (i4_i3 << 2) ^ i2_i1
                if not swapped:
                    return ret
                else:
                    return swap_bits(swap_bits(ret, 0, 2), 1, 3)

            return position

        self.base_paths = base_paths
        self.sufficient_permutations = sufficient_permutations
        self.gen_path = gen_path
        j_to_kernel_pt = {1: 0, 2: 1, 4: 2, 8: 3, 3: 4, 12: 4}

        def _swaps_from_permutation(transpositions: Any) -> tuple[Any, Any]:
            # TODO move to utilities, make it accept j_to_kernel_pt as parameter,
            # use same interface in all codomain circuits instead of rewriting
            """
            Apply an automorphism of the hypercube on the coordinates of hadsq_ker
            e.g. if we apply a permutation swapping directions 0 and 1 on the hypercube
            we swap the order-0 and order-1 bit in all components:
            0 = 0000 -> 0 = 0000
            1 = 0001 -> 2 = 0010
            5 = 0101 -> 6 = 0110
            The points of hadsq_ker are indexed by a label: T1 -> 1, T2 -> 2, T3 -> 4, T4 -> 8, T1+2 -> 3, T3+4 -> 12
            and each of these points has 16 coordinates
            We apply the bit permutations on (label, coordinate index) and collect the corresponding swaps we need to make on (point, coordinate)
            """

            # we expect transpositions = [(0,1), (2,3)]
            # or = [(0, 2), (1, 3)]
            def swap_bits_ij(string: int, i: int, j: int) -> int:
                """
                take an integer and swap its i-th and j-th bit.
                """
                ith_bit = (string >> i) & 1
                jth_bit = (string >> j) & 1
                bits_remove = (ith_bit << i) ^ (jth_bit << j)
                bits_swapped = (ith_bit << j) ^ (jth_bit << i)
                return string ^ bits_remove ^ bits_swapped

            def swap_bits(string: int) -> int:
                for i, j in transpositions:
                    string = swap_bits_ij(string, i, j)
                return string

            swaps: list[tuple[tuple[int, int], tuple[int, int]]] = []
            for ker_pt_idx in (1, 2, 4, 8, 3, 12):
                for component in range(16):
                    a, b = ker_pt_idx, component
                    c, d = swap_bits(ker_pt_idx), swap_bits(component)

                    a_, c_ = (j_to_kernel_pt[x] for x in (a, c))
                    if (
                        ((a_, b) != (c_, d))
                        and ((c_, d), (a_, b)) not in swaps
                        and ((a_, b), (c_, d)) not in swaps
                    ):
                        swaps.append(((a_, b), (c_, d)))

            swaps_coord_only: list[tuple[int, int]] = []
            for b in range(16):
                d = swap_bits(b)
                if (
                    (b != d)
                    and ((b, d) not in swaps_coord_only)
                    and ((d, b) not in swaps_coord_only)
                ):
                    swaps_coord_only.append((b, d))

            return swaps, swaps_coord_only

        def pi_mat(i: int, j: int) -> Register:
            return hadsq_ker.a[j_to_kernel_pt[j]].a[i]

        # reduce to the case Taux = T1 + T2
        swaps_1w3_2w4, swaps_1w3_2w4_coord_only = _swaps_from_permutation(
            [(0, 2), (1, 3)]
        )
        for swap in swaps_1w3_2w4:
            (a, b), (c, d) = swap
            qc_cswap(
                c_Taux_is_34,
                hadsq_ker.a[a].a[b],
                hadsq_ker.a[c].a[d],
            )

        for path_idx, (base_path, sigma1, sigma2) in enumerate(
            itertools.product(base_paths, range(2), sufficient_permutations)
        ):
            path = gen_path(sigma1, sigma2, base_path)
            path_register = qc_empty_reg(self, BitVectorType(0))
            for i in range(15):
                edge = path(i + 1) ^ path(i)
                path_register += pi_mat(path(i), edge)
                path_register += pi_mat(path(i + 1), edge)

            # check if the path has a zero component (invalid path)
            qc_is_any_coord_zero(
                qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                path_has_zero.a[path_idx],
                range(30),
            )

        # ASSUMPTION: there should be at least a valid path (i.e. a path_has_zero.a[i] = 0) for 0 <= i <= 23

        ## find the index i of the first valid path
        self.x_reg(path_has_zero)
        qc_xor(path_has_zero.a[0], path_selector.a[0])
        for i in range(1, 24):
            qc_or(path_selector.a[i - 1], path_has_zero.a[i], path_selector.a[i])
        # path_selector = 0001111111
        for i in range(24 - 1, 0, -1):
            qc_xor(path_selector.a[i - 1], path_selector.a[i])

        # now only one bit in path_selector is 1. Do a conditional addition on all of the paths.
        for path_idx, (base_path, sigma1, sigma2) in enumerate(
            itertools.product(base_paths, range(2), sufficient_permutations)
        ):
            path = gen_path(sigma1, sigma2, base_path)
            path_register = qc_empty_reg(self, BitVectorType(0))
            for i in range(15):
                edge = xor(path(i + 1), path(i))
                path_register += pi_mat(path(i), edge)
                path_register += pi_mat(path(i + 1), edge)

            qc_cxor(
                path_selector.a[path_idx],
                qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                path_coords,
            )

        # path_coords contains the correct path, in the following order:
        # (path(0), edge(0, 1)), (path(1), edge(0, 1))
        # then (path(1), edge(1, 2)), (path(2), edge(1, 2)), ...

        ## operations
        # x^{−1}_{η(i)} = ρ_L(i) * ρ_R(i)
        rho_left = self.add_anc(coords_type)
        rho_right = self.add_anc(coords_type)
        self.x(rho_left.a[0][0])
        self.x(rho_right.a[16 - 1][0])

        # qc_add_modint(pi_mat(path(0), edge(0)), rho_left.a[1])
        qc_add_modint(path_coords.a[0], rho_left.a[1])
        # qc_add_modint(pi_mat(path(16 - 1), edge(16 - 2)), rho_right(16 - 2))
        qc_add_modint(path_coords.a[2 * (16 - 2) + 1], rho_right.a[16 - 2])

        for i in range(2, 16):
            # ρ_L(i) = ρ_L(i - 1) * π_{η(i), edge(i)}
            qc_muladd_modint(
                rho_left.a[i - 1],
                # pi_mat(path(i - 1), edge(i - 1)),
                path_coords.a[2 * (i - 1)],
                rho_left.a[i],
            )
            # ρ_R(2^g - 1 - i) = ρ_R(2^g - i) * π_{η(2^g - i), edge(2^g - 1)}
            qc_muladd_modint(
                rho_right.a[16 - i],
                # pi_mat(path(16 - i), edge(16 - 1 - i)),
                path_coords.a[2 * (16 - 1 - i) + 1],
                rho_right.a[16 - 1 - i],
            )

        for path_idx, (base_path, sigma1, sigma2) in enumerate(
            itertools.product(base_paths, range(2), sufficient_permutations)
        ):
            path = gen_path(sigma1, sigma2, base_path)

            qc_cadd_modint(path_selector.a[path_idx], rho_right.a[0], out.a[path(0)])
            qc_cadd_modint(
                path_selector.a[path_idx], rho_left.a[16 - 1], out.a[path(16 - 1)]
            )
            for i in range(1, 16 - 1):
                qc_cmuladd_modint(
                    path_selector.a[path_idx],
                    rho_left.a[i],
                    rho_right.a[i],
                    out.a[path(i)],
                )

        ## uncompute and release ancillas
        # undo HIIP
        for i in range(15, 1, -1):
            qc_mulsub_modint(
                rho_right.a[16 - i],
                # pi_mat(path(16 - i), edge(16 - 1 - i)),
                path_coords.a[2 * (16 - 1 - i) + 1],
                rho_right.a[16 - 1 - i],
            )
            qc_mulsub_modint(
                rho_left.a[i - 1],
                # pi_mat(path(i - 1), edge(i - 1)),
                path_coords.a[2 * (i - 1)],
                rho_left.a[i],
            )
        qc_sub_modint(path_coords.a[2 * (16 - 2) + 1], rho_right.a[16 - 2])
        qc_sub_modint(path_coords.a[0], rho_left.a[1])

        self.x(rho_left.a[0][0])
        self.x(rho_right.a[16 - 1][0])
        self.release_anc(rho_left, rho_right)

        # undo copy of path in the right position
        for path_idx, (base_path, sigma1, sigma2) in enumerate(
            itertools.product(base_paths, range(2), sufficient_permutations)
        ):
            path = gen_path(sigma1, sigma2, base_path)
            path_register = qc_empty_reg(self, BitVectorType(0))
            for i in range(15):
                edge = xor(path(i + 1), path(i))
                path_register += pi_mat(path(i), edge)
                path_register += pi_mat(path(i + 1), edge)

            qc_cxor(
                path_selector.a[path_idx],
                qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                path_coords,
            )

        # undo path selector
        for i in range(1, 24):
            qc_xor(path_selector.a[i - 1], path_selector.a[i])
        for i in range(24 - 1, 0, -1):
            qc_or(path_selector.a[i - 1], path_has_zero.a[i], path_selector.a[i])
        qc_xor(path_has_zero.a[0], path_selector.a[0])
        self.x_reg(path_has_zero)

        for path_idx, (base_path, sigma1, sigma2) in enumerate(
            itertools.product(base_paths, range(2), sufficient_permutations)
        ):
            path = gen_path(sigma1, sigma2, base_path)
            path_register = qc_empty_reg(self, BitVectorType(0))
            for i in range(15):
                edge = path(i + 1) ^ path(i)
                path_register += pi_mat(path(i), edge)
                path_register += pi_mat(path(i + 1), edge)

            qc_is_any_coord_zero(
                qc_reg_cast(path_register, TupleType(30, ModIntType(p))),
                path_has_zero.a[path_idx],
                range(30),
            )

        for swap in swaps_1w3_2w4[::-1]:
            (a, b), (c, d) = swap
            qc_cswap(
                c_Taux_is_34,
                hadsq_ker.a[a].a[b],
                hadsq_ker.a[c].a[d],
            )

        for swap in swaps_1w3_2w4_coord_only:
            b, d = swap
            qc_cswap(c_Taux_is_34, out.a[b], out.a[d])

        self.release_anc(path_coords, path_has_zero, path_selector)

    def dummy_classical_function(
        self,
        args: tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
            QartonBool,
        ],
    ) -> tuple[
        tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
        QartonBool,
        CoordsDim4,
    ]:
        return theta2isogenydim4second_codomain_classical_function(self.p, args)

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4],
            QartonBool,
            CoordsDim4,
        ],
    ) -> tuple[
        tuple[CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4, CoordsDim4], QartonBool
    ]:
        return args[:-1]
