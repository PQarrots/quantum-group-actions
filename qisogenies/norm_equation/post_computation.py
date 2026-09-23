"""
Post-computation circuit (to happen after the quantum search for the solution of the
norm equation).

"""

from typing import Any

from qarton.arithmetic import (
    isqrt_output_size,
    qc_div_uint,
    qc_expand_uint,
    qc_truncate_uint,
)
from qarton.binary_operations import (
    qc_cswap,
    qc_mcx,
    qc_xor,
)
from qarton.circuit import (
    BackendSpecifier,
    BoolType,
    Circuit,
    ITupleType,
    QartonBool,
    Register,
    UInt,
    UIntType,
    dummify,
    memoize,
    qc_make_ituple,
    qc_make_tuple,
    qc_reg_cast,
    qc_reg_from_bits,
)
from qarton.modular_arithmetic import (
    ModInt,
    ModIntType,
    qc_add_modint,
    qc_cneg_modint,
    qc_muladd_modint,
    qc_mulsub_modint,
)
from qarton.signed_arithmetic import (
    SInt,
    SIntType,
    qc_abs,
    qc_abs_uncompute,
)

from .ring_integers import (
    RingInteger,
    RingIntegerType,
    qc_conj,
    qc_norm,
)
from .util import FAST_BACKENDS, InstanceData

__all__ = [
    "postcomp_dummy",
    "postcomp_from_qlapoti_output",
    "qc_xor_uint_to_modint",
    "qc_sint_to_modint",
    "qc_modint_to_sint",
    "qc_uint_to_modint",
    "qc_modint_to_uint",
    "qc_ensure_1mod4",
    "qc_ensure_1mod4_erase",
    "qc_cond_ensure_1mod4",
    "qc_cond_ensure_1mod4_erase",
    "PostComp",
]


def postcomp_dummy(
    d: InstanceData,
    delta_1: RingInteger,
    delta_2: RingInteger,
    z: UInt,
    which_case_when_2mod8: QartonBool,
    DD1: SInt,
    DD2: SInt,
    b1: UInt,
    b2: UInt,
    N: UInt,
) -> tuple[tuple[ModInt, ModInt, ModInt, ModInt, ModInt], int]:
    """
    Post-computation in the algorithm.
    Takes everything that was computed so far and produces the outputs:

    N1, sigma1, sigma2, sigma3, sigma4 as defined in qt-Pegasis, eq. (12)
    via the intermediate values B1, B2, C1, C2, E1, E2 defined in qt-Pegasis, eq. (8)
    """

    D1, D2 = DD1.v, DD2.v

    if z % 4 == 1 and b1 % 2 == 0:
        b2, b1 = b1, b2

    if z % 8 == 2:
        # Impose conditions for later
        # b1 = 1 mod 4
        if b1 % 4 == 3:
            b1 = -b1  # type: ignore
        # b2 = 1 mod 4 for (1) / 3 mod 4 for (2)
        if (which_case_when_2mod8 == 0 and b2 % 4 == 3) or (
            which_case_when_2mod8 == 1 and b2 % 4 == 1
        ):
            b2 = -b2  # type: ignore

    assert z % 8 == 2 or z % 4 == 1

    d1 = delta_1.norm() // N

    # compute xC1, xC2, xE1, xE2
    tmp = delta_2.conjugate()
    (xC1, xC2) = (tmp.a.v, tmp.b.v)
    tmp = delta_1.conjugate()
    (xE1, xE2) = (tmp.a.v, tmp.b.v)

    beta_1 = b1 * N
    beta_2 = b2 * N
    # solving the norm equation!!!!
    assert beta_1**2 + beta_2**2 + delta_1.norm() + delta_2.norm() == 2 ** (d.e) * N

    A1, A2 = delta_1.a.v * b1, delta_1.b.v * b1

    splitting_type = (
        2 if ((A1 % 2) != 0 and (A2 % 2) != 0) else 1 if (A2 % 2) != 0 else 0
    )

    B1, B2 = b1 * b2 * N, 0  # b2bar_b1
    C1, C2 = b1 * xC1, b1 * xC2  # c2_b1bar
    E1, E2 = b2 * xE1, b2 * xE2  # b2bar_c1

    if N % 2 == 0:
        E1, C1 = C1, E1  # type: ignore
        E2, C2 = C2, E2  # type: ignore

        B1, D1 = D1, B1  # type: ignore
        B2, D2 = D2, B2  # type: ignore

    M = 1 << (d.e + 2)
    # Nb1 = b1**2 * N  # n(frak_b1)
    N1 = ModInt((d1 + b1**2 * N) % M, M)  # n(frak_b1) + n(frak_c1)
    # alpha = ModInt(N1, 2**(d.e + 2))**(-1)

    # N2 will be computed later for splitting, can be recovered from N1 (as it's 2**e - N1)
    # N2 = d2 + b2**2 * N  # n(frak_b2) + n(frak_c2)
    # N2 = 2**e - N1
    # beta = ModInt(N2, 2**(d.e + 2))**(-1)

    s1 = ModInt((C1 - E1) % M, M)
    s2 = ModInt((C1 + C2 - E1 - E2) % M, M)
    s3 = ModInt((B1 + B2 + D1 + D2) % M, M)
    s4 = ModInt((B1 + D1) % M, M)
    # s5 = (C2 - E2) % 2
    # s6 = (B2 + D2) % 2

    sigmas = (s1, s2, s3, s4)
    return (N1, *sigmas), splitting_type


def postcomp_from_qlapoti_output(N: Any, e_sol: Any, qlapoti_output: Any) -> Any:
    A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, N1, _N2, _Nb1 = qlapoti_output
    M = 1 << (e_sol + 2)

    # N2 will be computed later for splitting, can be recovered from N1 (as it's 2**e - N1)
    # N2 = d2 + b2**2 * N  # n(frak_b2) + n(frak_c2)
    # N2 = 2**e - N1
    # beta = ModInt(N2, 2**(d.e + 2))**(-1)

    assert N1 % 2 == 1

    if N % 2 == 0:
        E1, C1 = C1, E1  # type: ignore
        E2, C2 = C2, E2  # type: ignore

        B1, D1 = D1, B1  # type: ignore
        B2, D2 = D2, B2  # type: ignore

    s1 = ModInt((C1 - E1) % M, M)
    s2 = ModInt((C1 + C2 - E1 - E2) % M, M)
    s3 = ModInt((B1 + B2 + D1 + D2) % M, M)
    s4 = ModInt((B1 + D1) % M, M)
    # s5 = (C2 - E2) % 2
    # s6 = (B2 + D2) % 2

    splitting_type = (
        2 if ((A1 % 2) != 0 and (A2 % 2) != 0) else 1 if (A2 % 2) != 0 else 0
    )

    sigmas = (s1, s2, s3, s4)
    return (N1, *sigmas), splitting_type


def qc_xor_uint_to_modint(x: Register, dest: Register) -> None:
    """
    Copies a little-endian x: UInt(w) into dest: ModInt(p)
    dest assumed to have enough space: p > 2**w.
    """

    w = min(
        UIntType.get_from_register(x).width, ModIntType.get_from_register(dest).width
    )
    qc_xor(x[:w], dest[:w])


def qc_sint_to_modint(x: Register, ww: int) -> Register:
    xt = SIntType.get_from_register(x)
    if xt.width >= ww - 2:
        raise Exception("Not supported")
    parent = x.parent_circuit
    x_abs, x_sgn = qc_abs(x)
    x_ext = qc_expand_uint(x_abs, ww)
    x_ext = qc_reg_cast(x_ext, datatype=ModIntType(1 << ww))
    qc_cneg_modint(x_sgn, x_ext)
    # negative <=> last bit is 1
    parent.cx(x_ext[-1], x_sgn[0])
    parent.assert_anc(x_sgn)
    return x_ext


def qc_modint_to_sint(x: Register, ww: int) -> Register:
    xt = ModIntType.get_from_register(x)
    assert xt.p.bit_count() == 1
    w = xt.p.bit_length() - 1
    if ww >= w - 2:
        raise Exception("Not supported")
    parent = x.parent_circuit
    sgn = parent.get_anc(BoolType())
    parent.cx(x[-1], sgn[0])
    qc_cneg_modint(sgn, x)
    x_small = qc_reg_cast(x, datatype=UIntType(w))
    x_small = qc_truncate_uint(x_small, ww)
    x_sint = qc_abs_uncompute(x_small, sgn)
    return x_sint


def qc_uint_to_modint(x: Register, ww: int) -> Register:
    xt = UIntType.get_from_register(x)
    if xt.width >= ww - 2:
        raise Exception("Not supported")
    x_ext = qc_expand_uint(x, ww)
    x_ext = qc_reg_cast(x_ext, datatype=ModIntType(1 << ww))
    return x_ext


def qc_modint_to_uint(x: Register, ww: int) -> Register:
    xt = ModIntType.get_from_register(x)
    assert xt.p.bit_count() == 1
    w = xt.p.bit_length() - 1
    if ww >= w - 2:
        raise Exception("Not supported")
    x_small = qc_reg_cast(x, datatype=UIntType(w))
    x_small = qc_truncate_uint(x_small, ww)
    return x_small


def qc_ensure_1mod4(x: Register, ww: int) -> tuple[Register, Register]:
    """
    Converts odd Uint to a modint that is 3 mod 4 (not 1 mod 4). A bit
    information is returned.
    """
    parent = x.parent_circuit
    xt = UIntType.get_from_register(x)
    if xt.width >= ww - 2:
        raise Exception("Not supported")
    x_ext = qc_expand_uint(x, ww)
    x_ext = qc_reg_cast(x_ext, datatype=ModIntType(1 << ww))
    is_1mod4 = parent.get_anc(BoolType())
    parent.cx(x_ext[1], is_1mod4[0])  # is now 1 <=> is 3 mod 4
    qc_cneg_modint(is_1mod4, x_ext)
    return x_ext, is_1mod4


def qc_ensure_1mod4_erase(x: Register, is_1mod4: Register, ww: int) -> Register:
    parent = x.parent_circuit
    xt = ModIntType.get_from_register(x)
    assert xt.p.bit_count() == 1
    w = xt.p.bit_length() - 1
    qc_cneg_modint(is_1mod4, x)
    parent.cx(x[1], is_1mod4[0])
    parent.test_anc(is_1mod4)
    x_red = qc_reg_cast(x, datatype=UIntType(w))
    x_red = qc_truncate_uint(x_red, ww)
    return x_red


def qc_cond_ensure_1mod4(
    x: Register, ww: int, cond: Register, flip: Register | None = None
) -> tuple[Register, Register, Register]:
    """
    Like ``qc_ensure_1mod4``, but the negation (to move away from 3 mod 4) is
    only applied when ``cond`` is 1; when ``cond`` is 0, ``x`` is left
    untouched (up to the type conversion to ModInt).

    If ``flip`` is given (a BoolType register), the target residue is
    swapped: the value is instead moved away from 1 mod 4 (i.e. towards 3 mod
    4) whenever ``flip`` is 1.

    Returns ``(x_as_modint, is_3mod4, neg_trigger)``: the two ancilla
    registers must be passed back to ``qc_cond_ensure_1mod4_erase`` (along
    with the same ``cond``/``flip``) to restore ``x`` and release them.
    """
    parent = x.parent_circuit
    xt = UIntType.get_from_register(x)
    if xt.width >= ww - 2:
        raise Exception("Not supported")
    x_ext = qc_expand_uint(x, ww)
    x_ext = qc_reg_cast(x_ext, datatype=ModIntType(1 << ww))
    is_3mod4 = parent.get_anc(BoolType())
    parent.cx(x_ext[1], is_3mod4[0])
    if flip is not None:
        parent.cx(flip[0], is_3mod4[0])
    neg_trigger = parent.get_anc(BoolType())
    parent.ccx(cond[0], is_3mod4[0], neg_trigger[0])
    if flip is not None:
        parent.cx(flip[0], is_3mod4[0])
    qc_cneg_modint(neg_trigger, x_ext)
    return x_ext, is_3mod4, neg_trigger


def qc_cond_ensure_1mod4_erase(
    x: Register,
    is_3mod4: Register,
    neg_trigger: Register,
    cond: Register,
    ww: int,
    flip: Register | None = None,
) -> Register:
    parent = x.parent_circuit
    xt = ModIntType.get_from_register(x)
    assert xt.p.bit_count() == 1
    w = xt.p.bit_length() - 1
    qc_cneg_modint(neg_trigger, x)
    if flip is not None:
        parent.cx(flip[0], is_3mod4[0])
    parent.ccx(cond[0], is_3mod4[0], neg_trigger[0])
    if flip is not None:
        parent.cx(flip[0], is_3mod4[0])
    parent.test_anc(neg_trigger)
    parent.cx(x[1], is_3mod4[0])
    parent.test_anc(is_3mod4)
    x_red = qc_reg_cast(x, datatype=UIntType(w))
    x_red = qc_truncate_uint(x_red, ww)
    return x_red


@memoize
@dummify
class PostComp(
    Circuit[
        tuple[RingInteger, RingInteger, UInt, QartonBool, SInt, SInt, UInt, UInt, UInt],
        tuple[
            tuple[
                RingInteger,
                RingInteger,
                UInt,
                QartonBool,
                SInt,
                SInt,
                UInt,
                UInt,
                UInt,
            ],
            tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
            UInt,
        ],
    ]
):
    # Output registers are added in order to output
    """
    Postcomputation step 1/2, after we run the norm equation algorithm.

    Setting: compute action of the ideal frak_a = (N, alpha)

    The input is:

    * delta_1
    * delta_2
    * z: UIntType(w//2 + 20). The value found during the search; only its low
      bits are read, to decide whether/how b1, b2 must be adjusted (swapped,
      and their sign fixed modulo 4).
    * which_case_when_2mod8: QartonBool. Only relevant when z % 8 == 2: picks
      which of the two valid residues mod 4 b2 must be brought to.
    * D_1
    * D_2
    * b_1
    * b_2
    * N: UIntType(w//2). Norm of the ideal, which is an integer. Positive, odd.

    Since the value z computed previously has size (w//2 + 20), the sizes of b1 and b2
    are precisely given by: isqrt_output_size(w//2 + 20)

    All inputs are preserved, even if none are needed later except N

    The output is N1, the sigmas and the splitting type.
    """

    def __init__(
        self, d: InstanceData, backends: BackendSpecifier = FAST_BACKENDS
    ) -> None:
        """
        :param d: Parameters.
        :type d: InstanceData
        """
        super().__init__()
        w, e, p = d.w, d.e, d.p
        self.d = d
        self.e = e
        self.w = w

        b1_and_b2_size = isqrt_output_size(w // 2 + 20)

        algo_input = self.add_input_output(
            ITupleType(
                RingIntegerType(p, w // 2 + w // 4 + 10),  # delta1
                RingIntegerType(p, w // 2 + w // 4 + 10),  # delta2
                UIntType(w // 2 + 20),  # z
                BoolType(),  # which_case_when_2mod8
                SIntType(w),  # D1
                SIntType(w),  # D2
                UIntType(b1_and_b2_size),  # b1
                UIntType(b1_and_b2_size),  # b2
                UIntType(w // 2),  # N
            )
        )
        delta1, delta2, z_reg, which_case, D1, D2, b1_int, b2_int, N_int = (
            algo_input.a[i] for i in range(9)
        )
        ring_int_type = delta1.datatype

        # ------------------------------------------------------------------
        # if z % 4 == 1 and b1 % 2 == 0: swap(b1, b2)
        # ------------------------------------------------------------------
        swap_flag = self.get_anc(BoolType())
        self.x(z_reg[1])
        self.x(b1_int[0])
        qc_mcx(qc_reg_from_bits([z_reg[0], z_reg[1], b1_int[0]]), swap_flag)
        self.x(b1_int[0])
        self.x(z_reg[1])

        qc_cswap(swap_flag, b1_int, b2_int)

        # ------------------------------------------------------------------
        # if z % 8 == 2: ensure b1 = 1 mod 4, and b2 = 1 mod 4 (which_case ==
        # 0) or 3 mod 4 (which_case == 1); otherwise b1, b2 are left as-is.
        # ------------------------------------------------------------------
        is_z8eq2 = self.get_anc(BoolType())
        self.x(z_reg[0])
        self.x(z_reg[2])
        qc_mcx(qc_reg_from_bits([z_reg[0], z_reg[1], z_reg[2]]), is_z8eq2)
        self.x(z_reg[2])
        self.x(z_reg[0])

        b1, b1_is3mod4, b1_negtrig = qc_cond_ensure_1mod4(b1_int, e + 2, is_z8eq2)
        b2, b2_is3mod4, b2_negtrig = qc_cond_ensure_1mod4(
            b2_int, e + 2, is_z8eq2, flip=which_case
        )

        # d1 = delta1.norm() // N (NOTE delta1.norm() is supposed to be a multiple of N)
        delta1_norm = qc_norm(delta1)

        d1, rem = qc_div_uint(N_int, delta1_norm)
        self.test_anc(rem)
        self.release_anc(rem)  # remainder is assumed to be 0
        # we can truncate d1 to the size of N1, and we will use that as N1 later on
        N1 = qc_truncate_uint(d1, e + 2)
        N1 = qc_reg_cast(N1, ModIntType(1 << (e + 2)))  # type: ignore
        # quite importantly, we will not need to recompute the norm.

        splitting_type = self.add_anc(UIntType(2))

        self.cx(delta1.a["b"][0], splitting_type[0])
        self.cx(delta1.a["a"][0], splitting_type[1])
        self.cx(splitting_type[1], splitting_type[0])

        # -------------------------------------
        # compute xC1, xC2, xE1, xE2
        # delta_2.conjugate() = xC1 + omega xC2
        qc_conj(delta2)
        d2a, d2b = delta2.a["a"], delta2.a["b"]

        xC1 = qc_sint_to_modint(d2a, e + 2)
        xC2 = qc_sint_to_modint(d2b, e + 2)

        # delta_1.conjugate() = xE1 + omega xE2
        qc_conj(delta1)
        d1a, d1b = delta1.a["a"], delta1.a["b"]

        xE1 = qc_sint_to_modint(d1a, e + 2)
        xE2 = qc_sint_to_modint(d1b, e + 2)

        # auxiliary value b1N = b1 * N
        b1N = self.get_anc(ModIntType(2 ** (e + 2)))
        N = qc_uint_to_modint(N_int, e + 2)  # type: ignore
        qc_muladd_modint(b1, N, b1N)

        sigma4 = self.get_anc(ModIntType(1 << (e + 2)))
        # B1 = b1 * b2 * N  # b2bar_b1
        # sigma4 = B1 + D1
        D1_abs, D1_sgn = qc_abs(D1)
        qc_xor_uint_to_modint(D1_abs, sigma4)
        qc_cneg_modint(D1_sgn, sigma4)
        qc_muladd_modint(b1N, b2, sigma4)

        # N1 = d1 + b1**2 * N  # n(frak_b1) + n(frak_c1)
        qc_muladd_modint(b1N, b1, N1)
        # N1 is an output, but we can just erase b1N and restore N here immediately
        qc_mulsub_modint(b1, N, b1N)
        self.release_anc(b1N)
        # also don't need the modular version of N anymore, this releases space
        N_int = qc_modint_to_uint(N, w // 2)

        # sigma3 = sigma4 + D2 # + B2, but B2 = 0
        sigma3 = self.get_anc(ModIntType(1 << (e + 2)))
        D2_abs, D2_sgn = qc_abs(D2)
        qc_xor_uint_to_modint(D2_abs, sigma3)
        qc_cneg_modint(D2_sgn, sigma3)
        qc_add_modint(sigma4, sigma3)

        # C1, C2 = b1 * xC1, b1 * xC2  # c2_b1bar
        # E1, E2 = b2 * xE1, b2 * xE2  # b2bar_c1
        # sigma1 = C1 - E1
        # sigma2 = sigma1 + C2 - E2
        sigma1 = self.get_anc(ModIntType(1 << (e + 2)))
        qc_muladd_modint(b1, xC1, sigma1)
        qc_mulsub_modint(b2, xE1, sigma1)

        sigma2 = self.get_anc(ModIntType(1 << (e + 2)))
        qc_xor(sigma1, sigma2)
        qc_muladd_modint(b1, xC2, sigma2)
        qc_mulsub_modint(b2, xE2, sigma2)

        # if N % 2 == 0: sigma1 and sigma2 change sign
        self.x(N_int[0])
        qc_cneg_modint(qc_reg_from_bits(N_int[0]), sigma1)
        qc_cneg_modint(qc_reg_from_bits(N_int[0]), sigma2)
        self.x(N_int[0])

        ################ mark outputs

        N1_sigmas = qc_make_tuple(N1, sigma1, sigma2, sigma3, sigma4)
        self.add_output(N1_sigmas)
        self.add_output(splitting_type)

        ################ Uncompute ##############

        # uncompute xC1, xC2, xE1, xE2
        d2a = qc_modint_to_sint(xC1, w // 2 + w // 4 + 10)
        d2b = qc_modint_to_sint(xC2, w // 2 + w // 4 + 10)

        delta2 = qc_reg_cast(d2a + d2b, ring_int_type)
        qc_conj(delta2)

        d1a = qc_modint_to_sint(xE1, w // 2 + w // 4 + 10)
        d1b = qc_modint_to_sint(xE2, w // 2 + w // 4 + 10)

        delta1 = qc_reg_cast(d1a + d1b, ring_int_type)
        qc_conj(delta1)

        b1_int = qc_cond_ensure_1mod4_erase(
            b1, b1_is3mod4, b1_negtrig, is_z8eq2, b1_and_b2_size
        )
        b2_int = qc_cond_ensure_1mod4_erase(
            b2, b2_is3mod4, b2_negtrig, is_z8eq2, b1_and_b2_size, flip=which_case
        )

        # undo the z % 8 == 2 selector
        self.x(z_reg[0])
        self.x(z_reg[2])
        qc_mcx(qc_reg_from_bits([z_reg[0], z_reg[1], z_reg[2]]), is_z8eq2)
        self.x(z_reg[2])
        self.x(z_reg[0])
        self.test_anc(is_z8eq2)

        # undo the swap, then the swap selector (using the now-restored,
        # pre-swap b1_int, matching the state it was computed from initially)
        qc_cswap(swap_flag, b1_int, b2_int)

        self.x(z_reg[1])
        self.x(b1_int[0])
        qc_mcx(qc_reg_from_bits([z_reg[0], z_reg[1], b1_int[0]]), swap_flag)
        self.x(b1_int[0])
        self.x(z_reg[1])
        self.test_anc(swap_flag)

        D1 = qc_abs_uncompute(D1_abs, D1_sgn)  # type: ignore
        D2 = qc_abs_uncompute(D2_abs, D2_sgn)  # type: ignore

        ######################
        # remap outputs
        self.remap(
            qc_make_ituple(
                delta1, delta2, z_reg, which_case, D1, D2, b1_int, b2_int, N_int
            ),
            N1_sigmas,
            splitting_type,
        )

    def validate_input(
        self,
        args: tuple[
            RingInteger, RingInteger, UInt, QartonBool, SInt, SInt, UInt, UInt, UInt
        ],
    ) -> bool:
        """
        Conditions for the input to be valid.
        We check that d1,d2,b1,b2 are non-zero quaternions/integers, that their norms sums to the correct value
        Also we check that D1,D2 were correctly computed
        We cannot check that d1,d2,b1,b2 are in the correct ideal, as we are not given the idea
        We could maybe add a check that there is a N-norm ideal generated by these 4 elements.
        """
        d = self.d
        delta_1, delta_2, z, _which_case_when_2mod8, D1, D2, b1, b2, N = args
        if (delta_1 == 0) or (delta_2 == 0) or (b1 == 0) or (b2 == 0):
            return False
        if not (z % 8 == 2 or z % 4 == 1):
            return False
        if delta_1.norm() + delta_2.norm() + b1**2 * N**2 + b2**2 * N**2 != N * 2**d.e:
            return False
        d_ri = (delta_1 * delta_2.conjugate()) // N
        if D1 % 2**d.e != d_ri.a % 2**d.e:
            return False
        return bool(D2 % 2**d.e == d_ri.b % 2**d.e)

    def dummy_classical_function(
        self,
        args: tuple[
            RingInteger, RingInteger, UInt, QartonBool, SInt, SInt, UInt, UInt, UInt
        ],
    ) -> tuple[
        tuple[RingInteger, RingInteger, UInt, QartonBool, SInt, SInt, UInt, UInt, UInt],
        tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
        UInt,
    ]:

        d = self.d
        sigmas, splitting_type = postcomp_dummy(d, *args)
        return (args, sigmas, UInt(splitting_type))

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            tuple[
                RingInteger,
                RingInteger,
                UInt,
                QartonBool,
                SInt,
                SInt,
                UInt,
                UInt,
                UInt,
            ],
            tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
            UInt,
        ],
    ) -> tuple[
        RingInteger, RingInteger, UInt, QartonBool, SInt, SInt, UInt, UInt, UInt
    ]:
        tmp, _, _ = args
        return tmp
