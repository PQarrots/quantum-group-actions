"""
Addition of points on Montgomery curves. Adapted from Qarton's code for addition
of points on Weierstrass curves, with different formulas and types.

"""

from qarton.binary_operations import (
    qc_cswap,
    qc_cxor,
    qc_mcx,
    qc_mcx_neg,
    qc_or,
    qc_swap,
)
from qarton.circuit import (
    BackendSpecifier,
    BoolType,
    Circuit,
    EmptyBackendSpecifier,
    InPlaceCircuit,
    PreCircuit,
    QartonBool,
    Register,
    dummify,
    memoize,
)
from qarton.circuit.backends import (
    BackendHandler,
    OnCallOutputType,
    define_on_call,
)
from qarton.dispatch import (
    qc_add,
    qc_add_var,
    qc_cadd,
    qc_cadd_var,
    qc_csub,
    qc_csub_var,
    qc_sub,
    qc_sub_var,
)
from qarton.modular_arithmetic import (
    KaliskiInverseFixedP,
    ModInt,
    ModIntType,
    qc_add_modint,
    qc_cadd_modint,
    qc_cdbl_modint,
    qc_chalve_modint,
    qc_cmuladd_modint,
    qc_cmulsub_modint,
    qc_csqradd_modint,
    qc_csqrsub_modint,
    qc_csub_modint,
    qc_sub_modint,
)

from .montgomery_curve import (
    AffMontgomeryPoint,
    AffMontgomeryPointType,
    AffMontgomeryPointVariable,
    AffMontgomeryPointVariableType,
    ECMontgomery,
    ECMontgomeryType,
)
from .montgomery_neg import (
    qc_cneg_affmontgomerypoint,
    qc_cneg_affmontgomerypoint_var,
    qc_neg_affmontgomerypoint,
    qc_neg_affmontgomerypoint_var,
)

__all__ = [
    "AffMontgomeryAdd",
    "AffMontgomeryAddIP",
    "qc_add_affmontgomerypoint",
    "qc_sub_affmontgomerypoint",
    "ControlledAffMontgomeryAddIP",
    "qc_cadd_affmontgomerypoint",
    "qc_csub_affmontgomerypoint",
    "AffMontgomeryAddIPVar",
    "qc_add_affmontgomerypoint_var",
    "qc_sub_affmontgomerypoint_var",
    "ControlledAffMontgomeryAddIPVar",
    "qc_cadd_affmontgomerypoint_var",
    "qc_csub_affmontgomerypoint_var",
]


qc_add_affmontgomerypoint = BackendHandler[
    [AffMontgomeryPoint | Register, Register], None
]("qc_add_affmontgomerypoint")
qc_add.define(
    (AffMontgomeryPointType, AffMontgomeryPointType), qc_add_affmontgomerypoint
)
qc_add.define((AffMontgomeryPoint, AffMontgomeryPointType), qc_add_affmontgomerypoint)
qc_add_affmontgomerypoint.add_default("qisogenies.montgomery.AffMontgomeryAddIP")


qc_sub_affmontgomerypoint = BackendHandler[
    [AffMontgomeryPoint | Register, Register], None
]("qc_sub_affmontgomerypoint")
qc_sub.define(
    (AffMontgomeryPointType, AffMontgomeryPointType), qc_sub_affmontgomerypoint
)
qc_sub.define((AffMontgomeryPoint, AffMontgomeryPointType), qc_sub_affmontgomerypoint)
qc_sub_affmontgomerypoint.add_default("qisogenies.montgomery.AffMontgomeryAddIP")


qc_cadd_affmontgomerypoint = BackendHandler[
    [Register, AffMontgomeryPoint | Register, Register], None
]("qc_cadd_affmontgomerypoint")
qc_cadd.define(
    (BoolType, AffMontgomeryPointType, AffMontgomeryPointType),
    qc_cadd_affmontgomerypoint,
)
qc_cadd.define(
    (BoolType, AffMontgomeryPoint, AffMontgomeryPointType), qc_cadd_affmontgomerypoint
)
qc_cadd_affmontgomerypoint.add_default(
    "qisogenies.montgomery.ControlledAffMontgomeryAddIP"
)


qc_csub_affmontgomerypoint = BackendHandler[
    [Register, AffMontgomeryPoint | Register, Register], None
]("qc_csub_affmontgomerypoint")
qc_csub.define(
    (BoolType, AffMontgomeryPointType, AffMontgomeryPointType),
    qc_csub_affmontgomerypoint,
)
qc_csub.define(
    (BoolType, AffMontgomeryPoint, AffMontgomeryPointType), qc_csub_affmontgomerypoint
)
qc_csub_affmontgomerypoint.add_default(
    "qisogenies.montgomery.ControlledAffMontgomeryAddIP"
)


qc_add_affmontgomerypoint_var = BackendHandler[
    [AffMontgomeryPointVariable | Register, Register, Register], None
]("qc_add_affmontgomerypoint_var")
qc_add_var.define(
    (AffMontgomeryPointVariableType, AffMontgomeryPointVariableType, ECMontgomeryType),
    qc_add_affmontgomerypoint_var,
)
qc_add_var.define(
    (AffMontgomeryPointVariable, AffMontgomeryPointVariableType, ECMontgomeryType),
    qc_add_affmontgomerypoint,
)
qc_add_affmontgomerypoint_var.add_default("qisogenies.montgomery.AffMontgomeryAddIPVar")


qc_sub_affmontgomerypoint_var = BackendHandler[
    [AffMontgomeryPointVariable | Register, Register, Register], None
]("qc_sub_affmontgomerypoint_var")
qc_sub_var.define(
    (AffMontgomeryPointVariableType, AffMontgomeryPointVariableType, ECMontgomeryType),
    qc_sub_affmontgomerypoint,
)
qc_sub_var.define(
    (AffMontgomeryPointVariable, AffMontgomeryPointVariableType, ECMontgomeryType),
    qc_sub_affmontgomerypoint,
)
qc_sub_affmontgomerypoint_var.add_default("qisogenies.montgomery.AffMontgomeryAddIPVar")


qc_cadd_affmontgomerypoint_var = BackendHandler[
    [Register, AffMontgomeryPointVariable | Register, Register, Register], None
]("qc_cadd_affmontgomerypoint_var")
qc_cadd_var.define(
    (
        BoolType,
        AffMontgomeryPointVariableType,
        AffMontgomeryPointVariableType,
        ECMontgomeryType,
    ),
    qc_cadd_affmontgomerypoint_var,
)
qc_cadd_var.define(
    (
        BoolType,
        AffMontgomeryPointVariable,
        AffMontgomeryPointVariableType,
        ECMontgomeryType,
    ),
    qc_cadd_affmontgomerypoint,
)
qc_cadd_affmontgomerypoint_var.add_default(
    "qisogenies.montgomery.ControlledAffMontgomeryAddIPVar"
)


qc_csub_affmontgomerypoint_var = BackendHandler[
    [Register, AffMontgomeryPointVariable | Register, Register, Register], None
]("qc_csub_affmontgomerypoint_var")
qc_csub_var.define(
    (
        BoolType,
        AffMontgomeryPointVariableType,
        AffMontgomeryPointVariableType,
        ECMontgomeryType,
    ),
    qc_csub_affmontgomerypoint,
)
qc_csub_var.define(
    (
        BoolType,
        AffMontgomeryPointVariable,
        AffMontgomeryPointVariableType,
        ECMontgomeryType,
    ),
    qc_csub_affmontgomerypoint,
)
qc_csub_affmontgomerypoint_var.add_default(
    "qisogenies.montgomery.ControlledAffMontgomeryAddIPVar"
)


def _aff_add_montgomery_content(
    qc: Circuit,
    pp: Register,
    qq: Register,
    res: Register,
    a: ModInt | Register,
    b: ModInt | Register,
    q: int,
) -> None:
    """
    Addition of points on curves in Montgomery representation.

    Contains 10 multiplications and 2 inverses (forward and backward).

    """

    ppx, ppy = pp.a["x"], pp.a["y"]
    qqx, qqy = qq.a["x"], qq.a["y"]
    resx, resy = res.a["x"], res.a["y"]
    ppnotinf, qqnotinf = pp.a["not_infty"], qq.a["not_infty"]
    resnotinf = res.a["not_infty"]

    # cases where we have to do something:
    # 1. (x,y,1) (0,0,0) -> (x,y,1) (0,0,0) (x,y,1) via xor
    # 2. (0,0,0) (x,y,1) -> (0,0,0) (x,y,1) (x,y,1) via xor

    # 3. (x,y,1) (x',y',1) -> (x,y,1) (x',y',1) (result, 1)
    # 4. (x,y,1) (x,y,1) -> (x,y,1) (x,y,1) (result, 1)
    # 5. (x,y,1) (x,-y,1) ->  (x,y,1) (x,-y,1) (0,0,0) (nothing happens)

    # -- case 2 if pp is inf: copy qq to output
    qc.x(ppnotinf[0])
    qc_cadd_modint(ppnotinf, qqx, resx)
    qc_cadd_modint(ppnotinf, qqy, resy)
    qc_cxor(ppnotinf, qqnotinf, resnotinf)
    qc.x(ppnotinf[0])
    # -- case 1 if qq is inf: copy pp to output
    qc.x(qqnotinf[0])
    qc_cadd_modint(qqnotinf, ppx, resx)
    qc_cadd_modint(qqnotinf, ppy, resy)
    qc_cxor(qqnotinf, ppnotinf, resnotinf)
    qc.x(qqnotinf[0])

    # --- compute the controls for case 3 and 4
    qqeqminuspp = qc.get_anc(BoolType())
    qqeqpp = qc.get_anc(BoolType())

    qc_sub_modint(qqx, ppx)  # x1-x2
    qc_sub_modint(qqy, ppy)  # y1-y2
    qc_mcx_neg(ppx + ppy, qqeqpp)  # indicates that qq = pp
    qc_add_modint(qqy, ppy)  # y1
    qc_add_modint(qqy, ppy)  # y1 + y2
    qc_mcx_neg(ppx + ppy, qqeqminuspp)  # indicates that qq = -pp
    qc_sub_modint(qqy, ppy)  # y1
    qc_add_modint(qqx, ppx)

    # self.print(qqeqpp, qqeqminuspp, msg="eq, eqmin")
    # if qqeqminuspp, then c3 and c4 should be 0.
    # if qqeqpp and (not qqeqminuspp) and ppnotinf and qqnotinf, then c4 should be 1.
    # if (not qqeqpp) and (not qqeqminuspp) and ppnotinf and qqnotinf, then c3 should be 1.

    c3 = qc.get_anc(BoolType())
    c4 = qc.get_anc(BoolType())

    qc.x(qqeqminuspp[0])
    qc_mcx(ppnotinf + qqnotinf + qqeqpp + qqeqminuspp, c4)
    qc.x(qqeqpp[0])
    qc_mcx(ppnotinf + qqnotinf + qqeqminuspp + qqeqpp, c3)
    qc.x(qqeqpp[0])
    qc.x(qqeqminuspp[0])

    qc.cx(c3[0], resnotinf[0])
    qc.cx(c4[0], resnotinf[0])

    c3orc4 = qc.get_anc(BoolType())
    qc_or(c3, c4, c3orc4)

    # --- operations in case 3 & 4 (mixed)

    # compute lambda
    # in case 3: lambda = (y1-y2)/(x1-x2)
    # in case 4: lambda = (3x1^2 + 2a x1 + 1) / (2y1 b)

    # in both cases: x3 = b * lambda^2 - a - x1 - x2 and y3 = lambda(x1-x3) - y1

    lamnum = qc.get_anc(ModIntType(q))  # reduce space later please
    lamden = qc.get_anc(ModIntType(q))

    # put (y1-y2) in lamnum
    qc_cadd_modint(c3, ppy, lamnum)
    qc_csub_modint(c3, qqy, lamnum)

    # at the moment lamden is 0, put 3x1 + 2a in lamden
    qc.test_anc(lamden)
    qc_cadd_modint(c4, ppx, lamden)
    qc_cadd_modint(c4, a, lamden)
    qc_cdbl_modint(c4, lamden)
    qc_cadd_modint(c4, ppx, lamden)
    # lamden now contains c4(2a + 3x1)

    # add x1(3x1 + 2a) to lamnum
    qc_cmuladd_modint(c4, ppx, lamden, lamnum)  # ----- 1 multiplication
    # remove 3x1 + 2a from lamden
    qc_csub_modint(c4, ppx, lamden)
    qc_chalve_modint(c4, lamden)
    qc_csub_modint(c4, a, lamden)
    qc_csub_modint(c4, ppx, lamden)
    qc.test_anc(lamden)
    # now lamden is 0, lamnum contains c3(y1-y2) + c4( x1(3x1 + 2a) )

    # add 1 to lamnum
    qc_cadd_modint(c4, ModInt(1, q), lamnum)
    # now lamnum is c3(y1-y2) + c4(3x1^2 + 2a x1 + 1), and completed

    qc_cadd_modint(c3, ppx, lamden)
    qc_csub_modint(c3, qqx, lamden)
    # now lamden is c3(x1-x2), need to add c4 (2y1 b)
    qc_cdbl_modint(c4, ppy)
    qc_cmuladd_modint(c4, b, ppy, lamden)  # --------------------- 1 multiplication
    qc_chalve_modint(c4, ppy)
    # lamden created in both cases

    # inverse lamden
    (_, garbage) = qc.append(PreCircuit(KaliskiInverseFixedP, q), lamden)
    # keep the garbage around
    lam = qc.get_anc(ModIntType(q))
    qc_cmuladd_modint(c3orc4, lamnum, lamden, lam)  # ---------- 1 multiplication

    # ----------------------
    qc_csqradd_modint(c3orc4, lam, resy)  # ------------------ 1 multiplication
    # resy contains lam^2, need to multiply with b
    qc_cmuladd_modint(c3orc4, b, resy, resx)  # -------------- 1 multiplication
    # resx contains b * lambda^2, erase resy
    qc_csqrsub_modint(c3orc4, lam, resy)  # ------------------- 1 multiplication
    qc_csub_modint(c3orc4, a, resx)
    qc_csub_modint(c3orc4, ppx, resx)
    qc_csub_modint(c3orc4, qqx, resx)  # resx now contains x3 = b lam^2 - a - x1 - x2

    qc_csub_modint(c3orc4, resx, ppx)  # ------------------- 1 multiplication
    # ppx contains x1 - x3
    qc_cmuladd_modint(c3orc4, ppx, lam, resy)  # ------------------- 1 multiplication
    # resy: lam(x1-x3)
    qc_cadd_modint(c3orc4, resx, ppx)
    qc_csub_modint(c3orc4, ppy, resy)  # resy is now correct

    # --------------------------
    # now we need to erase lamnum and lamden

    qc_cmulsub_modint(
        c3orc4, lamnum, lamden, lam
    )  # ------------------- 1 multiplication
    # self.test_anc(lam, msg="lam not 0")
    qc.assert_anc(lam)
    qc.append(PreCircuit(KaliskiInverseFixedP, q, inverse=True), lamden, garbage)

    # ----------------------------
    qc_cdbl_modint(c4, ppy)
    qc_cmulsub_modint(c4, b, ppy, lamden)  # ------------------- 1 multiplication
    qc_chalve_modint(c4, ppy)
    qc_cadd_modint(c3, qqx, lamden)
    qc_csub_modint(c3, ppx, lamden)
    qc.test_anc(lamden)
    # now we need to erase lamnum

    qc_csub_modint(c4, ModInt(1, q), lamnum)

    qc.test_anc(lamden)
    # add c4(2a + 3x1) to lamden
    qc_cadd_modint(c4, ppx, lamden)
    qc_cadd_modint(c4, a, lamden)
    qc_cdbl_modint(c4, lamden)
    qc_cadd_modint(c4, ppx, lamden)
    # lamden now contains c4(2a + 3x1)

    # remove x1(3x1 + 2a) from lamnum
    qc_cmulsub_modint(c4, ppx, lamden, lamnum)  # ----- 1 multiplication

    # remove c4(3x1 + 2a) from lamden
    qc_csub_modint(c4, ppx, lamden)
    qc_chalve_modint(c4, lamden)
    qc_csub_modint(c4, a, lamden)
    qc_csub_modint(c4, ppx, lamden)
    qc.test_anc(lamden)

    # ------
    qc_cadd_modint(c3, qqy, lamnum)
    qc_csub_modint(c3, ppy, lamnum)
    qc.test_anc(lamnum, msg="lamnum not 0")
    qc.test_anc(lamden, msg="lamden not 0")

    # ---------------------
    qc_or(c3, c4, c3orc4)
    # self.test_anc(c3orc4, msg="c3orc4 not 0")
    qc.assert_anc(c3orc4)

    qc.x(qqeqpp[0])
    qc.x(qqeqminuspp[0])
    qc_mcx(ppnotinf + qqnotinf + qqeqminuspp + qqeqpp, c3)
    qc.x(qqeqpp[0])
    qc_mcx(ppnotinf + qqnotinf + qqeqpp + qqeqminuspp, c4)
    qc.x(qqeqminuspp[0])

    # self.test_anc(c3, c4, msg="c3-c4 not 0")
    qc.assert_anc(c3, c4)

    qc_sub_modint(qqx, ppx)  # x1-x2
    qc_add_modint(qqy, ppy)  # y1 + y2
    qc_mcx_neg(ppx + ppy, qqeqminuspp)  # indicates that qq = -pp
    qc_sub_modint(qqy, ppy)  # y1
    qc_sub_modint(qqy, ppy)  # y1 - y2
    qc_mcx_neg(ppx + ppy, qqeqpp)  # indicates that qq = pp
    qc_add_modint(qqx, ppx)  # x1
    qc_add_modint(qqy, ppy)  # y1

    qc.assert_anc(qqeqminuspp, qqeqpp)


@dummify
@memoize
class AffMontgomeryAdd(
    Circuit[
        tuple[AffMontgomeryPoint, AffMontgomeryPoint],
        tuple[AffMontgomeryPoint, AffMontgomeryPoint, AffMontgomeryPoint],
    ]
):
    """
    Out of place place addition of points.
    """

    def __init__(
        self, ec: ECMontgomery, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.ec = ec
        pp = self.add_input_output(AffMontgomeryPointType(ec))
        qq = self.add_input_output(AffMontgomeryPointType(ec))
        res = self.add_anc_output(AffMontgomeryPointType(ec))
        _aff_add_montgomery_content(
            self, pp, qq, res, ModInt(ec.A, ec.q), ModInt(ec.B, ec.q), ec.q
        )

    def dummy_classical_function(
        self, args: tuple[AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[AffMontgomeryPoint, AffMontgomeryPoint, AffMontgomeryPoint]:
        pp, qq = args
        return pp, qq, qq + pp

    def validate_output(
        self, args: tuple[AffMontgomeryPoint, AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> bool:
        pp, qq, res = args
        return res == (pp + qq)

    def dummy_classical_function_inverse(
        self, args: tuple[AffMontgomeryPoint, AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[AffMontgomeryPoint, AffMontgomeryPoint]:
        pp, qq, _ = args
        return pp, qq


@dummify
@memoize
class AffMontgomeryAddVar(
    Circuit[
        tuple[AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery],
        tuple[
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            ECMontgomery,
            AffMontgomeryPointVariable,
        ],
    ]
):
    """
    Out of place place addition of points, variable curve.
    """

    def __init__(
        self, q: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.q = q
        pp = self.add_input_output(AffMontgomeryPointVariableType(q))
        qq = self.add_input_output(AffMontgomeryPointVariableType(q))
        ec = self.add_input_output(ECMontgomeryType(q))
        res = self.add_anc_output(AffMontgomeryPointVariableType(q))
        a = ec.a["A"]
        b = ec.a["B"]

        _aff_add_montgomery_content(self, pp, qq, res, a, b, q)

    def validate_input(
        self,
        args: tuple[
            AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> bool:
        """
        Input is valid if the two points are indeed on the curve.
        """
        pp, qq, ec = args
        return pp.is_on_curve(ec) and qq.is_on_curve(ec)

    def dummy_classical_function(
        self,
        args: tuple[
            AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> tuple[
        AffMontgomeryPointVariable,
        AffMontgomeryPointVariable,
        ECMontgomery,
        AffMontgomeryPointVariable,
    ]:
        pp, qq, ec = args
        # convert to affine points on the curve and add the points
        rr = AffMontgomeryPointVariable.from_AffMontgomeryPoint(
            pp.to_AffMontgomeryPoint(ec) + qq.to_AffMontgomeryPoint(ec)
        )
        return pp, qq, ec, rr

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            ECMontgomery,
            AffMontgomeryPointVariable,
        ],
    ) -> tuple[AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery]:
        pp, qq, ec, _ = args
        return pp, qq, ec


@dummify
@memoize
class AffMontgomeryAddIP(InPlaceCircuit[tuple[AffMontgomeryPoint, AffMontgomeryPoint]]):
    """
    In-place addition of points. We implement it using two out-of-place additions.

    """

    def __init__(
        self, ec: ECMontgomery, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.ec = ec
        pp = self.add_input_output(AffMontgomeryPointType(ec))
        qq = self.add_input_output(AffMontgomeryPointType(ec))

        # step 1: add out of place to obtain P, Q, P+Q
        _, _, res = self.append(
            PreCircuit(
                AffMontgomeryAdd,
                ec,
            ),
            pp,
            qq,
        )
        # step 2: negate P
        qc_neg_affmontgomerypoint(pp)
        # step 3: consider Q as the output of the addition between P+Q and -P. We will
        # thus destroy it. As a consequence we must first swap P+Q and Q
        qc_swap(res, qq)
        self.append(PreCircuit(AffMontgomeryAdd, ec, inverse=True), pp, qq, res)
        qc_neg_affmontgomerypoint(pp)

    def dummy_classical_function(
        self, args: tuple[AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[AffMontgomeryPoint, AffMontgomeryPoint]:
        pp, qq = args
        return pp, qq + pp

    def dummy_classical_function_inverse(
        self, args: tuple[AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[AffMontgomeryPoint, AffMontgomeryPoint]:
        pp, qq = args
        return pp, qq - pp

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_add_affmontgomerypoint)
        def _(x: Register | AffMontgomeryPoint, y: Register) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            ec = AffMontgomeryPointType.get_from_register(y).ec
            return PreCircuit(cls, ec)

        @define_on_call(cls, qc_sub_affmontgomerypoint)
        def _(x: Register | AffMontgomeryPoint, y: Register) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            ec = AffMontgomeryPointType.get_from_register(y).ec
            return PreCircuit(cls, ec, inverse=True)


@dummify
@memoize
class ControlledAffMontgomeryAddIP(
    InPlaceCircuit[tuple[QartonBool, AffMontgomeryPoint, AffMontgomeryPoint]]
):
    """
    In-place addition of points, controlled.
    """

    def __init__(
        self, ec: ECMontgomery, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.ec = ec
        c = self.add_input_output(BoolType())
        pp = self.add_input_output(AffMontgomeryPointType(ec))
        qq = self.add_input_output(AffMontgomeryPointType(ec))

        # step 1: add out of place to obtain P, Q, P+Q
        _, _, res = self.append(PreCircuit(AffMontgomeryAdd, ec), pp, qq)
        # step 2: negate P, controlled.
        qc_cneg_affmontgomerypoint(c, pp)
        # step 3: controlled swap. If control is not set, we just uncompute P + Q.
        qc_cswap(c, res, qq)
        self.append(PreCircuit(AffMontgomeryAdd, ec, inverse=True), pp, qq, res)
        qc_cneg_affmontgomerypoint(c, pp)

    def dummy_classical_function(
        self, args: tuple[QartonBool, AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[QartonBool, AffMontgomeryPoint, AffMontgomeryPoint]:
        c, pp, qq = args
        return c, pp, qq + pp if c else qq

    def dummy_classical_function_inverse(
        self, args: tuple[QartonBool, AffMontgomeryPoint, AffMontgomeryPoint]
    ) -> tuple[QartonBool, AffMontgomeryPoint, AffMontgomeryPoint]:
        c, pp, qq = args
        return c, pp, qq - pp if c else qq

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_cadd_affmontgomerypoint)
        def _(
            c: Register, x: Register | AffMontgomeryPoint, y: Register
        ) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            ec = AffMontgomeryPointType.get_from_register(y).ec
            return PreCircuit(cls, ec)

        @define_on_call(cls, qc_csub_affmontgomerypoint)
        def _(
            c: Register, x: Register | AffMontgomeryPoint, y: Register
        ) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            ec = AffMontgomeryPointType.get_from_register(y).ec
            return PreCircuit(cls, ec, inverse=True)


@dummify
@memoize
class AffMontgomeryAddIPVar(
    InPlaceCircuit[
        tuple[AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery]
    ]
):
    """
    In-place addition of points, variable case.
    """

    def __init__(
        self, q: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.q = q
        pp = self.add_input_output(AffMontgomeryPointVariableType(q))
        qq = self.add_input_output(AffMontgomeryPointVariableType(q))
        ec = self.add_input_output(ECMontgomeryType(q))

        # step 1: add out of place to obtain P, Q, P+Q
        _, _, _, res = self.append(
            PreCircuit(
                AffMontgomeryAddVar,
                q,
            ),
            pp,
            qq,
            ec,
        )
        # step 2: negate P
        qc_neg_affmontgomerypoint_var(pp, ec)
        # step 3: consider Q as the output of the addition between P+Q and -P. We will
        # thus destroy it. As a consequence we must first swap P+Q and Q
        qc_swap(res, qq)
        self.append(PreCircuit(AffMontgomeryAddVar, q, inverse=True), pp, qq, ec, res)
        qc_neg_affmontgomerypoint_var(pp, ec)

    def validate_input(
        self,
        args: tuple[
            AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> bool:
        """
        Input is valid if the two points are indeed on the curve.
        """
        pp, qq, ec = args
        return pp.is_on_curve(ec) and qq.is_on_curve(ec)

    def dummy_classical_function(
        self,
        args: tuple[
            AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> tuple[AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery]:
        pp, qq, ec = args
        rr = AffMontgomeryPointVariable.from_AffMontgomeryPoint(
            pp.to_AffMontgomeryPoint(ec) + qq.to_AffMontgomeryPoint(ec)
        )
        return pp, rr, ec

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
        ],
    ) -> tuple[AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery]:
        pp, qq, ec = args
        rr = AffMontgomeryPointVariable.from_AffMontgomeryPoint(
            qq.to_AffMontgomeryPoint(ec) - pp.to_AffMontgomeryPoint(ec)
        )
        return pp, rr, ec

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_add_affmontgomerypoint_var)
        def _(
            x: Register | AffMontgomeryPointVariable, y: Register, e: Register
        ) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            q = AffMontgomeryPointVariableType.get_from_register(y).q
            return PreCircuit(cls, q)

        @define_on_call(cls, qc_sub_affmontgomerypoint_var)
        def _(
            x: Register | AffMontgomeryPointVariable, y: Register, e: Register
        ) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            q = AffMontgomeryPointVariableType.get_from_register(y).q
            return PreCircuit(cls, q, inverse=True)


@dummify
@memoize
class ControlledAffMontgomeryAddIPVar(
    InPlaceCircuit[
        tuple[
            QartonBool,
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            ECMontgomery,
        ]
    ]
):
    """
    In-place addition of points, variable case.
    """

    def __init__(
        self, q: int, backends: BackendSpecifier = EmptyBackendSpecifier
    ) -> None:
        super().__init__()
        self.q = q
        c = self.add_input_output(BoolType())
        pp = self.add_input_output(AffMontgomeryPointVariableType(q))
        qq = self.add_input_output(AffMontgomeryPointVariableType(q))
        ec = self.add_input_output(ECMontgomeryType(q))

        # step 1: add out of place to obtain P, Q, P+Q
        _, _, _, res = self.append(PreCircuit(AffMontgomeryAddVar, q), pp, qq, ec)
        # step 2: negate P
        qc_cneg_affmontgomerypoint_var(c, pp, ec)
        # step 3: consider Q as the output of the addition between P+Q and -P. We will
        # thus destroy it. As a consequence we must first swap P+Q and Q
        qc_cswap(c, res, qq)
        self.append(PreCircuit(AffMontgomeryAddVar, q, inverse=True), pp, qq, ec, res)
        qc_cneg_affmontgomerypoint_var(c, pp, ec)

    def validate_input(
        self,
        args: tuple[
            QartonBool,
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            ECMontgomery,
        ],
    ) -> bool:
        """
        Input is valid if the two points are indeed on the curve.
        """
        _, pp, qq, ec = args
        return pp.is_on_curve(ec) and qq.is_on_curve(ec)

    def dummy_classical_function(
        self,
        args: tuple[
            QartonBool,
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            ECMontgomery,
        ],
    ) -> tuple[
        QartonBool, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
    ]:
        c, pp, qq, ec = args
        rr = (
            AffMontgomeryPointVariable.from_AffMontgomeryPoint(
                pp.to_AffMontgomeryPoint(ec) + qq.to_AffMontgomeryPoint(ec)
            )
            if c
            else qq
        )
        return c, pp, rr, ec

    def dummy_classical_function_inverse(
        self,
        args: tuple[
            QartonBool,
            AffMontgomeryPointVariable,
            AffMontgomeryPointVariable,
            ECMontgomery,
        ],
    ) -> tuple[
        QartonBool, AffMontgomeryPointVariable, AffMontgomeryPointVariable, ECMontgomery
    ]:
        c, pp, qq, ec = args
        rr = (
            AffMontgomeryPointVariable.from_AffMontgomeryPoint(
                qq.to_AffMontgomeryPoint(ec) - pp.to_AffMontgomeryPoint(ec)
            )
            if c
            else qq
        )
        return c, pp, rr, ec

    @classmethod
    def initialize_backend_rules(cls) -> None:

        @define_on_call(cls, qc_cadd_affmontgomerypoint_var)
        def _(
            c: Register,
            x: Register | AffMontgomeryPointVariable,
            y: Register,
            e: Register,
        ) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            q = AffMontgomeryPointVariableType.get_from_register(y).q
            return PreCircuit(cls, q)

        @define_on_call(cls, qc_csub_affmontgomerypoint_var)
        def _(
            c: Register,
            x: Register | AffMontgomeryPointVariable,
            y: Register,
            e: Register,
        ) -> OnCallOutputType:
            if not isinstance(x, Register):
                return None
            q = AffMontgomeryPointVariableType.get_from_register(y).q
            return PreCircuit(cls, q, inverse=True)
