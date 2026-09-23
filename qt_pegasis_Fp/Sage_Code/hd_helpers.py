import itertools as itt
import logging
from time import time

from sage.all import *

from .theta_lib.pkg.basis_change.base_change_dim4 import (
    apply_block_hadamard,
    base_change_theta_dim4,
    random_symplectic_matrix,
)
from .theta_lib.pkg.isogenies.gluing_isogeny_dim4 import (
    GluingIsogenyDim4,
    SuperglueIsogenyDim4,
    proj_equal,
)
from .theta_lib.pkg.isogenies.isogeny_chain_dim4 import IsogenyChainDim4
from .theta_lib.pkg.isogenies.isogeny_dim4 import IsogenyDim4, SpecialIsogenyDim4
from .theta_lib.pkg.theta_structures.montgomery_theta import (
    hadamard_null_point_to_montgomery_coeff,
    null_point_to_montgomery_coeff,
)
from .theta_lib.pkg.theta_structures.Theta_dim1 import (
    ThetaPointDim1,
    ThetaStructureDim1,
)
from .theta_lib.pkg.theta_structures.Theta_dim2 import (
    ProductThetaStructureDim2,
    ThetaPointDim2,
)
from .theta_lib.pkg.theta_structures.Theta_dim4 import ProductThetaStructureDim2To4
from .theta_lib.pkg.theta_structures.theta_helpers_dim4 import (
    hadamard,
    proj_batch_inversion,
)
from .theta_lib.pkg.theta_structures.Tuple_point import TuplePoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger_sh = logging.StreamHandler()
formatter = logging.Formatter("%(name)s [%(levelname)s] %(message)s")
logger_sh.setFormatter(formatter)
logger.addHandler(logger_sh)


class ChainHelper:
    """
    General 4D isogeny wrapper for `theta_lib`
    """

    def __init__(self, e, T, T16, B, M, BPQ, ePQ4, M2, strategy):
        """
        Input:
        - e: the total number of (2*)-steps
        - T: 4 TuplePoints representing the T_i
        - T16: TuplePoints for the T_i mod 16
        - B: Canonical basis (R, S) on E[4]
        - M: Change of coordinates from B^4 to (S, T)
        - BPQ: basis (P, Q) of E[4*2^e]
        - ePQ4: 4-Weil Pairing between P4 and Q4 (4-torsion of P,Q)
        - M2: splitting change of basis matrix
        """
        # Gluing
        tt0 = time()
        glue_4d = GlueHelper(e, T16, B, M, BPQ, ePQ4)
        tt1 = time()
        logger.info(f"\t- Gluing: {tt1 - tt0:.3f}s")

        # Chain
        Phi = IsogenyChainDim4(T, glue_4d, e, 1, splitting=True, strategy=strategy)

        tt2 = time()
        logger.info(f"\t- Chain: {tt2 - tt1:.3f}s for {e = }")
        cod = Phi._isogenies[-1]._codomain

        # Splitting
        P, Q = BPQ

        # assert ePQ4 == P.weil_pairing(Q, 2**(e+2))**(2**e)
        N = base_change_theta_dim4(M2, ePQ4)
        codom_prod = cod.base_change_struct(N)

        thetaA, thetaB = find_product_4(codom_prod.null_point())
        a1, b1 = thetaA[0], thetaA[1]
        a2, b2 = thetaA[0], thetaA[2]

        A1 = null_point_to_montgomery_coeff(a1, b1)
        A2 = null_point_to_montgomery_coeff(a2, b2)

        Fp = A1.parent().base()
        if Fp(A2 + 2).is_square():
            A2 = -A2
        if Fp(A1 + 2).is_square():
            A1 = -A1
        self.Ea = A1
        self.Eabar = A2
        tt3 = time()
        logger.info(f"\t- Splitting: {tt3 - tt2:.3f}s")


class GlueHelper:
    def __init__(self, e, T16, B, M, BPQ, ePQ4):
        """
        Input:
        - T16: 4 TuplePoints representing the T_i
        - B: Canonical basis (R, S) on E[4]
        - M: Change of coordinates from B^4 to (S, T)
        - BPQ: basis (P, Q) of E[4*2^e]
        - ePQ4: e4(P4, Q4)
        """
        logger.debug("\t- Starting Gluing")
        _t0 = time()
        # Init
        self.E = B[0].curve()  # Over Fp2
        self.Fp2 = self.E.base_ring()
        self.Fp = self.Fp2.base_ring()

        R, S = B
        P, Q = BPQ

        # Make dim 1 and 2 theta structures
        Theta1 = ThetaStructureDim1(self.E, R, S)
        Theta2 = ProductThetaStructureDim2(Theta1, Theta1)

        # Turn dimension 1 theta structure into dim 4
        dom_prod = ProductThetaStructureDim2To4(Theta2, Theta2)

        # Change of coordinate to the "good" basis
        self.e4 = ePQ4  # e4(P4, Q4) == e4(R, S)
        N_dim4 = base_change_theta_dim4(M, self.e4)
        dom_base_change = dom_prod.base_change_struct(N_dim4)

        self.N_dim4 = N_dim4
        self.dom_base_change = dom_base_change
        self.dom_prod = dom_prod
        self.Theta1 = Theta1
        _t1 = time()
        logger.debug(f"\t- Change of basis: {_t1 - _t0:.3f}s")

        # Prepare kernel
        L_K = list([2 * _T for _T in T16])
        L_K_indexes = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)]

        L_K = [
            [TuplePoint(L_K[k][0], L_K[k][1]), TuplePoint(L_K[k][2], L_K[k][3])]
            for k in range(len(L_K))
        ]

        # Translates for evaluation of the first gluing
        self.L_trans = [
            [2 * (L_K[1][0] + L_K[2][0]), 2 * (L_K[1][1] + L_K[2][1])],
            [2 * (L_K[0][0] + L_K[3][0]), 2 * (L_K[0][1] + L_K[3][1])],
        ]  # T2+T3, T1+T4
        self.L_ind = [6, 9]  # (0,1,1,0), (1,0,0,1)

        # Turn the theta dim 1 structure into dim 2
        L_K_theta = []
        for k in range(len(L_K)):
            _P1 = mont_to_theta(L_K[k][0][0], Theta1)
            _P2 = mont_to_theta(L_K[k][0][1], Theta1)
            thetaQ1 = theta_pt_prod_1to2(_P1, _P2)

            _P1 = mont_to_theta(L_K[k][1][0], Theta1)
            _P2 = mont_to_theta(L_K[k][1][1], Theta1)
            thetaQ2 = theta_pt_prod_1to2(_P1, _P2)
            L_K_theta.append((thetaQ1, thetaQ2))

        L_K = L_K_theta
        L_K = [dom_prod.product_theta_point(_R, _S) for _R, _S in L_K]
        L_K = [dom_base_change.base_change_coords(N_dim4, _RS) for _RS in L_K]
        _t2 = time()
        logger.debug(f"\t- Prepare kernel: {_t2 - _t1:.3f}s")

        # Gluing
        self.glue_4d = GluingIsogenyDim4(
            dom_base_change, L_K, L_K_indexes, coerce=self.Fp
        )
        self._domain = self.glue_4d._domain
        self._codomain = self.glue_4d._codomain

        # Second isogeny
        L_K_2 = list(T16)
        L_K_2.append(L_K_2[0] + L_K_2[1])

        L_K_2_indexes = [
            (1, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
            (1, 1, 0, 0),
        ]

        L_K_2 = [self.eval_gluing(_T) for _T in L_K_2]

        self.second_isogeny = GluingIsogenyDim4(
            self.glue_4d._codomain, L_K_2, L_K_2_indexes, coerce=self.Fp
        )
        self._codomain = self.second_isogeny._codomain
        _t3 = time()
        logger.debug(f"\t- Isogeny: {_t3 - _t2:.3f}s")

    def codomain(self):
        return self._codomain

    def eval_gluing(self, P):
        """
        We need to pass, together with P, translates P+T by points of 4-torsion
        T above the kernel. The index is turn into a number, so (1, 0, 1, 0) is
        T1 + T3 and corresponds to the index 2^0 + 2^2 = 5.
        All inputs to that are in theta coordinates, while P is a TuplePoint.
        """
        # Add translates to the points
        P = [TuplePoint(P[0], P[1]), TuplePoint(P[2], P[3])]

        P_trans = [[P[0] + Ti[0], P[1] + Ti[1]] for Ti in self.L_trans]

        # Turn the theta dim 1 structure into dim 2
        # TODO: write more unified classes
        _P1 = mont_to_theta(P[0][0], self.Theta1)
        _P2 = mont_to_theta(P[0][1], self.Theta1)
        thetaQ1 = theta_pt_prod_1to2(_P1, _P2)

        _P1 = mont_to_theta(P[1][0], self.Theta1)
        _P2 = mont_to_theta(P[1][1], self.Theta1)
        thetaQ2 = theta_pt_prod_1to2(_P1, _P2)
        P_theta = (thetaQ1, thetaQ2)

        # Turning the translates into theta
        P_trans_theta = []
        for Q_tr in P_trans:
            _P1 = mont_to_theta(Q_tr[0][0], self.Theta1)
            _P2 = mont_to_theta(Q_tr[0][1], self.Theta1)
            thetaQ1 = theta_pt_prod_1to2(_P1, _P2)

            _P1 = mont_to_theta(Q_tr[1][0], self.Theta1)
            _P2 = mont_to_theta(Q_tr[1][1], self.Theta1)
            thetaQ2 = theta_pt_prod_1to2(_P1, _P2)
            P_trans_theta.append((thetaQ1, thetaQ2))

        P = P_theta
        P = self.dom_prod.product_theta_point(P_theta[0], P[1])
        P = self.dom_base_change.base_change_coords(self.N_dim4, P)

        P_trans = P_trans_theta
        P_trans = [self.dom_prod.product_theta_point(_R, _S) for _R, _S in P_trans]
        P_trans = [
            self.dom_base_change.base_change_coords(self.N_dim4, _RS) for _RS in P_trans
        ]

        return self.glue_4d.special_image(P, P_trans, self.L_ind)

    def __call__(self, P):
        Q = self.second_isogeny(self.eval_gluing(P))
        return Q


def theta_pt_prod_1to2(P1, P2):
    """
    Given two ThetaPointDim1 P1 and P2 on the same curve return the
    ThetaPointDim2 corresponding to (P1, P2) on E^2.
    """
    a1, b1 = P1.coords()
    a2, b2 = P2.coords()

    theta1 = P1._parent
    theta2 = ProductThetaStructureDim2(theta1, theta1)

    coords = [a1 * a2, b1 * a2, a1 * b2, b1 * b2]
    thetaP12 = ThetaPointDim2(theta2, coords)
    return thetaP12


def mont_to_theta(P, T):
    """
    Given a point P on E and a ThetaStructureDim1 for E return theta
    coordinates for P.
    From [Pegasis, Appendix A.1]
    Input:
    - a point P on E
    - a ThetaStrucureDim1 for E
    Output:
    - a ThetaPointDim1 representing P
    """
    R = T.P
    assert R.curve()(0, 0) != R * 2
    r = R.x()
    a, b = r + 1, r - 1

    if P != 0:
        xP, zP = P[0], P[2]
    else:
        xP, zP = 1, 0
    t1 = a * (xP - zP)
    t2 = b * (xP + zP)
    assert any([t1, t2]), "Invalid point"

    thetaP = ThetaPointDim1(T, (t1, t2))
    return thetaP


def find_product_4(theta_null):
    """
    [Dartois Phd, Alg. 6.9] We have theta_ij = (theta_i)(theta_j) where the
    two components are on the two surfaces. Fix an index i0j0 to start such
    that theta_i0j0 != 0, and then set thetaA_i = theta_ij0 and thetaB_j =
    theta_i0j. Can be generalized to split anything in half dimension.
    """
    sA = []
    sB = []
    for i0, j0 in itt.product(range(4), repeat=2):
        idx = (j0 << 2) + i0  # Theta_i0j0
        if theta_null[idx] == 0:
            continue
        thetaA = [theta_null[(j0 << 2) + i] for i in range(4)]
        thetaB = [theta_null[(j << 2) + i0] for j in range(4)]
        return thetaA, thetaB


def int_to_rabbit(N1, s1, s2, s3, s4):
    int_id = (N1 << 8) + (s1 << 6) + (s2 << 4) + (s3 << 2) + s4

    if int_id in [279, 317, 371, 465, 791, 829, 883, 977]:
        return {
            "path": [3, 11, 9, 5, 4, 6, 2, 10, 8, 0, 1, 13, 12, 14],
            "r1": 7,
            "r2": 15,
        }
    elif int_id in [327, 380, 461, 468, 839, 892, 973, 980]:
        return {
            "path": [3, 7, 6, 10, 8, 9, 1, 5, 4, 0, 2, 14, 12, 13],
            "r1": 11,
            "r2": 15,
        }
    elif int_id in [333, 372, 455, 476, 845, 884, 967, 988]:
        return {
            "path": [12, 14, 6, 5, 1, 9, 8, 10, 2, 0, 4, 7, 3, 11],
            "r1": 13,
            "r2": 15,
        }
    elif int_id in [285, 311, 369, 467, 797, 823, 881, 979]:
        return {
            "path": [12, 13, 9, 10, 2, 6, 4, 5, 1, 0, 8, 11, 3, 7],
            "r1": 14,
            "r2": 15,
        }
    else:
        raise ValueError("Did not find a rabbit structure.")


def int_to_change_theta_coords_matrix(N1, s1, s2, s3, s4):
    int_id = (N1 << 8) + (s1 << 6) + (s2 << 4) + (s3 << 2) + s4

    if int_id in [279, 285, 311, 317, 791, 797, 823, 829]:
        # N = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        # [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
        # [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0],
        # [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
        # [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        # [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
        # [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1],
        # [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1],
        # [0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        # [0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0],
        # [0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0],
        # [0, 0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0],
        # [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        # [0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0],
        # [0, 0, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        # [0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0]]
        row_blocks = list(range(16))
        col_blocks = [0, 5, 11, 14, 4, 1, 10, 15, 12, 2, 7, 9, 8, 3, 6, 13]
        # First block columns reversed to have all 1 (eg. 1<->4 for 2nd block)
        had_perm = [[0, 3, 2, 1], [0, 1, 2, 3], [0, 2, 3, 1], [0, 1, 3, 2]]
    elif int_id in [327, 333, 455, 461, 839, 845, 967, 973]:
        # N = [[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        # [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 0, 0],
        # [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0],
        # [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, -1, 0, 0],
        # [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        # [0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 1, 0, 0, 0],
        # [0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 1, 0, 0, 0],
        # [0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        # [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        # [0, 0, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1],
        # [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1],
        # [0, 0, -1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        # [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
        # [0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0]]
        row_blocks = list(range(16))
        col_blocks = [0, 7, 10, 13, 12, 1, 6, 11, 8, 2, 5, 15, 4, 3, 9, 14]
        had_perm = [[0, 1, 3, 2], [0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2]]
    elif int_id in [369, 371, 465, 467, 881, 883, 977, 979]:
        # N = [[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
        # [1, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 1, 0],
        # [1, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0],
        # [1, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0],
        # [0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        # [0, -1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, -1],
        # [0, 1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, -1],
        # [0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
        # [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        # [0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
        # [0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
        # [0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        # [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        # [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0],
        # [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
        # [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0]]
        row_blocks = list(range(16))
        col_blocks = [0, 7, 9, 14, 8, 1, 6, 15, 12, 2, 5, 11, 4, 3, 10, 13]
        had_perm = [[0, 3, 1, 2], [0, 1, 2, 3], [0, 2, 1, 3], [0, 1, 3, 2]]
    elif int_id in [372, 380, 468, 476, 884, 892, 980, 988]:
        # N = [[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        # [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, -1, 0, 0],
        # [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0],
        # [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, -1, 0, 0],
        # [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        # [0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 1, 0, 0, 0],
        # [0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, 0, 0],
        # [0, -1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 1, 0, 0, 0],
        # [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        # [0, 0, 1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1],
        # [0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1],
        # [0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 1],
        # [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, -1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        # [0, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0],
        # [0, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0]]
        row_blocks = list(range(16))
        col_blocks = [0, 6, 11, 13, 12, 1, 7, 10, 4, 2, 9, 15, 8, 3, 5, 14]
        had_perm = [[0, 2, 3, 1], [0, 3, 2, 1], [0, 2, 1, 3], [0, 3, 1, 2]]
    else:
        raise ValueError("Did not find a gluing change of theta coordinates matrix.")
    return (row_blocks, col_blocks, had_perm)


class GlueHelperFp:
    def __init__(self, N1, s1, s2, s3, s4, T16, P16, Q16, E, Et):
        """
        Input:
        - N1, s1, s2, s3, s4: integers mod 4 defining the gluing
        - T16: 4 TuplePoints representing the T_i
        - P16, Q16: basis of E[16]
        - E: starting curve
        - Et: twist of E
        """
        logger.debug("\t- Starting first isogenies")
        _t0 = time()

        d_rabbit = int_to_rabbit(N1, s1, s2, s3, s4)
        aux_point_index = d_rabbit["path"][0]
        change_theta_coords_matrix = int_to_change_theta_coords_matrix(
            N1, s1, s2, s3, s4
        )
        self.glue_4d = SuperglueIsogenyDim4(
            T16, P16, Q16, E, Et, aux_point_index, d_rabbit, change_theta_coords_matrix
        )
        self.d_rabbit = d_rabbit
        self._domain = self.glue_4d._domain

        _t1 = time()
        logger.debug(f"\t- Gluing: {_t1 - _t0} s")

        if aux_point_index == 3:
            # auxiliary point is T1+T2
            K_8 = [
                self.glue_4d.special_eval(T16[0], False),
                self.glue_4d.special_eval(T16[1], False),
                self.glue_4d.eval(T16[2], True),
                self.glue_4d.eval(T16[3], True),
            ]
        elif aux_point_index == 12:
            # auxiliary point is T3+T4
            K_8 = [
                self.glue_4d.eval(T16[0], False),
                self.glue_4d.eval(T16[1], False),
                self.glue_4d.special_eval(T16[2], True),
                self.glue_4d.special_eval(T16[3], True),
            ]
        else:
            raise ValueError("Wrong value of aux_point_index.")

        K_8.append(hadamard(proj_batch_inversion(self.glue_4d._inv_im_aux_pt_dual)))

        # Test
        if False:
            if aux_point_index == 3:
                # auxiliary point is T1+T2
                K_2 = [
                    self.glue_4d.special_eval(4 * T16[0], False),
                    self.glue_4d.special_eval(4 * T16[1], False),
                    self.glue_4d.eval(4 * T16[2], True),
                    self.glue_4d.eval(4 * T16[3], True),
                ]
            elif aux_point_index == 12:
                # auxiliary point is T3+T4
                K_2 = [
                    self.glue_4d.eval(4 * T16[0], False),
                    self.glue_4d.eval(4 * T16[1], False),
                    self.glue_4d.special_eval(4 * T16[2], True),
                    self.glue_4d.special_eval(4 * T16[3], True),
                ]

            def scal_prod(i, j):
                return (
                    (i & 1) * (j & 1)
                    + ((i >> 1) & 1) * ((j >> 1) & 1)
                    + ((i >> 2) & 1) * ((j >> 2) & 1)
                    + ((i >> 3) & 1) * ((j >> 3) & 1)
                ) % 2

            K_2bis = []
            for i in range(4):
                T = list(K_2[i])
                for j in range(16):
                    if scal_prod(2**i, j):
                        T[j] = -T[j]
                K_2bis.append(T)

            for i in range(1, 4):
                print(proj_equal(K_2bis[i], K_2bis[0]))

        self._second_isogeny = SpecialIsogenyDim4(K_8, aux_point_index)

        self._codomain = self._second_isogeny._codomain

        _t2 = time()
        logger.debug(f"\t- Second isogeny: {_t2 - _t1} s")

    def eval(self, P, twistP):
        if twistP == self.glue_4d.twist_aux_pt:
            return self._second_isogeny(self.glue_4d.special_eval(P, twistP))
        else:
            return self._second_isogeny(self.glue_4d.eval(P, twistP))

    def eval_basis(self, T):
        return [
            self.eval(T[0], False),
            self.eval(T[1], False),
            self.eval(T[2], True),
            self.eval(T[3], True),
        ]


def int_to_splitting_change_of_theta_coords_matrix(N1, Nb1, A1, A2):
    int_id = (N1 << 6) + (Nb1 << 4) + (A1 << 2) + A2
    if int_id in [80, 82, 88, 90, 240, 242, 248, 250]:
        # [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0]]
        row_blocks = list(range(16))
        col_blocks = list(range(16))
        # First block columns reversed to have all 1 (eg. 1<->4 for 2nd block)
        had_perm = [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
    elif int_id in [81, 83, 121, 123, 217, 219, 241, 243]:
        row_blocks = list(range(16))
        col_blocks = list(range(16))
        # First block columns reversed to have all 1 (eg. 1<->4 for 2nd block)
        had_perm = [[0, 3, 2, 1], [1, 2, 3, 0], [2, 1, 0, 3], [3, 0, 1, 2]]
    elif int_id in [87, 93, 117, 127, 213, 223, 247, 253]:
        row_blocks = list(range(16))
        col_blocks = list(range(16))
        # First block columns reversed to have all 1 (eg. 1<->4 for 2nd block)
        had_perm = [[0, 1, 3, 2], [1, 0, 2, 3], [2, 3, 1, 0], [3, 2, 0, 1]]
    else:
        raise ValueError("Did not find a splitting change of theta coordinates matrix.")
    return (row_blocks, col_blocks, had_perm)


def is_product(null_point, theta1, theta2):
    null_point_dim2 = []
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


def codomain_null_point_to_curves(null_point):
    i0 = 0
    for i in range(4):
        if null_point[4 * i0] != 0:
            i0 = i

    a1, b1 = null_point[4 * i0], null_point[4 * i0 + 1]
    a2, b2 = null_point[4 * i0], null_point[4 * i0 + 2]

    assert is_product(null_point, (a1, b1), (a2, b2))

    A1 = null_point_to_montgomery_coeff(a1, b1)
    A2 = null_point_to_montgomery_coeff(a2, b2)

    A1bis = hadamard_null_point_to_montgomery_coeff(a1, b1)
    A2bis = hadamard_null_point_to_montgomery_coeff(a2, b2)

    if (A1 + 2).is_square():
        A1 = A1bis
    if (A2 + 2).is_square():
        A2 = A2bis

    return A1, A2


def count_even_coeff_dim4(domain):
    count = 0
    for i in range(16):
        for j in range(16):
            if bin(i & j).count("1") % 2 == 0:
                S = 0
                for k in range(16):
                    S += ((-1) ** (bin(k & i).count("1"))) * domain[k] * domain[k ^ j]
                if S != 0:
                    count += 1
    return count


def random_product(null_point):
    p = F.characteristic()
    Fp2 = GF((p, 2), name="i", modulus=var("x") ** 2 + 1)

    ii = Fp2.gen()
    for i in range(1000):
        M = random_symplectic_matrix(4)
        N = base_change_theta_dim4(M, zeta)


class ChainHelperFp:
    """
    General 4D isogeny wrapper for `theta_lib`
    """

    def __init__(self, e, N1, s1, s2, s3, s4, Nb1, A1, A2, T, T16, P16, Q16, strategy):
        """
        Input:
        - e: the total number of (2*)-steps
        - N1, s1, s2, s3, s4, Nb1, A1, A2: integers mod 4 defining the gluing
        - T: 4 TuplePoints representing the T_i
        - T16: TuplePoints for the T_i mod 16
        - P16, Q16: basis of E[16]
        """
        # Gluing
        tt0 = time()
        E = P16.curve()
        Et = Q16.curve()
        self.first_isogenies = GlueHelperFp(N1, s1, s2, s3, s4, T16, P16, Q16, E, Et)
        self._domain = self.first_isogenies.glue_4d._domain
        tt1 = time()
        logger.info(f"\t- Gluing: {tt1 - tt0:.3f}s")

        # Chain
        self.strategy = (
            strategy  # precompute_strategy_with_first_eval(e,1,M=1,S=0.8,I=10)
        )
        self.e = e
        self.T = T
        Phi = self.isogeny_chain()

        tt2 = time()
        logger.info(f"\t- Chain: {tt2 - tt1:.3f}s for {e = }")
        cod = Phi[-1]._codomain

        # assert ePQ4 == P.weil_pairing(Q, 2**(e+2))**(2**e)
        row_blocks, col_blocks, had_perm = (
            int_to_splitting_change_of_theta_coords_matrix(N1, Nb1, A1, A2)
        )
        codom_prod_null_point = apply_block_hadamard(
            cod.null_point(), row_blocks, col_blocks, had_perm
        )

        self.Ea, self.Eabar = codomain_null_point_to_curves(codom_prod_null_point)
        tt3 = time()
        logger.info(f"\t- Splitting: {tt3 - tt2:.3f}s")

    def isogeny_chain(self):
        """
        Compute the isogeny chain and store intermediate isogenies for evaluation
        Adapted from theta_lib with special twist handling for first_isogenies.
        """
        # Store chain of 2-isogenies
        isogeny_chain = []

        # Bookkeeping for optimal strategy
        strat_idx = 0
        level = [0]
        ker = self.T
        kernel_elements = [ker]

        # Length of the chain
        n = self.e - 1

        for k in range(n):
            # print("chain k = {}".format(k))
            prev = sum(level)
            ker = kernel_elements[-1]

            while prev != (n - 1 - k):
                level.append(self.strategy[strat_idx])
                prev += self.strategy[strat_idx]

                # Perform the doublings and update kernel elements
                # Prevent the last unnecessary doublings for first isogeny computation
                if k > 0 or prev != n - 1:
                    ker = [
                        ker[i].double_iter(self.strategy[strat_idx]) for i in range(4)
                    ]
                    kernel_elements.append(ker)

                # Update bookkeeping variable
                strat_idx += 1

            # Compute the codomain from the 8-torsion
            if k == 0:
                phi = self.first_isogenies
            elif k <= 10:
                phi = IsogenyDim4(Th, ker, new_method=True, search_hamilton=True)
            else:
                phi = IsogenyDim4(Th, ker, new_method=True, search_hamilton=False)

            # if k>0:
            # ker2 = [T.double_iter(2) for T in ker]
            # print(count_even_coeff_dim4(Th.null_point()))
            # print(ker2[0]==Th.act_null((0,0,0,0),(1,0,0,0)))
            # print(ker2[1]==Th.act_null((0,0,0,0),(0,1,0,0)))
            # print(ker2[2]==Th.act_null((0,0,0,0),(0,0,1,0)))
            # print(ker2[3]==Th.act_null((0,0,0,0),(0,0,0,1)))

            # Update the chain of isogenies
            Th = phi._codomain
            # print(parent(Th.null_point().coords()[0]))
            isogeny_chain.append(phi)

            # Remove elements from list
            if k > 0:
                kernel_elements.pop()
            level.pop()

            # Push through points for the next step
            if k == 0:
                kernel_elements = [phi.eval_basis(kernel) for kernel in kernel_elements]
            else:
                kernel_elements = [
                    [phi(T) for T in kernel] for kernel in kernel_elements
                ]
            # print([[parent(T.coords()[0]) for T in kernel] for kernel in kernel_elements])

        return isogeny_chain
