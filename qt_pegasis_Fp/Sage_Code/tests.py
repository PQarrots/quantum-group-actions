from time import time
import logging

from sage.all import *
proof.all(False)


from .theta_lib.pkg.theta_structures.theta_helpers_dim4 import hadamard, squared
from .xonly import xPoint, random_xPoint, MontgomeryA, isWeierstrass, translate_by_T
from .params import *
from .norm_eq import qlapoti

from .hd_helpers import GlueHelper
from .theta_lib.pkg.basis_change.canonical_basis_dim1 import make_canonical
from .theta_lib.pkg.theta_structures.Tuple_point import TuplePoint
from .theta_lib.pkg.basis_change.base_change_dim4 import is_symplectic_matrix_dim4

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
logger_sh = logging.StreamHandler()
formatter = logging.Formatter('%(name)s [%(levelname)s] %(message)s')
logger_sh.setFormatter(formatter)
logger.addHandler(logger_sh)

class qtPegasis:
    def __init__(self, level):
        r"""
        Main qt-Pegasis class. Takes as input the security level:
        - 500|1000|15000|2000|4000: qt-Pegasis parameters
        - 500P|1000P|1500P|2000P|4000P: PEGASIS parameters
        """
        logger.info('Initialization')
        _t0 = time()
        params = qt_params(level)
        self.f = params['f']
        self.e = params['e']
        self.p = self.f * 2**self.e - 1
        self.A = params['A']

        self.Fp = GF(self.p)
        self.Fp2 = GF((self.p, 2), name='i', modulus=var('x')**2 + 1)
        self.K = NumberField(name="pi", polynomial = var('x')**2 + self.p)
        self.pi = self.K.gens()[0]

        E_start = EllipticCurve(self.Fp, [0, self.A, 0, 1, 0])
        self.E_start = E_start

        self.e_sol = self.e - 3

        _t1 = time()
        logger.info(f' - Done in {_t1-_t0:.3f}s')
        logger.info('Precomputations')
        EE = E_start.change_ring(self.Fp2)

        # Loading basis from params (generated with pgen.py)
        self.P = EE([params['Px'], params['Py']])
        self.Q = EE([params['Qx'], self.Fp2([0, params['Qy']])])
        self.TP = EE([params['TPx'], 0])
        self.TQ = EE([params['TQx'], 0])
        self.P16 = EE([params['P16x'], params['P16y']])
        self.Q16 = EE([params['Q16x'], self.Fp2([0, params['Q16y']])])
        self.P16 = EE([params['P16x'], params['P16y']])
        self.Q16 = EE([params['Q16x'], self.Fp2([0, params['Q16y']])])
        self.S = EE([params['Sx'], self.Fp2([params['Sy'], params['Syi']])])
        self.R = EE([params['Rx'], self.Fp2([params['Ry'], params['Ryi']])])
        self.ePQ4 = self.Fp2([0, params['ePQ4']])
        _x, _y, _z, _w = params['_xyzw']

        M0 = Matrix(Integers(4),[
            [_x, 0, 0, 0, _y, 0, 0, 0], # (P, 0, 0, 0)
            [0, _x, 0, 0, 0, _y, 0, 0], # (0, P, 0, 0)
            [0, 0, _x, 0, 0, 0, _y, 0],
            [0, 0, 0, _x, 0, 0, 0, _y],
            [_z, 0, 0, 0, _w, 0, 0, 0], # (S, 0, 0, 0)
            [0, _z, 0, 0, 0, _w, 0, 0],
            [0, 0, _z, 0, 0, 0, _w, 0],
            [0, 0, 0, _z, 0, 0, 0, _w]
        ]).transpose()
        self.M0 = M0
        _t2 = time()
        logger.info(f' - Done in {_t2-_t1:.3f}s')

    def sage_action(self, frak_a, E=None, timings=False, ret_twist=False):
        """
        Compute the action of frak_a from E using qt-Pegasis.
        Wrapper that calls qt_action (which assumes the ideal is
        of a certain form, and reduced).
        Warning: sage structures may cause a significant slowdown especially at
        higher levels.

        Input:
        - frak_a = an instance of Sage's fractional_ideal class
        - E: an elliptic curve; if None, self.E_start is used instead
        - timings, ret_twist: qt_action flags

        Output:
        - `Ea`, the montgomery coefficient of the codomain curve
        """
        #print('Inside the sage part')
        if E:
            E_0 = E
        else:
            E_0 = self.E_start

        frak_a_reduced = frak_a.reduce_equiv()
        frak_a_red_gens = frak_a_reduced.gens()
        if len(frak_a_red_gens) == 1:
            return E_0

        N, alpha = frak_a_red_gens
        assert N == frak_a_reduced.norm()

        # 2-isogeny step
        tt0 = time()
        if ZZ(N) % 4 == 2:
            E_0 = self.two_isogeny(E_0, alpha)
            N //= 2
        tt1 = time()

        out = self.qt_action([ZZ(N), alpha], E=E_0, timings=timings,
                             ret_twist=ret_twist)

        if timings:
            E_a = EllipticCurve(self.Fp, [0, out[0], 0, 1, 0])
            return E_a, out[1]

        if ret_twist:
            return [EllipticCurve(self.Fp, [0, A, 0, 1, 0]) for A in out]

        return EllipticCurve(self.Fp, [0, out, 0, 1, 0])

    def qt_action(self, frak_a, E=None, timings=False, ret_twist=False):
        """
        Compute the action of frak_a from E using qt-Pegasis.

        Input:
        - frak_a = (ell, omega - lambda) an ideal represented as two elements
          in Z[omega]; we assume ell to be smaller than sqrt(p), so that the
          ideal is (close) to reduced and we can use it directly
        - E: an elliptic curve; if None, self.E_start is used instead
        - timings: if True, return an additional tuple (t1, t2, t3) with the
          timings of the corresponding steps
        - `ret_twist`: if True, return also Eabar

        Output:
        - `Ea`, the montgomery coefficient of the codomain curve
        """
        logger.info(f'Starting qt-Pegasis action')
        # Step 1: solve norm equation
        _t1 = time()

        A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, N1, N2, Nb1 = qlapoti(
                frak_a, self.e_sol)

        # Step 2: compute basis
        # Step 2.1: Curve basis generation
        _t2 = time()
        logger.info(f'Step 1: {_t2-_t1:.3f}s')
        e_sol = self.e_sol

        if not E:
            E = self.E_start

        if E == self.E_start:
            P, Q = self.P, self.Q
            TP, TQ = self.TP, self.TQ
            P16, Q16 = self.P16, self.Q16
            ePQ4 = self.ePQ4
            R, S = self.R, self.S
            M0 = self.M0

        else:
            P, Q, TP, TQ = self.TwoTorsBasis(E, e_sol+2)

            # 4-torsion
            dt_two = 2**(e_sol-2)
            P16, Q16 = dt_two*P, dt_two*Q
            P4, Q4 = 4*P16, 4*Q16
            ePQ4 = P4.weil_pairing(Q4, 4)

            # Change of basis
            _, _, R, S, M0 = make_canonical(P4, Q4, 4, preserve_pairing=True)
            (_x, _y), (_z, _w) = M0
            # assert _x*R + _y*S == P4 and _z*R + _w*S == Q4
            M0 = Matrix(Integers(4),[
                [_x, 0, 0, 0, _y, 0, 0, 0], # (P, 0, 0, 0)
                [0, _x, 0, 0, 0, _y, 0, 0], # (0, P, 0, 0)
                [0, 0, _x, 0, 0, 0, _y, 0],
                [0, 0, 0, _x, 0, 0, 0, _y],
                [_z, 0, 0, 0, _w, 0, 0, 0], # (S, 0, 0, 0)
                [0, _z, 0, 0, 0, _w, 0, 0],
                [0, 0, _z, 0, 0, 0, _w, 0],
                [0, 0, 0, _z, 0, 0, 0, _w]
            ])
            M0 = M0.transpose()
        _t21 = time()
        logger.info(f'Step 2.1: {_t21 - _t2:.3f}s')

        # Step 2.2: 4D basis generation
        alpha = inverse_mod(N1, 2**(e_sol + 2))
        beta = inverse_mod(N2, 2**(e_sol + 2))
        q_cff = (1 - alpha*2**e_sol) % 2**(e_sol+2)
        s1 = (C1 - E1) % 2**(e_sol+2)
        s2 = (C1 + C2 - E1 - E2) % 2**(e_sol+2)
        s3 = (B1 + B2 + D1 + D2) % 2**(e_sol+2)
        s4 = (B1 + D1) % 2**(e_sol+2)
        s5 = (C2 - E2) % 2
        s6 = (B2 + D2) % 2

        M = Matrix(Integers(4), [
            [0,0,0,0,-alpha,0,0,0],
            [0,0,0,0,0,-alpha,0,0],
            [0,0,beta*s3,-beta*s1,0,0,0,0],
            [0,0,beta*s2, beta*s4,0,0,0,0],
            [N1,0,s3,-s1,0,0,0,0],
            [0,N1,s2,s4,0,0,0,0],
            [0,0,0,0,q_cff,0,alpha*s4,-alpha*s2],
            [0,0,0,0,0,q_cff,alpha*s1,alpha*s3]]).transpose()

        # Compute the compatible basis from [qt-Pegasis, Eq. (x), Section 4.y]
        OE = P.curve()(0)

        # Basis mod 16
        N1P_16 = N1*P16

        alphas1_16 = (alpha*s1) % 16
        alphas2_16 = (alpha*s2) % 16
        alphas3_16 = (alpha*s3) % 16
        alphas4_16 = (alpha*s4) % 16
        q_cffQ_16=(q_cff % 16)*Q16

        T1_16 = TuplePoint(N1P_16,OE,(s3%16)*P16,-(s1%16)*P16)
        T2_16 = TuplePoint(OE, N1P_16, (s2%16)*P16,(s4%16)*P16)
        T3_16 = TuplePoint(q_cffQ_16,OE,alphas4_16*Q16,-alphas2_16*Q16)
        T4_16 = TuplePoint(OE,q_cffQ_16,alphas1_16*Q16,alphas3_16*Q16)

        glue_4d = GlueHelper(e_sol, (T1_16, T2_16, T3_16, T4_16), (R, S),
                M0 * M, (P, Q), ePQ4)

        L_K_8 = [T1_16, T2_16, T3_16, T4_16]
        L_K_8 += [L_K_8[0]+L_K_8[1],L_K_8[2]+L_K_8[3]]
        HSfL_K_8 = [hadamard(squared(glue_4d.eval_gluing(T))) for T in L_K_8]

        d_zeros = {}
        L_ind_to_zero = [1,2,4,8,3,12]
        for i in range(6):
            L_zeros=[]
            for j in range(16):
                if HSfL_K_8[i][j]==0:
                    L_zeros.append(j)

            d_zeros[L_ind_to_zero[i]]=L_zeros

        return d_zeros

    def first_two_isog(self, N1,s1,s2,s3,s4, E=None):
        r"""
        Inputs given mod 16 to test the first two isogenies only.
        """

        if not E:
            E = self.E_start

        if E == self.E_start:
            P, Q = self.P, self.Q
            TP, TQ = self.TP, self.TQ
            P16, Q16 = self.P16, self.Q16
            ePQ4 = self.ePQ4
            R, S = self.R, self.S
            M0 = self.M0

        else:
            P, Q, TP, TQ = self.TwoTorsBasis(E, e_sol+2)

            # 4-torsion
            dt_two = 2**(e_sol-2)
            P16, Q16 = dt_two*P, dt_two*Q
            P4, Q4 = 4*P16, 4*Q16
            ePQ4 = P4.weil_pairing(Q4, 4)

            # Change of basis
            _, _, R, S, M0 = make_canonical(P4, Q4, 4, preserve_pairing=True)
            (_x, _y), (_z, _w) = M0
            # assert _x*R + _y*S == P4 and _z*R + _w*S == Q4
            M0 = Matrix(Integers(4),[
                [_x, 0, 0, 0, _y, 0, 0, 0], # (P, 0, 0, 0)
                [0, _x, 0, 0, 0, _y, 0, 0], # (0, P, 0, 0)
                [0, 0, _x, 0, 0, 0, _y, 0],
                [0, 0, 0, _x, 0, 0, 0, _y],
                [_z, 0, 0, 0, _w, 0, 0, 0], # (S, 0, 0, 0)
                [0, _z, 0, 0, 0, _w, 0, 0],
                [0, 0, _z, 0, 0, 0, _w, 0],
                [0, 0, 0, _z, 0, 0, 0, _w]
            ])
            M0 = M0.transpose()

        # Step 2.2: 4D basis generation
        alpha = inverse_mod(N1, 16)
        beta = inverse_mod(-N1, 16)

        M = Matrix(Integers(4), [
            [0,0,0,0,-alpha,0,0,0],
            [0,0,0,0,0,-alpha,0,0],
            [0,0,beta*s3,-beta*s1,0,0,0,0],
            [0,0,beta*s2, beta*s4,0,0,0,0],
            [N1,0,s3,-s1,0,0,0,0],
            [0,N1,s2,s4,0,0,0,0],
            [0,0,0,0,1,0,alpha*s4,-alpha*s2],
            [0,0,0,0,0,1,alpha*s1,alpha*s3]]).transpose()

        assert(is_symplectic_matrix_dim4(M))

        # Compute the compatible basis from [qt-Pegasis, Eq. (x), Section 4.y]
        OE = P.curve()(0)

        # Basis mod 16
        N1P_16 = N1*P16

        alphas1_16 = (alpha*s1) % 16
        alphas2_16 = (alpha*s2) % 16
        alphas3_16 = (alpha*s3) % 16
        alphas4_16 = (alpha*s4) % 16

        T1_16 = TuplePoint(N1P_16,OE,(s3)*P16,-(s1)*P16)
        T2_16 = TuplePoint(OE, N1P_16, (s2)*P16,(s4)*P16)
        T3_16 = TuplePoint(Q16,OE,alphas4_16*Q16,-alphas2_16*Q16)
        T4_16 = TuplePoint(OE,Q16,alphas1_16*Q16,alphas3_16*Q16)

        glue_4d = GlueHelper(self.e_sol, (T1_16, T2_16, T3_16, T4_16), (R, S),
                M0 * M, (P, Q), ePQ4, False)

        L_K_8 = [T1_16, T2_16, T3_16, T4_16]
        L_K_8 += [L_K_8[0]+L_K_8[1],L_K_8[2]+L_K_8[3]]
        HSfL_K_8 = [hadamard(squared(glue_4d.eval_gluing(T))) for T in L_K_8]

        d_zeros = {}
        L_ind_to_zero = [1,2,4,8,3,12]
        for i in range(6):
            L_zeros=[]
            for j in range(16):
                if HSfL_K_8[i][j]==0:
                    L_zeros.append(j)

            d_zeros[L_ind_to_zero[i]]=L_zeros

        return d_zeros


    def sample_ideal(self):
        r"""
        Samples a random ideal frak_ell = (ell, omega + lambda) with ell a
        random prime slightly smaller than sqrt(p), so that the sampled ideal
        is (close to) reduced
        """
        ub = isqrt(self.p)//2

        ell = next_prime(randint(0, ub))
        while kronecker(-self.p, ell) != 1:
            ell = next_prime(ell)

        Fl = GF(ell)
        x = Fl.polynomial_ring().gens()[0]
        lam = (x**2 + x + Fl(self.p+1)/4).any_root()
        b2 = (1 - self.pi)/2 + ZZ(lam)
        if (b2-ell).norm() < b2.norm():
            b2 -= ell

        frak_ell = (ell, b2)
        # frak_ell = self.max_order*ell + self.max_order*b2
        return frak_ell

def find_hamilton_rec(boolS,v,n,E):
    # V = [0, ..., n-1]
    # boolS[i]== True if i\in S and False otherwise
    boolT = boolS.copy()
    boolT[v] = False
    m = sum(boolT)
    if m==0:
        return [v]
    for w in range(n):
        # if w\in S and v<-->w
        if boolT[w] and ((v,w) in E or (w,v) in E):
            P=find_hamilton_rec(boolT,w,n,E)
            if len(P)==m:
                return P+[v]
    return []

def find_hamilton(n,E):
    boolS=[True for i in range(n)]
    for v in range(n):
        P=find_hamilton_rec(boolS,v,n,E)
        if len(P)==n:
            return P
    return False

def zeros_to_edges(d_zeros):
    E=[]
    for i in range(16):
        for j in d_zeros:
            k=i^j
            if (i not in d_zeros[j]) and ((i,k) not in E) and ((k,i) not in E):
                E.append((i,k))
    return E

P_Gray = [0,1,3,2,6,7,5,4,12,13,15,14,10,11,9,8]

def is_correct_path(P,d_zeros):
    for i in range(15):
        t = P[i+1]^P[i]
        if (t not in d_zeros) or (i in d_zeros[t]):
            return False
    return True

def act_on_path(P,aux,sigma1,v,sigma2):
    Q=[]
    if aux == 12:
        for bin1 in P:
            bin2=0
            for i in range(2):
                bin2 += ((bin1 & (1<<(sigma1[i])))>>(sigma1[i]))<<i
            bin2 = bin2^v
            bin2 = bin2+sigma2[bin1>>2]<<2
            Q.append(bin2)
    else:
        for bin1 in P:
            bin2 = sigma2[bin1&3]^v
            for i in range(2):
                bin2 += ((bin1 & (1<<(sigma1[i]+2)))>>(sigma1[i]+2))<<(i+2)
            Q.append(bin2)
    return Q

Sigmas = [[0, 1, 2, 3],
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
 [3, 2, 1, 0]]

def find_auto(d_zeros,aux):
    for sigma1 in [[0,1],[1,0]]:
        for v in range(4):
            for sigma2 in Sigmas:
                P=act_on_path(P_Gray,aux,sigma1,v,sigma2)
                if is_correct_path(P,d_zeros):
                    return sigma1,v,sigma2
    return False

if __name__=="__main__":
    EGA = qtPegasis(500)
    d_hamilton = {}
    nb_4 = 0
    nb_5 = 0
    nb_3_only = 0
    nb_12_only = 0
    nb_tot = 0
    nb_fail_aut = 0
    for N1 in [3]:#[1,3,5,7]:
        for s1 in [7]:#range(8):
            for s2 in [1]:#range(8):
                for s3 in [0]:#range(8):
                    for s4 in [5]:#range(8):
                        if (s1%2 == 0 and s2%2 == 0) or (s3%2 == 0 and s4%2 == 0):
                            continue
                        if s1%4 == 2 or s2%4 == 2 or s3%4 == 2 or s4%4 == 2:
                            continue

                        alpha = inverse_mod(N1, 16)
                        beta = inverse_mod(-N1, 16)

                        M = Matrix(Integers(4), [
                            [0,0,0,0,-alpha,0,0,0],
                            [0,0,0,0,0,-alpha,0,0],
                            [0,0,beta*s3,-beta*s1,0,0,0,0],
                            [0,0,beta*s2, beta*s4,0,0,0,0],
                            [N1,0,s3,-s1,0,0,0,0],
                            [0,N1,s2,s4,0,0,0,0],
                            [0,0,0,0,1,0,alpha*s4,-alpha*s2],
                            [0,0,0,0,0,1,alpha*s1,alpha*s3]]).transpose()

                        if is_symplectic_matrix_dim4(M):
                            print("N1 = {}, s1 = {}, s2 = {}, s3 = {}, s4 = {} mod 8.".format(N1,s1,s2,s3,s4))
                            nb_tot += 1
                            print(nb_tot)
                            d_zeros = EGA.first_two_isog(N1,s1,s2,s3,s4)

                            d_zeros_4={1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8]}
                            d_zeros_3={1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],3:d_zeros[3]}
                            d_zeros_12={1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],12:d_zeros[12]}

                            E_4=zeros_to_edges(d_zeros_4)
                            E_3=zeros_to_edges(d_zeros_3)
                            E_12=zeros_to_edges(d_zeros_12)

                            L_ham_4 = find_hamilton(16,E_4)
                            L_ham_3 = find_hamilton(16,E_3)
                            L_ham_12 = find_hamilton(16,E_12)

                            aut_3 = find_auto(d_zeros_3,3)
                            aut_12 = find_auto(d_zeros_12,12)

                            if (not aut_3) or (not aut_12):
                                nb_fail_aut += 1

                            d_hamilton[(N1,s1,s2,s3,s4)] = [L_ham_4,L_ham_3,L_ham_12]

                            if L_ham_4:
                                nb_4 += 1
                            elif L_ham_3 and L_ham_12:
                                nb_5 += 1
                            elif L_ham_3:
                                nb_3_only += 1
                            elif L_ham_12:
                                nb_12_only +=1




                            















if False:
    L_d_zeros=[]
    L_path_4=[]
    L_path_3=[]
    L_path_12=[]
    for i in range(n):
        frak_a = EGA.sample_ideal()
        d_zeros=EGA.qt_action(frak_a)
        if d_zeros not in L_d_zeros:
            typ+=1
            L_d_zeros.append(d_zeros)
            d_zeros_4={1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8]}
            d_zeros_3={1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],3:d_zeros[3]}
            d_zeros_12={1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],12:d_zeros[12]}

            L_path_4.append(find_auto(d_zeros_4,0))
            L_path_3.append(find_auto(d_zeros_3,3))
            L_path_12.append(find_auto(d_zeros_12,12))
            print("Type {}:".format(typ))
            print("Zeros:")
            for x in d_zeros:
                print("{}: {}".format(x,d_zeros[x]))
            
            print("Hamiltonian path with 4 points:")
            if L_path_4[-1]==False:
                print(False)
            else:
                sigma1,s=L_path_4[-1]
                print("sigma1 = {}".format(sigma1))
                print("s = {}".format(s))

            print("Hamiltonian path with T_1+T_2:")
            if L_path_3[-1]==False:
                print(False)
            else:
                sigma1,s,sigma2=L_path_3[-1]
                print("sigma1 = {}".format(sigma1))
                print("s = {}".format(s))
                print("sigma2 = {}".format(sigma2))

            print("Hamiltonian path with T_3+T_4:")
            if L_path_12[-1]==False:
                print(False)
            else:
                sigma1,s,sigma2=L_path_12[-1]
                print("sigma1 = {}".format(sigma1))
                print("s = {}".format(s))
                print("sigma2 = {}\n".format(sigma2))
























