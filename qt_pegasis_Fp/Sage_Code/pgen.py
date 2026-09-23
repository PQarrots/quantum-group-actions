from sage.all import *
proof.all(False)
from .theta_lib.pkg.basis_change.canonical_basis_dim1 import make_canonical
from .xonly import xPoint, random_xPoint, MontgomeryA, isWeierstrass, translate_by_T

from pathlib import Path
import json

def gen_params(p, A):
    """
    Given p and the montgomery coefficient of the starting curve A, generate
    and print all the precomputation for the starting curve.
    """
    e = valuation(p+1, 2)
    f = (p+1)//(2**e)

    Fp = GF(p)
    Fp2 = GF((p, 2), name='i', modulus=var('x')**2 + 1)

    E_start = EllipticCurve(Fp, [0, A, 0, 1, 0])
    print('Setup done')

    e_sol = e - 3

    EE = E_start.change_ring(Fp2)

    P, Q, TP, TQ = TwoTorsBasis(E_start, e_sol+2)

    dt_two = 2**(e_sol-2)
    P16, Q16 = dt_two*P, dt_two*Q
    P4, Q4 = 4*P16, 4*Q16
    ePQ4 = P4.weil_pairing(Q4, 4)

    assert TP.y() == 0 and TP.x() in Fp
    assert TQ.y() == 0 and TQ.x() in Fp

    # Change of basis
    _, _, R, S, M0 = make_canonical(P4, Q4, 4, preserve_pairing=True)
    (_x, _y), (_z, _w) = M0
    assert _x*R + _y*S == P4 and _z*R + _w*S == Q4

    # Strategy
    strategy = optimised_strategy_qt_pegasis(e-3)
    data = {
        'f': int(f),
        'e': int(e),
        'A': int(A),
        'Px': int(P.x()),
        'Py': int(P.y()),
        'Qx': int(Q.x()),
        'Qy': int(Q.y()[1]),
        'TPx': int(TP.x()),
        'TQx': int(TQ.x()),
        'P16x': int(P16.x()),
        'P16y': int(P16.y()),
        'Q16x': int(Q16.x()),
        'Q16y': int(Q16.y()[1]),
        'Rx': int(R.x()),
        'Ry': int(R.y()[0]),
        'Ryi': int(R.y()[1]),
        'Sx': int(S.x()),
        'Sy': int(S.y()[0]),
        'Syi': int(S.y()[1]),
        'ePQ4': int(ePQ4[1]),
        '_xyzw': [int(_x), int(_y), int(_z), int(_w)],
        'strategy': strategy
    }
    return data

    # Print all stuff in hex
    # print(f"'f':{f},")
    # print(f"'e':{e},")
    # print(f"'A':{hex(ZZ(A))},")
    # ...

# Strategies imported from C_Code/scripts/generate_Fp
def optimised_strategy_with_first_eval(n,mul_c=1,first_eval_c=1,precomp_dbl=1):
    r"""
    Adapted from optimised_strategy when the fist isogeny evaluation is more costly.
    This is well suited to gluing comptations. Computes optimal strategies with constraint
    at the beginning. This takes into account the fact that doublings on the codomain of 
    the first isogeny are impossible (because of zero dual theta constants).

    INPUT:
    - n: number of leaves of the strategy (length of the isogeny).
    - mul_c: relative cost of one doubling compared to one generic 2-isogeny evaluation.
    - first_eval_c: relative cost of an evaluation of the first 2-isogeny (gluing) 
    compared to one generic 2-isogeny evaluation.
    - precomp_dbl: precomputation performed on the theta structure before doublings.

    OUTPUT:
    - S_left[n]: an optimal strategy of depth n with constraint at the beginning
    represented as a sequence [s_0,...,s_{t-2}], where there is an index i for every 
    internal node of the strategy, where indices are ordered depth-first left-first 
    (as the way we move on the strategy) and s_i is the number of leaves to the right 
    of internal node i (see https://sike.org/files/SIDH-spec.pdf, pp. 16-17).
    """

    eval_c = 1.000
    first_eval_c = first_eval_c
    mul_c  = mul_c

    S_left = {1:[], 2:[1]} # Optimal strategies "on the left" i.e. meeting the first left edge 
    S_right = {1:[]} # Optimal strategies "on the right" i.e. not meeting the first left edge
    C_left = {1:0, 2:mul_c+first_eval_c } # Cost of strategies on the left
    C_right = {1:0 } # Cost of strategies on the right
    for i in range(2, n+1):
        # Optimisation on the right
        # If b>1, then a precomputation is necessary to initialize the theta structure at root of the left part of the tree
        b, cost = min(((b, C_right[i-b] + C_right[b] + b*mul_c + (i-b)*eval_c+min(1,b-1)*precomp_dbl) for b in range(1,i)), key=lambda t: t[1])
        S_right[i] = [b] + S_right[i-b] + S_right[b]
        C_right[i] = cost

    for i in range(3,n+1):
        # Optimisation on the left
        # If b>1, then a precomputation is necessary to initialize the theta structure at root of the left part of the tree
        b, cost = min(((b, C_left[i-b] + C_right[b] + b*mul_c + (i-1-b)*eval_c+first_eval_c+min(1,b-1)*precomp_dbl) for b in range(1,i)), key=lambda t: t[1])
        S_left[i] = [b] + S_left[i-b] + S_right[b]
        C_left[i] = cost

    return S_left[n]

def optimised_strategy_qt_pegasis(e):
    r"""
    INPUT: 
    - e: length of the chain (the first isogeny counts twice). This 
    is NOT the prime exponent e2 (e=e2-3).

    OUTPUT: optimised strategy with first isogeny evaluation taking 
    accurate costs into account.
    """
    pname = 500*int(round(e/500,0))
    # Costs in cycles from a benchmark ran on a 2,3 GHz Intel Core i7 (4 core)
    if pname == 500:
        M = 92
        S = 67
        a = 56
    elif pname == 1000:
        M = 292
        S = 191
        a = 129
    elif pname == 1500:
        M = 1176
        S = 417
        a = 176
    elif pname == 2000:
        M = 2260
        S = 693
        a = 233
    elif pname == 4000:
        M = 7531
        S = 3336
        a = 336
    else:
        print("Unrecognized exponent. Set to default: 500.")
        M = 92
        S = 67
        a = 56
    def balanced_strategy(n: int) -> tuple[int, ...]:
        if n < 1:
            raise ValueError(f"{n} must be positive")
        if n == 1:
            return tuple()
        return (
            int((n + 1) // 2),
            *balanced_strategy(n - (n + 1) // 2),
            *balanced_strategy((n + 1) // 2),
        )
        

    eval_c = 16*M + 16*S + 128*a
    first_eval_c = 159*M + 52*S + 263*a
    mul_c = 32*M + 32*S + 128*a
    precomp_dbl = 132*M + 16*S + 128*a

    # The first isogeny counts twice so the chain has apparent length e-1
    return optimised_strategy_with_first_eval(e-1,mul_c/eval_c,first_eval_c/eval_c,precomp_dbl/eval_c)


# Basis generation from PEGASIS
def TwoTorsBasis(E, e):
    T0, Tm1, T1 = find_Ts(E)

    A = MontgomeryA(E)
    F = E.base_field()
    p = F.characteristic()

    R = F["X"]
    X = R.gens()[0]
    f = X**2 + A*X + 1

    xT0 = T0.x()
    xP = xT0 + F.random_element()**2
    while not (f(xP)*xP).is_square() or (xP-Tm1.x()).is_square():
        xP = xT0 + F.random_element()**2

    xQ = xT0 - F.random_element()**2
    while (f(xQ)*xQ).is_square() or not ((xQ-T1.x()).is_square()):
        xQ = xT0 - F.random_element()**2

    P = xPoint(xP, E)
    Q = xPoint(xQ, E)

    assert (p+1) % 2**(e+1) == 0
    cofac = (p+1)/2**(e+1)
    P = P.xMUL(cofac)
    Q = Q.xMUL(cofac)

    assert P.xMUL(2**(e-1))
    assert not P.xMUL(2**e)
    assert Q.xMUL(2**(e-1))
    assert not Q.xMUL(2**e)
    assert Q.xMUL(2**(e-1)) != P.xMUL(2**(e-1))

    return eval_omega_and_lift(E, P, Q, T0, Tm1, T1)

def eval_omega_and_lift(E, P, Q, T0, Tm1, T1):
    Plift = E.lift_x(P.X)

    mu1 = Tm1.tate_pairing(Plift,2,1)
    mu2 = T1.tate_pairing(Plift,2,1)

    if mu1 == 1 and mu2 == 1:
        TP = E(0)
    elif mu1 == -1 and mu2 == 1:
        TP = T1
    elif mu1 == 1 and mu2 == -1:
        TP = Tm1
    elif mu1 == -1 and mu2 == -1:
        TP = T0
    else:
        raise ValueError("Wrong Tate pairing P.")

    A = MontgomeryA(E)
    F =  E.base_field()
    Et = EllipticCurve(F,[0,-A,0,1,0])

    Tm1t = Et(-T1.x(),0)
    T1t = Et(-Tm1.x(),0)

    Qliftt = Et.lift_x(-Q.X)

    mu1 = Tm1t.tate_pairing(Qliftt,2,1)
    mu2 = T1t.tate_pairing(Qliftt,2,1)

    if mu1 == 1 and mu2 == 1:
        TQ = E(0)
    elif mu1 == -1 and mu2 == 1:
        TQ = Tm1
    elif mu1 == 1 and mu2 == -1:
        TQ = T1
    elif mu1 == -1 and mu2 == -1:
        TQ = T0
    else:
        raise ValueError("Wrong Tate pairing Q.")

    p = F.characteristic()
    Fp2 = GF((p, 2), name='i', modulus=var('x')**2 + 1)
    EE = E.change_ring(Fp2)
    ii = Fp2.gen()
    Qlift = EE(-Qliftt.x(),-ii*Qliftt.y())
    Plift = EE(Plift)
    TP = EE(TP)
    TQ = EE(TQ)

    return Plift, Qlift, TP, TQ

def find_Ts(E, only_T0 = False):
    r"""
    Given a curve E, finds and marks the non-trivial
    2-torsion points according to Lemma D.1
    """
    A = MontgomeryA(E)
    F =  E.base_field()
    R = F["X"]
    X = R.gens()[0]
    f = X**2 + A*X + 1
    lam1, lam2 = f.roots(multiplicities=False)

    R1 = E(lam1, 0)
    R2 = E(lam2, 0)
    R3 = E(0, 0)

    #Find T0
    Rs = [R1, R2]
    for T in Rs:
        if T.tate_pairing(T, 2, 1) != 1:
            T0 = T
            Rs.remove(T)
            break

    assert T0
    if only_T0:
        return T0

    assert T0.tate_pairing(T0, 2, 1) == -1
    Rs.append(R3)
    for T in Rs:
        if T.tate_pairing(T0, 2, 1) == 1:
            Tm1 = T
            Rs.remove(T)
            break

    assert Tm1
    T1 = Rs[0]
    assert T1.tate_pairing(T0, 2, 1) != 1

    return T0, Tm1, T1

def starting_curve(p, E0=None):
    """
    Given a prime p, compute the montgomery coefficient of a valid starting
    curve A.
    """
    # Starting curve y^3 = x^3 - x
    if not E0:
        Fp = GF(p)
        # Fp2 = GF((p, 2), name='i', modulus=var('x')**2 + 1)
        E0 = EllipticCurve(Fp, [-1, 0])

    # Fid the cofactor
    val2 = valuation(p+1, 2)
    ell = (p+1)//(2**val2)
    assert ell != 1
    ell = factor(ell)[0][0]
    cof = (p+1)//ell
    print(f'{ell = }')

    # Act with some odd ideal
    n_act = ZZ(p).nbits() // 2
    n_act += n_act // 10

    E = E0
    E._order = p+1
    #from tqdm import tqdm
    # for i in range(n_act):
    for i in range(n_act):
        print(f'Action {i+1}/{n_act} ({100*i/n_act:.1f}%)')
        while True:
            # !!!! E.random_point() is very very very slow (at the first iteration) !!!
            P = E.random_point() * cof
            if P != 0:
                assert P * ell == 0, "something is very wrong"
                break
        P._order = ell
        phi = E.isogeny(P)
        E = phi.codomain()
        E._order = p+1

    A = E.montgomery_model().a_invariants()[1]
    if (A-2).is_square():
        A = -A
    return A

def starting_curve_new(p, E0=None):
    Fp = GF(p)
    sqrt2 = Fp(2).sqrt()
    A0 = -sqrt2-1/sqrt2
    if is_square(A0-2):
        A0=-A0

    return A0

# Parameter sets for PEGASIS
peg_primes = {
    '500':33*2**503-1,
    '1000':15*2**1004-1,
    '1500':9*2**1551-1,
    '2000':51*2**2026-1,
    '4000':63*2**4084-1
}

peg_curves = {
    '500':0x102bad9db56de678020d816f2c566b728c4cf1515fda7c3687a864824ea9a8c7a03b686a2d80253ad0b82d8923200c4b48c28e67bb66d4eb586cff837915a121,
    '1000':0xe474e3ef79088b1bbaccf7ae89f9f54915df70255a113272315c2cf8848e52c753403bec0cf386862b1cb4512510bc748ca96efa6158e543dfa151df2ddba05437796407a5a463b172da19daf51de831508e6f8af1b101559164aa023111c0b58fe9b2891d8335d6dde2f9c15dd8fa51a3a7387f120f038ae45d33b499ed,
    '1500':0x1253a191007c7828135448acd08b424b1d35435105a3c3591a4f0999f36500e7f47aaefae38133f57aa7e4085fee410080818f993ae0e749c86fbfc1979797da273f6792fadb3c6bc9119410d1ad223d4795d8fa4796fe1ed6c8efdc9b291f2f6e7ab04bbdb42029b510a52fcb9d16a4ac984e28903b0db2ca83b549a050615aaaa2fc5bb8dfd9d7c2e76920bceb0d76f5d6d65ddba4ab99d1eaaf490a1a1cf2edec36db979f0b48d6f05c160572a67f18a3357140dd268e0fd6fb2a02816b,
    '2000':0x2a5264129fc329c574f17d7355b096d99dff8cd2f46b295ce5198f9278342bbe5810709c09dd2d8427eced058237c4b89bd102fe1c21eecfef5f203bf49127c887126051d51678f098c0e9a99fe85b9535df355596e5292e85d46451ad110e7955c650e592e2420a9d243ee3a4187eea1a88761b83db13e364ca9a32426072f59a684329e74f81a9504fb6359332a730ef51201182da362032ce0f10aaae693bfd7207730be75d29bbde01d3c2153abce23af50a535041c129a86c0de8c94dda5ea3b00caefb3ad0b73bd6113bfe2cfb5a9ea8aa6bba5a788c06b0f3963241b851ae65a2eafd933c076cee39c63657f8056efe9858c84a715e8f1985a3be,
    '4000':0xf24a99b4c09c41e66e81d92bbf7dc29ca187aec88c97dd176d1d33a517e0055c1462f209f9fd0292126c4a075f6bbc5a9deae336faa329abc903b0e578f3d07c0ee74531e654843e1d8580aa0ad4e7c2089e74b4032d3d98cd1717adbc328c266baaebd9800854ca5778821a1c2acc8bd9df86a4dc979e0a5deadfe6dfab34fe134b1645a0309692e6b4695ba7b9cecefebe43d16146a9b1b0c55753611857b6ccabeab15f2c18856dd1d47bc9c33dd0494fa97fa5e583ded4c31eefd949b1ba957c20e01c5c72a564a28fcf9f40368801a4f3c60ee2639636696a654ba98fb2adf486e3794355639c315c6f1349324443dec0f1c591947f3a4380fb0756fa65cb3e34e9111f5807b7a15cb6980b20cfbf446c8fa8c4d36259a6ef1a0aa543509112e7b56b9b8e04bf74b761deee6e0f51fec35f2f59d0f0474b2da20f752659780accbd930f50fbd049b20e4886f35cdae2cce8f06387be15b5a5e10a662293d1f71129a6547b741bd5f106d33e836bdcbb1a8fdd223aa8cfc204bb587fa60fb41f28fa2a01f26634699fcdce70c5b2ef93e6523e4daeeb5a79f044f4dbc545d3ecc20109514acdaa48f4bfb1901a408199742be8333901ac9f4b8dbbb2ec4b25e55a497bfd5e23c181dc167c022a5908183cb9f203176538a2ccc6b9c23cd20fe037bc4776ec4f561e9974abf7e84cdcea0fbc61bce901302b961966250b
}

# Parameter sets for qt-Pegasis
qtp_primes = {
    '20': 2**19 - 1,
    '30': 2**31 - 1,
    '128': 2**127 - 1,
    '500': 27 * 2**500 - 1,
    '1000': 15 * 2**1004 - 1, # same as in PEGASIS
    '1500': 5 * 2**1522 - 1,
    '2000': 5 * 2**2014 - 1,
    '4000': 45 * 2**4024 - 1
}

qtp_curves = {
    '20': 0,
    '30': 0,
    '128': 0,
    '500':0x4b788b08cbbbcc90ab57a927737ecaf2cffd8b0bae759f5abb96ed039acc2a6f09a3a5d244579422e527f18182052b31dbffad24c0ef0b7bbf0ff27f8ca0cb,
    '1000':0x82bfdbd34abc199d651f717570393d0a11eae8080825a5ae548a251645fcb04d72ff684d2e842adc314309201fb4e5110f017df28606ab4b1afa61df953cf85ee7b298259f4a15df9755749199f49254ff390fc9fa6c6b1ffb21ff56173776bbaa5c5145917b1904c1cddedd8e0107854372e241e5d1019d7e733b146a62,
    '1500':0x1253a191007c7828135448acd08b424b1d35435105a3c3591a4f0999f36500e7f47aaefae38133f57aa7e4085fee410080818f993ae0e749c86fbfc1979797da273f6792fadb3c6bc9119410d1ad223d4795d8fa4796fe1ed6c8efdc9b291f2f6e7ab04bbdb42029b510a52fcb9d16a4ac984e28903b0db2ca83b549a050615aaaa2fc5bb8dfd9d7c2e76920bceb0d76f5d6d65ddba4ab99d1eaaf490a1a1cf2edec36db979f0b48d6f05c160572a67f18a3357140dd268e0fd6fb2a02816b,
    '2000':0x1215e3b3750a9eb22a2a2dc072b571e84e6b460f73d3cb5f2b5401466fb1ed9905d9fa70cd68e76552430f84727062204d0eb919f2c7968e2c2ac5ffad354328ce71fb96cadee4f69488370a226cdb2c341fbafd6dcf6e8830301dd31ff76ff76a726f13f55f7eb37764b1f4116a770adf25af971d380d0f3fabc82f164ab9891cd4661698c2c9dfcb81e1088ff78f41a7b6219b93d251573a3082c15f79631ecbe34afc600719d79bd8eddd6cf515eda02024cd6b4a59b4c86b4ca08112012cff26dfe0327ed4d804146cb8869a87ace6596d440a90acab27902427d50f2158b994b6b1cc834859d63ff274f89dec3692940bc4d44ab3ffe2e0e5110,
    '4000':0x145be586ca055838ef371d4b11248effb7de554ef07e28887d947d0cbb045004bdd6a2dfdd425245b075be5d4aae9bc32f236763e1b95b111df5798cf78e1cab370aacf29b18b8ca5433a6549f1d98bb47a4377ff5218c0e81b3fef5d4be8f102702db6080577bee99692b8b3b09136ac0c5de9c9763efaa7c025dae520af36edf324c7870cb30a2127f05d421919e6e561f00c11464dee7b2b15693d2224b44b80c8220050de5404528b0329450a24d691c54ca017b38fed89b01e419fb190a6b7b89a5533152e566616b2820c00fbdc945c0c0f8f7782d570872a4ddca46bd4713ca47190423371c5aa7967b41754e12d25bc07edebdc88f7ec3f05cebf2a5df9a18dc5796d43d9c422f8a025d561cf6d81697554b07b101dd6b690f46869cf26f6e76ab015ffe0ad0ede4af6fd9681d66c67743800d6ab1e298f3acdbf20497769af3a0fcebadc70e2db4df435a6ae94affd5a052b1e7bf20ec2f303a71aa04e19af92f7dbb34734cbd4e2ee8dcb4e1bf45c55f66f793497cdb7b268ff26b8728d420507fd890a3059c318208608b54d1318d886937f443aa5ecbe5b740432402e942a28bd77867f202b9661e4f99add593aae23b0ea9f1654cb8bf635a193aa49972c1d5f57e76be2623997bfa94fe2c09a8d0beb176fe18d347e65dddc92d93efca1eefd9c070de030b9c589ebd49f847cca25a20ff
}


if __name__ == "__main__":
    set_random_seed(42) # Always get same curve

    # Choose security level and parameter set name
    # lvl = '128'
    #lvl = '500'
    #lvl = '1000'
    #lvl = '1500'
    #lvl = '2000'
    lvl = '4000'

    # PEGASIS setup
    # name = lvl + 'P'
    # p = peg_primes[lvl]
    # A = peg_curves[lvl]

    # qt-P setup
    name = lvl
    p = qtp_primes[lvl]
    # A = qtp_curves[lvl]

    print(f'Parameter set {name}')
    assert is_prime(p)

    # If needed, generate new starting curve
    A = starting_curve_new(p)
    print(f'{hex(A) = }')

    # Compute parameters
    data = gen_params(p, A)

    # Automatically generates `params/{name}.json` file that can be loaded with
    # PEGASIS(name)
    fname = Path(__file__).parent / 'params' / f'{name}.json'
    if fname.is_file():
        _x = input(f'Level {name} already existing; override? [y/N]')
        if _x.upper() != 'Y':
            exit()
    with open(fname, 'w') as fh:
        fh.write(json.dumps(data))






