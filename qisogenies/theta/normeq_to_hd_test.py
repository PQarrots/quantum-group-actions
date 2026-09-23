from qarton.modular_arithmetic import ModInt
from qarton.tests import run_real

from qisogenies.montgomery import AffMontgomeryPoint, ECMontgomery

from .normeq_to_hd import NormeqOutputToHdKernel, NormeqOutputToRabbitType

################
# from qarton.circuit import Circuit
# from qarton.modular_arithmetic import ModInt, ModIntType, qc_inv_modint
# class Circ(Circuit):
#     def __init__(self, p):
#         super().__init__()
#         a = self.add_input_output(ModIntType(p))
#         qc_inv_modint(a)
#     def dummy_classical_function(self, args):
#         a = args
#         return a**(-1)
#     def dummy_classical_function_inverse(self, args):
#         a = args
#         return a**(-1)
# qc = Circ(16)
# import random
# for x in (random.randint(1, 7) for _ in range(10)):
#     n = ModInt(2*x + 1, 16)
#     assert qc.simulate(n,) == qc.dummy_classical_function(n)

# qc = Circ(31)
# import random
# for x in (random.randint(1, 7) for _ in range(10)):
#     n = ModInt(x, 31)
#     assert qc.simulate(n,) == qc.dummy_classical_function(n)

# qc = Circ(17)
# import random
# for x in (random.randint(1, 16) for _ in range(10)):
#     n = ModInt(x, 17)
#     assert qc.simulate(n,) == qc.dummy_classical_function(n)


# qc = Circ(17*19)
# import random
# for x in (random.randint(1, 16) for _ in range(10)):
#     n = ModInt(x, 17*19)
#     assert qc.simulate(n,) == qc.dummy_classical_function(n)

# qc = Circ(34)
# import random
# for x in (random.randint(0, 7) for _ in range(10)):
#     n = ModInt(2*x + 1, 34)
#     assert qc.simulate(n,) == qc.dummy_classical_function(n)

###################################

""" # Run this in qt_Pegasis_Fp to get input data

from qt_pegasis_Fp.[...].qt_pegasis import qtPegasisFp

def test_basis_gen():
    from sage.all import EllipticCurve, GF, choice
    lvl = "500"
    EGA = qtPegasisFp(lvl)
    print(f"e={EGA.e - 3}")
    print(f"p={EGA.p}")

    p, e = EGA.p, EGA.e - 3

    # random Fp curve that's not j=1728
    A = EGA.qt_action(EGA.sample_ideal())
    Em = EllipticCurve(GF(p), [0, A, 0, 1, 0])
    P_, Q_, TP_, TQ_, Emt = EGA.TwoTorsBasis(Em, e + 2)

    # transform to Weierstrass to generate examples for Qarton
    E = Em.short_weierstrass_model()
    iso = Em.isomorphism_to(E)
    P, TP = map(iso, (P_, TP_))
    assert P in E and TP in E
    Et = E.quadratic_twist()
    isot = Emt.isomorphism_to(Et)
    Q, TQ = map(isot, (Q_, TQ_))
    assert Q in Et and TQ in Et

    print(f"{p = }")
    print(f"E = EC(p, {E.a4()}, {E.a6()})")
    print(f"Et = EC(p, {Et.a4()}, {Et.a6()})")

    for pt, name, curve in zip((P, Q,  TP, TQ), ("P", "Q", "TP", "TQ"), ("E", "Et", "E", "Et")):
        print(f"{name} = AffPoint({pt.x()}, {pt.y()}, {curve})")

test_basis_gen()
"""


@run_real(NormeqOutputToHdKernel)
def test_NormeqOutputToHdKernel() -> None:
    e = 497
    p = 88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151
    M = 2 ** (e + 2)
    inputs = (
        ModInt(  # N1
            271965841585097467492828140505618316501832768566942316799131547856625873682042067933014981670916149914767673273504778391978052501825243699138622787721,
            M,
        ),
        ModInt(  # sigma1
            1470541006353854950935788223815838361419326454674183354564551999975554457192092066872276170407155449854046881534090136244222939734999210286586450694361,
            M,
        ),
        ModInt(  # sigma2
            1470541006353854950935788223815838361419326454674183354564551999975554457192074324979373206457418749042734336217014271748492470772591000473902546660295,
            M,
        ),
        ModInt(  # sigma3
            52474306520799205031150884068158020575384380008760475650479698031636230275170578009767955628395550006809201114450554990292008184092445440625843282775,
            M,
        ),
        ModInt(  # sigma4
            52474306520799205031150884068158020575384380008760475650479698031636230275170555297829712481271840642636383501051932458931356636366182917088071822592,
            M,
        ),
    )
    E = ECMontgomery(
        p,
        37736885071025214817332768723342824736266080999352595341106952477881295281652207918506128806919468446407825435245265362104704784136894378114079285399799,
        1,
    )
    Et = ECMontgomery(
        p,
        50644661342170615673023353091002352373583254243810153974941914460714317221274005063342163685879943796666115036198856555144161142602011234325790959513352,
        1,
    )
    P = AffMontgomeryPoint(
        75842956333104591486362497705518581862413486069457268759035178333250938214116831897530472963163407866486709591443158395578555782818416115605709813408331,
        24990257162407806697624109847882868309478533166179065227790041348520517712306242436664437817840001826675965008383181777515071517711597480517129563395937,
        E,
    )
    Q = AffMontgomeryPoint(
        71540364242722791201132302916298739316641094987971415881090191005639930900909566182585128003509916083067279724458627997898736996663377163535139255370480,
        24990257162407806697624109847882868309478533166179065227790041348520517712306242436664437817840001826675965008383181777515071517711597480517129563395937,
        Et,
    )
    TP = AffMontgomeryPoint(
        46342069251788815387793158301782509827810863162324301096996927133103309908066739348396818726226452013246685169214326157464342356446972282255220401475501,
        0,
        E,
    )
    TQ = AffMontgomeryPoint(
        42039477161407015102562963512562667282038472080838448219051939805492302594859473633451473766572960229827255302229795759784523570291933330184649843437650,
        0,
        Et,
    )

    qc = NormeqOutputToHdKernel(e, P, Q, TP, TQ)
    # print(qc.dummy_classical_function(inputs))
    sim = qc.simulate(inputs)
    assert sim == qc.dummy_classical_function(inputs)


# test_NormeqOutputToHdKernel()


@run_real(NormeqOutputToRabbitType)
def test_NormeqOutputToRabbitType() -> None:
    import random

    zero_idx = random.randint(0, 3)
    vals = [random.choice([1, 3]) for _ in range(3)]
    vals.insert(zero_idx, 0)
    inputs = tuple(vals)

    qc = NormeqOutputToRabbitType()
    sim = qc.simulate(inputs)  # type: ignore
    assert sim == qc.dummy_classical_function(inputs)  # type: ignore


# test_NormeqOutputToRabbitType()
