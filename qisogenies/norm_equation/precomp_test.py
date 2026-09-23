from qarton.circuit import UInt
from qarton.tests import run_real

from qisogenies.norm_equation.ring_integers import RingInteger

from .precomp import Precomp
from .util import InstanceData


@run_real(Precomp)
def test_Precomp() -> None:
    instance_data = InstanceData(
        w=512,
        x=22,
        e=497,
        p=88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
        nb_primes=20,
    )

    N = UInt(
        1319151932912851252711038610754328142703683672798922447731831499702203058047
    )
    alpha = RingInteger(
        instance_data.p,
        instance_data.w // 2,
        a=423292816347847510813540138861214179632975392658722134304591198996210288974,
        b=-1,
    )

    qc = Precomp(instance_data)
    outputs = qc.simulate((N, alpha))
    assert outputs[0] == (N, alpha)
    assert outputs[1] == (
        1235635957169073885777546167146029175,
        21748518120581998297125348544233471294480841448286905013213612014716530104,
        308210978739082153555666083286211760468688137227667358018107710673394587936,
        16885517775413214593640358021273065434886827357231582692125720406885063852170,
    )
    assert qc.simulate((N, alpha)) == qc.dummy_classical_function((N, alpha))
