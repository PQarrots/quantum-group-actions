"""
Tests of the circuits with random inputs which were obtained by running the original
qt-pegasis code.
"""

from qarton.circuit import UInt
from qarton.tests import run_real

from .compute_stzk import ComputeSTZK, ComputeSTZKStep2
from .util import InstanceData


@run_real(ComputeSTZKStep2)
def test_ComputeSTZKStep2() -> None:
    instance_data = InstanceData(
        w=512,
        x=22,
        e=497,
        p=88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
        nb_primes=20,
    )

    inputs = (
        (
            UInt(1235635957169073885777546167145055163),
            UInt(
                21748518120581998297125348544233471294480841448286905013213612014716530104
            ),
        ),
        (
            UInt(
                312096645971942027628782403112681938008980932050952590274978708363732209189
            ),
            UInt(
                33439986929351778198844967746852441818893112932616425633187132840643030413
            ),
        ),
    )

    qc = ComputeSTZKStep2(instance_data)
    assert qc.simulate(inputs) == qc.dummy_classical_function(inputs)


@run_real(ComputeSTZK)
def test_ComputeSTZK() -> None:
    instance_data = InstanceData(
        w=512,
        x=22,
        e=497,
        p=88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
        nb_primes=20,
    )
    n_tralpha = (
        UInt(
            1319151932912851252711038610754328142703683672798922447731831499702203058047
        ),
        UInt(
            846585632695695021627080277722428359265950785317444268609182397992420577947
        ),
    )
    precomp_result = (
        UInt(1235635957169073885777546167146029175),
        UInt(
            21748518120581998297125348544233471294480841448286905013213612014716530104
        ),
        UInt(
            308210978739082153555666083286211760468688137227667358018107710673394587936
        ),
        UInt(
            16885517775413214593640358021273065434886827357231582692125720406885063852170
        ),
    )
    rchoice = UInt(3296705)

    qc = ComputeSTZK(instance_data)
    t = (n_tralpha, precomp_result, rchoice)
    assert qc.simulate(t) == qc.dummy_classical_function(t)
