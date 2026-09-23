from qarton.circuit import ITupleType, UIntType
from qarton.signed_arithmetic import SIntType

from qisogenies.norm_equation.norm_equation_circuit import NormEquation
from qisogenies.norm_equation.ring_integers import RingIntegerType
from qisogenies.norm_equation.util import FAST_BACKENDS, InstanceData, IterationsData


def test_NormEquation() -> None:

    # construct the full circuit for the 500-bit parameter.
    # It works (so the types agree everywhere), but the construction is a bit slow.

    instance_data = InstanceData(
        w=512,
        x=20,
        e=500,
        p=88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
        nb_primes=20,
    )

    iterations_data = IterationsData(30, 4)

    qc = NormEquation(instance_data, iterations_data, backends=FAST_BACKENDS)
    print(qc.nbr_qubits())


def test_types() -> None:

    t1 = ITupleType(
        RingIntegerType(
            88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
            394,
        ),
        RingIntegerType(
            88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
            394,
        ),
        SIntType(
            512,
        ),
        SIntType(
            512,
        ),
        UIntType(
            139,
        ),
        UIntType(
            139,
        ),
        UIntType(
            256,
        ),
    )
    t2 = ITupleType(
        RingIntegerType(
            88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
            394,
        ),
        RingIntegerType(
            88381546413195830490356121814345177109849335243162749316048866938595612502926212981848292492799412243073940471444121917248865926738905612439870244913151,
            394,
        ),
        SIntType(
            512,
        ),
        SIntType(
            512,
        ),
        UIntType(
            139,
        ),
        UIntType(
            139,
        ),
        UIntType(
            256,
        ),
    )
    assert t1 == t2
