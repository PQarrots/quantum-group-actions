"""
Successive search levels of our quantum norm equation circuit.

* The first level (QuantumSearchLevel1) amplifies the amplitude over the choices that
  pass the first test. It is a QAA where the amplified algorithm produces the uniform
  superposition.

* The second level (QuantumSearchLevel2) is a QAA where the amplified algorithm is
  QuantumSearchLevel1, and it amplifies the amplitude over the choices that pass the
  first and the second test.

* The third level (QuantumSearchLevel3) is a QAA where the amplified algorithm is
  QuantumSearchLevel2, and it amplifies the amplitude over the choices that pass the
  first, second and third tests.

"""

from qarton.binary_operations import qc_flip, qc_flip_neg
from qarton.circuit import (
    BackendSpecifier,
    EmptyBackendSpecifier,
    InPlaceCircuit,
    ITupleType,
    PreCircuit,
    UInt,
    UIntType,
    memoize,
)

from qisogenies.arithmetic.cornacchia import CornacchiaTestOnly
from qisogenies.arithmetic.trial_division_sieve import TrialDivisionSieveNotTwo

from .first_check import FirstCheckTestOnly
from .ring_integers import RingInteger, RingIntegerType
from .util import InstanceData, IterationsData

__all__ = ["QuantumSearchLevel1", "QuantumSearchLevel2"]


@memoize
class QuantumSearchLevel1Iterate(
    InPlaceCircuit[
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ]
):
    """
    Iterate of the quantum search at level 1. Takes as input the precomputation result,
    and the choice register. Maps back to the same by uncomputation
    - computation with flip in the middle.
    """

    def __init__(
        self,
        d: InstanceData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        w, p, x = d.w, d.p, d.x

        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )

        choice = self.add_input_output(UIntType(x))

        # ---------------------

        # test choice
        algo_input, precomp_result_reg, choice, z, bv, which_case_when_2mod8 = (
            self.append(
                PreCircuit(FirstCheckTestOnly, d),
                algo_input,
                precomp_result_reg,
                choice,
            )
        )

        # now do second test: we only need z for that
        z, bits = self.append(
            PreCircuit(TrialDivisionSieveNotTwo, len(z), d.nb_primes), z
        )
        # flip the bits: we want all of them to be 0
        self.x_reg(bits)
        # qc_flip(bv + bits)
        # self.flip_global_phase()
        qc_flip(bv + bits)
        self.flip_global_phase()
        self.x_reg(bits)
        self.append(
            PreCircuit(TrialDivisionSieveNotTwo, len(z), d.nb_primes, inverse=True),
            z,
            bits,
        )

        self.append(
            PreCircuit(FirstCheckTestOnly, d).inverse(),
            algo_input,
            precomp_result_reg,
            choice,
            z,
            bv,
            which_case_when_2mod8,
        )

        # diffuse choice
        self.h_reg(choice)
        qc_flip_neg(choice)
        self.h_reg(choice)


@memoize
class QuantumSearchLevel1(
    InPlaceCircuit[
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ]
):
    """
    On input the precomputation result, produce a filtered superposition of x.
    (Does not include the computation result of the first
    test, it will have to be re-done on the output).
    """

    def __init__(
        self,
        d: InstanceData,
        it: IterationsData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        x, w, p = d.x, d.w, d.p
        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )

        # initialize choice
        choice = self.add_input_output(UIntType(x))
        self.h_reg(choice)

        iterate = PreCircuit(QuantumSearchLevel1Iterate, d)
        self.append_iterated(
            iterate, algo_input, precomp_result_reg, choice, iterations=it.level1
        )


@memoize
class QuantumSearchLevel2Iterate(InPlaceCircuit):
    """
    Iterate of the quantum search at level 2. Takes as input the precomputation result,
    and the choice register.

    Computes - uncomputes the search at level 1 & test on its output.
    """

    def __init__(
        self,
        d: InstanceData,
        it: IterationsData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        w, p, x = d.w, d.p, d.x

        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )

        choice = self.add_input_output(UIntType(x))

        # ---------------------

        # test choice
        # redo first test
        algo_input, precomp_result_reg, choice, z, bv, which_case_when_2mod8 = (
            self.append(
                PreCircuit(FirstCheckTestOnly, d),
                algo_input,
                precomp_result_reg,
                choice,
            )
        )
        # now do second test: we only need z for that
        z, bits = self.append(
            PreCircuit(TrialDivisionSieveNotTwo, len(z), d.nb_primes), z
        )
        # flip the bits: we want all of them to be 0
        self.x_reg(bits)
        # now do third test: compute cornacchia
        z, b, garbage = self.append(PreCircuit(CornacchiaTestOnly, len(z)), z)
        qc_flip(bv + bits + b)
        self.flip_global_phase()
        self.x_reg(bits)
        self.append(PreCircuit(CornacchiaTestOnly, len(z), inverse=True), z, b, garbage)
        self.append(
            PreCircuit(TrialDivisionSieveNotTwo, len(z), d.nb_primes, inverse=True),
            z,
            bits,
        )
        self.append(
            PreCircuit(FirstCheckTestOnly, d, inverse=True),
            algo_input,
            precomp_result_reg,
            choice,
            z,
            bv,
            which_case_when_2mod8,
        )

        # apply inverse of level 2
        self.append(
            PreCircuit(QuantumSearchLevel1, d, it, inverse=True),
            algo_input,
            precomp_result_reg,
            choice,
        )
        qc_flip_neg(choice)
        self.append(
            PreCircuit(QuantumSearchLevel1, d, it),
            algo_input,
            precomp_result_reg,
            choice,
        )


@memoize
class QuantumSearchLevel2(
    InPlaceCircuit[
        tuple[
            tuple[UInt, RingInteger],
            tuple[UInt, UInt, UInt, UInt],
            UInt,
        ],
    ]
):
    """
    On input the precomputation result, produce a filtered superposition of x.
    """

    def __init__(
        self,
        d: InstanceData,
        it: IterationsData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        super().__init__()
        w, p, x = d.w, d.p, d.x

        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )

        precomp_result_reg = self.add_input_output(
            ITupleType(
                UIntType(w // 4),
                UIntType(w // 2),
                UIntType(w // 2),
                UIntType(w // 2 + 10),
            )
        )

        # initialize choice
        choice = self.add_input_output(UIntType(x))

        # --------------------------
        self.append(
            PreCircuit(QuantumSearchLevel1, d, it),
            algo_input,
            precomp_result_reg,
            choice,
        )

        iterate = PreCircuit(QuantumSearchLevel2Iterate, d, it)
        self.append_iterated(
            iterate,
            algo_input,
            precomp_result_reg,
            choice,
            iterations=it.level2,
        )
