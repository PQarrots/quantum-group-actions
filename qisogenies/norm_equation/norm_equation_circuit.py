"""
This is the main circuit that, on input (N, alpha):

* performs the precomputation

* performs the quantum search for a good value of k. Actually the search is on a value
  x, drawn at random from a small integer interval, which determines k.

* performs the post-computation on the obtained result

"""

from qarton.binary_operations import qc_mcx
from qarton.circuit import (
    BackendSpecifier,
    BoolType,
    Circuit,
    EmptyBackendSpecifier,
    ITupleType,
    PreCircuit,
    QartonBool,
    UInt,
    UIntType,
    memoize,
    qc_make_ituple,
)
from qarton.modular_arithmetic import ModInt

from qisogenies.arithmetic import Cornacchia
from qisogenies.norm_equation.post_computation import PostComp

from .first_check import FirstCheck
from .precomp import Precomp
from .ring_integers import RingInteger, RingIntegerType
from .search_levels import QuantumSearchLevel2
from .util import InstanceData, IterationsData

__all__ = ["NormEquation"]


@memoize
class NormEquation(
    Circuit[
        tuple[UInt, RingInteger],
        tuple[
            UInt,
            RingInteger,
            UInt,
            QartonBool,
            tuple[ModInt, ModInt, ModInt, ModInt, ModInt],
            UInt,
        ],
    ]
):
    """
    On input the input of norm_eq, returns a filtered superposition of x. The
    amplitude over the good x depends on the number of iterations.

    Outputs are:

    - algorithm input (preserved)
    - choice register
    - Boolean indicating if the choice was correct
    - N1_sigmas (from post-computation)
    - splitting type (from post-computation)

    """

    def __init__(
        self,
        d: InstanceData,
        it: IterationsData,
        backends: BackendSpecifier = EmptyBackendSpecifier,
    ) -> None:
        """
        :param d: Data of the instance (w,p,e,x)
        :type d: InstanceData
        :param it: Number of iterations to set
        :type it: IterationsData
        """
        super().__init__()
        w, p, x = d.w, d.p, d.x
        self.d = d

        # input of the algorithm is N and alpha
        algo_input = self.add_input_output(
            ITupleType(UIntType(w // 2), RingIntegerType(p, w // 2))
        )
        n_reg = algo_input.a[0]
        # ------------------------

        _, precomp_result_reg = self.append(PreCircuit(Precomp, d), algo_input)

        # initialize choice
        choice = self.add_anc_output(UIntType(x))  # has to be kept

        # sample good choice
        self.append(
            PreCircuit(QuantumSearchLevel2, d, it),
            algo_input,
            precomp_result_reg,
            choice,
        )

        # now choice contains a good choice (w.h.p). We redo all the computations:
        (_, _, _, delta_1, delta_2, d1, d2, z, bvv, which_case_when_2mod8) = self.append(
            PreCircuit(FirstCheck, d), algo_input, precomp_result_reg, choice
        )

        # compute Cornacchia (full)
        _, b, b1, b2 = self.append(
            PreCircuit(
                Cornacchia,
                len(z),
            ),
            z,
        )

        is_solution = self.add_anc_output(BoolType())
        qc_mcx(bvv + b, is_solution)

        post_computation_input = qc_make_ituple(
            delta_1, delta_2, z, which_case_when_2mod8, d1, d2, b1, b2, n_reg
        )
        post_computation_output, N1_sigmas, splitting_type = self.append(
            PreCircuit(PostComp, d), post_computation_input
        )
        (
            delta_1,
            delta_2,
            z,
            which_case_when_2mod8,
            d1,
            d2,
            b1,
            b2,
            n_reg,
        ) = (post_computation_output.a[i] for i in range(9))
        # we only need N1_sigmas and splitting_type. The rest can be uncomputed
        self.add_output(N1_sigmas)
        self.add_output(splitting_type)

        # erase b, b1, b2
        self.append(PreCircuit(Cornacchia, len(z), inverse=True), z, b, b1, b2)
        assert b1._destroyed  # type: ignore
        assert b2._destroyed  # type: ignore
        assert b._destroyed  # type: ignore

        # uncompute firstcheck. Only algo_input, precomp_result_reg, choice, remain
        self.append(
            PreCircuit(FirstCheck, d, inverse=True),
            algo_input,
            precomp_result_reg,
            choice,
            delta_1,
            delta_2,
            d1,
            d2,
            z,
            bvv,
            which_case_when_2mod8,
        )
        assert bvv._destroyed  # type: ignore

        self.append(
            PreCircuit(Precomp, d, inverse=True), algo_input, precomp_result_reg
        )
