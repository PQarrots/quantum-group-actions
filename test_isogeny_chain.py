"""
This script constructs the 4-dimensional isogeny evaluation and checks its correctness.

Because the Qarton simulator that we use is slow, and there are a lot of gates in
the circuit, we "dummify" most of the sub-circuits.

* In the "chain" construction (memory-heavy), the only "real" circuits during
  simulation are: ``IsogenyChain`` and ``IsogenyChainWithGarbage``, which is its main
  component. All their sub-circuits are replaced by their ``dummy_classical_function``.
  They are tested independently using our unit tests.

* In the "pebbled" construction (memory-efficient), the main circuit ``IsogenyChainWithPebbling``
  is "real", as well as its pebbling steps. The circuits inside each pebbling step
  are essentially the same sub-circuits as in ``IsogenyChainWithGarbage``, and all of these
  are "dummified".

The construction of the circuit and simulation will take some of time. Simulation
will only be possible for a reasonable parameter size, due to the large memory
required by our current implementation (which may break the python integer size limit,
and in the end, crash python and / or the computer).

The parametrization given in this script (128-bit prime) allows the simulation to run
in a reasonable time and memory usage for both parameters.

Usage
-----

To run with the "pebbled" circuit (this will take a longer time, less than 1 hour):

    python test_isogeny_chain.py --circuit pebbled

To run with the "chain" circuit:

    python test_isogeny_chain.py --circuit chain

In both cases, the simulation will display a progress bar.

Note
----

The curve we are starting from is the default E_start from the qtPegasis instance.
We did not test with another curve.

"""

import argparse
from random import seed
from typing import Any

from qarton.circuit import ResourceReporter, UInt
from qarton.modular_arithmetic import ModInt
from sage.all import GF, set_random_seed  # type: ignore

from qisogenies.montgomery import AffMontgomeryPoint, ECMontgomery
from qisogenies.norm_equation.norm_eq import qlapoti as qlapoti_qisogenies
from qisogenies.norm_equation.post_computation import postcomp_from_qlapoti_output
from qisogenies.norm_equation.ring_integers import RingInteger
from qisogenies.norm_equation.util import FAST_BACKENDS
from qisogenies.theta.isogeny_chain_dim4 import (
    InputData,
    IsogenyChain,
    IsogenyChainWithGarbage,
)
from qisogenies.theta.isogeny_chain_dim4_pebbled import IsogenyChainWithPebbling
from qisogenies.theta.normeq_to_hd import (
    NormeqOutputToHdKernel,
    normeqoutputtorabbittype_classical_function,
)
from qisogenies.theta.theta_util import CoordsDim4, balanced_strategy
from qt_pegasis_Fp.Sage_Code.qt_pegasis import qtPegasisFp
from qt_pegasis_Fp.Sage_Code.theta_lib.pkg.theta_structures.montgomery_theta import (
    torsion_to_theta_null_point,  # type: ignore
)
from qt_pegasis_Fp.Sage_Code.theta_lib.pkg.theta_structures.Theta_dim4 import (
    ThetaStructureDim4,
)
from qt_pegasis_Fp.Sage_Code.theta_lib.pkg.theta_structures.theta_helpers_dim4 import (
    product_theta_point_dim4,  # type: ignore
)
from test_norm_equation import sample_sage_ideal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the isogeny chain circuit for a small parameter and simulate it."
    )
    parser.add_argument(
        "--circuit",
        choices=["chain", "pebbled"],
        default="chain",
        help=(
            "Which isogeny chain circuit to build: 'chain' for IsogenyChain "
            "(default), or 'pebbled' for IsogenyChainWithPebbling."
        ),
    )
    return parser.parse_args()


def fp_to_modint(x: Any) -> ModInt:
    """
    Convert a sage Fp object into a Qarton ModInt.
    """
    p = x.parent().characteristic()
    assert x in GF(p)
    return ModInt(int(x.lift()), int(p))


def prepare_params_isogeny_chain(
    qtPegasis_instance: qtPegasisFp,
    domain_qtp: ThetaStructureDim4,
    E_start: Any | None = None,
) -> InputData:
    p = int(qtPegasis_instance.p)
    if E_start is None:
        E_start = qtPegasis_instance.E_start  # type: ignore
    assert E_start.a2() in GF(p)  # type: ignore
    A = int(E_start.a2().lift())  # type: ignore

    E = ECMontgomery(p, A, 1)
    E_twist = ECMontgomery(p, (-A) % p, 1)
    prod_np = CoordsDim4(fp_to_modint(x) for x in domain_qtp.null_point())  # type: ignore
    inv_np = CoordsDim4(fp_to_modint(x ** (-1)) for x in domain_qtp.null_point())  # type: ignore
    inv_dual_np = CoordsDim4(
        fp_to_modint(x ** (-1))
        for x in domain_qtp.null_point_dual()  # type: ignore
    )

    return InputData(E, E_twist, prod_np, inv_np, inv_dual_np)


def prepare_input_norm_eq_to_hd(
    P: Any, Q: Any, TP: Any, TQ: Any
) -> tuple[
    AffMontgomeryPoint, AffMontgomeryPoint, AffMontgomeryPoint, AffMontgomeryPoint
]:
    p = int(P.curve().base_ring().characteristic())
    res = tuple(
        AffMontgomeryPoint(
            int(R.x().lift()),
            int(R.y().lift()),
            ECMontgomery(p, int(R.curve().a2().lift()), 1),
        )
        for R in (P, Q, TP, TQ)
    )
    assert len(res) == 4  # for type checking
    return res


def compute_classical_input(
    qtPeg_instance: qtPegasisFp,
) -> tuple[Any, tuple[Any, Any, Any, Any]]:
    e = qtPeg_instance.e
    P, Q, TP, TQ, _ = qtPeg_instance.TwoTorsBasis(qtPeg_instance.E_start, e - 1)  # type: ignore
    P4 = P * (1 << (e - 3))  # type: ignore

    a, b = torsion_to_theta_null_point(P4)  # type: ignore
    OE4 = product_theta_point_dim4([a, b], [a, b], [a, b], [a, b])  # type: ignore

    # (Product) theta structure on E^4
    domain_prod_np = ThetaStructureDim4(OE4)  # type: ignore

    return domain_prod_np, (P, Q, TP, TQ)  # type: ignore


def ideal_sage_to_qisogenies(frak_a: Any) -> tuple[int, RingInteger]:
    N, alpha = frak_a

    # convert to ringinteger element
    p: Any = -(alpha.parent().discriminant())  # type: ignore
    p = int(p)  # type: ignore
    tr_alpha, _ = list(alpha)  # type: ignore

    # Choose sign of trace positive
    if tr_alpha < 0:
        alpha = -alpha  # type: ignore

    w = ((p.bit_length() + 10) // 4) * 4

    tmp1, tmp2 = list(alpha)  # type: ignore
    a, b = int(tmp1 + tmp2), int(-2 * tmp2)  # type: ignore
    assert abs(b) == 1
    return int(N), RingInteger(p, w, a, b)


if __name__ in ("__main__", "sage.all"):
    SEED = 2
    set_random_seed(SEED)
    seed(SEED)

    args = _parse_args()

    level = 128
    qtPeg = qtPegasisFp(level)  # type: ignore
    e, e_sol, p, A, E_start = (  # type: ignore
        int(qtPeg.e),
        int(qtPeg.e_sol),
        int(qtPeg.p),
        int(qtPeg.A),
        qtPeg.E_start,  # type: ignore
    )
    strategy_balanced = balanced_strategy(e_sol)
    strategy_optimize_memory = (*tuple(e_sol - 1 - i for i in range(e_sol - 1)), 1)

    frak_a_sage = sample_sage_ideal(qtPeg)
    frak_a = ideal_sage_to_qisogenies(frak_a_sage)

    ##########
    # start from norm equation output
    normeq_output: tuple[int, ...] = qlapoti_qisogenies(frak_a_sage, e_sol)

    codom_qtp_A = qtPeg.qt_action(frak_a_sage, normeq_output=normeq_output)  # type: ignore
    codom_qtp_Jdiv256 = (codom_qtp_A**2 - 3) ** 3 / (codom_qtp_A**2 - 4)  # type: ignore
    domain_qtp, (P, Q, TP, TQ) = compute_classical_input(qtPeg)

    #############
    # Generate N1, sigmas, rabbit type, splitting type (norm eq post-computation)
    (N1, *sigmas), splitting_type = postcomp_from_qlapoti_output(
        frak_a[0], e_sol, normeq_output
    )
    sigmas_mod4 = tuple(UInt(sigma.v % 4) for sigma in sigmas)
    assert len(sigmas_mod4) == 4  # for typing
    N1_mod2 = int(N1 % 2)
    rabbit_type = normeqoutputtorabbittype_classical_function(sigmas_mod4)[1]

    ###############
    # Test generation of HD kernel

    _tmp = (
        ModInt(N1, (1 << (e_sol + 2))),
        *(ModInt(x.v, 2 ** (e_sol + 2)) for x in sigmas),
    )
    assert len(_tmp) == 5  # for typing
    T1, T2, T3, T4, T1p2, T3p4 = NormeqOutputToHdKernel(
        e_sol, *prepare_input_norm_eq_to_hd(P, Q, TP, TQ)
    ).dummy_classical_function(_tmp)[1]

    ###########################
    # Test dim-4 isogeny chain
    ###### 1. instantiation
    print("Building the circuit...")

    if args.circuit == "chain":
        _cclass: type[IsogenyChain] | type[IsogenyChainWithPebbling] = IsogenyChain
        IsogenyChain.make_real()
        IsogenyChainWithGarbage.make_real()
        # by making both these circuits "real", we will test the proper behavior
        # of the isogeny chain
    else:
        _cclass = IsogenyChainWithPebbling
        IsogenyChainWithPebbling.make_real()

    isogeny_chain_qc = _cclass(
        e_sol,
        prepare_params_isogeny_chain(qtPeg, domain_qtp),
        strategy_balanced,
        backends=FAST_BACKENDS,
    )

    ####### Resource usage:
    r = ResourceReporter(isogeny_chain_qc)
    print("Nbr of qubits", r.nbr_qubits())
    print("Nbr of CCX gates or equivalent (log2)", r.ccx_count_log2())

    # print("Most costly sub-circuits:")
    # d = r.sub_circuits_by_class_proportion(30)
    # for k in d:
    #    print(k)

    # isogeny_chain_opt_memory_qc = IsogenyChain(
    #    e_sol, prepare_params_isogeny_chain(qtPeg, domain_qtp), strategy_optimize_memory
    # )
    # r = ResourceReporter(isogeny_chain_opt_memory_qc)
    # print("Nbr of qubits", r.nbr_bits())
    # print("Nbr of CCX gates or equivalent (log2)", r.ccx_count_log2())
    del r

    ###### 2. the result is consistent with the sage implementation
    print("Testing consistency with the Sage implementation of qt-Pegasis... ")
    inputs = ((T1, T2, T3, T4, T1p2, T3p4), N1_mod2, sigmas_mod4, splitting_type)
    J_codom = isogeny_chain_qc.dummy_classical_function(inputs)[-1]  # type: ignore
    assert codom_qtp_Jdiv256 == (J_codom).v, (
        f"{codom_qtp_Jdiv256 = }, {(J_codom).v = }"
    )  # the classical input is consistent with sage!
    print("It's consistent with Sage!")

    ###### 3. check on the full circuit
    # Note that sub-circuits are dummified, but the main circuits IsogenyChain
    # and IsogenyChainWithGarbage are *both* real in this run, so we do test the
    # proper behavior of the isogeny chain.

    print("Simulating the circuit... ")
    classical = *inputs, J_codom
    simulated = isogeny_chain_qc.simulate(inputs, progress_bar=True)  # type: ignore
    assert simulated == classical

    print("Success!")
