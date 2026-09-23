"""
Testing our python implementation of qlapoti and the circuit implementations
of its building blocks.

Usage
-----

Run with the defaults (4 ideals, the qarton-based classical implementation,
no circuit simulation)::

    python test_norm_equation.py

Test a different number of ideals::

    python test_norm_equation.py --num-ideals 10

Test the *original*, unmodified qt-Pegasis Sage implementation::

    python test_norm_equation.py --test original

Test the qarton-based classical re-implementation (this is the default, so
this is the same as running with no arguments)::

    python test_norm_equation.py --test qarton

Test the qarton-based implementation while additionally replacing calls to
one of its building blocks with a simulation of the corresponding quantum
circuit (also cross-checked against the classical function on every call).
The available building blocks are "fix_coeff", "compute_stzk_step2",
"compute_stzk" and "first_test"::

    python test_norm_equation.py --test first_test
    python test_norm_equation.py --num-ideals 10 --test fix_coeff

Note
----

This script does not test the full norm equation circuit NormEquation, since it would
require a much larger computation (running the entire Grover search), and it does not
test the post-computation circuit PostComp, since it does not have the same interface as
the original qt-Pegasis norm equation (it merges together operations in the qlapoti
function and afterwards).

"""

import argparse
import random
from typing import Any

from sage.all import set_random_seed  # type: ignore

import qt_pegasis_Fp.Sage_Code.norm_eq as qt_pegasis_norm_eq_module
import qt_pegasis_Fp.Sage_Code.qt_pegasis as qt_pegasis_module
from qisogenies.arithmetic.cornacchia_classical import cornacchia_classical_function
from qisogenies.norm_equation.norm_eq import qlapoti as qisogenies_qlapoti

_CIRCUIT_CHOICES = ["fix_coeff", "compute_stzk_step2", "compute_stzk", "first_test"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test our python implementation of qlapoti and the circuit "
            "implementations of its building blocks."
        ),
    )
    parser.add_argument(
        "--num-ideals",
        type=int,
        default=4,
        help="Number of ideals to sample and test (default: 4).",
    )
    parser.add_argument(
        "--test",
        choices=["original", "qarton", *_CIRCUIT_CHOICES],
        default="qarton",
        help=(
            "What to test (default: qarton). 'original' uses the original "
            "qt-Pegasis Sage implementation, unmodified. 'qarton' uses our "
            "qarton-based classical re-implementation in place of it. Any "
            "other value also uses the qarton-based implementation, but "
            "additionally replaces calls to that specific function with a "
            "simulation of its quantum circuit."
        ),
    )
    # This module is also imported (for `sample_odd_sage_ideal` and its
    # import-time patching of the qt-Pegasis implementation) by other
    # top-level scripts, such as build_isogeny_chain_circuit.py, which have
    # their own command-line arguments. Only parse `sys.argv` -- and
    # potentially fail on arguments meant for the importing script -- when
    # this module is actually run directly; otherwise just fall back to the
    # defaults declared above.
    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_args([])


_args = _parse_args()

NUM_IDEALS = _args.num_ideals

# 'original' keeps the unmodified qt-Pegasis implementation; every other
# choice ('qarton', or one of the circuit names) switches to our own code.
SWITCH_TO_QARTON_IMPLEMENTATION = _args.test != "original"

# If not None, will use the genuine circuit implementation instead of the
# classical function. This may make the code very slow.
TEST_CIRCUIT = _args.test if _args.test in _CIRCUIT_CHOICES else None

# ----------------------------------------------------


def sum_of_squares(z: int) -> list[int] | None:
    z = int(z)
    n = z.bit_length() + 10
    succ, a, b, _ = cornacchia_classical_function(z, n)
    if succ:
        return [a, b]
    else:
        return []


# Dynamically patch the sum of squares function to use our own, so the
# results should match
qt_pegasis_norm_eq_module.sum_of_squares = sum_of_squares  # type: ignore

# Dynamically patch qt_pegasis.py to use qisogenies' qlapoti instead of the
# one from Sage_Code/norm_eq.py. `qt_pegasis.py` imports `qlapoti` via
# `from .norm_eq import qlapoti`, binding it as a module-level name that is
# looked up at call time, so overwriting it here (before any call) is enough
# to redirect both call sites without touching either source file.
if SWITCH_TO_QARTON_IMPLEMENTATION:
    qt_pegasis_module.qlapoti = qisogenies_qlapoti  # type: ignore

if TEST_CIRCUIT == "fix_coeff":  # type: ignore
    import qisogenies.norm_equation.first_check as first_check_module
    from qisogenies.norm_equation.first_check import FixCoefft

    # keep a handle on the original classical function, so we can cross-check
    # the circuit's output against it below (before overwriting the module
    # attribute with the circuit-based replacement)
    _classical_fix_coeff = first_check_module.fix_coeff

    def _fix_coeff_via_circuit(
        d: Any, delta_1: Any, delta_2: Any, N: Any, z: Any
    ) -> Any:
        # FixCoefft is memoized on `d`, so this only builds the circuit once per
        # distinct InstanceData (all calls within a single qlapoti run share the
        # same `d`) and reuses the cached, frozen instance afterwards.
        qc = FixCoefft(d)
        qc.make_real()
        _, _, _, _, conds, D1, D2, which_case_when_2mod8 = qc.simulate(
            (delta_1, delta_2, N, z)
        )
        circuit_output = (conds, D1, D2, which_case_when_2mod8)

        # check that the circuit genuinely agrees with the classical function
        # it's replacing, on every call, not just in an isolated test
        classical_output = _classical_fix_coeff(d, delta_1, delta_2, N, z)
        assert circuit_output == classical_output, (
            "FixCoefft circuit output disagrees with classical fix_coeff: "
            f"circuit={circuit_output!r} classical={classical_output!r}"
        )

        return circuit_output

    # `first_test` (in first_check.py) calls `fix_coeff` as a bare name looked up
    # in its own module's globals at call time, so patching it here (before any
    # call) is enough to redirect that call site to the circuit simulation.
    first_check_module.fix_coeff = _fix_coeff_via_circuit  # type: ignore
elif TEST_CIRCUIT == "compute_stzk_step2":  # type: ignore
    import qisogenies.norm_equation.compute_stzk as compute_stzk_module
    from qisogenies.norm_equation.compute_stzk import ComputeSTZKStep2

    _classical_compute_stzk_step2 = compute_stzk_module.compute_stzk_step2

    def _compute_stzk_step2_via_circuit(d: Any, args: Any) -> Any:
        # ComputeSTZKStep2 is memoized on `d`, so this only builds the circuit
        # once per distinct InstanceData and reuses the cached, frozen
        # instance afterwards.
        qc = ComputeSTZKStep2(d)
        qc.make_real()
        _, s, t, z, bv = qc.simulate(args)
        circuit_output = (s, t, z, bv)

        # check that the circuit genuinely agrees with the classical function
        # it's replacing, on every call, not just in an isolated test
        classical_output = _classical_compute_stzk_step2(d, args)
        assert circuit_output == classical_output, (
            "ComputeSTZKStep2 circuit output disagrees with classical "
            f"compute_stzk_step2: circuit={circuit_output!r} "
            f"classical={classical_output!r}"
        )

        return circuit_output

    # `compute_stzk` (in compute_stzk.py) calls `compute_stzk_step2` as a bare
    # name looked up in its own module's globals at call time, so patching it
    # here (before any call) is enough to redirect that call site.
    compute_stzk_module.compute_stzk_step2 = _compute_stzk_step2_via_circuit  # type: ignore
elif TEST_CIRCUIT == "compute_stzk":  # type: ignore
    import qisogenies.norm_equation.first_check as first_check_module
    from qisogenies.norm_equation.compute_stzk import ComputeSTZK

    # `compute_stzk` is imported into first_check.py's namespace (via
    # `from .compute_stzk import ComputeSTZK, compute_stzk`), which is where
    # `first_test` looks it up, so that's the module attribute to capture and
    # patch, not the one in compute_stzk.py itself.
    _classical_compute_stzk = first_check_module.compute_stzk_classical  # type: ignore

    def _compute_stzk_via_circuit(
        d: Any, n_tralpha: Any, precomp_result: Any, rchoice: Any
    ) -> Any:
        # ComputeSTZK is memoized on `d`, so this only builds the circuit once
        # per distinct InstanceData and reuses the cached, frozen instance
        # afterwards.
        qc = ComputeSTZK(d)
        qc.make_real()
        _, _, _, s, t, z, k, bv = qc.simulate((n_tralpha, precomp_result, rchoice))
        circuit_output = (s, t, z, k, bv)

        # check that the circuit genuinely agrees with the classical function
        # it's replacing, on every call, not just in an isolated test
        classical_output = _classical_compute_stzk(
            d, n_tralpha, precomp_result, rchoice
        )
        assert circuit_output == classical_output, (
            "ComputeSTZK circuit output disagrees with classical compute_stzk: "
            f"circuit={circuit_output!r} classical={classical_output!r}"
        )

        return circuit_output

    # `first_test` (in first_check.py) calls `compute_stzk` as a bare name
    # looked up in its own module's globals at call time, so patching it here
    # (before any call) is enough to redirect that call site.
    first_check_module.compute_stzk_classical = _compute_stzk_via_circuit  # type: ignore
elif TEST_CIRCUIT == "first_test":  # type: ignore
    import qisogenies.norm_equation.norm_eq as qisogenies_norm_eq_module
    from qisogenies.norm_equation.first_check import FirstCheck

    _classical_first_test = qisogenies_norm_eq_module.first_test  # type: ignore

    def _first_test_via_circuit(
        d: Any,
        algo_input: Any,
        precomp_result: Any,
        rchoice: Any,
        test_bounds: bool = False,
    ) -> Any:
        # FirstCheck is memoized on `d`, so this only builds the circuit once
        # per distinct InstanceData and reuses the cached, frozen instance
        # afterwards. Note this does not force its ComputeSTZK/FixCoefft
        # sub-circuits to run as real gates too (they stay in their own
        # default dummy/real state) -- only FirstCheck's own top-level gates
        # are guaranteed to actually run here.
        qc = FirstCheck(d)
        qc.make_real()
        (
            _,
            _,
            _,
            delta_1,
            delta_2,
            D1,
            D2,
            z,
            bvv,
            which_case_when_2mod8,
        ) = qc.simulate((algo_input, precomp_result, rchoice))
        circuit_output = (delta_1, delta_2, D1, D2, z, bvv, which_case_when_2mod8)

        # check that the circuit genuinely agrees with the classical function
        # it's replacing, on every call, not just in an isolated test
        classical_output = _classical_first_test(
            d, algo_input, precomp_result, rchoice, test_bounds
        )
        assert circuit_output == classical_output, (
            "FirstCheck circuit output disagrees with classical first_test: "
            f"circuit={circuit_output!r} classical={classical_output!r}"
        )

        return circuit_output

    # `qlapoti_internal` (in norm_eq.py) calls `first_test` as a bare name
    # looked up in its own module's globals at call time (imported there via
    # `from .first_check import first_test`), so patching it here (before any
    # call) is enough to redirect that call site.
    qisogenies_norm_eq_module.first_test = _first_test_via_circuit  # type: ignore
elif TEST_CIRCUIT is not None:
    raise ValueError(f"Unknown TEST_CIRCUIT: {TEST_CIRCUIT!r}")

qtPegasisFp = qt_pegasis_module.qtPegasisFp


def sample_sage_ideal(EGA: qtPegasisFp) -> tuple[Any, Any]:
    """
    Samples a reduced ideal of norm N such that N is = 0 mod 4, or N is odd.
    """
    not_good = True
    while not_good:
        frak_a: Any = EGA.sample_sage_ideal()  # type: ignore
        frak_a_reduce: Any = frak_a.reduce_equiv()  # type: ignore
        n = frak_a_reduce.norm()  # type: ignore
        not_good = n % 4 == 2  # type: ignore
        # not the right congruence
        a, b = frak_a_reduce.gens_two()  # type: ignore
    return int(a), b  # type: ignore


def sample_odd_sage_ideal(EGA: qtPegasisFp) -> tuple[Any, Any]:
    return EGA.sample_ideal()  # type: ignore


if __name__ == "__main__":
    seed = 15
    set_random_seed(seed)
    random.seed(seed)

    lvl = "500"
    EGA = qtPegasisFp(lvl)  # type: ignore

    for _ in range(NUM_IDEALS):
        frak_a = sample_sage_ideal(EGA)
        EGA.qt_action(frak_a)  # type: ignore
