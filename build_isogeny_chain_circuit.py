"""
Build the isogeny chain circuit.

Usage
-----

Run with the default security level (500)::

    python build_isogeny_chain_circuit.py

Choose a different security level (500, 1000, 2000 or 4000)::

    python build_isogeny_chain_circuit.py --level 500

Build the circuit for all security levels, in order::

    python build_isogeny_chain_circuit.py --level all

Choose which circuit construction to build (default: chain)::

    python build_isogeny_chain_circuit.py --circuit chain
    python build_isogeny_chain_circuit.py --circuit pebbled

The total runtime for all levels is around one hour.
At 4000 bits the script will end up using a significant amount of RAM (20 GB in our run).

Example output for the chain construction
-----------------------------------------

    Level: 500
    Building the circuit...
    Done! Time: 83.0311963558197
    Nbr of qubits 99072594
    Nbr of CCX gates (log2) 42.86020621282368
    Level: 1000
    Building the circuit...
    Done! Time: 179.39419651031494
    Nbr of qubits 431447379
    Nbr of CCX gates (log2) 45.92948863990189
    Level: 2000
    Building the circuit...
    Done! Time: 404.12567806243896
    Nbr of qubits 1867110890
    Nbr of CCX gates (log2) 49.02340395690248
    Level: 4000
    Building the circuit...
    Done! Time: 1040.7245831489563
    Nbr of qubits 7977429557
    Nbr of CCX gates (log2) 52.11550865483638

Example output for the pebbled construction
-------------------------------------------

    Level: 500
    Building the circuit...
    Done! Time: 21.11715579032898
    Nbr of qubits 2853795
    Nbr of CCX gates (log2) 45.971225528369494
    Level: 1000
    Building the circuit...
    Done! Time: 45.62153697013855
    Nbr of qubits 8421880
    Nbr of CCX gates (log2) 49.038285341756996
    Level: 2000
    Building the circuit...
    Done! Time: 101.47556233406067
    Nbr of qubits 26114139
    Nbr of CCX gates (log2) 52.137891491419104
    Level: 4000
    Building the circuit...
    Done! Time: 252.53792142868042
    Nbr of qubits 88156290
    Nbr of CCX gates (log2) 55.26275307521455

"""

import argparse
import time
from random import seed

from sage.all import set_random_seed  # type: ignore

from qisogenies.norm_equation.util import FAST_BACKENDS
from qisogenies.theta.isogeny_chain_dim4 import (
    IsogenyChain,
)
from qisogenies.theta.isogeny_chain_dim4_pebbled import IsogenyChainWithPebbling
from qisogenies.theta.theta_util import balanced_strategy
from qt_pegasis_Fp.Sage_Code.qt_pegasis import qtPegasisFp  # type: ignore
from test_isogeny_chain import (
    compute_classical_input,
    prepare_params_isogeny_chain,
)

SEED = 2
set_random_seed(SEED)
seed(SEED)


LEVELS = ["500", "1000", "2000", "4000"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the isogeny chain circuit and report its resource usage."
    )
    parser.add_argument(
        "--level",
        choices=[*LEVELS, "all"],
        default="500",
        help=(
            "Security level to use for the qt-Pegasis parameters (default: 500). "
            "Use 'all' to build the circuit for every level, in order."
        ),
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


def build_and_report(level: str, circuit: str) -> None:
    """
    Build the isogeny chain circuit for the given security level, and print
    its resource usage.
    """
    print("Level:", level)

    qtPeg = qtPegasisFp(level)  # type: ignore
    _e, e_sol, _p, _A, _E_start = (  # type: ignore
        int(qtPeg.e),
        int(qtPeg.e_sol),
        int(qtPeg.p),
        int(qtPeg.A),
        qtPeg.E_start,  # type: ignore
    )
    strategy_balanced = balanced_strategy(e_sol)
    _strategy_optimize_memory = (
        *tuple(e_sol - 1 - i for i in range(e_sol - 1)),
        1,
    )

    ##########
    domain_qtp, (_P, _Q, _TP, _TQ) = compute_classical_input(qtPeg)

    ###########################
    # Instantiate the isogeny chain circuit

    print("Building the circuit...")
    t1 = time.time()

    CC = IsogenyChain if circuit == "chain" else IsogenyChainWithPebbling

    isogeny_chain_qc = CC(
        e_sol,
        prepare_params_isogeny_chain(qtPeg, domain_qtp),
        strategy_balanced,
        backends=FAST_BACKENDS,
    )
    print("Done! Time:", time.time() - t1)

    ####### Resource usage:
    r = isogeny_chain_qc.resources
    print("Nbr of qubits", r.nbr_qubits())
    print("Nbr of CCX gates (log2)", r.ccx_count_log2())

    print("Most costly sub-circuits:")
    d = r.sub_circuits_by_class_proportion(30)
    for k in d:
        print(k)


if __name__ in ("__main__", "sage.all"):
    args = _parse_args()
    chosen_level = args.level
    levels_to_run = LEVELS if chosen_level == "all" else [chosen_level]

    for lvl in levels_to_run:
        build_and_report(lvl, args.circuit)
