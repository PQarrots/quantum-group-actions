r"""
Script to construct the entire norm equation circuit for a given parameter size,
and display its count metrics.

The norm equation circuit contains a quantum search, and its number of iterations
is taken from ``estimate_p1_p2.py``, where it was obtained using experimental results
on the p1 / p2 probabilities.

Building norm equation circuit for 500 ...
Done! Elapsed time: 22.467968463897705
['500', 11743, 36.25, 0.9]
Building norm equation circuit for 1000 ...
Done! Elapsed time: 47.825364112854004
['1000', 23205, 39.36, 0.95]
Building norm equation circuit for 2000 ...
Done! Elapsed time: 137.440256357193
['2000', 46151, 42.51, 0.98]
Building norm equation circuit for 4000 ...
Done! Elapsed time: 494.348019361496
['4000', 91973, 45.84, 0.98]
\begin{tabular}{rrrr}
\hline
   level &   qubits &   log2(CCX count) &   Sqrt proportion \\
\hline
     500 &   11,743 &             36.25 &              0.9  \\
    1000 &   23,205 &             39.36 &              0.95 \\
    2000 &   46,151 &             42.51 &              0.98 \\
    4000 &   91,973 &             45.84 &              0.98 \\
\hline
\end{tabular}



"""

from time import time

from qarton.circuit import CCX_GATE, Circuit, UInt
from tabulate import tabulate

from estimate_p1_p2 import find_instance_data, get_iterations_data, sample_ideal
from qisogenies.arithmetic.mod_sqrt import MinusOneSqrtGeneral
from qisogenies.norm_equation.norm_equation_circuit import NormEquation
from qisogenies.norm_equation.util import FAST_BACKENDS


def norm_equation_circuit(level: str) -> NormEquation:
    t1 = time()
    print("Building norm equation circuit for", level, "...")
    it_data = get_iterations_data(level)
    _, d = find_instance_data(level, x=14, nb_primes=20)
    out = NormEquation(d, it_data, backends=FAST_BACKENDS)
    print("Done! Elapsed time:", time() - t1)
    return out


def sub_circuit_and_inverse_proportion(
    qc: Circuit, cls: type[Circuit] | list[type[Circuit]], target: str
) -> float:
    acls = cls if isinstance(cls, list) else [cls]
    names: list[str] = []
    for c in acls:
        names.append(c.name_of_class())
        names.append(c.name_of_class_inverse())
    total = 0
    for k, v in qc.resources.sub_circuits_by_name().items():
        if Circuit.name_exists(k):
            # this is a circuit (not a basic gate)
            _qc = Circuit.from_name(k)
            if _qc.class_name in names:
                cost = _qc.resources.get_by_class(target)
                total += v * cost
    x = qc.resources.get_by_class(target)
    return round(total / x, 2)


def resource_table(levels: list[str] | None = None, fmt: str = "github") -> str:
    """Build the norm equation circuit for several parameter sizes and return a
    table with the CCX gate count and number of qubits for each.

    :param levels: Parameter sizes to build. Defaults to
        ``["500", "1000", "2000", "4000"]``.
    :param fmt: A ``tabulate`` table format. ``"github"`` (default) for a plain
        text table, ``"latex"`` / ``"latex_booktabs"`` for LaTeX.
    :return: The rendered table (also printed to stdout).
    """
    if levels is None:
        levels = ["500", "1000", "2000", "4000"]

    rows: list[list[object]] = []
    for level in levels:
        qc = norm_equation_circuit(level)

        r = qc.resources
        p = sub_circuit_and_inverse_proportion(
            qc, [MinusOneSqrtGeneral], CCX_GATE.class_name
        )
        row: list[object] = [
            level,
            r.nbr_qubits(),
            round(r.ccx_count_log2(), 2),
            round(p, 2),
        ]
        print(row)
        rows.append(row)

    table = tabulate(
        rows,
        headers=["level", "qubits", "log2(CCX count)", "Sqrt proportion"],
        tablefmt=fmt,
        intfmt=",",
    )
    print(table)
    return table


def test_norm_equation_circuit(level: str) -> None:

    qc = norm_equation_circuit(level)
    ega, _ = find_instance_data(level, x=14, nb_primes=20)
    n, alpha = sample_ideal(ega, 5)

    outputs = qc.simulate((UInt(n), alpha), progress_bar=True)
    for o in outputs:
        print(o)


if __name__ == "__main__":
    resource_table(fmt="latex")
