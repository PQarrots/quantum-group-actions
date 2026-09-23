r"""
Script to estimate the probabilities to pass the successive tests in the norm equation
algorithm. The number of iterations of the two-level search are deduced from them.

* The probability p1 varies. We take the minimum over all our trials, and this will
  serve to compute the number of controlled iterations in the first amplification layer in the
  two-level search.

* The probability p2 does not (seem to) vary. We take the average over all our trials,
  and this will serve to compute the number of non-controlled iterations in the
  second amplification layer in the two-level search.

To obtain all the data, simply run this script:

    python estimate_p1_p2.py

It will take some time and use up a lot of computing power (you can change the value
of NB_PROC to modify the number of concurrent threads, default is 6).

You can also import it and use ``get_iterations_data()`` to obtain the number of iterations
for different security levels, experimentally computed.


"""

from collections.abc import Callable
from math import ceil, log2, pi, sqrt
from multiprocessing import Process, Queue
from statistics import mean
from typing import Any

from sage.all import set_random_seed  # type: ignore
from tqdm import tqdm

from qisogenies.norm_equation.norm_eq import qlapoti_internal
from qisogenies.norm_equation.ring_integers import RingInteger
from qisogenies.norm_equation.util import InstanceData, IterationsData
from qt_pegasis_Fp.Sage_Code.qt_pegasis import qtPegasisFp
from test_norm_equation import sample_odd_sage_ideal, sample_sage_ideal

# Number of instances run in parallel when estimating p1, p2.
NB_PROC = 6

# ---------------------------------------------------------


def find_instance_data(
    level: str, x: int = 15, nb_primes: int = 20
) -> tuple[Any, InstanceData]:
    """
    Obtain the EGA and map hard-coded parameters into an InstanceData which contains all the
    parameters.
    """
    EGA = qtPegasisFp(level)  # type: ignore
    p = int(EGA.p)  # type: ignore
    e = int(EGA.e)  # type: ignore
    w = ((p.bit_length() + 10) // 4) * 4
    return EGA, InstanceData(w, x, e, p, nb_primes)


def sample_ideal(
    ega: Any, seed: int = 5, always_odd: bool = False
) -> tuple[int, RingInteger]:
    """
    Uses the EGA object to sample an ideal and convert into our representation.

    By default we sample ideals of odd or even norm, but we can also sample ideals
    of odd norm which are *already* reduced (this avoids having to perform reduction in Sage).
    """
    set_random_seed(seed)
    if always_odd:
        N, alpha = sample_odd_sage_ideal(ega)  # type: ignore
    else:
        N, alpha = sample_sage_ideal(ega)  # type: ignore
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
    return int(N), RingInteger(p, w, a, b)  # type: ignore


def estimate_probabilities(
    ega: Any,
    instance_data: InstanceData,
    seed: int = 5,
    progress: bool = True,
    only_odd: bool = False,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[float, float]:
    """Estimates the probabilities to pass the two tests.
    To make this estimate, we select a random input to the algorithm and try all
    possible choices in a certain range. We count the number of choices that pass the
    successive tests.

    Running this function with different seeds will take different inputs, and thus
    yield different estimates. Increasing the size of the choice set refines the
    estimates.

    :param ega: The EGA object
    :param seed: Random seed, defaults to 5
    :param only_first_step: if True, will return only (p', 0) where p' is the probability
           to pass the very first tests in the algorithm (those which are responsible
           for the variation in p1, per our estimate).
    :type seed: int, optional
    :return: The probabilities p1, p2.
    :rtype: tuple[float, float]
    """
    set_random_seed(seed)
    N, ring_int = sample_ideal(ega, seed, only_odd)
    t = qlapoti_internal(
        instance_data,
        N,
        ring_int,
        test_all=True,
        progress=progress,
        progress_callback=progress_callback,
    )[0]
    ct, ctc, cts = t
    # ctc corresponds to the first test in the quantum search (p1)
    # cts is the second test in the quantum search (so a full success) (p2)
    return (ctc / ct), (cts / ctc)


def _p1_p2_worker(
    level: str, x: int, nb_primes: int, seeds: list[int], queue: Any
) -> None:
    """Worker process: estimates (p1, p2) for each seed in ``seeds``.

    Pushes ``("progress", n)`` messages as iterations are done (so the parent can
    aggregate a precise bar over all workers), ``("result", seed, p1, p2)`` per
    instance, and a final ``("done", ...)``.

    If level = 4000 we only sample ideals of odd norm without doing a reduction,
    at the moment.
    """
    ega, instance_data = find_instance_data(level, x, nb_primes)

    def report(n: int) -> None:
        queue.put(("progress", n))

    for seed in seeds:
        p1, p2 = estimate_probabilities(
            ega,
            instance_data,
            seed,
            progress=False,
            only_odd=(level == "4000"),
            progress_callback=report,
        )
        queue.put(("result", seed, p1, p2))
    queue.put(("done", 0, 0.0, 0.0))


def estimate_p1_p2_using_many_instances(
    level: str, x: int, nb_primes: int, nbr: int
) -> tuple[list[float], list[float]]:
    """Estimates the probabilities p1 and p2 over many instances.

    The ``nbr`` instances are spread over ``NB_PROC`` parallel processes, each with
    its own seed. A single global tqdm bar aggregates the inner-loop iterations
    (``(1 << x) + 1`` per instance) reported by all workers.

    :param level: Security level (choose from 500, 1000, 2000, 4000)
    :type level: str
    :param x: Size of the search space in bits. A larger x means a slower computation,
              but it also needs to be large enough for the p2 computation to make sense
              (there needs to be solutions).
    :type x: int
    :param nb_primes: Number of primes in the first divisibility test
    :type nb_primes: int
    :param nbr: Number of reruns of the p1,p2 estimation.
    :type nbr: int
    :return: The lists of p1 and p2 values for all trials.
    :rtype: tuple[list[float], list[float]]
    """

    seeds = [i + 10 for i in range(nbr)]
    nb_proc = min(NB_PROC, nbr)
    # Round-robin the seeds over the workers.
    chunks = [seeds[i::nb_proc] for i in range(nb_proc)]

    queue: Any = Queue()  # type: ignore
    processes = [
        Process(target=_p1_p2_worker, args=(level, x, nb_primes, chunks[i], queue))
        for i in range(nb_proc)
    ]
    for p in processes:
        p.start()

    results: dict[int, tuple[float, float]] = {}
    done = 0
    iters_per_instance = (1 << x) + 1
    with tqdm(total=nbr * iters_per_instance, unit="it", unit_scale=True) as pbar:
        while done < nb_proc:
            msg = queue.get()
            tag = msg[0]
            if tag == "done":
                done += 1
            elif tag == "progress":
                pbar.update(msg[1])
            else:  # ("result", seed, p1, p2)
                _, seed, p1, p2 = msg
                results[seed] = (p1, p2)

    for p in processes:
        p.join()

    p1_values = [results[s][0] for s in seeds]
    p2_values = [results[s][1] for s in seeds]
    for p1, p2 in zip(p1_values, p2_values, strict=True):
        print(to_log(p1), to_log(p2))
    return p1_values, p2_values


def to_log(x: float) -> str:
    return f"2^{{{round(log2(x), 2)}}}"


def run_level(level: str) -> None:

    x = 22
    nb_primes = 20

    print("level, x, nb_primes:", level, x, nb_primes)
    l1, l2 = estimate_p1_p2_using_many_instances(level, x, nb_primes, 20)
    print(
        "Min, max, average of p1:", to_log(min(l1)), to_log(max(l1)), to_log(mean(l1))
    )
    print(
        "Min, max, average of p2:", to_log(min(l2)), to_log(max(l2)), to_log(mean(l2))
    )
    print("Iterations data:")
    it1 = (pi / 4) * 1 / sqrt(min(l1))
    it2 = (pi / 4) * 1 / sqrt(mean(l2))
    print(to_log(it1), to_log(it2))


# obtained by running this script
ITERATION_NUMERS_LOG2 = {
    "500": (4.95, 1.89),
    "1000": (4.55, 2.4),
    "2000": (3.67, 2.88),
    "4000": (5.03, 3.44),
}


def get_iterations_data(level: str) -> IterationsData:
    it1, it2 = ITERATION_NUMERS_LOG2[level]
    return IterationsData(ceil(2 ** (it1)), ceil(2 ** (it2)))


if __name__ == "__main__":
    for level in ["500", "1000", "2000", "4000"]:
        run_level(level)
        # print(get_iterations_data(level))
