"""
This is the main file for the classical norm equation algorithm. It exposes the
following functions:

* qlapoti_internal: our re-implementation of the qlapoti function found in qt-Pegasis,
  with parameters compatible with our code. It also allows to compute the number
  of successes in the different tests, and can be used for success probability estimations.

* qlapoti: exposes an interface equivalent to the qlapoti function in qt-Pegasis, allowing
  to plug it in the original code.

In the code of ``qlapoti_internal``, the functions:

* precomp
* first_test

and some components of first_test, can be replaced at runtime by circuit simulation
(to test the circuits). To do so, you can edit the flags in ``flags.py``. But this
will make the code *much* slower. You should expect a handful of minutes for a single
isogeny evaluation.

"""

import random
from collections.abc import Callable
from typing import Any

from qarton.circuit import QartonBool, UInt
from qarton.signed_arithmetic import SInt

from qisogenies.arithmetic.cornacchia_classical import cornacchia_classical_function
from qisogenies.arithmetic.trial_division_sieve import is_almostprime_but_not_two

from .first_check import first_test
from .precomp import precomp_classical
from .ring_integers import RingInteger
from .util import InstanceData

__all__ = ["qlapoti_internal", "qlapoti"]

# When a progress callback is given, iterations are reported to it in batches of
# this size (reporting every single iteration would flood an inter-process queue).
_PROGRESS_CHUNK = 1 << 13


def qlapoti_internal(
    d: InstanceData,
    N: int,
    ring_int: RingInteger,
    test_all: bool = False,
    progress: bool = True,
    progress_callback: Callable[[int], None] | None = None,
    verb: bool = False,
) -> tuple[tuple[int, int, int], tuple[int, ...]]:
    """
    Algorithm for solving the norm equation N1 + N2 = 2^e.

    Input:
    - frak_a = (ell, omega - lambda) an ideal represented as two elements
    in Z[omega]; we assume ell to be smaller than sqrt(p), so that the
    ideal is (close) to reduced and we can use it directly

    Also trace is positive now, N is odd.
    - e: target exponent of 2 for the norm equation

    Progress reporting (only when ``test_all`` is set, since the loop then runs the
    full ``(1 << x) + 1`` iterations):
    - ``progress``: show a local tqdm bar.
    - ``progress_callback``: instead, call ``progress_callback(n)`` after every ``n``
      iterations (in batches), e.g. to aggregate progress across parallel workers.
      Takes precedence over ``progress``.

    Output:
    - A1, A2 generating frak_c1_bar * frak_b1
    - B1, B2 generating frak_b2_bar * frak_b1
    - C1, C2 generating frak_c2 * frak_c1_bar
    - D1, D2 generating frak_c1_bar * frak_c2
    - E1, E2 generating frak_b2_bar * frak_c1
    - N1 = frak_b1.norm() + frak_c1.norm()
    - N2 = frak_b2.norm() + frak_c2.norm()
    - Nb1 = frak_b1.norm()
    """
    assert N % 4 != 2, "unsupported case"

    algo_input = (N, ring_int)
    algo_input = (
        UInt(algo_input[0]),
        RingInteger(d.p, d.w // 2, algo_input[1].a, algo_input[1].b),
    )
    assert algo_input[1].w == d.w // 2

    x = d.x
    w, nb_primes = d.w, d.nb_primes

    # --------- precomputation step
    precomp_result = precomp_classical(d, algo_input)

    ct = 0  # number of iterations
    ctc = 0  # number of Cornacchia calls
    cts = 0  # number of solutions

    b1, b2 = 0, 0

    output: tuple[int, ...] = tuple()

    # ct takes the values 1, 2, ..., (1 << x) + 1 (same bounds as the original
    # while loop). Progress reporting is only meaningful when testing all inputs;
    # in the random case the loop usually stops early.
    # - progress_callback: report iterations to a caller-supplied sink (used to
    #   aggregate progress across parallel workers).
    # - otherwise, when progress is set, show a local tqdm bar.
    loop_iter: Any = range(1, (1 << x) + 2)
    if test_all and progress_callback is None and progress:
        from tqdm import tqdm

        loop_iter = tqdm(loop_iter)

    report_callback = progress_callback if test_all else None
    since_report = 0

    for ct in loop_iter:
        if report_callback is not None:
            since_report += 1
            if since_report >= _PROGRESS_CHUNK:
                report_callback(since_report)
                since_report = 0

        # If we are testing all the inputs, we just take the counter as value.
        # Otherwise try a random value.
        i = random.randrange(1 << x) if not test_all else ct

        step_1_output = first_test(d, algo_input, precomp_result, UInt(i))

        # first testing step: various computations
        (delta_1, delta_2, D1, D2, z, bvv, which_case_when_2mod8) = step_1_output
        assert len(bvv) == 6
        # bvv[0] is the constraint that v < bound and bvv[1] is that z > 0
        if not (bvv[0] and bvv[1]):
            continue

        if not all(list(bvv)):
            continue

        # pseudo-primality check
        if not is_almostprime_but_not_two(int(z), nb_primes):
            continue

        # -------- this is where our "first test" stops

        ctc += 1
        # third and final test: Cornacchia
        sol, b1, b2, _ = cornacchia_classical_function(int(z), w)

        if sol == 0:
            continue

        assert b1**2 + b2**2 == z
        cts += 1

        if verb:
            print("delta_1", delta_1)
            print("delta_2", delta_2)
            print("D1", D1)
            print("D2", D2)
            print("z", z)
            print("b1", b1)
            print("b2", b2)

        output = post_computation_in_qlapoti(
            d, delta_1, delta_2, z, which_case_when_2mod8, D1, D2, b1, b2, N
        )

        # If we are testing all inputs, we shouldn't break here
        if not test_all:
            break

    if report_callback is not None and since_report:
        report_callback(since_report)

    if cts == 0:
        raise Exception("No solution found")

    return (ct, ctc, cts), output


def post_computation_in_qlapoti(
    d: InstanceData,
    delta_1: RingInteger,
    delta_2: RingInteger,
    z: UInt,
    which_case_when_2mod8: QartonBool,
    DD1: SInt,
    DD2: SInt,
    b1: int,
    b2: int,
    N: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]:
    """
    Post-computation in the algorithm. Takes everything that was computed so far and
    produces the outputs:

    (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, N1, N2, Nb1)
    """
    D1, D2 = DD1.v, DD2.v

    if z % 4 == 1 and b1 % 2 == 0:
        b2, b1 = b1, b2

    if z % 8 == 2:
        # Impose conditions for later
        # b1 = 1 mod 4
        if b1 % 4 == 3:
            b1 = -b1
        # b2 = 1 mod 4 for (1) / 3 mod 4 for (2)
        if (which_case_when_2mod8 == 0 and b2 % 4 == 3) or (
            which_case_when_2mod8 == 1 and b2 % 4 == 1
        ):
            b2 = -b2

    assert z % 8 == 2 or z % 4 == 1

    d1 = delta_1.norm() // N
    d2 = delta_2.norm() // N
    if N % 2 == 1:
        assert d1 % 2 == 0
    else:
        assert d1 % 2 == 1

    # compute xC1, xC2, xE1, xE2
    tmp = delta_2.conjugate()
    (xC1, xC2) = (tmp.a.v, tmp.b.v)
    tmp = delta_1.conjugate()
    (xE1, xE2) = (tmp.a.v, tmp.b.v)

    beta_1 = b1 * N
    beta_2 = b2 * N
    # check that we solved the norm equation!!!!
    assert beta_1**2 + beta_2**2 + delta_1.norm() + delta_2.norm() == 2 ** (d.e) * N

    A1, A2 = delta_1.a.v * b1, delta_1.b.v * b1
    B1, B2 = b1 * b2 * N, 0  # b2bar_b1
    C1, C2 = b1 * xC1, b1 * xC2  # c2_b1bar
    E1, E2 = b2 * xE1, b2 * xE2  # b2bar_c1

    Nb1 = b1**2 * N  # n(frak_b1)
    N1 = d1 + b1**2 * N  # n(frak_b1) + n(frak_c1)
    N2 = d2 + b2**2 * N  # n(frak_b2) + n(frak_c2)

    assert N1 % 2 == 1

    if N % 2 == 0:
        Nb1 = d1
        E1, C1 = C1, E1  # type: ignore
        E2, C2 = C2, E2  # type: ignore

        B1, D1 = D1, B1  # type: ignore
        B2, D2 = D2, B2  # type: ignore

    return (A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, N1, N2, Nb1)


def qlapoti(frak_a: Any, e: Any, verb: bool = True) -> Any:
    """
    Wrapper around our Qarton-compatible Qlapoti implementation.
    This function can be used as drop-in replacement in the original
    qt-Pegasis code, and we do this to test its correctness.
    """
    N, alpha = frak_a
    if verb:
        print("Entering the new norm equation computation!")
        print("N=", N)
    # convert to ringinteger element
    p: Any = -(alpha.parent().discriminant())
    p = int(p)
    # assert N % 4 != 2, "N is 2 mod 4, so you must act with a single 2-isogeny first"

    tr_alpha, _ = list(alpha)

    # Choose sign of trace positive
    if tr_alpha < 0:
        alpha = -alpha

    w = ((p.bit_length() + 10) // 4) * 4

    tmp1, tmp2 = list(alpha)
    # since omega = (1 + sqrt(-p))/2 the mapping is:
    # a, b = tmp1 - tmp2, 2*tmp2
    a, b = int(tmp1 - tmp2), int(2 * tmp2)
    ring_int = RingInteger(p, w, a, b)
    assert abs(a) < (1 << (w // 2))
    assert abs(b) < (1 << (w // 2))
    assert ring_int.trace() > 0

    instance_data = InstanceData(w=w, x=22, e=e, p=p, nb_primes=20)

    return qlapoti_internal(instance_data, int(N), ring_int, test_all=False)[1]
