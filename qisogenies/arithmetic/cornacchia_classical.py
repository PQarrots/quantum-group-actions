"""
Classical version of our quantum Cornacchia algorithm (implemented in simple python).
This is actually a restricted version of Cornacchia which solves only the equation
:math:`x^2 + y^2 = m` on input :math:`m`.


"""

from math import isqrt

from qarton.circuit import BitVector

from .mod_sqrt import minusone_sqrt_general

__all__ = [
    "cornacchia_iterations",
    "cornacchia_classical_function",
    "cornacchia_iterations_1",
    "cornacchia_iterations_2",
]


def cornacchia_iterations_1(a: int, m: int) -> int:
    """Steps of the Euclidean algorithm in Cornacchia's algorithm.
    (Basic implementation, for tests).

    :param a: Input integer.
    :type a: int
    :param m: Input integer.
    :type m: int
    :return: One part of the solution.
    :rtype: int
    """

    # assumes: a, b < m
    b = m % a
    r0, r1 = a, b
    while r1**2 > m:
        assert r1 < r0
        r0, r1 = r1, r0 % r1
    return r1


def cornacchia_iterations_2(a: int, m: int) -> int:
    """
    Constant-time implementation inspired by
    https://eprint.iacr.org/2023/807.pdf (algorithm 8).
    (Also only for tests).

    """
    it = 4 * m.bit_length()
    stop = 0
    l = isqrt(m)
    b = m % a

    for _ in range(it):
        # print(a, b, stop, l)
        if b > a:
            a, b = b, a
        # here a is always the biggest one
        # we test b (the remainder)
        stop += int(b <= l)
        if stop == 0:
            shift = 0
            while (b << (shift + 1)) < a:
                shift += 1
            a = a - (b << shift)

    return b


def cornacchia_iterations(a: int, m: int) -> int:
    """Constant-time, quantum-friendly implementation of the Euclidean algorithm
    steps in Cornacchia's algorithm. This is the function that corresponds to
    our quantum implementation.

    :param a: Input integer.
    :type a: int
    :param m: Input integer.
    :type m: int
    :return: One part of the solution.
    :rtype: int
    """

    # seems an OK choice for the number of iterations (but it's heuristic)
    it = 3 * m.bit_length() // 2
    shift = 0
    stop = 0
    l = isqrt(m)
    b = m % a

    # -----

    for _ in range(it):
        if b > a and shift == 0:
            # if shift == 0 then b is not shifted, this is the original value
            a, b = b, a
            # this corresponds to the end of a quotient-remainder computation.

        # after this we can update Stop. Note that this:
        # stop += int(b <= l and shift == 0)
        # is equivalent to this:
        stop += int(b <= l)
        # since a non-zero shift will only make b bigger anyway.
        is_stop_zero = int(stop == 0)

        if b > a and shift > 0:
            # otherwise we're moving b back to its original value. Shift is always
            # positive, so this is correct.
            shift -= 1
            b >>= 1

        if 2 * b <= a and is_stop_zero:
            # otherwise we're currently shifting b because it's too small.
            shift += 1
            b <<= 1

        if b <= a and a < 2 * b and is_stop_zero:
            # in this case b' <= a < 2*b' (where b is the shifted value of b)
            # so a - b' < b'
            # do the subtraction. Shifting back will be handled by the next iterations.
            a = a - b
            # assert b > a

        # erase is_stop_zero

    return b


def cornacchia_classical_function(m: int, n: int) -> tuple[int, int, int, BitVector]:
    """
    Cornacchia's algorithm. Accepts the following inputs:

    * m = 2 mod 8
    * m = 1 mod 4

    Only guaranteed to succeed if m is twice a prime, or a prime. This is the function
    that corresponds to our quantum implementation.

    :param m: Input integer.
    :type m: int
    :param n: Current bit-size (for the circuit)
    :type n: int
    :rtype: tuple[int, int, int, BitVector]
    """
    if m % 8 != 2 and m % 4 != 1:
        raise ValueError("Wrong m congruence")

    # find a square root of -1 mod m
    r0, bv = minusone_sqrt_general(m, n)
    # r0 is a in the rest of the algorithm
    if r0 == 0:
        raise ValueError("Wrong m?")

    b = cornacchia_iterations(r0, m)

    sol = isqrt(m - b**2)
    if b**2 + int(sol) ** 2 == m:
        return 1, b, int(sol), bv + BitVector.from_int(r0, n)
    else:
        return 0, b, int(sol), bv + BitVector.from_int(r0, n)
