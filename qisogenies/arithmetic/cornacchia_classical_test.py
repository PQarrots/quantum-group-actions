import random

from sympy import isprime  # type: ignore

from .cornacchia_classical import (
    cornacchia_classical_function,
    cornacchia_iterations,
    cornacchia_iterations_1,
    cornacchia_iterations_2,
)


def test_cornacchia2_2() -> None:
    for _ in range(1 << 16):
        m = random.randrange(3, 1 << 10)
        a = random.randrange(1, m)
        # print(a, m)
        # print(a, m)
        # a, m = 726, 860
        assert cornacchia_iterations_1(a, m) == cornacchia_iterations_2(a, m)


def test_cornacchia3() -> None:
    for _ in range(1 << 16):
        m = random.randrange(3, 1 << 10)
        a = random.randrange(1, m)
        # print(a, m)
        # print(a, m)
        # a, m = 726, 860
        assert cornacchia_iterations_1(a, m) == cornacchia_iterations(a, m)


def test_large() -> None:
    for _ in range(10):
        m = random.randrange(1 << 1000)
        a = random.randrange(1, m)
        assert cornacchia_iterations_1(a, m) == cornacchia_iterations(a, m)


def test_cornacchia() -> None:
    n = 50
    random.seed(0)

    for _ in range(100):
        m = random.randrange(1 << n)
        if m % 8 == 2 or m % 4 == 1:
            b, x, y, _ = cornacchia_classical_function(m, n)
            if b == 1:
                assert x**2 + y**2 == m

            if (m % 2 == 1 and isprime(m)) or (m % 2 == 0 and isprime(m // 2)):
                assert b
            #    assert x**2 + y**2 == m
