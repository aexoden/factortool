# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Jason Lynch <jason@aexoden.com>

import math

from functools import cache

from factortool.util import SMALL_PRIMES, is_prime, log_factor_result


@cache
def factor_tf(n: int) -> list[int]:
    original_n = n
    factors: list[int] = []

    for p in SMALL_PRIMES:
        while n % p == 0:
            factors.append(p)
            n //= p
        if n == 1:
            break

    # Check if n is a perfect square.
    if n > 1:
        sqrt_n = int(math.isqrt(n))
        if sqrt_n * sqrt_n == n:
            factors.extend([sqrt_n, sqrt_n])
            n = 1

    # Include the remaining n if it's greater than one.
    if n > 1:
        factors.append(n)

    # Log the factoring result.
    if len(factors) > 1:
        log_factor_result("TF", original_n, factors)

    return factors


class Number:
    n: int
    prime_factors: list[int]
    composite_factors: list[int]
    methods: set[str]

    def __init__(self, n: int) -> None:
        if is_prime(n):
            self.composite_factors = []
            self.prime_factors = [n]
        else:
            self.composite_factors = [n]
            self.prime_factors = []

        self.methods = set()

    def factor_tf(self) -> None:
        composite_factors = self.composite_factors.copy()
        self.composite_factors = []

        for n in composite_factors:
            factors = factor_tf(n)

            if len(factors) > 1:
                self.methods.add("TF")

            for factor in factors:
                if is_prime(factor):
                    self.prime_factors.append(factor)
                else:
                    self.composite_factors.append(factor)
