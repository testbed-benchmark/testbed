"""One shared prior-maximum integral, never estimated per policy."""

import math
from functools import lru_cache

from scipy.integrate import quad


@lru_cache(maxsize=None)
def expected_prior_max(d, mu, tau):
    def integrand(x):
        tail = 0.5 * math.erfc((x - mu) / (tau * math.sqrt(2)))
        if tail >= 1:
            return 1.0
        return -math.expm1(d * math.log1p(-tail))

    value, error = quad(integrand, 0, math.inf, epsabs=1e-10, epsrel=1e-10)
    return {"value": value, "quadrature_error": error, "epsabs": 1e-10, "epsrel": 1e-10}
