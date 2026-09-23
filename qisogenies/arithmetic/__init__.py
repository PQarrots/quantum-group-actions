"""
Arithmetic building blocks of our algorithms and circuits: modular square root,
sieving by small primes, Cornacchia's algorithm.

This module contains both classical functions and quantum circuit definitions
(which match the functions).
"""

from . import (
    cornacchia,
    cornacchia_classical,
    efficient_mod_arithmetic,
    mod4,
    mod_inverse_pow2,
    mod_sqrt,
    trial_division_sieve,
)
from .cornacchia import *  # noqa: F403
from .cornacchia_classical import *  # noqa: F403
from .efficient_mod_arithmetic import *  # noqa: F403
from .mod4 import *  # noqa: F403
from .mod_inverse_pow2 import *  # noqa: F403
from .mod_sqrt import *  # noqa: F403
from .trial_division_sieve import *  # noqa: F403

__all__: list[str] = []
__all__.extend(cornacchia.__all__)
__all__.extend(cornacchia_classical.__all__)
__all__.extend(mod_inverse_pow2.__all__)
__all__.extend(mod_sqrt.__all__)
__all__.extend(mod4.__all__)
__all__.extend(trial_division_sieve.__all__)
__all__.extend(efficient_mod_arithmetic.__all__)
