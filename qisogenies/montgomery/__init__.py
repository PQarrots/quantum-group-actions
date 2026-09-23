"""
Elliptic curves in Montgomery form.

The code was adapted from Qarton's package on Weierstrass curves. The formulas are
diferent, leading to different circuits for point addition and other operations.

We define both fixed-curve and variable-curve types for points in affine coordinates,
which are used in our circuits for dimension-4 isogeny chains.

"""

from . import montgomery_add, montgomery_curve, montgomery_neg, montgomery_prod
from .montgomery_add import *  # noqa: F403
from .montgomery_curve import *  # noqa: F403
from .montgomery_neg import *  # noqa: F403
from .montgomery_prod import *  # noqa: F403

__all__: list[str] = []
__all__.extend(montgomery_neg.__all__)
__all__.extend(montgomery_curve.__all__)
__all__.extend(montgomery_add.__all__)
__all__.extend(montgomery_prod.__all__)
