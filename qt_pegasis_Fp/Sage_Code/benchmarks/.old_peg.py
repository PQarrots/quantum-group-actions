import random
from tqdm import tqdm
from time import time
import os
"""
This file requires the original pegasis repository to be imported as a
subdirectory to cross check results. It can be downloaded with
```
git clone --recurse-submodules git@github.com:pegasis4d/pegasis.git old_pegasis
```

Usage:
    - sage .old_peg.py ell lambda lvl : run the old version of PEGASIS on the
      ideal a = (ell, (pi + lambda)/2) for level lvl
"""
if not os.path.isdir('.old_pegasis'):
    print('Original PEGASIS code not found')
    print('Try git clone --recurse-submodules git@github.com:pegasis4d/pegasis.git old_pegasis')
    exit(1)

import sys
sys.path.insert(0, '.old_pegasis')

from sage.all import *

from pegasis import PEGASIS

ell = int(sys.argv[1])
# lb = int(sys.argv[2])
x = int(sys.argv[2])
lvl = int(sys.argv[3])

EGA = PEGASIS(lvl)
pi = EGA.order.gens()[1]

ids = [I[0] for I in factor(EGA.order.fractional_ideal(ell))]
assert len(ids) == 2, "non split prime provided"
a = None
for I in ids:
    check = I.gens()[1][0]
    if (check > 0) == (x > 0):
        a = I
        break
if not a:
    print('Ideal not found')
    exit(1)

a = EGA.order.ideal(a.gens())
# a = EGA.order.ideal([ell, pi + lb])

Ea = EGA.action(EGA.E_start, a)
print(f'{Ea.a_invariants()[1]}')


