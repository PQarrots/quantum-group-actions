import sys
sys.path.insert(0, '..')
from time import time
import logging

from sage.all import *

from qt_pegasis import qtPegasis

# Setup logging
#logging.getLogger('qt_pegasis').setLevel(logging.INFO)
logging.getLogger('hd_helpers').setLevel(logging.WARNING)

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
set_random_seed(rr)

# Benchmark parameters

lvl = '500P' # add P for PEGASIS parameter set

# Load sage
EGA = qtPegasis(lvl)

for i in range(10):
    print(f"\n~~~~~~ RUN {i + 1} ~~~~~~~")
    frak_a = EGA.sample_sage_ideal()
    frak_b = EGA.sample_sage_ideal()
    E_a = EGA.sage_action(frak_a)
    E_ab = EGA.sage_action(frak_b, E_a)
    frak_c = frak_a*frak_b
    E_c = EGA.sage_action(frak_c)
    assert E_ab.j_invariant() == E_c.j_invariant()
    print(" > PASSED!")
