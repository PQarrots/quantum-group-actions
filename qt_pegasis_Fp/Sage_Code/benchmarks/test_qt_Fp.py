import sys
sys.path.insert(0, '..')
from time import time
import logging

from sage.all import *

from qt_pegasis import qtPegasisFp

# Setup logging
#logging.getLogger('qt_pegasis').setLevel(logging.INFO)
logging.getLogger('hd_helpers').setLevel(logging.WARNING)

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
print("Testing commutativity\n")
set_random_seed(rr)

# Benchmark parameters

lvl = '500' # add P for PEGASIS parameter set

# Load sage
EGA = qtPegasisFp(lvl)

for i in range(10):
    print(f"\n~~~~~~ RUN {i + 1} ~~~~~~~")
    frak_a = EGA.sample_sage_ideal()
    frak_b = EGA.sample_sage_ideal()
    print("[b]*[a]*E")
    E_a = EGA.sage_action(frak_a)
    E_ab = EGA.sage_action(frak_b, E_a)
    print("[a]*[b]*E")
    E_b = EGA.sage_action(frak_b)
    E_ba = EGA.sage_action(frak_a,E_b)
    A_ab = E_ab.a_invariants()[1]
    A_ba = E_ba.a_invariants()[1]
    assert A_ab == A_ba
    print("Test [b]*[a]*E == [a]*[b]*E > PASSED!")
