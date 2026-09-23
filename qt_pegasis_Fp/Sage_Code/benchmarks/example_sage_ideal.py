import sys
sys.path.insert(0, '..')
from time import time
import logging

from sage.all import *

from qt_pegasis import qtPegasis

# Setup logging
logging.getLogger('qt_pegasis').setLevel(logging.INFO)
logging.getLogger('hd_helpers').setLevel(logging.WARNING)

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
set_random_seed(rr)

# Benchmark parameters

lvl = '500P' # add P for PEGASIS parameter set

# Load sage
EGA = qtPegasis(lvl)
frak_a = EGA.sample_sage_ideal()
EGA.sage_action(frak_a)
