import sys
sys.path.insert(0, '..')
from tqdm import tqdm
from time import time
import logging

from sage.all import *

from qt_pegasis import qtPegasis
import os

"""
This file requires the original pegasis repository to be imported as a
subdirectory to cross check results. It can be downloaded with
```
git clone --recurse-submodules git@github.com:pegasis4d/pegasis.git .old_pegasis
```
"""
if not os.path.isdir('.old_pegasis'):
    print('Original PEGASIS code not found')
    print('Try:')
    print('\tgit clone --recurse-submodules git@github.com:pegasis4d/pegasis.git .old_pegasis')
    exit(1)

# Setup logging
logging.getLogger('qt_pegasis').setLevel(logging.INFO)

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
set_random_seed(rr)

# Benchmark parameters
lvl = '500'
n_runs = 3

EGA = qtPegasis(lvl + 'P') # adding P for PEGASIS parameter set
order = EGA.K.order_of_conductor(2)

print(f'\nSetup done - starting {n_runs} runs for {lvl = }\n')



for _ in range(n_runs):
    print('\n' + '='*20 + f'\nRun {_+1}/{n_runs}\n' + '='*20)

    # Hacky way to consistently pass the same ideal to qtp and PEGASIS
    while True:
        ell = random_prime(2**100)
        ids = [I[0] for I in factor(order.fractional_ideal(ell))]
        if len(ids) == 2:
            break
    a = None
    for I in ids:
        check = I.gens()[1][0]
        if check < 0:
            a = I
            break
    assert a

    ell, alpha = a.gens_two()
    Ea, Eabar = EGA.qt_action((ZZ(ell), alpha), ret_twist=True)

    print('-----')
    # Test Ea
    os.system(f'sage .old_peg.py {ell} -1 {lvl} > tmp.txt')
    Ea_old = int(open('tmp.txt', 'r').read().strip())
    os.remove('tmp.txt')
    assert Ea_old == Ea
    print('Ea ok')

    # Test Eabar
    os.system(f'sage .old_peg.py {ell} 1 {lvl} > tmp.txt')
    Eabar_old = int(open('tmp.txt', 'r').read().strip())
    os.remove('tmp.txt')
    assert Eabar_old == Eabar
    print('Eabar ok')

