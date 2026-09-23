import sys
sys.path.insert(0, '..')
from time import time
import logging

from sage.all import randint, set_random_seed

from qt_pegasis import qtPegasis

# Setup logging
logging.getLogger('qt_pegasis').setLevel(logging.INFO)
logging.getLogger('hd_helpers').setLevel(logging.WARNING)

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
set_random_seed(rr)

# Benchmark parameters
lvl = '500P' # add P for PEGASIS parameters
n_runs = 10

# Load sage
EGA = qtPegasis(lvl)
frak_a = EGA.sample_ideal()
EGA.qt_action(frak_a)

print(f'\nSetup done - starting {n_runs} runs for {lvl = }\n')

t_labels = ['T1', 'T2.1', 'T2.2', 'T3']
times = [0 for _ in t_labels]

for _ in range(n_runs):
    print('\n' + '='*20 + f'\nRun {_+1}/{n_runs}\n' + '='*20)
    frak_a = EGA.sample_ideal()
    _, run_times = EGA.qt_action(frak_a, timings=True)

    for i in range(len(times)):
        times[i] += run_times[i]

print('\n' + '='*20 + '\n')

for i in range(len(t_labels)):
    print(f'Avg {t_labels[i]}: {times[i]/n_runs:.3f}s')
print(f'Tot: {sum(times)/n_runs:.3f}s')

