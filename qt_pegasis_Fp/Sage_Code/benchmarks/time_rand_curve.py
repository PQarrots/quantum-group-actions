import sys
sys.path.insert(0, '..')
from time import time
import logging

from sage.all import *
proof.all(False)

from qt_pegasis import qtPegasis

# Setup logging
logging.getLogger('qt_pegasis').setLevel(logging.INFO)

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
set_random_seed(rr)

# Benchmark parameters
n_runs = 10

for lvl in ['500P', '1000P', '1500P', '2000P', '4000P']:
    print(f'{lvl = }')
    EGA = qtPegasis(lvl)

    # First curve
    frak_a = EGA.sample_ideal()
    EA = EGA.qt_action(frak_a)

    EA = EllipticCurve(EA.parent().base(), [0, EA, 0, 1, 0])

    print(f'\nSetup done - starting {n_runs} runs for {lvl = }\n')

    t_labels = ['T1', 'T2.1', 'T2.2', 'T3']
    times = [0 for _ in t_labels]

    # for _ in tqdm(range(n_runs)):
    for _ in range(n_runs):
        print('\n' + '='*20 + f'\nRun {_+1}/{n_runs}  -  {lvl = }\n' + '='*20)
        frak_a = EGA.sample_ideal()
        _, run_times = EGA.qt_action(frak_a, EA, timings=True)

        for i in range(len(times)):
            times[i] += run_times[i]

    print('\n' + '='*20 + '\n')
    for i in range(len(t_labels)):
        print(f'Avg {t_labels[i]}: {times[i]/n_runs:.3f}s')
    print(f'Tot: {sum(times)/n_runs:.3f}s')
    print(f'\n' + '='*20)

    with open('times.txt', 'a') as fh:
        fh.write('='*50 + f'\n{lvl = }\n')
        for i in range(len(t_labels)):
            fh.write(f'Avg {t_labels[i]}: {times[i]/n_runs:.3f}s\n')
        fh.write(f'Tot: {sum(times)/n_runs:.3f}s\n')
        fh.write('='*50 + f'\n')
    break


