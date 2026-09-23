import sys
sys.path.insert(0, '..')
from tqdm import tqdm
from time import time

from sage.all import *

from params import qt_params
from norm_eq import qlapoti

rr = randint(1, 2**16)
print(f'{rr = }\n====================')
set_random_seed(rr)

# Benchmark parameters
lvl = '500P' # add P for PEGASIS parameters
n_runs = 10

print(f'\nSetup done - starting {n_runs} runs for {lvl = }\n')

stats = {}
tot_time = 0

print(f'Initialization')

params = qt_params(lvl)
f = params['f']
e = params['e']
e_sol = e - 3
p = f * 2**e - 1

K = NumberField(name="pi", polynomial = var('x')**2 + p)
pi = K.gens()[0]

ub = isqrt(p)
lb = ub // 2**30

def random_ideal():

    ell = next_prime(lb, ub)
    while kronecker(-p, ell) != 1:
        ell = next_prime(lb, ub)

    Fl = GF(ell)
    x = Fl.polynomial_ring().gens()[0]
    lam = (x**2 + x + Fl(p+1)/4).any_root()
    b2 = (1 - pi)/2 + ZZ(lam)
    if (b2-ell).norm() < b2.norm():
        b2 -= ell

    return (ell, b2)

frak_a = random_ideal()
_ = qlapoti(frak_a, e_sol) # Warm up sage

for _ in tqdm(range(n_runs)):
    frak_a = random_ideal()

    t0 = time()
    run_stats = qlapoti(frak_a, e_sol, stats=True)
    t1 = time()
    tot_time += t1-t0

    for k in run_stats:
        stats[k] = stats.get(k, 0) + run_stats[k]

sing_lab = ['er_1_1', 'er_1_2', 'er_5_1', 'er_5_2', 'er_2_3']
stats['Singular cases'] = sum(stats[k] for k in sing_lab)

print(f'Time: {tot_time/n_runs:.3f}s')
for k in stats:
    avg = stats[k] / n_runs
    print(f'{k}: {avg:.3f}')

