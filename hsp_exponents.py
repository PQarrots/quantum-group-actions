"""
Script to compute the trade-offs in Kuperberg's algorithm.
"""

from math import*

from scipy import optimize

def quantum_1(t0,r,n): return (t0**2+2*log(r,2)/(r-1)*n)**0.5

def classical_1(t0, r,n): return r/2*(t0**2+2*log(r,2)/(r-1)*n)**0.5

def query_1(t0,r,n): return quantum_1(t0,r,n)-t0+log(t0,2)

def classical_2(t0, r,n): return r/2*(t0**2+4*log(r,2)/r/(r-1)*n)**0.5

def quantum_2(t0,r,n): return t0*(1-r/2)+classical_2(t0,r,n)

def query_2(t0,r,n): return quantum_2(t0,r,n) -t0 + log(t0,2)

def target_function_1(t0,q,t,r,n):return max(classical_1(t0,r,n) -t, quantum_1(t0,r,n), query_1(t0,r,n)+q)


def target_function_2(t0,q,t,r,n):return max(classical_2(t0,r,n) -t, quantum_2(t0,r,n), query_2(t0,r,n)+q)

def run_n(n,q):
    for r in range(2,12,2):
        print(f"r = {r}")
        for t in (0,10,20,30,40,48):
            print(f"\tt = {t}")
            a =        optimize.minimize_scalar(target_function_1, args=(q,t,r,n), bounds=(1,n))
            b =        optimize.minimize_scalar(target_function_2, args=(q,t,r,n), bounds=(1,n))
            if a.success:
                print(f"\t\t1 : t0 = {a.x}, C = {a.fun}")
            if b.success:
                print(f"\t\t2 : t0 = {b.x}, C = {b.fun}")

