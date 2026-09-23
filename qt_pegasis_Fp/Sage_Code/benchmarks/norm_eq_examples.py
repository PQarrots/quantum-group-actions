import sys
sys.path.insert(0, '..')
from time import time
import logging

from sage.all import *
proof.arithmetic(False)

from norm_eq import qlapoti
from qt_pegasis import qtPegasis

# Setup logging
logging.getLogger('qt_pegasis').setLevel(logging.INFO)
logging.getLogger('hd_helpers').setLevel(logging.WARNING)

lines = []
d_niter = {'500':10,'1000':10,'1500':10,'2000':10,'4000':10}
d_NWORDS_ORDER = {'500':8,'1000':16,'1500':24,'2000':32,'4000':63}
for lvl in ['500','1000','1500','2000','4000']:
	print("Level "+lvl)
	lines += ["#ifdef P_"+lvl]
	niter = d_niter[lvl]
	lines += ["const norm_eq_sol_t SOL["+str(niter)+"] = { {"]
	EGA = qtPegasis(lvl)
	

	for i in range(niter):
		print("Iteration number "+str(i))

		print("Sampling ideal")
		t1 = time()
		frak_a = EGA.sample_sage_ideal()
		t2 = time()
		print("End sampling: "+str(t2-t1)+" s")

		print("Reducing ideal")
		t1 = time()
		frak_a_reduced = frak_a.reduce_equiv()
		frak_a_red_gens = frak_a_reduced.gens()
		N, alpha = frak_a_red_gens

		assert alpha[1] == 1/2
		a = alpha[0]-alpha[1]
		Nmod4 = int(N)%4

		if Nmod4 == 2:
			N = N//2
		t2=time()
		print("End reducing: "+str(t2-t1)+" s")

		print("Starting qlapoti")
		t1=time()
		A1, A2, B1, B2, C1, C2, D1, D2, E1, E2, N1, N2, Nb1 = qlapoti([ZZ(N),alpha], EGA.e_sol)
		t2=time()
		print("End qlapoti: "+str(t2-t1)+" s")
		d_val = {"N1":N1,"Nb1":Nb1,"A1":A1,"A2":A2,"B1":B1,"B2":B2,"C1":C1,"C2":C2,
		"D1":D1,"D2":D2,"E1":E1,"E2":E2}
		for x in d_val:
			val = d_val[x]%(2**(EGA.e_sol+2))
			line = "."+x+" = {"
			for j in range(d_NWORDS_ORDER[lvl]):
				line += hex(val%2**64)
				val = val // (2**64)
				if j<d_NWORDS_ORDER[lvl]-1:
					line += ", "
				else:
					line += "},"

			lines += [line]

		lines += [f".Nmod4 = {Nmod4},"]
		
		if int(a)%2:
			boolean = "true"
		else:
			boolean = "false"
		lines += [".two_isog_choice = "+boolean+" }"]

		if i<niter-1:
			lines[-1] += ", {"
		else:
			lines[-1] += "};"

	lines += ["#endif"]

filename = "solutions.h"
with open(filename, "w") as file:
	file.writelines([line + "\n" for line in lines])