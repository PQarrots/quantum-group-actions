from sage.all import *
import itertools

Qi = QQ['I']
I = Qi.gen()
Qi = QQ.extension(I**2+1, names=('I',))
I = Qi.gen()

@cached_function
def multindex_to_index(*args):
	r"""
	Input: 4 elements i0,i1,i2,i3 in {0,1}.

	Output: k=i0+2*i1+4*i2+8*i3.
	"""
	if len(args)==4:
		i0,i1,i2,i3=args
	else:
		i0,i1,i2,i3=args[0]
	return i0+2*i1+4*i2+8*i3

@cached_function
def index_to_multindex(k):
	r"""
	Input: 
	- k: integer between 0 and 15.

	Output: binary decomposition of k.
	"""
	L_ind=[]
	l=k
	for i in range(4):
		L_ind.append(l%2)
		l=l//2
	return tuple(L_ind)

def bloc_decomposition(M):
	I1=[0,1,2,3]
	I2=[4,5,6,7]

	A=M[I1,I1]
	B=M[I2,I1]
	C=M[I1,I2]
	D=M[I2,I2]

	return A,B,C,D

def mat_prod_vect(A,I):
	J=[]
	for i in range(4):
		J.append(0)
		for k in range(4):
			J[i]+=A[i,k]*I[k]
	return tuple(J)

def add_tuple(I,J):
	K=[]
	for k in range(4):
		K.append(I[k]+J[k])
	return tuple(K)

def scal_prod_tuple(I,J):
	s=0
	for k in range(4):
		s+=I[k]*J[k]
	return s

def red_mod_2(I):
	J=[]
	for x in I:
		J.append(ZZ(x)%2)
	return tuple(J)

def choose_non_vanishing_index(C,D,zeta):
	for I0 in itertools.product([0,1],repeat=4):
		L=[0 for k in range(16)]
		for J in itertools.product([0,1],repeat=4):
			CJ=mat_prod_vect(C,J)
			DJ=mat_prod_vect(D,J)
			e=-scal_prod_tuple(CJ,DJ)-2*scal_prod_tuple(I0,DJ)

			I0pDJ=add_tuple(I0,CJ)

			L[multindex_to_index(red_mod_2(I0pDJ))]+=zeta**(ZZ(e))
		for k in range(16):
			if L[k]!=0:
				return I0,L


def base_change_theta_dim4(M,zeta):
	
	Z4=Integers(4)
	A,B,C,D=bloc_decomposition(M.change_ring(Z4))

	I0,L0=choose_non_vanishing_index(C,D,zeta)


	N=[L0]+[[0 for j in range(16)] for i in range(15)]
	for I in itertools.product([0,1],repeat=4):
		if I!=(0,0,0,0):
			AI=mat_prod_vect(A,I)
			BI=mat_prod_vect(B,I)
			for J in itertools.product([0,1],repeat=4):
				CJ=mat_prod_vect(C,J)
				DJ=mat_prod_vect(D,J)

				AIpCJ=add_tuple(AI,CJ)
				BIpDJ=add_tuple(BI,DJ)

				e=scal_prod_tuple(I,J)-scal_prod_tuple(AIpCJ,BIpDJ)-2*scal_prod_tuple(I0,BIpDJ)
				N[multindex_to_index(I)][multindex_to_index(red_mod_2(add_tuple(AIpCJ,I0)))]+=zeta**(ZZ(e))

	Fp2=zeta.parent()
	return matrix(Fp2,N)


def is_symplectic_matrix_dim4(M):
	A,B,C,D=bloc_decomposition(M)
	if B.transpose()*A!=A.transpose()*B:
		return False
	if C.transpose()*D!=D.transpose()*C:
		return False
	if A.transpose()*D-B.transpose()*C!=identity_matrix(4):
		return False
	return True

def compute_splittings():
	d_matrices = {}
	n = 0
	# Splitting change of basis
	for N1, Nb1 in itertools.product([1,3],repeat=2):
		N2 = 4-N1
		alpha = N1
		beta = N2
		t = Nb1
		for A1, A2 in itertools.product([0,1,2,3], repeat=2):
			A3 = (A1 + A2) % 4
			
			if (Nb1**2+A1*A3)%2 == 0 or (1+t**2*A1*A3)%2 ==0:
				continue

			mu = inverse_mod(Nb1**2+A1*A3,4)
			nu = inverse_mod(1+t**2*A1*A3,4)

			M = Matrix(Integers(4), [[0,0,0,0,-Nb1*N1*mu,A3*N1*mu,0,0],
				[0,0,0,0,-A1*N1*mu,-Nb1*N1*mu,0,0],
				[0,0,alpha*N2,0,0,0,0,0],
				[0,0,0,alpha*N2,0,0,0,0],
				[nu,-t*A1*nu,-alpha*N2,0,0,0,0,0],
				[t*A3*nu,nu,0,-alpha*N2,0,0,0,0],
				[0,0,0,0,Nb1*N1*mu,-A3*N1*mu,beta*N1,0],
				[0,0,0,0,A1*N1*mu,Nb1*N1*mu,0,beta*N1]])

			if not is_symplectic_matrix_dim4(M):
				continue

			N = base_change_theta_dim4(M,I)

			print("Case N1 = {}, Nb1 = {}, A1 = {}, A2 = {}:".format(N1,Nb1,A1,A2))
			id_case = N1*2**6+Nb1*2**4+A1*2**2+A2
			print("id = {}".format(id_case))
			found = False
			for x in d_matrices:
				if d_matrices[x] == N:
					found = True
					print(x)
			if not found:
				x = "Matrix of type {}".format(n)
				d_matrices[x] = N
				n+=1
				print(x)

	print("\n")
	for x in d_matrices:
		print(x+":")
		N = d_matrices[x]
		print(N)

		L_pos=[]
		L_vals=[]
		for i in range(16):
			for j in range(16):
				pos = 16*i+j
				if N[i][j]==4:
					L_pos.append(pos)
					L_vals.append(True)
				elif N[i][j]==-4:
					L_pos.append(pos)
					L_vals.append(False)
		print(L_pos)
		print(L_vals)


if __name__=="__main__":
	compute_splittings()




