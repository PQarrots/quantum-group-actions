import sys
sys.path.insert(0, 'precomp')

from sage.all import *
import itertools
from .Find_Hamilton_path import *

Qi = QQ['I']
I = Qi.gen()
Qi = QQ.extension(I**2+1, names=('I',))
I = Qi.gen()

R = PolynomialRing(Qi,'a,b',2)
a, b = R.gens()

def double(P):
	xP = P[0]
	zP = P[1]
	return [b*(a**2*xP**4-2*b**2*xP**2*zP**2+a**2*zP**4),
	-a*(b**2*xP**4-2*a**2*xP**2*zP**2+b**2*zP**4)]

def diff_add(P,Q,PmQ):
	xP, zP = P[0], P[1]
	xQ, zQ = Q[0], Q[1]
	xPmQ, zPmQ = PmQ[0], PmQ[1]

	HSxP = xP**2
	HSzP = zP**2
	HSxP, HSzP = HSxP+HSzP, HSxP-HSzP
	HSxQ = xQ**2
	HSzQ = zQ**2
	HSxQ, HSzQ = HSxQ+HSzQ, HSxQ-HSzQ

	xPpQ = HSxP*HSxQ*(a**2-b**2)
	zPpQ = HSzP*HSzQ*(a**2+b**2)

	xPpQ, zPpQ = xPpQ+zPpQ, xPpQ-zPpQ

	xPpQ = xPpQ*zPmQ
	zPpQ = zPpQ*xPmQ

	return [xPpQ,zPpQ]

#P8 = [u,z]
#Q8 = [v,b]
#P4 = [1,1]
#Q4 = [1,0]
#P8mQ8 = [w,z2]
#P4mQ4 = [1,I]

#P8x3 = [z,u]
#Q8x3 = [b,-v]

#P8pQ8 = diff_add(P8,Q8,P8mQ8)
#P4pQ8 = diff_add(P8pQ8,P8,Q8)
#P8x3pQ8 = diff_add(P8pQ8,P4,P8mQ8)
#P8pQ4 = diff_add(P8pQ8,Q8,P8)
#P4pQ4 = diff_add(P4,Q4,P4mQ4)
#P8x3pQ4 = diff_add(P4pQ4,P8,P8pQ4)
#P8pQ8x3 = diff_add(P8pQ8,Q4,P8mQ8)
#P4pQ8x3 = diff_add(P4pQ4,Q8,P4pQ8)
#P8x3pQ8x3 = diff_add(P8x3pQ4,Q8,P8x3pQ8)

#LC_grid = [[[a,b],Q8,Q4,Q8x3],
#[P8,P8pQ8,P8pQ4,P8pQ8x3],
#[P4,P4pQ8,P4pQ4,P4pQ8x3],
#[P8x3,P8x3pQ8,P8x3pQ4,P8x3pQ8x3]]


def hadamard2(x,y):
    return (x+y, x-y)

def hadamard4(x,y,z,t):
    x,y=hadamard2(x,y)
    z,t=hadamard2(z,t)
    return (x+z, y+t, x-z, y-t)

def hadamard8(a,b,c,d,e,f,g,h):
    a,b,c,d=hadamard4(a,b,c,d)
    e,f,g,h=hadamard4(e,f,g,h)
    return (a+e, b+f, c+g, d+h, a-e, b-f, c-g, d-h)

def hadamard16(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p):
    a,b,c,d,e,f,g,h=hadamard8(a,b,c,d,e,f,g,h)
    i,j,k,l,m,n,o,p=hadamard8(i,j,k,l,m,n,o,p)
    return (a+i, b+j, c+k, d+l, e+m, f+n, g+o, h+p, a-i, b-j, c-k, d-l, e-m, f-n, g-o, h-p)

def hadamard(P):
    return hadamard16(*P)

def squared(P):
	return [x**2 for x in P]

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

def apply_base_change_theta_dim4(N,P):
	Q=[]
	for i in range(16):
		Q.append(0)
		for j in range(16):
			Q[i]+=N[i,j]*P[j]
	return Q

def complete_symplectic_matrix_dim4(C,D,n=4):
	Zn=Integers(n)

	Col_I4=[matrix(Zn,[[1],[0],[0],[0]]),matrix(Zn,[[0],[1],[0],[0],[0]]),
	matrix(Zn,[[0],[0],[1],[0],[0],[0]]),matrix(Zn,[[0],[0],[0],[1],[0],[0],[0]])]

	L_DC=block_matrix([[D.transpose(),-C.transpose()]])
	Col_AB_i=L_DC.solve_right(Col_I4[0])
	A_t=Col_AB_i[[0,1,2,3],0].transpose()
	B_t=Col_AB_i[[4,5,6,7],0].transpose()

	for i in range(1,4):
		F=block_matrix(2,1,[L_DC,block_matrix(1,2,[B_t,-A_t])])
		Col_AB_i=F.solve_right(Col_I4[i])
		A_t=block_matrix(2,1,[A_t,Col_AB_i[[0,1,2,3],0].transpose()])
		B_t=block_matrix(2,1,[B_t,Col_AB_i[[4,5,6,7],0].transpose()])

	A=A_t.transpose()
	B=B_t.transpose()

	M=block_matrix([[A,C],[B,D]])

	return M

def is_symplectic_matrix_dim4(M):
	A,B,C,D=bloc_decomposition(M)
	if B.transpose()*A!=A.transpose()*B:
		return False
	if C.transpose()*D!=D.transpose()*C:
		return False
	if A.transpose()*D-B.transpose()*C!=identity_matrix(4):
		return False
	return True


class Tree:
	def __init__(self,node):
		self._node=node
		self._edges=[]
		self._children=[]

	def add_child(self,child,edge):
		self._children.append(child)
		self._edges.append(edge)

	def look_node(self,node):
		if self._node==node:
			return self
		elif len(self._children)>0:
			for child in self._children:
				t_node=child.look_node(node)
				if t_node!=None:
					return t_node

	def edge_product(self,L_factors,factor_node=ZZ(1)):
		n=len(self._children)
		L_prod=[(factor_node,self._node)]
		for i in range(n):
			L_prod+=self._children[i].edge_product(L_factors,factor_node*L_factors[self._edges[i]])
		return L_prod

	def edge_num_den(self,L_num,L_den,factor_num=ZZ(1),factor_den=ZZ(1)):
		n=len(self._children)
		L_prod=[(factor_num,factor_den,self._node)]
		for i in range(n):
			L_prod+=self._children[i].edge_num_den(L_num,L_den,factor_num*L_num[self._edges[i]],factor_den*L_den[self._edges[i]])
		return L_prod

def batch_inversion(L):
	r"""Does n inversions in 3(n-1)M+1I.

	Input:
	- L: list of elements to invert.

	Output:
	- [1/x for x in L]
	"""
	# Given L=[a0,...,an]
	# Computes multiples=[a0, a0.a1, ..., a0...an]
	multiples=[L[0]]
	for ai in L[1:]:
		multiples.append(multiples[-1]*ai)

	# Computes inverses=[1/(a0...an),...,1/a0]
	inverses=[1]#[1/multiples[-1]]
	for i in range(1,len(L)):
		inverses.append(inverses[-1]*L[-i])

	# Finally computes [1/a0,...,1/an]
	result=[inverses[-1]]
	for i in range(2,len(L)+1):
		result.append(inverses[-i]*multiples[i-2])
	return result

def special_compute_codomain(L_K_8,L_K_8_ind,L_zeros):
	r"""
	Input:
	- L_K_8: list of points of 8-torsion in the kernel.
	- L_K_8_ind: list of corresponding multindices (i0,i1,i2,i3) of points in L_K_8
	(L_K_8[i]=i0*P0+i1*P1+i2*P2+i3*P3, where (P0,..,P3) is a basis of K_8 (compatible with
	the canonical basis of K_2) and L_K_8_ind[i]=(i0,i1,i2,i3)).
	
	Output:
	- codomain of the isogeny.
	Also initializes self._precomputation, containing the inverse of theta-constants.
	"""
	HSK_8=[hadamard(squared(P)) for P in L_K_8]
	L_zeros = []
	for x in HSK_8:
		L = []
		for i in range(16):
			if x[i]==0:
				L.append(i)
		L_zeros.append(L)
	print(L_zeros)
		
	# Choice of reference index j_0<->chi_0 corresponding to a non-vanishing theta-constant.
	found_tree=False
	j_0=0
	while not found_tree:
		found_k0=False
		for k in range(len(L_K_8)):
			j0pk=j_0^multindex_to_index(L_K_8_ind[k])
			if HSK_8[k][j_0]!=0 and j_0 not in L_zeros and j0pk not in L_zeros:
				k_0=k
				found_k0=True
				break
		if not found_k0:
			j_0+=1
		else:
			j0pk0=j_0^multindex_to_index(L_K_8_ind[k_0])
			# List of tuples of indices (index chi of the denominator: HS(f(P_k))_chi, 
			#index chi.chi_k of the numerator: HS(f(P_k))_chi.chi_k, index k).
			L_ratios_ind=[(j_0,j0pk0,k_0)]
			L_covered_ind=[j_0,j0pk0]

			# Tree containing the the theta-null points indices as nodes and the L_ratios_ind reference indices as edges.
			tree_ratios=Tree(j_0)
			tree_ratios.add_child(Tree(j0pk0),0)

			# Filling in the tree
			tree_filled=False
			while not tree_filled:
				found_j=False
				for j in L_covered_ind:
					for k in range(len(L_K_8)):
						jpk=j^multindex_to_index(L_K_8_ind[k])
						if (jpk not in L_covered_ind) and (jpk not in L_zeros) and HSK_8[k][j]!=0:
							print(jpk, jpk not in L_zeros)
							L_covered_ind.append(jpk)
							L_ratios_ind.append((j,jpk,k))
							tree_j=tree_ratios.look_node(j)
							tree_j.add_child(Tree(jpk),len(L_ratios_ind)-1)
							found_j=True
							#break
						#if found_j:
							#break
				if not found_j or len(L_covered_ind)==16-len(L_zeros):
					tree_filled=True
			if len(L_covered_ind)!=16-len(L_zeros):
				j_0+=1
			else:
				found_tree=True

	print(L_covered_ind)
	print(L_ratios_ind)

	L_denom=[HSK_8[t[2]][t[0]] for t in L_ratios_ind]
	L_num=[HSK_8[t[2]][t[1]] for t in L_ratios_ind]
	L_num_den_prod = tree_ratios.edge_num_den(L_num,L_denom)
	L_denom_inv = batch_inversion([t[1] for t in L_num_den_prod])
	L_ratios=[L_num_den_prod[i][0]*L_denom_inv[i] for i in range(len(L_ratios_ind))]

	O_coords=[ZZ(0) for i in range(16)]
	for i in range(len(L_ratios)):
		O_coords[L_num_den_prod[i][2]]=L_ratios[i]

	L_zeros = []
	for i in range(16):
		if O_coords[i]==0:
			L_zeros.append(i)
	print(L_zeros)

	return hadamard(O_coords)


def four_isogeny(M):
	#M = complete_symplectic_matrix_dim4(C,D)
	N = base_change_theta_dim4(M,I)
	A,B,C,D = bloc_decomposition(M)

	assert is_symplectic_matrix_dim4(M)

	OE = [a,b]
	OE4 = []
	for i in range(2):
		for j in range(2):
			for k in range(2):
				for l in range(2):
					OE4.append(OE[i]*OE[j]*OE[k]*OE[l])

	OE4N = apply_base_change_theta_dim4(N,OE4)
	Uf12 = hadamard(squared(OE4N))

	
	# comb_PQ[i][j]=[i]P+[j]Q
	# P = [a+b,a-b], Q = [-1,1] (A+2 not a square, pi(P)=P, pi(Q)=-Q)
	# theta(P) = [1, 1], theta(Q) = [1,0], theta(P-Q)=[1,I] 
	comb_PQ = [[[a,b],[1,0],[a,-b], [1,0]],
	[[1,1],[1,-I],[1,-1],[1,I]],
	[[b,a],[0,1],[b,-a],[0,1]],
	[[1,1],[1,I],[1,-1],[1,-I]]]
	#for i in range(4):
		#for j in range(4):
			#x,y=comb_PQ[i][j]
			#comb_PQ[i][j]=[y,x]

	K = [None for i in range(16)]
	for i in range(2):
		for j in range(2):
			for k in range(2):
				for l in range(2):
					C_ijkl = [i*C[m,0]+j*C[m,1]+k*C[m,2]+l*C[m,3] for m in range(4)]
					D_ijkl = [i*D[m,0]+j*D[m,1]+k*D[m,2]+l*D[m,3] for m in range(4)]

					T = []
					for ind in range(16):
						i0, i1, i2, i3 = index_to_multindex(ind)
						T.append(comb_PQ[C_ijkl[0]][D_ijkl[0]][i0]
							*comb_PQ[C_ijkl[1]][D_ijkl[1]][i1]
							*comb_PQ[C_ijkl[2]][D_ijkl[2]][i2]
							*comb_PQ[C_ijkl[3]][D_ijkl[3]][i3])
					T=apply_base_change_theta_dim4(N,T)
					K[i+2*j+4*k+8*l]=T

	Uf22 = [T[0] for T in K]


	#HSK = [hadamard(squared(T)) for T in K]
	#correct = True
	#for i in range(16):
		#for j in range(16):
			#test = (((HSK[0][j]*HSK[0][j^i]==0) and (HSK[i][j]==0)) or ((HSK[0][j]*HSK[0][j^i]!=0) and (HSK[i][j]!=0)))
			#print(i,j,test,HSK[0][j]*HSK[0][j^i],HSK[i][j])
			#correct = correct and test
	#print(correct)

	return N,Uf12,Uf22

def Subgraph_codomain(M):
	N = base_change_theta_dim4(M,I)
	A,B,C,D = bloc_decomposition(M)

	assert is_symplectic_matrix_dim4(M)

	OE = [a,b]
	OE4 = []
	for i in range(2):
		for j in range(2):
			for k in range(2):
				for l in range(2):
					OE4.append(OE[i]*OE[j]*OE[k]*OE[l])

	OE4N = apply_base_change_theta_dim4(N,OE4)

	# comb_PQ[i][j]=[i]P+[j]Q
	# P = [a+b,a-b], Q = [-1,1] (A+2 not a square, pi(P)=P, pi(Q)=-Q)
	# theta(P) = [1, 1], theta(Q) = [1,0], theta(P-Q)=[1,I] 
	comb_PQ = [[[a,b],[1,0],[a,-b], [1,0]],
	[[1,1],[1,-I],[1,-1],[1,I]],
	[[b,a],[0,1],[b,-a],[0,1]],
	[[1,1],[1,I],[1,-1],[1,-I]]]

	K = [None for i in range(16)]
	for i in range(2):
		for j in range(2):
			for k in range(2):
				for l in range(2):
					C_ijkl = [i*C[m,0]+j*C[m,1]+k*C[m,2]+l*C[m,3] for m in range(4)]
					D_ijkl = [i*D[m,0]+j*D[m,1]+k*D[m,2]+l*D[m,3] for m in range(4)]

					T = []
					for ind in range(16):
						i0, i1, i2, i3 = index_to_multindex(ind)
						T.append(comb_PQ[C_ijkl[0]][D_ijkl[0]][i0]
							*comb_PQ[C_ijkl[1]][D_ijkl[1]][i1]
							*comb_PQ[C_ijkl[2]][D_ijkl[2]][i2]
							*comb_PQ[C_ijkl[3]][D_ijkl[3]][i3])
					T=apply_base_change_theta_dim4(N,T)
					K[i+2*j+4*k+8*l]=T


	add_play = [1,2,4,8,3,12]
	image_8_torsion = []
 
	for i in add_play:
		# H(Theta(f1(P)))**2=H(Theta(2P)*Theta(0))
		P = [K[0][j]*K[i][j] for j in range(16)]
		image_8_torsion.append(hadamard(P))
  

	Adgency_matrix = matrix(16) 
 
	for j in range(len(add_play)):
		for i in range(16):
			if image_8_torsion[j][i] !=0:
				Adgency_matrix[i,(i ^ add_play[j])] = 1
    

  
	return Adgency_matrix




def Compute_rabbits():
	# s1 = (C1 - E1) % 4
	# s2 = (C1 + C2 - E1 - E2) % 4
	# s3 = (B1 + B2 + D1 + D2) % 4
	# s4 = (B1 + D1) % 4
	Adgency_matrices = []
	Case_names = []

	print("Searching for rabbits...")
	for N1 in [1,3]:
		alpha = N1
		beta = 4-alpha
		for s1, s2, s3, s4 in itertools.product([0,1,2,3],repeat=4):
			A=matrix(Integers(4),[[0,0,0,0],[0,0,0,0],
				[0,0,beta*s3,-beta*s1],[0,0,beta*s2,beta*s4]]).transpose()
			B=matrix(Integers(4),[[-alpha,0,0,0],[0,-alpha,0,0],
				[0,0,0,0],[0,0,0,0]]).transpose()
			C=matrix(Integers(4),[[N1,0,s3,-s1],[0,N1,s2,s4],
				[0,0,0,0],[0,0,0,0]]).transpose()
			D=matrix(Integers(4),[[0,0,0,0],[0,0,0,0],
				[1,0,alpha*s4,-alpha*s2],[0,1,alpha*s1,alpha*s3]]).transpose()

			if (s1 in [0,2] and s2 in [0,2]) or (s3 in [0,2] and s4 in [0,2]):# Ensure gluing
				continue
			if s1==2 or s2==2 or s3==2 or s4==2:# Avoid singularity in f2
				continue

			M=block_matrix([[A,C],[B,D]])
			if not is_symplectic_matrix_dim4(M):
				continue


			Case_names.append((N1,s1,s2,s3,s4))
			print("Found rabbit for this basis Case N1={}, s1={}, s2={}, s3={}, s4={} (mod 4):".format(N1,s1,s2,s3,s4))
			Adgency_matrice = Subgraph_codomain(M)
			Adgency_matrices.append(Adgency_matrice)

			#codomain_graph = Graph(Adgency_matrice)
			#codomain_graph.show()
			#codomain_graphs.append(codomain_graph)


			#for x in d_zeros:


			#d_zeros_6 = {1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],6:d_zeros[6]}
			#d_zeros_9 = {1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],9:d_zeros[9]}
			#d_zeros_69 = {1:d_zeros[1],2:d_zeros[2],4:d_zeros[4],8:d_zeros[8],6:d_zeros[6],9:d_zeros[9]}

			#E6 = zeros_to_edges(d_zeros_6)
			#E9 = zeros_to_edges(d_zeros_9)
			#E69 = zeros_to_edges(d_zeros_69)

			#P6 = find_hamilton(16,E6)
			#print("Hamilton path with index 1,2,4,8 and 6:{}".format(P6))

			#P9 = find_hamilton(16,E9)
			#print("Hamilton path with index 1,2,4,8 and 9:{}".format(P9))

			#P69 = find_hamilton(16,E69)
			#print("Hamilton path with index 1,2,4,8 and 9:{}".format(P69))

	return Case_names, Adgency_matrices


def bit_flip(n, type):
    b0 = n % 2
    b1 = (n // 2) % 2
    b2 = (n // 4) % 2
    b3 = (n // 8) % 2 
    if type == "identity":
        return 8*b3 + 4*b2 + 2*b1 + b0
    elif type == "(0 1)(2 3)":
        return 8*b2 + 4*b3 + 2*b0 + b1
    elif type == "(0 2)(1 3)":
        return 8*b1 + 4*b0 + 2*b3 + b2
    elif type == "(0 3)(1 2)":    
        return 8*b0 + 4*b1 + 2*b2 + b3

    


class Rabbit:
	def __init__(self,Case_name,Adgency_matrix):
		self._Case_name=Case_name
		self._Adgency_matrix=Adgency_matrix
		self.Graph=Graph(Adgency_matrix)
    
		self._rabbit_type=None
		for i in [7,11,13,14]:
			if sum(self._Adgency_matrix[i])== 1:
				self._rabbit_type=i
				break

		if self._rabbit_type==7 or self._rabbit_type==11:
			self._8_torsion_points = [(1,2,4,8,12)] 
		if self._rabbit_type==13 or self._rabbit_type==14:
			self._8_torsion_points = [(1,2,4,8,3)] 
   
		if self._rabbit_type==7:
			self.isomorphism_type = "identity"
		if self._rabbit_type==14:
			self.isomorphism_type = "(0 3)(1 2)"
		if self._rabbit_type==11:
			self.isomorphism_type = "(0 1)(2 3)"
		if self._rabbit_type==13:
			self.isomorphism_type = "(0 2)(1 3)"
  
		central_spanning_tree = [(15,3),(7,3),(3,11),(11,9),(9,5),(5,4),(4,6),(6,2),(2,10),(10,8),(8,0),(0,1),(1,13),(13,12),(12,14)]
		self.spanning_tree = [(bit_flip(edge[0],self.isomorphism_type),bit_flip(edge[1],self.isomorphism_type)) for edge in central_spanning_tree]
  
	def is_isomorphic_to(self,other):
		if self.Graph.is_isomorphic(other.Graph):
			return True
		return False
  
	def show(self):
		self.Graph.show()
  
	def __repr__(self):
		N1, s1, s2, s3, s4 = self._Case_name
		ans = "Rabbit Case: N1={}, s1={}, s2={}, s3={}, s4={} (mod 4).\n".format(*self._Case_name)
		ans += "id: {}.\n".format(N1*2**8+s1*2**6+s2*2**4+s3*2**2+s4)
		ans += "Rabbit type: {}.\n".format(self._rabbit_type)
		ans += "Computable using the following 8-torsion:{}\n".format(self._8_torsion_points)
		ans += "Isomorphism to a type 7 (on bits): {}\n".format(self.isomorphism_type)
		ans += "Rabbit skeleton (edge list) :{}\n".format(self.spanning_tree)
		return ans

    




if __name__=="__main__":
	Case_names, Adgency_matrices = Compute_rabbits()

	print("\n\nAll Rabbits found:\n")
 
	Rabbits = []
	for i in range(len(Case_names)):
		Rabbits.append(Rabbit(Case_names[i],Adgency_matrices[i]))
		print(Rabbits[i])
		print("\n")
		#codomain_graphs[i].show()
	
	#check all isomorphism
	for i in range(len(Case_names)):
		for j in range(i+1,len(Case_names)):
			assert Rabbits[i].is_isomorphic_to(Rabbits[j])
   

"""
   ⠀⠀⠀⠀⠀⣠⣤⣦⣤⣄⡀⠀⠀⠀⠀⢀⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⣰⠟⠙⠀⠀⠀⠈⢻⡆⠀⣴⠞⠋⠉⠉⠙⠳⣦⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢸⡛⠂⠀⠀⠀⠀⠀⠈⣿⣾⠋⠀⠀⠀⠀⠀⠀⠈⣿⡄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣽⠁⠀⠀⠀⠀⠀⠀⠀⣽⢇⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢰⣿⠄⠀⠀⠀⠀⠀⠀⠐⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⢺⡇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⢨⡟⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠇⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠈⣿⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣿⡆⠀⠀⢀⣀⣀⡀⢸⣇⠀⠀⠀⠀⠀⠀⠀⢀⣾⠃⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣘⡟⠰⠛⠛⠉⠙⠉⠈⠃⠀⠀⠀⠀⠀⠀⢰⣾⡟⠚⢶⣄⠀⠀⠀⠀⠀
⠀⠀⠀⣤⡾⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡁⠀⢀⡬⢹⡇⠀⠀⠀⠀
⠀⠀⣴⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⠀⠚⢷⣼⡷⠀⠀⠀⠀
⠀⣼⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢙⣷⠀⠀⠘⢿⣷⠀⠀⠀
⢸⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣇⠀⠀⠀⢹⣧⠀⠀
⣿⢣⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡏⣡⠀⠀⠀⠻⣧⠀
⣿⡾⡿⠖⠀⠀⠀⠀⠀⠀⠀⠀⢀⣶⣿⣤⠀⠀⠀⠀⠀⠀⠀⣼⡇⠃⠀⠀⠀⠀⢹⣇
⠹⣧⡀⠀⠀⠰⣦⣸⣶⠄⠀⠀⠸⡿⠿⠇⠀⠀⠀⠀⠀⠀⢢⡿⠅⠀⠀⠀⠀⠀⠀⣿
⠀⠈⠻⣦⣒⠸⠛⠻⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠟⠁⠀⠀⠀⠀⣄⠀⠀⣾
⠀⠀⠀⠈⢙⣷⢶⣤⣀⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⡶⠟⠁⠀⠀⠀⠀⠀⣼⢏⣠⣾⠟
⠀⠀⠀⢀⣾⠃⠀⠀⠉⠛⠛⠻⠶⠶⠶⠶⠞⠋⠁⠀⠀⠀⠀⠀⠀⣰⡾⠛⠛⠉⠀⠀
⠀⠀⠀⠘⣿⠀⠀⠀⠀⠀⢲⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⣠⡾⠏⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠻⣧⡀⠀⠀⣡⣿⠛⠻⠶⣾⠀⠀⠀⠀⠀⠀⠈⢾⡟⠆⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠉⠛⠛⠛⠋⠁⠀⠀⠀⢿⣦⠀⠀⠀⠀⠀⣠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣶⣤⣀⣦⣴⡟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
"""
#Congratulations, you have found the hidden rabbit !