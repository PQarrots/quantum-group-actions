from sage.all import *

from ..theta_structures.Theta_dim4 import ThetaStructureDim4, ThetaPointDim4
from ..theta_structures.theta_helpers_dim4 import hadamard, squared, batch_inversion, proj_batch_inversion, multindex_to_index, dot_prod_dim4, product_theta_point_dim4
from ..theta_structures.Tuple_point import TupleAddComponents
from ..theta_structures.montgomery_theta import torsion_to_theta_null_point
from ..isogenies.tree import Tree, Tree_new, fill_tree
from ..isogenies.isogeny_dim4 import IsogenyDim4, DualIsogenyDim4
from ..isogenies.tesseract import solve_HIIP_rabbit, adjacency_matrix
from ..basis_change.base_change_dim4 import dim4_vector_to_theta_point_superglue

def proj_equal(P1, P2):
    if len(P1) != len(P2):
        return False
    for i in range(0, len(P1)):
        if P1[i]==0:
            if P2[i] != 0:
                return False
        else:
            break
    r=P1[i]
    s=P2[i]
    for i in range(0, len(P1)):
        if P1[i]*s != P2[i]*r:
            return False
    return True

def find_zeros(O):
	L=[]
	for i in range(16):
		if O[i]==0:
			L.append(i)
	return L

class GluingIsogenyDim4(IsogenyDim4):
	def __init__(self,domain,L_K_8,L_K_8_ind, coerce=None, L_zeros=None):
		r"""
		Input:
		- domain: a ThetaStructureDim4.
		- L_K_8: list of points of 8-torsion in the kernel.
		- L_K_8_ind: list of corresponding multindices (i0,i1,i2,i3) of points in L_K_8
		(L_K_8[i]=i0*P0+i1*P1+i2*P2+i3*P3, where (P0,..,P3) is a basis of K_8 (compatible with
		the canonical basis of K_2) and L_K_8_ind[i]=(i0,i1,i2,i3)).
		"""

		if not isinstance(domain, ThetaStructureDim4):
			raise ValueError("Argument domain should be a ThetaStructureDim4 object.")
		self._domain = domain
		self._inv_null_point_dual=None
		self._coerce=coerce
		self._special_compute_codomain(L_K_8,L_K_8_ind,L_zeros)

		#a_i2=squared(self._domain.zero())
		#HB_i2=hadamard(squared(hadamard(self._codomain.zero())))
		#for i in range(16):
			#print(HB_i2[i]/a_i2[i])

	def _special_compute_codomain(self,L_K_8,L_K_8_ind,L_zeros=None):
		r"""
		Input:
		- L_K_8: list of points of 8-torsion in the kernel.
		- L_K_8_ind: list of corresponding multindices (i0,i1,i2,i3) of points in L_K_8
		(L_K_8[i]=i0*P0+i1*P1+i2*P2+i3*P3, where (P0,..,P3) is a basis of K_8 (compatible with
		the canonical basis of K_2) and L_K_8_ind[i]=(i0,i1,i2,i3)).
		- L_zeros (optional): pretedermined list of zero indices of codomain dual theta null
		point.
		
		Output:
		- codomain of the isogeny.
		Also initializes self._inv_null_point_dual, containing the inverse of theta-constants.
		"""

		HSK_8=[hadamard(squared(P.coords())) for P in L_K_8]

		if L_zeros is None:
			U2 = hadamard(squared(self._domain._null_point.coords()))
			L_zeros = find_zeros(U2)

		L_ind = [multindex_to_index(x) for x in L_K_8_ind]

		tree = fill_tree(HSK_8, L_ind, L_zeros)

		tree.edge_product()

		lamb = tree.invert_denom()

		inv_null_point_dual = [None for i in range(16)]
		inv_null_point_dual[tree._root] = lamb

		nz_dual = [lamb]

		for i in range(len(tree._ind_to_edge)):
			ind = tree._ind_to_edge[i]
			j = ind//16
			inv_null_point_dual[j] = tree._edge_num[i]*tree._edge_denom[i]
			nz_dual.append(inv_null_point_dual[j])

		null_point_dual = [self._coerce(0) for i in range(16)]
		nz_dual = proj_batch_inversion(nz_dual)
		null_point_dual[tree._root] = nz_dual[0]
		for i in range(len(tree._ind_to_edge)):
			ind = tree._ind_to_edge[i]
			j = ind//16
			null_point_dual[j] = nz_dual[i+1]
		null_point = hadamard(null_point_dual)

		# Duplicate memory for quick access but useless in C
		self._inv_null_point_dual = inv_null_point_dual

		#assert proj_equal(squared(null_point_dual),U2)

		self._codomain=ThetaStructureDim4(null_point,null_point_dual=null_point_dual,inv_null_point_dual=inv_null_point_dual)

	def _special_compute_codomain_old(self,L_K_8,L_K_8_ind):
		r"""Deprecated.

		Input:
		- L_K_8: list of points of 8-torsion in the kernel.
		- L_K_8_ind: list of corresponding multindices (i0,i1,i2,i3) of points in L_K_8
		(L_K_8[i]=i0*P0+i1*P1+i2*P2+i3*P3, where (P0,..,P3) is a basis of K_8 (compatible with
		the canonical basis of K_2) and L_K_8_ind[i]=(i0,i1,i2,i3)).
		
		Output:
		- codomain of the isogeny.
		Also initializes self._inv_null_point_dual, containing the inverse of theta-constants.
		"""
		HSK_8=[hadamard(squared(P.coords())) for P in L_K_8]

		# Choice of reference index j_0<->chi_0 corresponding to a non-vanishing theta-constant.
		found_tree=False
		j_0=0
		while not found_tree:
			found_k0=False
			for k in range(len(L_K_8)):
				if HSK_8[k][j_0]!=0:
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
							if jpk not in L_covered_ind and HSK_8[k][j]!=0:
								L_covered_ind.append(jpk)
								L_ratios_ind.append((j,jpk,k))
								tree_j=tree_ratios.look_node(j)
								tree_j.add_child(Tree(jpk),len(L_ratios_ind)-1)
								found_j=True
								#break
							#if found_j:
								#break
					if not found_j or len(L_covered_ind)==16:
						tree_filled=True
				if len(L_covered_ind)!=16:
					j_0+=1
				else:
					found_tree=True

		L_denom=[HSK_8[t[2]][t[0]] for t in L_ratios_ind]
		L_denom_inv=batch_inversion(L_denom)
		L_num=[HSK_8[t[2]][t[1]] for t in L_ratios_ind]
		L_ratios=[L_num[i]*L_denom_inv[i] for i in range(15)]

		L_coords_ind=tree_ratios.edge_product(L_ratios)

		O_coords=[ZZ(0) for i in range(16)]
		for t in L_coords_ind:
			if self._coerce:
				O_coords[t[1]]=self._coerce(t[0])
			else:
				O_coords[t[1]]=t[0]

		# Precomputation
		# TODO: optimize inversions and give inv_null_point_dual to the codomain _arithmetic_precomputation
		L_prec=[]
		L_prec_ind=[]
		for i in range(16):
			if O_coords[i]!=0:
				L_prec.append(O_coords[i])
				L_prec_ind.append(i)
		L_prec_inv=batch_inversion(L_prec)
		inv_null_point_dual=[None for i in range(16)]
		for i in range(len(L_prec)):
			inv_null_point_dual[L_prec_ind[i]]=L_prec_inv[i]

		self._inv_null_point_dual=inv_null_point_dual

		for k in range(len(L_K_8)):
			for j in range(16):
				jpk=j^multindex_to_index(L_K_8_ind[k])
				#assert HSK_8[k][j]*O_coords[jpk]==HSK_8[k][jpk]*O_coords[j]
		
		#assert proj_equal(squared(self._domain._null_point.coords()), hadamard(squared(O_coords)))

		self._codomain=ThetaStructureDim4(hadamard(O_coords),null_point_dual=O_coords)

	def special_image(self,P,L_trans,L_trans_ind):
		r"""Used when we cannot evaluate the isogeny self because the codomain has zero 
		dual theta constants.

		Input:
		- P: ThetaPointDim4 of the domain.
		- L_trans: list of translates of P+T of P by points of 4-torsion T above the kernel.
		- L_trans_ind: list of indices of the translation 4-torsion points T.
		If L_trans[i]=\sum i_j*B_K4[j] then L_trans_ind[j]=\sum 2**j*i_j.

		Output:
		- the image of P by the isogeny self.
		"""
		HS_P=hadamard(squared(P.coords()))
		HSL_trans=[hadamard(squared(Q.coords())) for Q in L_trans]
		O_coords=self._codomain.null_point_dual()
		
		# L_lambda_inv: List of inverses of lambda_i such that: 
		# HS(P+Ti)=(lambda_i*U_{chi.chi_i,0}(f(P))*U_{chi,0}(0))_chi.
		L_lambda_inv_num=[]
		L_lambda_inv_denom=[]

		for k in range(len(L_trans)):
			for j in range(16):
				jpk=j^L_trans_ind[k]
				if HSL_trans[k][j]!=0 and O_coords[jpk]!=0:
					L_lambda_inv_num.append(HS_P[jpk]*O_coords[j])
					L_lambda_inv_denom.append(HSL_trans[k][j]*O_coords[jpk])
					break
		L_lambda_inv_denom=batch_inversion(L_lambda_inv_denom)
		L_lambda_inv=[L_lambda_inv_num[i]*L_lambda_inv_denom[i] for i in range(len(L_trans))]

		for k in range(len(L_trans)):
			for j in range(16):
				jpk=j^L_trans_ind[k]
				#assert HS_P[jpk]*O_coords[j]==L_lambda_inv[k]*HSL_trans[k][j]*O_coords[jpk]

		U_fP=[]
		for i in range(16):
			if self._inv_null_point_dual[i]!=None:
				U_fP.append(self._inv_null_point_dual[i]*HS_P[i])
			else:
				for k in range(len(L_trans)):
					ipk=i^L_trans_ind[k]
					if self._inv_null_point_dual[ipk]!=None:
						U_fP.append(self._inv_null_point_dual[ipk]*HSL_trans[k][ipk]*L_lambda_inv[k])
						break

		fP=hadamard(U_fP)
		if self._coerce:
			fP=[self._coerce(x) for x in fP]

		return self._codomain(fP)

	def dual(self):
		return DualIsogenyDim4(self._codomain,self._domain, hadamard=False)

class SuperglueIsogenyDim4:
	def __init__(self,T16,P16,Q16,E,Et,aux_point_index,d_rabbit,change_theta_coords_matrix):
		# Change of theta coordinates 16*16 matrix
		self._change_theta_coords_matrix = change_theta_coords_matrix
		self.special_compute_codomain(T16,P16,Q16,E,Et,aux_point_index,d_rabbit)

	def hadamard_product_of_alt_sum_theta_point(self,P,twistP):
		PT = TupleAddComponents(P,self._aux_pt,self.E,twistP,self.twist_aux_pt)
		u, v = PT.to_vectors()

		Nu = dim4_vector_to_theta_point_superglue(u,
			self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
		Nv = dim4_vector_to_theta_point_superglue(v,
			self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)

		Nu2 = squared(Nu)
		Nv2 = squared(Nv)

		twist = self.twist_aux_pt^twistP
		prod = []
		if twist:
			for i in range(16):
				prod.append(Nu2[i]+Nv2[i])
		else:
			for i in range(16):
				prod.append(Nu2[i]-Nv2[i])

		return hadamard(prod)

	def hadamard_product_of_alt_sum_theta_point_special(self,P,twistP):
		# Only for the case twistP == self.twist_aux_pt
		if twistP != self.twist_aux_pt:
			raise ValueError("twistP != self.twist_aux_pt .hadamard_product_of_alt_sum_theta_point_special method\n only applies to points on the auxiliary point parent curve.")
		PpT = P+self._aux_pt
		PmT = P-self._aux_pt

		# TODO: careful with the twist (something to do on coordinates probably)
		theta_PpT = PpT.to_dim4_vector(twistP)
		theta_PmT = PmT.to_dim4_vector(twistP)

		theta_PpT = dim4_vector_to_theta_point_superglue(theta_PpT,
			self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
		theta_PmT = dim4_vector_to_theta_point_superglue(theta_PmT,
			self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)

		# Test --> passes
		#TmP = self._aux_pt - P
		#theta_TmP = TmP.to_dim4_vector(twistP)
		#theta_TmP = dim4_vector_to_theta_point_superglue(theta_TmP,
			#self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)

		#print("theta_PmT == theta_TmP ? "+str(theta_PmT==theta_TmP))

		
		prod = dot_prod_dim4(theta_PpT,theta_PmT)
		return hadamard(prod)

	def special_compute_codomain(self,T16,P16,Q16,E,Et,aux_point_index,d_rabbit):
		# Elliptic curve defining the domain E^4
		self.E = E
		# Twist of E
		self.Et = Et

		P4 = 4*P16
		Q4 = 4*Q16

		a, b = torsion_to_theta_null_point(P4)

		# Twist for the symplectic basis of E[4] 
		# (0 if Q=(1:*:1) on E --> (-1:*:1) on Et and 15 if Q=(-1:*:1) on E --> (1:*:1) on Et)
		if Q4[0]==1: # Q = (-1:*:1) on E, so the order is normal and we use twisted Hadamard [Dartois & Duparc, Lemma 4.1?]
			self.basis_twist = 15
		elif Q4[0]==-1: # Q = (1:*:1) on E, so the order is swaped and we use normal Hadamard [Dartois & Duparc, Lemma 4.1?]
			self.basis_twist = 0
			a, b = b, a
		else:
			print(P4,Q4)
			raise ValueError("Wrong 4-torsion basis.")

		OE4 = product_theta_point_dim4([a,b],[a,b],[a,b],[a,b])

		# (Product) theta structure on E^4
		self._domain = ThetaStructureDim4(OE4)

		# For test --> Test passes
		if False:
			from ..basis_change.base_change_dim4 import apply_block_hadamard 
			NOE4 = apply_block_hadamard(OE4,*self._change_theta_coords_matrix)
		
			T2 = [8*T for T in T16]
			theta_T2 = [T2[0].to_dim4_vector(False), T2[1].to_dim4_vector(False),
			T2[2].to_dim4_vector(True), T2[3].to_dim4_vector(True)]
			theta_T2 = [dim4_vector_to_theta_point_superglue(theta_T,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix) for theta_T in theta_T2]

			def scal_prod(i,j):
				return ((i&1)*(j&1)+((i>>1)&1)*((j>>1)&1)+((i>>2)&1)*((j>>2)&1)+((i>>3)&1)*((j>>3)&1))%2

			theta_T2bis = [NOE4.copy(),NOE4.copy(),NOE4.copy(),NOE4.copy()]
			for i in range(4):
				for j in range(16):
					if scal_prod(2**i,j):
						theta_T2bis[i][j]=-NOE4[j]
				print(proj_equal(theta_T2bis[i],theta_T2[i]))

		if aux_point_index==3:
			# 16-torsion point T to translate
			self._aux_pt = T16[0]+T16[1]

			# Specify if T is on E^4 (False) or (E^t)^4 (True)
			self.twist_aux_pt = False

			T1T1T1pT2 = self._aux_pt+2*T16[0]
			T1mT2 = T16[0]-T16[1]
			T1pT2T2T2 = self._aux_pt+2*T16[1]

			theta_T1T1T1pT2 = T1T1T1pT2.to_dim4_vector(False)
			theta_T1mT2 = T1mT2.to_dim4_vector(False)
			theta_T1pT2T2T2 = T1pT2T2T2.to_dim4_vector(False)
			
			theta_T1T1T1pT2 = dim4_vector_to_theta_point_superglue(theta_T1T1T1pT2,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
			theta_T1mT2 = dim4_vector_to_theta_point_superglue(theta_T1mT2,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
			theta_T1pT2T2T2 = dim4_vector_to_theta_point_superglue(theta_T1pT2T2T2,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)


			# Include points on the same curve in HPK8
			HPK8 = [hadamard(dot_prod_dim4(theta_T1T1T1pT2,theta_T1mT2)),hadamard(dot_prod_dim4(theta_T1pT2T2T2,theta_T1mT2))]

			# Append points on the twist in HPK8
			HPK8.append(self.hadamard_product_of_alt_sum_theta_point(2*T16[2],True))
			HPK8.append(self.hadamard_product_of_alt_sum_theta_point(2*T16[3],True))
			HPK8.append(self.hadamard_product_of_alt_sum_theta_point(2*(T16[2]+T16[3]),True))

			# Support
			d_ind_to_ker = {1:0,2:1,4:2,8:3,12:4}
		if aux_point_index==12:
			# 16-torsion point T to translate
			self._aux_pt = T16[2]+T16[3]
			# Specify if T is on E^4 (False) or (E^t)^4 (True)
			self.twist_aux_pt = True
			
			# Include points on the same curve in HPK8
			HPK8=[self.hadamard_product_of_alt_sum_theta_point(2*T16[0],False),
			self.hadamard_product_of_alt_sum_theta_point(2*T16[1],False)]

			# Append points on the twist in HPK8
			T3T3T3pT4 = self._aux_pt+2*T16[2]
			T3mT4 = T16[2]-T16[3]
			T3pT4T4T4 = self._aux_pt+2*T16[3]

			theta_T3T3T3pT4 = T3T3T3pT4.to_dim4_vector(True)
			theta_T3mT4 = T3mT4.to_dim4_vector(True)
			theta_T3pT4T4T4 = T3pT4T4T4.to_dim4_vector(True)
			
			theta_T3T3T3pT4 = dim4_vector_to_theta_point_superglue(theta_T3T3T3pT4,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
			theta_T3mT4 = dim4_vector_to_theta_point_superglue(theta_T3mT4,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
			theta_T3pT4T4T4 = dim4_vector_to_theta_point_superglue(theta_T3pT4T4T4,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
			
			HPK8.append(hadamard(dot_prod_dim4(theta_T3T3T3pT4,theta_T3mT4)))
			HPK8.append(hadamard(dot_prod_dim4(theta_T3pT4T4T4,theta_T3mT4)))

			# Append last point point on the same curve in HPK8
			HPK8.append(self.hadamard_product_of_alt_sum_theta_point(2*(T16[0]+T16[1]),False))

			# Support
			d_ind_to_ker = {1:0,2:1,4:2,8:3,3:4}

		# Test adjacency
		if False:
			A = adjacency_matrix(HPK8,d_ind_to_ker)
		
			#L_rabbits = [{"path":[3, 11, 9, 5, 4, 6, 2, 10, 8, 0, 1, 13, 12, 14],"r1":7,"r2":15},
			#{"path":[3, 7, 6, 10, 8, 9, 1, 5, 4, 0, 2, 14, 12, 13],"r1":11,"r2":15},
			#{"path":[12, 14, 6, 5, 1, 9, 8, 10, 2, 0, 4, 7, 3, 11],"r1":13,"r2":15},
			#{"path":[12, 13, 9, 10, 2, 6, 4, 5, 1, 0, 8, 11, 3, 7],"r1":14,"r2":15}]

			def test_rabbit(A,d_rabbit):
				# Ears
				ret = True
				ret = ret and (A[d_rabbit["r1"]][d_rabbit["path"][0]])
				ret = ret and (A[d_rabbit["r2"]][d_rabbit["path"][0]])
				# Body
				for i in range(13):
					ret = ret and (A[d_rabbit["path"][i]][d_rabbit["path"][i+1]])
				return ret

			print(test_rabbit(A,d_rabbit))

			# Test location of zeros
			L_zeros = []
			for i in range(5):
				L_zeros.append([])
				for j in range(16):
					if HPK8[i][j]==0:
						L_zeros[i].append(j)
			print(L_zeros)
		

		# Inverse image U^B(f(T))^{-1}.
		self._inv_im_aux_pt_dual = solve_HIIP_rabbit(HPK8,d_rabbit,d_ind_to_ker)
		
		if False:
			from ..basis_change.base_change_dim4 import apply_block_hadamard 
			NOE4 = apply_block_hadamard(OE4,*self._change_theta_coords_matrix)
			U2 = hadamard(squared(NOE4))

			PpT = self._aux_pt
			PmT = -self._aux_pt

			# TODO: careful with the twist (something to do on coordinates probably)
			theta_PpT = PpT.to_dim4_vector(self.twist_aux_pt)
			theta_PmT = PmT.to_dim4_vector(self.twist_aux_pt)

			theta_PpT = dim4_vector_to_theta_point_superglue(theta_PpT,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)
			theta_PmT = dim4_vector_to_theta_point_superglue(theta_PmT,
				self._domain._null_point,self.basis_twist,self._change_theta_coords_matrix)

			theta_O = dot_prod_dim4(hadamard(dot_prod_dim4(theta_PpT,theta_PmT)),self._inv_im_aux_pt_dual)

			print(proj_equal(U2,squared(theta_O)))

		#print(self._inv_im_aux_pt_dual)

	def eval(self,P,twistP):
		prod = self.hadamard_product_of_alt_sum_theta_point(P,twistP)
		fP = dot_prod_dim4(prod,self._inv_im_aux_pt_dual)
		fP = hadamard(fP)
		return fP

	def special_eval(self,P,twistP):
		# Only for the case twistP == self.twist_aux_pt
		prod = self.hadamard_product_of_alt_sum_theta_point_special(P,twistP)
		fP = dot_prod_dim4(prod,self._inv_im_aux_pt_dual)
		fP = hadamard(fP)
		return fP


