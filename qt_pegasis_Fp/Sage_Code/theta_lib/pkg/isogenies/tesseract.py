from sage.all import *

def adjacency_matrix(HSK_8,d_int_to_ker):
	A = [[False for _ in range(16)] for _ in range(16)]
	for i in range(16):
		for j in d_int_to_ker:
			A[i][i^j] = (HSK_8[d_int_to_ker[j]][i^j]!=0)
			#A[i^j][i] = A[i][i^j]
	return A

Gray_path = [i^(i>>1) for i in range(16)]
Sigmas_4 = [[0, 1, 2, 3],
[0, 1, 3, 2],
[0, 2, 1, 3],
[0, 2, 3, 1],
[0, 3, 1, 2],
[0, 3, 2, 1],
[1, 0, 2, 3],
[1, 0, 3, 2],
[1, 2, 0, 3],
[1, 2, 3, 0],
[1, 3, 0, 2],
[1, 3, 2, 0],
[2, 0, 1, 3],
[2, 0, 3, 1],
[2, 1, 0, 3],
[2, 1, 3, 0],
[2, 3, 0, 1],
[2, 3, 1, 0],
[3, 0, 1, 2],
[3, 0, 2, 1],
[3, 1, 0, 2],
[3, 1, 2, 0],
[3, 2, 0, 1],
[3, 2, 1, 0]]

def act_on_index(i,s):
	ir = 0
	for m in range(4):
		ir += ((i>>m)&1)<<Sigmas_4[s][m]
	return ir

def act_on_path(path,s):
	pret=[]
	for i in path:
		pret.append(act_on_index(i,s))
	return pret

def is_hamilton_path(path,A):
	ret = True
	for i in range(15):
		ret = (ret and A[path[i]][path[i+1]] and A[path[i+1]][path[i]])
	return ret

def find_hamilton_path_tesseract(A):
	for s in range(24):
		path = act_on_path(Gray_path,s)
		if is_hamilton_path(path,A):
			New_path = path
	return New_path

def act_on_index_ext(i,aux_point_index,s1,v1,s2):
	# aux_point_index decides the support (e.g. {0,1} for 3=0011)
	# s1 automorphism outside the support (S_2), 0 is id and 1 is the transposition
	# v1 vector translate on the support (in (Z/2Z)^2)
	# s2 index of automorphism on the support (S((Z/2Z)^2) seen identified as Sigmas_4)
	if aux_point_index==3:
		ir = (((i>>2)&1)<<s1)+(((i>>3)&1)<<(s1^1))
		ir = (ir^v1)<<2
		ir += Sigmas_4[s2][i&3]
		return ir
	elif aux_point_index==12:
		ir = ((i&1)<<s1)+(((i>>1)&1)<<(s1^1))
		ir = ir^v1
		ir += Sigmas_4[s2][(i>>2)&3]<<2
		return ir
	else:
		raise NotImplementedError("aux_point_index should be 3 or 12.")

def act_on_path_ext(path,aux_point_index,s1,v1,s2):
	pret=[]
	for i in path:
		pret.append(act_on_index_ext(i,aux_point_index,s1,v1,s2))
	return pret

Path_3 = [6, 5, 4, 0, 1, 3, 2, 10, 11,  9,  8, 12, 13, 14, 15,  7]
Path_12 = [9, 5, 1, 0, 2, 3, 7,  6,  4,  8, 10, 11, 15, 14, 12, 13]

def find_hamilton_path_extended(A,aux_point_index):
	if aux_point_index==3:
		for s1 in range(2):
			for v1 in range(4):
				for s2 in range(24):
					path = act_on_path_ext(Gray_path,aux_point_index,s1,v1,s2)
					if is_hamilton_path(path,A):
						New_path = path
					path = act_on_path_ext(Path_3,aux_point_index,s1,v1,s2)
					if is_hamilton_path(path,A):
						New_path = path
	elif aux_point_index==12:
		for s1 in range(2):
			for v1 in range(4):
				for s2 in range(24):
					path = act_on_path_ext(Gray_path,aux_point_index,s1,v1,s2)
					if is_hamilton_path(path,A):
						New_path = path
					path = act_on_path_ext(Path_12,aux_point_index,s1,v1,s2)
					if is_hamilton_path(path,A):
						New_path = path
	else:
		raise NotImplementedError("aux_point_index should be 3 or 12.")
	return New_path

def solve_HIIP(HSK_8,path,d_int_to_ker):
	i0 = path[0]
	i1 = path[1]
	l = d_int_to_ker[i0^i1]
	red = [HSK_8[l][i0]]

	i2 = path[14]
	i3 = path[15]
	l = d_int_to_ker[i2^i3]
	blue = [HSK_8[l][i3]]

	for j in range(14):
		i0 = path[j+1]
		i1 = path[j+2]
		l = d_int_to_ker[i0^i1]
		red.append(red[-1]*HSK_8[l][i0])

		i2 = path[13-j]
		i3 = path[14-j]
		l = d_int_to_ker[i2^i3]
		blue.append(blue[-1]*HSK_8[l][i3])

	inv_null_pt_dual = [None]*16

	inv_null_pt_dual[path[0]]=blue[14]
	inv_null_pt_dual[path[15]]=red[14]

	for j in range(1,15):
		inv_null_pt_dual[path[j]]=red[j-1]*blue[14-j]

	return inv_null_pt_dual


def solve_HIIP_dichotomy(HSK_8,path,d_int_to_ker):
	# Deprecated, do not use
	inv_null_pt_dual = [None]*16
	# INIT LOOP: 
	for pad in range(8):
		i0 = path[2*pad]
		i1 = path[2*pad+1]
		l = d_int_to_ker[i0^i1]
		inv_null_pt_dual[i0] = HSK_8[l][i1]
		inv_null_pt_dual[i1] = HSK_8[l][i0]
	# MAIN LOOP
	length = 2
	while length<16:
		pad = 0
		while pad<16:
			mid_up = pad ^ length
			mid_low = pad ^ (length-1)
			i0 = path[mid_low]
			i1 = path[mid_up]
			l = d_int_to_ker[i0^i1]
			top_right = inv_null_pt_dual[i1]*HSK_8[l][i1]
			bot_left = inv_null_pt_dual[i0]*HSK_8[l][i0]
			for j in range(pad,mid_up):
				i0 = path[j]
				i1 = path[j^length]
				inv_null_pt_dual[i0] *= top_right
				inv_null_pt_dual[i1] *= bot_left
			pad += (length<<1)
		length = 2*length
	return inv_null_pt_dual

def solve_HIIP_rabbit(HSP_8,d_rabbit,d_int_to_ker):
	r"""
	- d_rabbit: rabbit structure dictionary with d_rabbit["r1"], d_rabbit["r2"]
	its ears and d_rabbit["path"] a path representing the body of the rabbit. 
	(r1 and r2 are connected to d_rabbit["path"][0]).
	"""
	#print(d_int_to_ker)
	#print(d_rabbit)

	## Rabbit's body
	path = d_rabbit["path"]

	# Additional part of the rabbit's body
	r1 = d_rabbit["r1"]
	r2 = d_rabbit["r2"]
	t1 = d_int_to_ker[r1^path[0]]
	t2 = d_int_to_ker[r2^path[0]]
	pi = HSP_8[t1][r1]*HSP_8[t2][r2]

	i0 = path[0]
	i1 = path[1]
	l = d_int_to_ker[i0^i1]
	red = [pi*HSP_8[l][i0]]

	i2 = path[12]
	i3 = path[13]
	l = d_int_to_ker[i2^i3]
	blue = [HSP_8[l][i3]]

	for j in range(12):
		i0 = path[j+1]
		i1 = path[j+2]
		l = d_int_to_ker[i0^i1]
		red.append(red[-1]*HSP_8[l][i0])

		i2 = path[11-j]
		i3 = path[12-j]
		l = d_int_to_ker[i2^i3]
		blue.append(blue[-1]*HSP_8[l][i3])

	inv_null_pt_dual = [None]*16

	inv_null_pt_dual[path[0]]=pi*blue[12]
	inv_null_pt_dual[path[13]]=red[12]

	for j in range(1,13):
		inv_null_pt_dual[path[j]]=red[j-1]*blue[12-j]

	## Rabbit's ears
	inv_null_pt_dual[r1]=blue[12]*HSP_8[t1][path[0]]*HSP_8[t2][r2]
	inv_null_pt_dual[r2]=blue[12]*HSP_8[t1][r1]*HSP_8[t2][path[0]]
	
	return inv_null_pt_dual

