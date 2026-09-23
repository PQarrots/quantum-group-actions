
def find_hamilton_rec(boolS,v,n,E):
	# V = [0, ..., n-1]
	# boolS[i]== True if i\in S and False otherwise
	boolT = boolS.copy()
	boolT[v] = False
	m = sum(boolT)
	if m==0:
		return [v]
	for w in range(n):
		# if w\in S and v<-->w
		if boolT[w] and ((v,w) in E or (w,v) in E):
			P=find_hamilton_rec(boolT,w,n,E)
			if len(P)==m:
				return P+[v]
	return []

def find_hamilton(n,E):
	boolS=[True for i in range(n)]
	for v in range(n):
		P=find_hamilton_rec(boolS,v,n,E)
		if len(P)==n:
			return P
	return False

def zeros_to_edges(d_zeros):
	E=[]
	for i in range(16):
		for j in d_zeros:
			k=i^j
			if (i not in d_zeros[j]) and ((i,k) not in E) and ((k,i) not in E):
				E.append((i,k))
	return E



