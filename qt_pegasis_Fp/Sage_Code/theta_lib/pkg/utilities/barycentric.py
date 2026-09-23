from sage.all import *

class AddComponents:
	def __init__(self,P,Q,E=None,twistP=False,twistQ=False):
		if E==None:
			E=P.curve()
		A = E.a_invariants()[1]
		F = A.parent()
		
		xP, yP, zP = P
		xQ, yQ, zQ = Q

		if xP == xQ and ((yP == yQ) or (yP == -yQ)):
			raise ValueError("Cannot create AddComponents when P = Q or -Q.")

		if twistP:
			xP = -xP
		if twistQ:
			xQ = -xQ
		
		#Gamma2 = xP-xQ
		#Gamma2 = Gamma2*Gamma2


		#self.u = yP*yP+yQ*yQ-(A+xP+xQ)*Gamma2 
		#self.v = yP*yQ
		#self.v = self.v + self.v
		#self.w = Gamma2

		t1 = zP * zQ # t1 = zP zQ
		t2 = xP * zQ # t1 = xP zQ
		t3 = zP * xQ # t3 = zP xQ
		t4 = yP * zQ # t4 = yP zQ
		t5 = zP * yQ # t5 = zP yQ
		t6 = t4 * t5 # t6 = t4 t5 = yP yQ zP zQ
		t6 = t6 * t1 # t6 = t6 t1 = yP yQ zP**2 zQ**2
		v  = 2 * t6 # v = 2 yP yQ zP**2 zQ**2

		t7 = A * t1 # t7 = A zP zQ
		t2_plus_t3 = t2 + t3 # t2_plus_t3 = xP zQ + zP xQ
		t7 = t7 + t2_plus_t3 # t7 = A zP zQ + xP zQ + zP xQ

		t3_doubled = 2 * t3 # t3_doubled = 2 zP xQ
		t2_diff = t2_plus_t3 - t3_doubled # t2_diff = xP zQ - zP xQ
		t2_squared = t2_diff**2 # t2_squared = (xP zQ - zP xQ)**2

		t7 = t7 * t2_squared # t7 = (A zP zQ + xP zQ + zP xQ)*(xP zQ - zP xQ)**2

		t4 = t4**2 # t4 = yP**2 zQ**2
		if twistP:
			t4 = -t4 # yP <- i yP so t4 = -t4
		t5 = t5**2 # t5 = zP**2 yQ**2
		if twistQ:
			t5 = -t5 # yQ <- i yQ so t5 = -t5
		t4 = t4 + t5 # t4 = yP**2 zQ**2 + zP**2 yQ**2
		t4 = t4 * t1 # t4 = (yP**2 zQ**2 + zP**2 yQ**2) zP zQ
		u  = t4 - t7 # u = (yP**2 zQ**2 + zP**2 yQ**2) zP zQ - (A zP zQ + xP zQ + zP xQ)*(xP zQ - zP xQ)**2

		w = t2_squared * t1 # w = (xP zQ - zP xQ)**2 zP zQ

		self.u, self.v, self.w = u, v, w

		if P[2]==0:
			self.u = xQ
			self.v = F(0)
			self.w = F(1)
		elif Q[2]==0:
			self.u = xP
			self.v = F(0)
			self.w = F(1)

		# Tests
		if False:
			p = F.characteristic()
			Fp2 = GF((p, 2), name='i', modulus=var('x')**2 + 1)

			ii = Fp2.gen()

			EE = E.change_ring(Fp2)

			if twistP:
				PE = EE(-P[0],ii*P[1],P[2])
			else:
				PE = EE(P)
			if twistQ:
				QE = EE(-Q[0],ii*Q[1],Q[2])
			else:
				QE = EE(Q)

			PpQ = PE+QE
			PmQ = PE-QE

			#print(PpQ)
			#print(PmQ)

			#print(self.u/self.w)
			#print(self.v/self.w)

			assert PpQ[0]*self.w == (self.u-ii*self.v)*PpQ[2]
			assert PmQ[0]*self.w == (self.u+ii*self.v)*PmQ[2]

	def __repr__(self):
		return "Barycentric coordinates (u:v:w)=({}, {}, {})".format(self.u,self.v,self.w)

	def __eq__(self,other):
		return self.u == other.u and self.v == other.v and self.w == other.w


		
