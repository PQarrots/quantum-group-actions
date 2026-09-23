from sage.all import *
from ..utilities.discrete_log import weil_pairing_pari
from ..utilities.barycentric import AddComponents
from ..theta_structures.theta_helpers_dim2 import product_theta_point

class TuplePoint:
	def __init__(self,*args):
		if len(args)==1:
			self._points=list(args[0])
		else:
			self._points=list(args)

	def points(self):
		return self._points

	def parent_curves(self):
		return [x.curve() for x in self._points]

	def parent_curve(self,i):
		return self._points[i].curve()

	def n_points(self):
		return len(self._points)

	def is_zero(self):
		return all([self._points[i]==0 for i in range(self.n_points())])

	def __repr__(self):
		return str(self._points)

	def __getitem__(self,i):
		return self._points[i]

	def __setitem__(self,i,P):
		self._points[i]=P

	def __eq__(self,other):
		n_self=self.n_points()
		n_other=self.n_points()
		return n_self==n_other and all([self._points[i]==other._points[i] for i in range(n_self)])

	def __add__(self,other):
		n_self=self.n_points()
		n_other=self.n_points()
		
		if n_self!=n_other:
			raise ValueError("Cannot add TuplePoint of distinct lengths {} and {}.".format(n_self,n_other))

		points=[]
		for i in range(n_self):
			points.append(self._points[i]+other._points[i])
		return self.__class__(points)

	def __sub__(self,other):
		n_self=self.n_points()
		n_other=self.n_points()
		
		if n_self!=n_other:
			raise ValueError("Cannot substract TuplePoint of distinct lengths {} and {}.".format(n_self,n_other))

		points=[]
		for i in range(n_self):
			points.append(self._points[i]-other._points[i])
		return self.__class__(points)

	def __neg__(self):
		n_self=self.n_points()
		points=[]
		for i in range(n_self):
			points.append(-self._points[i])
		return self.__class__(points)

	def __mul__(self,m):
		n_self=self.n_points()
		points=[]
		for i in range(n_self):
			points.append(m*self._points[i])
		return self.__class__(points)

	def __rmul__(self,m):
		return self*m

	def double_iter(self,n):
		result=self
		for i in range(n):
			result=2*result
		return result

	def weil_pairing(self,other,n):
		n_self=self.n_points()
		n_other=self.n_points()
		
		if n_self!=n_other:
			raise ValueError("Cannot compute the Weil pairing of TuplePoint of distinct lengths {} and {}.".format(n_self,n_other))

		zeta=1
		for i in range(n_self):
			zeta*=weil_pairing_pari(self._points[i],other._points[i],n)

		return zeta

	def to_dim4_vector(self,twist):
		r"""Returns the 16-coordinates vector
		[x1x2x3x4,z1x2x3x4,x1z2x3x4,...]
		"""
		if self.n_points()!=4:
			raise ValueError("The .to_dim4_vector method only works for quadruples of points.")
		P = []
		F = self._points[0].curve().base_field()
		OF = F(0)
		OneF = F(1)
		for i in range(4):
			if self._points[i]==0:
				P.append([OneF,OF])
			else:
				P.append([self._points[i][0],self._points[i][2]])# OneF would also work

		u12 = product_theta_point(P[0],P[1])
		u34 = product_theta_point(P[2],P[3])

		if twist:
			u12[1] = -u12[1]
			u12[2] = -u12[2]
			u34[1] = -u34[1]
			u34[2] = -u34[2]

		u = []
		for i in range(16):
			u.append(u12[i%4]*u34[i//4])

		return u


def uv_dim2(uvw1,uvw2,twist):
	r"""Takes two Addcomponents (ui:vi:wi) and returns two vectors
	u and v with 4 components.
	[Dartois & Duparc, Lemma 4.x, Equation (y)].
	"""
	u1, v1, w1 = uvw1.u, uvw1.v, uvw1.w
	u2, v2, w2 = uvw2.u, uvw2.v, uvw2.w

	F = u1.parent()

	u = [u1*u2,u2*w1,u1*w2,w1*w2]
	if twist:
		u[0] -= v1*v2
	else:
		u[0] += v1*v2
	v = [u2*v1+u1*v2,v2*w1,v1*w2,F(0)]

	return u, v

class TupleAddComponents:
	def __init__(self,P,Q,E=None,twistP=False,twistQ=False):
		if (not isinstance(P,TuplePoint)) or (not isinstance(Q,TuplePoint)):
			raise ValueError("TupleAddComponents constructor only accepts TuplePoint entries.")

		n = len(P._points)
		self.twistP = twistP
		self.twistQ = twistQ
		self._components = []
		for i in range(n):
			self._components.append(AddComponents(P[i],Q[i],E,twistP,twistQ))

	def __getitem__(self,i):
		return self._components[i]

	def __repr__(self):
		return str(self._components)

	def n_points(self):
		return len(self._components)

	def __eq__(self,other):
		n_self = self.n_points()
		n_other = other.n_points()
		return n_self == n_other and all([self._components[i]==other._components[i] for i in range(n_self)])

	def to_vectors(self):
		if self.n_points()!=4:
			raise ValueError("The .to_vectors method only works for quadruple of AddComponents.")
		twist = self.twistP^self.twistQ

		u12, v12 = uv_dim2(self._components[0],self._components[1],twist)
		u34, v34 = uv_dim2(self._components[2],self._components[3],twist)

		if twist:
			u = [u12[0]*u34[0]-v12[0]*v34[0],
			u12[1]*u34[0]-v12[1]*v34[0],
			u12[2]*u34[0]-v12[2]*v34[0],
			u12[3]*u34[0],
			u12[0]*u34[1]-v12[0]*v34[1],
			u12[1]*u34[1]-v12[1]*v34[1],
			u12[2]*u34[1]-v12[2]*v34[1],
			u12[3]*u34[1],
			u12[0]*u34[2]-v12[0]*v34[2],
			u12[1]*u34[2]-v12[1]*v34[2],
			u12[2]*u34[2]-v12[2]*v34[2],
			u12[3]*u34[2],
			u12[0]*u34[3],
			u12[1]*u34[3],
			u12[2]*u34[3],
			u12[3]*u34[3]]
		else:
			u = [u12[0]*u34[0]+v12[0]*v34[0],
			u12[1]*u34[0]+v12[1]*v34[0],
			u12[2]*u34[0]+v12[2]*v34[0],
			u12[3]*u34[0],
			u12[0]*u34[1]+v12[0]*v34[1],
			u12[1]*u34[1]+v12[1]*v34[1],
			u12[2]*u34[1]+v12[2]*v34[1],
			u12[3]*u34[1],
			u12[0]*u34[2]+v12[0]*v34[2],
			u12[1]*u34[2]+v12[1]*v34[2],
			u12[2]*u34[2]+v12[2]*v34[2],
			u12[3]*u34[2],
			u12[0]*u34[3],
			u12[1]*u34[3],
			u12[2]*u34[3],
			u12[3]*u34[3]]

		F = u12[0].parent()
		v = [u12[0]*v34[0]+v12[0]*u34[0],
		u12[1]*v34[0]+v12[1]*u34[0],
		u12[2]*v34[0]+v12[2]*u34[0],
		u12[3]*v34[0],
		u12[0]*v34[1]+v12[0]*u34[1],
		u12[1]*v34[1]+v12[1]*u34[1],
		u12[2]*v34[1]+v12[2]*u34[1],
		u12[3]*v34[1],
		u12[0]*v34[2]+v12[0]*u34[2],
		u12[1]*v34[2]+v12[1]*u34[2],
		u12[2]*v34[2]+v12[2]*u34[2],
		u12[3]*v34[2],
		v12[0]*u34[3],
		v12[1]*u34[3],
		v12[2]*u34[3],
		F(0)]

		return u, v

		








		