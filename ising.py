import numpy as np
from itertools import combinations

class ising:
	def __init__(self, netsize):	#Create ising model
	
		self.size=netsize
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.randomize_state()
	
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1
	
	def pdf(self):	#Get probability density function of ising model with parameters h, J
	
		self.P=np.zeros(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			self.P[n]=np.exp((np.dot(s,self.h) + np.dot(np.dot(s,self.J),s))/float(self.size))
		self.P/=np.sum(self.P)

	def random_wiring(self):	#Set random values for h and J
		self.h=np.random.randn(self.size)
		self.J=np.zeros((self.size,self.size))
		for i in np.arange(self.size):
			for j in np.arange(i+1,self.size):
				self.J[i,j]=np.random.randn(1)

	def independent_model(self, m):		#Set h to match an independen models with means m
		self.h=np.zeros((self.size))
		for i in range(self.size):
			P1=0.5*(1+m[i])
			P0=0.5*(1-m[i])
			self.h[i]=np.log(P1/P0)
		self.J=np.zeros((self.size,self.size))
		
	def observables(self):		#Get mean and correlations from probability density function
		self.pdf()
		self.m=np.zeros((self.size))
		self.C=np.zeros((self.size,self.size))
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			for i in range(self.size):
				self.m[i]+=self.P[n]*s[i]
				for j in np.arange(i+1,self.size):
					self.C[i,j]+=self.P[n]*s[i]*s[j]
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
				
	def inverse_exact(self,m1,C1,error):	#Solve exact inverse ising problem with gradient descent
		u=0.1
		count=0
		self.independent_model(m1)
		self.observables()
		fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C[np.abs(self.C)>0]-C1[np.abs(self.C)>0])))
		fmin=fit
		
		while fit>error:
			self.observables()

			dh=u*(m1-self.m)
			self.h+=dh
			dJ=u*(C1-self.C)

			self.J+=dJ
			fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C[np.abs(self.C)>0]-C1[np.abs(self.C)>0])))
			
			if count%10==0:
				print self.size,count,fit


			count+=1
		return fit
				
	def inverseMC(self,m1,C1,error):	#Solve inverse ising problem using Monte Carlo Samples

		u=0.1
		samples=200
		nT=40

		fit=1E10
		fmin=fit
		fitcount=0

#		Ps = self.MCsamples(samples)
		PS=[]
		fits=[]
		for i in range(nT):
			PS+=[self.MCsamples(samples)]
			fits+=[fit]

		count=0
		while fit>error:
			del PS[0]
			PS+=[self.MCsamples(samples)]
			Ps=PS[0]
			for i in np.arange(1,nT):
				Ps.update(PS[i])

			ns=Ps.keys()
			Pmc=Ps.values()
			Pmc/=np.sum(Pmc)
			self.observablesMC(ns,Pmc)
	
			dh=u*(m1-self.m)
			self.h+=dh
			dJ=u*(C1-self.C)
			self.J+=dJ
			fmin=np.min(fits)
			del fits[0]
			fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C[np.abs(self.C)>0]-C1[np.abs(self.C)>0])))
			fits+=[fit]
			if fit/fmin<1:
	#			fmin=fit
				fitcount=0
			else:
				fitcount+=1
				if fitcount>nT*2:
					if len(Ps)/2.0**self.size<1:
						samples+=samples/2
					fitcount=0
			if count%10==0:
				print self.size,count,len(Ps)/2.0**self.size,samples,fit
			count+=1
	
		return fit
	
				
	def MCsamples(self,samples):	#Generate a series of Monte Carlo samples
		self.randomize_state()
		# Main simulation loop:
		P={}
		for t in range(samples):
			self.MetropolisStep()
			n=bool2int((self.s+1)/2)
			if n<0:
				print n
				print ((self.s+1)/2)
			P[n]=np.exp((np.dot(self.s,self.h) + np.dot(np.dot(self.s,self.J),self.s))/float(self.size))
		return P
		
	def observablesMC(self, ns,P):	#Get mean and correlations from Monte Carlo samples

		self.m=np.zeros((self.size))
		self.C=np.zeros((self.size,self.size))
		for ind,n in enumerate(ns):
			s=bitfield(n,self.size)*2-1
			for i in range(self.size):
				self.m[i]+=P[ind]*s[i]
				for j in np.arange(i+1,self.size):
					self.C[i,j]+=P[ind]*s[i]*s[j]
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
				
		
				

			
	def MetropolisStep(self,i=None):	    #Execute step of Metropolis algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0 or np.random.rand(1) < np.exp(-eDiff):    # Metropolis
			self.s[i] = -self.s[i]
			
	def MetropolisStepT0(self,i=None):	    #Execute step of Metropolis algorithm with zero temperature (deterministic)
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0:
			self.s[i] = -self.s[i]
			
	def GlauberStep(self,i=None):			#Execute step of Glauber algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if np.random.rand(1) < 1.0/(1.0+np.exp(eDiff)):    # Glauber
			self.s[i] = -self.s[i]

	def deltaE(self,i):		#Compute energy difference between two states with a flip of spin i
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
 
			
				
	def metastable_states(self):	#Find the metastable states of the system
		self.pdf()
		ms=[]
		Pms=[]
		for n in range(2**self.size):
			m=1
			s=bitfield(n,self.size)
			for i in range(self.size):
				s1=s.copy()
				s1[i]=1-s1[i]
				n1=bool2int(s1)
				if self.P[n]<self.P[n1]:
					m=0
					break
			if m==1:
				ms+=[n]
				Pms+=[self.P[n]]
		return ms,Pms
		
	def get_valley(self,s):		#Find an attractor "valley" starting from state s
		ms,Pms=self.metastable_states()
		n=bool2int((s+1)/2)
		self.s=s.copy()
		while not n in ms:	
			self.MetropolisStepT0()
			n=bool2int((self.s+1)/2)
		ind=ms.index(n)
		valley=ind
		print ind,n,ms
		return valley
		
	def Energy(self):	#Compute energy function
		self.E=np.zeros(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,size)*2-1
			E[n]=(np.dot(s,self.h) + np.dot(np.dot(s,self.J),s))
		return E
		
	def MeanEnergy(self):	#Get mean energy of the model
		return np.sum(self.P*self.E)

	
def bool2int(x):				#Transform bool array into positive integer
	y = 0L
	for i,j in enumerate(np.array(x)[::-1]):
#        y += j<<i
		y += long(j*2**i)
	return y
    
def bitfield(n,size):			#Transform positive integer into bit array
	x = [int(x) for x in bin(n)[2:]]
	x = [0]*(size-len(x)) + x
	return np.array(x)


def subPDF(P,rng):
	subsize=len(rng)
	Ps=np.zeros(2**subsize)
	size=int(np.log2(len(P)))
	for n in range(len(P)):
		s=bitfield(n,size)
		Ps[bool2int(s[rng])]+=P[n]
	return Ps
	
def Entropy(P):
	E=0.0
	for n in range(len(P)):
		if P[n]>0:
			E+=-P[n]*np.log2(P[n])
	return E
	

def MI(Pxy, rngx, rngy):
	size=int(np.log2(len(Pxy)))
	Px=subPDF(Pxy,rngx)
	Py=subPDF(Pxy,rngy)
	I=0.0
	for n in range(len(Pxy)):
		s=bitfield(n,size)
		if Pxy[n]>0:
			I+=Pxy[n]*np.log(Pxy[n]/(Px[bool2int(s[rngx])]*Py[bool2int(s[rngy])]))
	return I
	
def TSE(P):
	size=int(np.log2(len(P)))
	C=0
	for npart in np.arange(1,0.5+size/2.0).astype(int):	
		bipartitions = list(combinations(range(size),npart))
		for bp in bipartitions:
			bp1=list(bp)
			bp2=list(set(range(size)) - set(bp))
			C+=MI(P, bp1, bp2)/float(len(bipartitions))
	return C
	
def KL(P,Q):
	D=0
	for i in range(len(P)):
		D+=P[i]*np.log(P[i]/Q[i])
	return D
    
def JSD(P,Q):
	return 0.5*(KL(P,Q)+KL(Q,P))

	
def PCA(h,J):
	size=len(h)
	P=get_PDF(h,J,size)
	m,C=observables(P,size)
	C=0.5*(C+np.transpose(C))
	w,v = np.linalg.eig(C)
	return w,v
	

class kinetic_ising:
	def __init__(self, netsize):	#Create ising model
	
		self.size=netsize
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.randomize_state()
	
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1
	

	def random_wiring(self):	#Set random values for h and J
		self.h=np.random.randn(self.size)
		self.J=np.random.randn(self.size,self.size)
				
	
	def GlauberStep(self):
		s=self.s.copy()
		for i in range(self.size):
			eDiff = self.deltaE(i,s)
			if np.random.rand(1) < 1.0/(1.0+np.exp(eDiff)):    # Glauber!
				self.s[i] = -self.s[i]
		
	def deltaE(self,i,s):
		return 2*(s[i]*self.h[i] + np.sum(s[i]*(self.J[:,i]*s)))
		
	def observablesMC(self,T):	#Get mean and correlations from Monte Carlo simulation of the kinetic ising model
		self.m=np.zeros(self.size)
		self.C=np.zeros((self.size,self.size))
		self.D=np.zeros((self.size,self.size))
	
		self.randomize_state()
		for t in range(T):
			self.GlauberStep()
			self.m+=self.s/float(T)
			for i in range(self.size):
				for j in np.arange(i+1,self.size):
					self.C[i,j]+=self.s[i]*self.s[j]/float(T)
#			print
#			print self.s
			for i in range(self.size):
				eDiff = self.deltaE(i,self.s)
				pflip = 1.0/(1.0+np.exp(eDiff))
				self.D[:,i]+=(self.s[i]*self.s*(1-pflip) - self.s[i]*self.s*pflip)/float(T)
#				print (self.s[i]*self.s*(1-pflip) - self.s[i]*self.s*pflip)
				
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
		for i in range(self.size):
			for j in range(self.size):
				self.D[i,j]-=self.m[i]*self.m[j]
			
	def observables(self):	#Get mean and correlations from Monte Carlo simulation of the kinetic ising model
		self.m=np.zeros(self.size)
		self.C=np.zeros((self.size,self.size))
		self.D=np.zeros((self.size,self.size))
	
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			self.m+=s*self.P[n]
			for i in range(self.size):
				for j in np.arange(i+1,self.size):
					self.C[i,j]+=s[i]*s[j]*self.P[n]
			for i in range(self.size):
				eDiff = self.deltaE(i,s)
				pflip = 1.0/(1.0+np.exp(eDiff))
				self.D[:,i]+=(s[i]*s*(1-pflip) - s[i]*s*pflip)*self.P[n]

		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
		for i in range(self.size):
			for j in range(self.size):
				self.D[i,j]-=self.m[i]*self.m[j]
				
		
				
	def pdf(self,T):	#Get mean and correlations from Monte Carlo simulation of the kinetic ising model
		self.P=np.zeros(2**self.size)
	
		self.randomize_state()
		for t in range(T):
			self.GlauberStep()
			n=bool2int((self.s+1)/2)
			self.P[n]+=1.0/float(T)
#		self.P[n]/=np.sum(self.P[n])


	def independent_model(self, m):		#Set h to match an independen models with means m
		self.h=np.zeros((self.size))
		for i in range(self.size):
			P1=0.5*(1+m[i])
			P0=0.5*(1-m[i])
			self.h[i]=np.log(P1/P0)
		self.J=np.zeros((self.size,self.size))
		
	def inverse(self,m1,D1, error):
		u=0.4
		count=0
		self.independent_model(m1)
		self.observables()
#		print self.m
#		print m1
#		print self.C
#		print D1
#		exit(0)
		fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.D-D1)))
		fmin=fit
		
		while fit>error:
			Dp=self.D.copy()
			self.observables()

			dh=u*(m1-self.m)
			self.h+=dh
			dJ=u*(D1-self.D)
			self.J+=dJ
			fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.D-D1)))
			
			if count%10==0:
				print self.size,count,fit
##				print (m1 -self.m)
#				print D1
#				print self.D
#				print self.h
#				print self.J


			count+=1
		return fit
