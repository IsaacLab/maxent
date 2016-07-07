#!/usr/bin/python

import numpy as np
from itertools import combinations

class ising:
	def __init__(self, netsize):
	
		self.size=netsize
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.s = np.random.randint(0,2,netsize)*2-1
	
	
	def pdf(self):	#Get probability density function of ising model with parameters h, J
	
		self.P=np.zeros(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			self.P[n]=np.exp((np.dot(s,self.h) + np.dot(np.dot(s,self.J),s))/float(self.size))
		self.P/=np.sum(self.P)

	def random_wiring(self):
		self.h=np.random.randn(self.size)
		self.J=np.zeros((self.size,self.size))
		for i in np.arange(self.size):
			for j in np.arange(i+1,self.size):
				self.J[i,j]=np.random.randn(1)

	def independent_model(self, m):
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
				
	def inverse_exact(self,m1,C1,error):
		u=0.1
		count=0
		self.observables()
		fit = np.mean((self.m-m1)**2)
		fit = 0.5*fit + 0.5*np.mean((self.C-C1)**2)
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
				
	def MCsamples(self,samples):
		self.s = np.random.randint(0,2,self.size)*2-1
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
		
	def observablesMC(self, ns,P):	#Get mean and correlations from Monte Carlo simulation of ising model

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
				
		
				
	def inverseMC(self,m1,C1,error):

		u=0.1
		samples=2000
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
			fit = np.sqrt( (np.sum((self.m-m1)**2)+np.sum((self.C[np.abs(self.C)>0]-C1[np.abs(self.C)>0])**2))/float(self.size+0.5*(self.size*(self.size+1))))
			fits+=[fit]
			if fit/fmin<1:
	#			fmin=fit
				fitcount=0
			else:
				fitcount+=1
				if fitcount>nT:
					if len(Ps)/2.0**self.size<1:
						samples+=samples/2
					fitcount=0
			if count%1==0:
				print self.size,count,len(Ps)/2.0**self.size,samples,fit
			count+=1
	
		return fit
	
			
	def MetropolisStep(self,i=None):
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0 or np.random.rand(1) < np.exp(-eDiff):    # Metropolis!
			self.s[i] = -self.s[i]
		
	def deltaE(self,i):
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
 
			
				
	def metastable_states(self):
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


def MetropolisStepT0(s,h,J,size,i=None):
	if i is None:
		i = np.random.randint(size)
	eDiff = deltaE(i,s,h,J)
	if eDiff <= 0:
		s[i] = -s[i]
	return s
	
def get_valley(s,ms,h,J,size):
	n=bool2int((s+1)/2)
	s1=s.copy()
	while not n in ms:	
		s1=MetropolisStepT0(s1,h,J,size)
		n=bool2int((s1+1)/2)
	ind=ms.index(n)
	valley=ind
	return valley

def GlauberStep(s,h,J,size,i=None):
	if i is None:
		i = np.random.randint(size)
	eDiff = deltaE(i,s,h,J)
	if np.random.rand(1) < 1.0/(1.0+np.exp(eDiff)):    # Glauber!
		s[i] = -s[i]
	return s
	


	
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

def Energy(h,J):

	size=len(h)
	P=get_PDF(h,J,size)
	E=0.0
	for n in range(len(P)):
		s=bitfield(n,size)*2-1
		E+=P[n]*(np.dot(s,h) + np.dot(np.dot(s,J),s))
	return E
	
def PCA(h,J):
	size=len(h)
	P=get_PDF(h,J,size)
	m,C=observables(P,size)
	C=0.5*(C+np.transpose(C))
	w,v = np.linalg.eig(C)
	return w,v
	
def get_E(h,J):	#Get probability density function of ising model with parameters h, J
	size=len(h)
	E=np.zeros(2**size)
	for n in range(2**size):
		s=bitfield(n,size)*2-1
		E[n]=(np.dot(s,h) + np.dot(np.dot(s,J),s))/float(size)
	return E
