import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from itertools import groupby
from argparse import ArgumentParser
from lib2Bspec import wronskian as wk2
from itertools import product

def wk(E, m, l1, l2, r0):
    x1 = r0**2/(2*l1**2)
    x2 = r0**2/(2*l2**2)
    
    a1 = -E*(l1**2) - m/2 + abs(m/2) + 1/2
    b1 = 1 + abs(m)
    wf1  = sc.hyp1f1(a1, b1, x1) #* x1**(abs(m)/2)
    #dwf1 = ((-x1**(abs(m)/2)+abs(m)*x1**(abs(m)/2-1))*sc.hyp1f1(a1, b1, x1) + x1**(abs(m)/2)*(a1/b1)*sc.hyp1f1(a1+1, b1+1, x1)) * (r0/l1**2)
    dwf1 = ((1/2)*(-1+abs(m)/x1)*sc.hyp1f1(a1, b1, x1) +(1/b1)*a1*sc.hyp1f1(a1+1, b1+1, x1)) / l1**2
    
    mu = m/2 - ((r0**2)/4)*((1/l1**2) - (1/l2**2))
    a2 = -E*(l2**2) - mu + abs(mu) + 1/2
    b2 = 1 + 2*abs(mu)
    wf2  = sc.hyperu(a2, b2, x2) #* x2**(abs(mu))
    #dwf2 = ((-x2**(abs(mu))+2*abs(mu)*x2**(abs(mu))-1)*sc.hyperu(a2, b2, x2) - x2**(abs(mu))*a2*sc.hyperu(a2+1, b2+1, x2)) * (r0/l2**2)
    dwf2 = ((1/2)*(-1+2*abs(mu)/x2)*sc.hyperu(a2, b2, x2)- a2*sc.hyperu(a2+1, b2+1, x2)) / l2**2
    return wf1*dwf2 - wf2*dwf1

def weighted_binary_search_interval(f,endpoints, tol=1e-20, max_iter=100000):
	fmin = f(endpoints[0])
	fmax = f(endpoints[1])
	if np.sign(fmin) * np.sign(fmax) > 0:
		print("WARNING: function evaluated at endpoints have the same sign! Terminating.")
		return float("nan")
	else:
		currentmin = endpoints[0]
		currentmax = endpoints[1]
		for i in range(max_iter):
			#ret  = (abs(fmin)*currentmin + abs(fmax)*currentmax)/(abs(fmin) + abs(fmax))
			ret = 0.5*(currentmin + currentmax)
			fret = f(ret)
			confidence = currentmax - currentmin
			if confidence < tol:
				print(f"Convergence after {i} iterations. Confidence range = {currentmax - currentmin}")
				return 0.5*(currentmax+currentmin)
			elif np.sign(fmin) * np.sign(fret) <= 0:
				currentmax = ret
				fmax = fret
			else:
				currentmin = ret
				fmin = fret
			if i == max_iter-1:
				print(f"WARNING: Max iteration number reached. Confidence range = {currentmax - currentmin}")
				return 0.5*(currentmax+currentmin)

def weighted_binary_search(f,xmin,xmax,resolution):
	test_array = np.arange(xmin,xmax,resolution)
	test_signs = np.sign(np.array(list(map(f, test_array))))
	intervals  = [[test_array[i], test_array[i+1]] for i in range(len(test_array)-1) if test_signs[i]*test_signs[i+1] <= 0]
	return [weighted_binary_search_interval(f,interval) for interval in intervals] 


if __name__ == "__main__":
	ap = ArgumentParser()
	ap.add_argument("--B_0", type=float, default=1.0, help="background magnetic field")
	ap.add_argument("--flux",nargs="+", type=float, default=[1.0], help="total flux within the r<R_0, can be multiple values")
	ap.add_argument("--flux_multiple", nargs="+", type=float, default=[], help="total flux multiple of flux quantum h/e. If specified, this option will override --flux.")
	ap.add_argument("--R_0", nargs="+",type=float, default=[5.0], help="radius of droplet, can be multiple values")
	ap.add_argument("--E_max", type=float, default=10.0, help="maximum energy to solve for")
	ap.add_argument("--m_min", type=int, default=-6, help="minimum angular momentum quantum number m")
	ap.add_argument("--m_max", type=int, default=60, help="maximum angular momentum quantum number m")

	aa = ap.parse_args()

	if len(aa.flux_multiple)==0:
		flux_range = aa.flux
	else:
		flux_range = [x*2*np.pi + 0.01 for x in aa.flux_multiple]

	for flux, R_0 in product(flux_range,aa.R_0):
		print(f"Working on flux = {flux}, R_0 = {R_0}")
		l_2 = 1/np.sqrt(aa.B_0)
		l_1 = 1/np.sqrt(abs(aa.B_0 + flux / (np.pi * R_0**2)))

		El = []
		ml = []

		Emin = 0.0
		Emax = aa.E_max
		delE = 0.05
		fig,ax = plt.subplots(figsize=(6,4.5))
		ax.tick_params(axis="y",direction="in", left="off",labelleft="on")
		with open(f"energies/eigen_B_0_{aa.B_0}_flux_{flux:.4f}_R0_{R_0:.4f}.dat", "w+") as f:
			f.write("")

		for m in range(aa.m_min,aa.m_max+1):
			print(f"m={m} -----------------")
			#f = lambda E: np.exp(-E)/R_0*wk2(E,m,l_1,l_2,R_0)
			f = lambda E: wk(E,m,l_1,l_2,R_0)
			E = weighted_binary_search(f,Emin,Emax,delE)
			#E = fsolve(f, np.linspace(0,10,30))
			#E = E[E>=0]
			plt.plot(m*np.ones(len(E)),E,"k_")
			El.append(E)
			ml.append(m)

			with open(f"energies/eigen_B_0_{aa.B_0}_flux_{flux:.4f}_R0_{R_0:.4f}.dat", "a+") as f:
				for En in E:
					if En != None:
						f.write(f"{En}\t{m}\n")
			
		fig.tight_layout
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel("m",fontsize=14)
		plt.ylabel("E",fontsize=14)
		plt.title(f"B_0 = {aa.B_0}, flux = {flux}, R_0 = {R_0}",fontsize=16)
		#plt.savefig(f"plots/B0_{aa.B_0}_flux_{flux:.4f}_R0_{R_0:.4f}_spectrum.svg")
		print("=====================================")