import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from itertools import groupby
from argparse import ArgumentParser
from lib2Bspec import weighted_binary_search
from itertools import product,repeat
import time

import os
from multiprocessing import Pool

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


def get_energies(params): # params is (m,flux,l1,l2,r0)
	# NOTE: This funciton returns a string object


	m    = params[0]
	flux = params[1]
	l1   = params[2]
	l2   = params[3]
	r0   = params[4]
	
	Emin = 0.0
	Emax = aa.E_max
	delE = 0.05

	
	if not aa.quiet:
		print(f"m={m} -----------------")

	f = lambda E: wk(E,m,l1,l2,r0)
	E = weighted_binary_search(f,Emin,Emax,delE,tol=aa.tolerance,max_iter=aa.max_iteration,quiet=aa.quiet)

	if len(E)>0:
		return "\n".join(f"{En}\t{m}" for En in E)
	else:
		return ""

if __name__ == "__main__":
	ap = ArgumentParser()
	ap.add_argument("--B_0", type=float, default=1.0, help="background magnetic field")
	ap.add_argument("--flux",nargs="+", type=float, default=[1.0], help="total flux within the r<R_0, can be multiple values")
	ap.add_argument("--flux_multiple", nargs="+", type=float, default=[], help="total flux multiple of flux quantum h/e. If specified, this option will override --flux.")
	ap.add_argument("--R_0", nargs="+",type=float, default=[5.0], help="radius of droplet, can be multiple values")
	ap.add_argument("--E_max", type=float, default=10.0, help="maximum energy to solve for")
	ap.add_argument("--m_min", type=int, default=-6, help="minimum angular momentum quantum number m")
	ap.add_argument("--m_max", type=int, default=60, help="maximum angular momentum quantum number m")
	ap.add_argument("--tolerance", type=float,default=1e-14, help="tolerance for the energy eigenvalues")
	ap.add_argument("--max_iteration", type=int, default=50000, help="maximum iteration for each energy eigenvalue calculation")
	ap.add_argument("--flux_shift", type=float, default=0.01)
	ap.add_argument("--quiet", "-q",action="store_true",default=False, help="use this to suppress all printouts")
	ap.add_argument("--num_cores", "-n", type=int, default=1, help="number of cores to use")

	aa = ap.parse_args()



	if len(aa.flux_multiple)==0:
		flux_range = aa.flux
	else:
		flux_range = [x*2*np.pi + aa.flux_shift for x in aa.flux_multiple]

	num_ms = aa.m_max - aa.m_min + 1
	num_cores_used = min(aa.num_cores,num_ms,os.cpu_count())

	if (num_cores_used < aa.num_cores) and (not quiet):
		print(f"NOTICE: Reducing the number of cores from {aa.num_cores} requested to {num_cores_used}.")


	if not os.path.isdir("energies"):
		os.mkdir("energies")
	for flux, R_0 in product(flux_range,aa.R_0):
		st = time.time()
		if not aa.quiet:
			print(f"Working on flux = {flux}, R_0 = {R_0}")
		l_2 = 1/np.sqrt(aa.B_0)
		l_1 = 1/np.sqrt(abs(aa.B_0 + flux / (np.pi * R_0**2)))

		El = []
		ml = []


		with Pool(num_cores_used) as p:
			output = "\n".join(p.map(get_energies,zip(range(aa.m_min,aa.m_max+1),repeat(flux),repeat(l_1),repeat(l_2),repeat(R_0))))

		with open(f"energies/eigen_B_0_{aa.B_0}_flux_{flux:.4f}_R0_{R_0:.4f}.dat", "w+") as f:
			f.write(output)
			

		print(f"_______ DONE flux = {flux:.5f}, R_0 = {R_0}:\t{time.time()-st:.5f} seconds. ________")