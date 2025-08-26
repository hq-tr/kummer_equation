import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from itertools import groupby
from argparse import ArgumentParser
from lib2Bspec import read_spectrum
from itertools import product


if __name__ == "__main__":
	ap = ArgumentParser()
	ap.add_argument("--B_0", type=float, default=1.0, help="background magnetic field")
	ap.add_argument("--flux",nargs="+", type=float, default=[1.0], help="total flux within the r<R_0, can be multiple values")
	ap.add_argument("--flux_multiple", nargs="+", type=float, default=[], help="total flux multiple of flux quantum h/e. If specified, this option will override --flux.")
	ap.add_argument("--flux_shift", type=float,default=0.01)
	ap.add_argument("--R_0", nargs="+",type=float, default=[5.0], help="radius of droplet, can be multiple values")
	ap.add_argument("--E_max", type=float, default=10.0, help="maximum energy to solve for")
	ap.add_argument("--m_min", type=int, default=-6, help="minimum angular momentum quantum number m")
	ap.add_argument("--m_max", type=int, default=60, help="maximum angular momentum quantum number m")

	aa = ap.parse_args()

	if len(aa.flux_multiple)==0:
		flux_range = aa.flux
	else:
		flux_range = [x*2*np.pi + aa.flux_shift for x in aa.flux_multiple]

	for flux, R_0 in product(flux_range,aa.R_0):
		print(f"Working on flux = {flux}, R_0 = {R_0}")

		fig,ax = plt.subplots(figsize=(6,4.5))
		ax.tick_params(axis="y",direction="in", left="off",labelleft="on")
		E,m = read_spectrum(aa.B_0,flux,R_0)
		plt.plot(m[E<=aa.E_max],E[E<=aa.E_max],"k_")
		fig.tight_layout
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)
		plt.xlabel("m",fontsize=14)
		plt.ylabel("E",fontsize=14)
		plt.title(f"B_0 = {aa.B_0}, flux = {flux/(2*np.pi):.2f}h/e, R_0 = {R_0}",fontsize=16)
		plt.savefig(f"plots/spectrum_B0_{aa.B_0}_flux_{flux:.4f}_R0_{R_0:.4f}.svg")
		print("=====================================")