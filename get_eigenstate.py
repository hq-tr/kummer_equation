import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from itertools import groupby
from argparse import ArgumentParser
from lib2Bspec import read_spectrum,eigenstate
from itertools import product

ap = ArgumentParser()
ap.add_argument("--B_0", type=float, default=1.0, help="Background magnetic field")
ap.add_argument("--flux", type=float, help="Added flux to the droplet")
ap.add_argument("--flux_multiplier", type=float, default=2*np.pi, help="Multiplier to the flux value")
ap.add_argument("--flux_shift", type=float,default=0.01, help="(Small) shift to the flux value")
ap.add_argument("--R_0", type=float, help="Radius of the domain wall")
ap.add_argument("--n", "-n", type=int, default=0, help="\"Landau level\" index to plot the wavefunction of")
ap.add_argument("--m", "-m", type=int, nargs ="+", help="L_z quantum number(s). Can input mulitple values.")
ap.add_argument("--r_max", type=float, default=10, help="Maximum r to plot")
ap.add_argument("--save_figure",action="store_true", help="Use this to save the figure as .svg file in plots/")

aa = ap.parse_args()

flux = aa.flux*aa.flux_multiplier + aa.flux_shift

B_1 = aa.B_0 + flux / (np.pi * aa.R_0**2)
l2 = 1/np.sqrt(aa.B_0)
l1 = 1/np.sqrt(abs(B_1))
r0 = aa.R_0


energies = []

if __name__ == "__main__":
	if not os.path.isdir("plots"):
		os.mkdir("plots")

	fig,ax = plt.subplots(figsize=(8,4.5))
	ax.tick_params(axis="y",direction="in", left="off",labelleft="on")
	ax.tick_params(axis="x",direction="in", left="off",labelleft="on")
	E_all,m_all = read_spectrum(aa.B_0, flux, r0)
	for m in aa.m:
		print(f"--- m = {m}")
		En = [E_all[i] for i in range(len(m_all)) if m_all[i]==m]
		energies.append(En)

		try:
			if m >= 0:
				E  = En[aa.n] # pick the n-th level energy
			else:
				E  = En[aa.n+m]
		except Exception as error_message:
			print(f"No {aa.n}-level energy at m = {m}.")
			continue

		wf,rc = eigenstate(E,aa.R_0,aa.B_0,B_1,m)
		print("** Critical points = ")
		print(rc)
		r_list = np.linspace(0,aa.r_max,200)
		#p_list = np.array([wf(r) for r in r_list])
		p_list = wf(r_list)

		plt.plot(r_list,p_list,label=f"m = {m}, E = {E:.5f}")

	plt.xlabel("r",fontsize=14)
	plt.ylabel(r"$\psi(r,\phi=0)$",fontsize=14)
	plt.legend()
	#plt.ylim([-1e-12,1e-10])
	plt.title(f"n={aa.n} level wavefunctions, $B_0$={aa.B_0}, $\Phi$={aa.flux}, $R_0$={aa.R_0}",fontsize=14)
	if os.environ.get('DISPLAY', '') == '':
		plt.savefig(f"plots/wf_B0_{aa.B_0}_flux_{flux:.4f}_R0_{aa.R_0:.4f}_n_{aa.n}.svg")
	else:
		if aa.save_figure:
			plt.savefig(f"plots/wf_B0_{aa.B_0}_flux_{flux:.4f}_R0_{aa.R_0:.4f}_n_{aa.n}.svg")
		plt.show()