import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
#from itertools import groupby
from argparse import ArgumentParser
from lib2Bspec import wronskian as wk2
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

aa = ap.parse_args()

flux = aa.flux*aa.flux_multiplier + aa.flux_shift

B_1 = aa.B_0 + flux / (np.pi * aa.R_0**2)
l2 = 1/np.sqrt(aa.B_0)
l1 = 1/np.sqrt(abs(B_1))
r0 = aa.R_0

try:
	with open(f"energies/eigen_B_0_{aa.B_0}_flux_{flux:.4f}_R0_{r0:.4f}.dat") as f:
		data = [list(map(float, x.split())) for x in f.readlines()]
except FileNotFoundError:
	print("Eigenvalue file not found.")


# aliases for confluent hypergeom functions
hypM = special.hyp1f1
hypU = special.hyperu

energies = []

# Define the function piece wise (L: r < R_0 , R: r > R_0)
def psiL(r,a1,b1,l1,m):
	x1 = r**2/(2*l1**2)
	return hypM(a1,b1,x1) * x1**(abs(m)/2) * np.exp(-x1/2)

def psiR(r,a2,b2,l2,mu):
	x2 = r**2/(2*l2**2)
	return hypU(a2,b2,x2) * x2**(abs(mu)) * np.exp(-x2/2)

# Define the full function with a given ratio. 
# Ratio is multiplied to the "R" component to ensure the function is continuous.
def psi(r,a1,b1,a2,b2,l1,l2,m,mu,ratio):
	if r < aa.R_0:
		#return 10**(-m) * hypM(a1,b1,r**2/(2*aa.l_1**2)) * r**m * np.exp(-r**2 / (4*aa.l_1**2))
		return psiL(r,a1,b1,l1,m)
	else:
		return psiR(r,a2,b2,l2,mu) * ratio
		#return 10**(-m) * ratio * hypU(a2,b2,r**2/(2*aa.l_2**2)) * r**abs(mu) * np.exp(-r**2 / (4*aa.l_2**2))

if __name__ == "__main__":
	fig,ax = plt.subplots(figsize=(6,4.5))
	ax.tick_params(axis="y",direction="in", left="off",labelleft="on")
	for m in aa.m:
		print(f"--- m = {m}")
		En = [thing[0] for thing in data if thing[1]==m]
		energies.append(En)

		try:
			if m >= 0:
				E  = En[aa.n] # pick the n-th level energy
			else:
				E  = En[aa.n+m]
		except Exception as error_message:
			print(f"No {aa.n}-level energy at m = {m}.")
			continue

		a1 = -E/aa.B_0 - m/2 + abs(m/2) + 1/2
		b1 = 1 + abs(m)

		mu = m/2 - ((r0**2)/4)*((1/l1**2) - (1/l2**2))
		a2 = -E/abs(B_1) - mu + abs(mu) + 1/2
		b2 = 1 + 2*abs(mu)

		dr = 0.002

		# There are two ratios: ratio between the values and ratio between the first derivatives. 
		# Ideally, these two ratio must be the same. We calculate both for sanity check.
		ratio = psiL(aa.R_0,a1,b1,l1,m) / psiR(aa.R_0,a2,b2,l2,mu)
		ratio2 = (psiL(aa.R_0,a1,b1,l1,m) - psiL(aa.R_0-dr,a1,b1,l1,m)) / (psiR(aa.R_0+dr,a2,b2,l2,mu)- psiR(aa.R_0,a2,b2,l2,mu))
		print(f"ratios: {ratio:.6f}\t{ratio2:.6f}")

		r_list = np.arange(0,aa.r_max, dr)
		p_list = np.array([psi(r,a1,b1,a2,b2,l1,l2,m,mu,ratio) for r in r_list])

		#p_list = np.array([psiL(r,a1,b1) for r in r_list])

		# roughly normalize the wavefunction using rectangle rule
		#dr     = r_list[1] - r_list[0]
		norm   = np.sum(0.5*(abs(p_list[:-1])**2 + abs(p_list[1:])**2) * dr)
		print(f"norm={norm}")
		p_list /= np.sqrt(norm)


		plt.plot(r_list,p_list,label=f"m = {m}, E = {E:.5f}")

	plt.xlabel("r")
	plt.ylabel(r"$\psi(r,\phi=0)$")
	plt.legend()
	#plt.ylim([-1e-12,1e-10])
	plt.title(f"n={aa.n} level wavefunctions, $B_0$={aa.B_0}, $\Phi$={aa.flux}, $R_0$={aa.R_0}")
	plt.savefig(f"plots/wf_B0_{aa.B_0}_flux_{flux:.4f}_R0_{aa.R_0:.4f}_n_{aa.n}.svg")