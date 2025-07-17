import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hyp1f1
from scipy.special import hyperu
from itertools import product

m_list = np.array([0,1,2])
E_list = np.array([0.5,0.7,1])
shift  = 0.5

def M(m,E,x):
	a = -E + 0.5
	b = m+1
	return hyp1f1(a,b,x)*np.exp(-x**2)

def U(m,E,x):
	a = -E + 0.5
	b = m+1-shift
	return hyp1f1(a,b,x)*np.exp(-x**2)

if __name__=="__main__":
	x = np.linspace(0,10)
	npts = len(x)
	for (m,E) in product(m_list,E_list):
		Mx = list(map(M,[m]*npts, [E]*npts,x))
		Ux = list(map(U,[m]*npts, [E]*npts,x))
		line, = plt.plot(x,Mx,label=f"M(x), E={E},m={m}")
		plt.plot(x,Ux,"--",c=line.get_color(), label=f"U(x), E={E},m={m}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Confluent Hypergeometric Functions")
plt.legend()
plt.show()