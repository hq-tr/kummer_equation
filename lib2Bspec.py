#!/usr/bin/env python
# coding: utf-8


#========================================================================================================
#                  ==================================================================
#                          Discontinuous Magnetic field single-particle spectrum
#                                             Mar 21  2024
#                                              Wang Yuzhu
#                  ==================================================================
#========================================================================================================
## ...Plot of tuning l1/l2/ adding potential well to be added
## v1.0: Plot the spectrum witn given (l1, l2), tuning m and r0 without potential well
##### Known bug: 1. the fsolve function may fail to find roots at some specific points and return the guesses,  
#####            which generates the flat stripes in the output figure
#####            2. Somehow it seems slower when run on the workstation than on my laptop... XD

import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from itertools import groupby
import time
from functools import partial

# The continuity condition is described by the Wronskian.
def wk(E, m, B1, B2, r0):
    l1 = 1/np.sqrt(abs(B1))
    l2 = 1/np.sqrt(abs(B2))

    x1 = r0**2/(2*l1**2)
    x2 = r0**2/(2*l2**2)
    
    a1 = -E/B1 - m/2 + abs(m/2) + 1/2
    b1 = 1 + abs(m)
    wf1  = sc.hyp1f1(a1, b1, x1) #* x1**(abs(m)/2)
    #dwf1 = ((-x1**(abs(m)/2)+abs(m)*x1**(abs(m)/2-1))*sc.hyp1f1(a1, b1, x1) + x1**(abs(m)/2)*(a1/b1)*sc.hyp1f1(a1+1, b1+1, x1)) * (r0/l1**2)
    dwf1 = ((1/2)*(-1+abs(m)/x1)*sc.hyp1f1(a1, b1, x1) +(1/b1)*a1*sc.hyp1f1(a1+1, b1+1, x1)) / l1**2
    
    #mu = m/2 - ((r0**2)/4)*((1/l1**2) - (1/l2**2))
    mu = m/2 - (B1-B2)*r0**2/4
    a2 = -E/B2 - mu + abs(mu) + 1/2
    b2 = 1 + 2*abs(mu)
    wf2  = sc.hyperu(a2, b2, x2) #* x2**(abs(mu))
    #dwf2 = ((-x2**(abs(mu))+2*abs(mu)*x2**(abs(mu))-1)*sc.hyperu(a2, b2, x2) - x2**(abs(mu))*a2*sc.hyperu(a2+1, b2+1, x2)) * (r0/l2**2)
    dwf2 = ((1/2)*(-1+2*abs(mu)/x2)*sc.hyperu(a2, b2, x2)- a2*sc.hyperu(a2+1, b2+1, x2)) / l2**2
    return wf1*dwf2 - wf2*dwf1

# The continuity condition is described by the Wronskian.
# Below is the old version that uses l1 and l2 instead of B1 and B2.
# This loses the information about the sign of B1 and B2
def wk2(E, m, l1, l2, r0):
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

# To flatten the list
def flatten(xss):
    return [x for xs in xss for x in xs]

# To group input arrays that are similar together
def nearby_groups(arr, tol_digits=6):
    for (_, grp) in groupby(arr, lambda x: round(x, tol_digits)):
        yield sorted(grp, key=lambda x: abs(round(x) - x))[0] # yield value from group that is closest to an integer


# The continuity condition is described by the Wronskian.
# This is the wronskian in the original code, which is missing the factor of 1/l^2 in the derivative.
# This is left here for legacy purpose.
def wronskian(E, m, l1, l2, r0):
    x1 = r0**2/(2*l1**2)
    x2 = r0**2/(2*l2**2)
    
    a1 = -E*(l1**2) - m/2 + abs(m/2) + 1/2
    b1 = 1 + abs(m)
    #psi1 = x1**(abs(m)/2)*np.exp(-x1**2/2)
    wf1  = sc.hyp1f1(a1, b1, x1)
    dwf1 = (1/2)*(-1+abs(m)/x1)*wf1 +(1/b1)*a1*sc.hyp1f1(a1+1, b1+1, x1)
    
    mu = m/2 - ((r0**2)/4)*((1/l1**2) - (1/l2**2))
    a2 = -E*(l2**2) - mu + abs(mu) + 1/2
    b2 = 1 + 2*abs(mu)
    #psi2 = x2**(abs(mu)/2)*np.exp(-x2**2/2)
    wf2  = sc.hyperu(a2, b2, x2)
    dwf2 = (1/2)*(-1+2*abs(mu)/x2)*wf2- a2*sc.hyperu(a2+1, b2+1, x2)
    
    return (wf1*dwf2 - wf2*dwf1)#*psi1*psi2

# Define this for the root finding command
def func(E):
    return wronskian(E, m, l1, l2, r0)

# Generate the x axis data for the plot
def plot_xpoints(solution, start, interval):
    counter=0
    list_output = []
    for x in solution[0]:
        dim_temp = len(x)
        num_temp = start+counter
            
        for i in range(dim_temp):
            list_output.append(num_temp)
            
        counter+=interval
    return list_output

# Generate the x axis data for different m (to be optimized because this is obviously inefficient)
def plot_xpoints_copies(solution, start, interval):
    counter=0
    list_output = []
    sol_dim = len(solution)
    for s in range(sol_dim):
        list_output.append([])
        for x in solution[s]:
            dim_temp = len(x)
            num_temp = start+counter
            
            for i in range(dim_temp):
                list_output[s].append(num_temp)
            
            counter+=interval
        counter = 0
    return list_output


####################################### MAIN PROGRAM ########################################
if __name__ == "__main__":
    # Change the magnetic length here (~1/sqrt(B))
    l1 = 0.5
    l2 = 1.0
    
    # Change the m values to plot here
    m_start = 0 
    m_end = 2

    # Generate the spectrum
    solution = []

    for m in range(m_start,m_end+1):
        solution.append([])
        for r0 in np.arange(0.009,10.09,0.05):
            ftemp =fsolve(func, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5],maxfev = 5000)
            root_temp1 = list(np.sort(ftemp))
            root_temp2 = [item for item in root_temp1 if item >= 0] # Only take the positive values
            root_temp  = list(nearby_groups(root_temp2))[0:5] # Tune how many roots we want here
            solution[m-m_start].append(root_temp)
    
    # Plot the spectrum
    xx = plot_xpoints_copies(solution, 0.1, 0.1)
    yy = solution
    fig, ax = plt.subplots()

    for mm in range(m_start,m_end+1):
        temp = mm-m_start
        ax.plot(xx[temp], flatten(yy[temp]), '.', label = 'm='+str(mm))

    ax.set(xlabel=r'$r_0$', ylabel=r'$E/(\hbar^2/m_e^*)$', title=r'$l_1$ = '+ str(l1) +', '+ r'$l_2 = $'+str(l2))
    ax.legend(loc='upper right', shadow=True, fontsize=18)
    ax.grid()

    plt.xlim([0, 10.1])
    plt.ylim([0, 10.5])

    output_name = 'spec_' + 'l1'+ '_'+ str(l1) + '_' + 'l2'+ '_' + str(l2) + '_' + 'm'+ '_' +str(m_start) +'-' + str(m_end) + ".png"

    fig.savefig(output_name)

    exit()

################################## IMPORTABLE FUNCTIONS ############################################3
# Read spectrum from data file
def read_spectrum(B0,flux,R0):
    try:
        with open(f"energies/eigen_B_0_{B0}_flux_{flux:.4f}_R0_{R0:.4f}.dat") as f:
            data = [list(map(float,x.split())) for x in f.readlines()]
            E = np.array([x[0] for x in data])
            m = np.array([int(x[1]) for x in data])
    except FileNotFoundError:
        print(f"Spectrum data not found for B0={B0}, flux={flux:.4f}, R0={R0:.4f}")
        E = []
        m = []

    return E,m

# Calculate the eigenstate given an energy E (already solved from the Wronskian)
# aliases for confluent hypergeom functions
hypM = sc.hyp1f1
hypU = sc.hyperu

# Define the function piece wise (L: r < R_0 , R: r > R_0)
def psiL(a1,b1,l1,m,r):
    x1 = r**2/(2*l1**2)
    if a1 >=0:
        return hypM(a1,b1,x1) * x1**(abs(m)/2) * np.exp(-x1/2)
    else:
        if abs(a1-round(a1)) > 1e-6:
            return hypM(a1,b1,x1) * x1**(abs(m)/2) * np.exp(-x1/2)
        else:
            #print("NOTE: Replacing the confluence hypergeometric M function with an associate Laguerre.")
            n = -round(a1)
            return sc.assoc_laguerre(x1,n,b1-1) * x1**(abs(m)/2) * np.exp(-x1/2)
    

def psiR(a2,b2,l2,mu,r):
    x2 = r**2/(2*l2**2)
    if a2 >=0 or abs(a2-round(a2)) > 1e-6:
        return hypU(a2,b2,x2) * x2**(abs(mu)) * np.exp(-x2/2)
    else:
        #print("NOTE: Replacing the confluence hypergeometric U function with an associate Laguerre.")
        n = -round(a2)
        return (-1)**n*sc.assoc_laguerre(x2,n,b2-1) * x2**(abs(mu)) * np.exp(-x2/2)

# Define the full function with a given ratio. 
# Ratio is multiplied to the "R" component to ensure the function is continuous.
def psi(r,r0,a1,b1,a2,b2,l1,l2,m,mu,ratio):
    return np.piecewise(r,[r<0, np.logical_and(r>=0,r<r0), r>=r0], [psiL(a1,b1,l1,m,0),partial(psiL,a1,b1,l1,m),lambda x: psiR(a2,b2,l2,mu,x) * ratio])
    # if r < 0: 
    #     return psiL(a1,b1,l1,m,0)
    # elif r < r0:
    #     return psiL(a1,b1,l1,m,r)
    # else:
    #     return psiR(a2,b2,l2,mu,r) * ratio


# Generate a wavefunction (in r) given an energy E
def eigenstate(E,r0,B0,B1,m):
    # Calculate the parameters:
    l2 = 1/np.sqrt(abs(B0))
    l1 = 1/np.sqrt(abs(B1))

    a1 = -E/B1 - m/2 + abs(m/2) + 1/2
    b1 = 1 + abs(m)

    mu = m/2 - (B1-B0)*r0**2/4
    #mu = m/2 - ((r0**2)/4)*((1/l1**2) - (1/l2**2))
    print(f"m = {m}\t2 mu = {2*mu}")
    a2 = -E*(l2**2) - mu + abs(mu) + 1/2
    b2 = 1 + 2*abs(mu)

    print(f"a1 = {a1}\ta2 = {a2}")
    # Calculate the ratio at r0:
    ddr = 0.0000000001 
    # There are two ratios: ratio between the values and ratio between the first derivatives. 
    # Ideally, these two ratio must be the same. We calculate both for sanity check.
    ratio = psiL(a1,b1,l1,m,r0) / psiR(a2,b2,l2,mu,r0)
    ratio2 = (psiL(a1,b1,l1,m,r0) - psiL(a1,b1,l1,m,r0-ddr)) / (psiR(a2,b2,l2,mu,r0+ddr)- psiR(a2,b2,l2,mu,r0))
    if abs(ratio-ratio2)>0.01:
        print(f"WARNING: (m={m}) the function may not be both continuous and differentiable at r={r0}")
        print(f"Check ratio:\t{ratio:.6f}\t{ratio2:.6f}")

    # normalize the wavefunction using rectangle rule
    r_list = np.concatenate((np.linspace(0,r0+0.5,1000,endpoint=False), np.linspace(r0+0.5,100,10000)))
    #r_list = np.linspace(0,100,20000)
    #p_list = np.array([psi(r,r0,a1,b1,a2,b2,l1,l2,m,mu,ratio) for r in r_list])
    p_list = psi(r_list,r0,a1,b1,a2,b2,l1,l2,m,mu,ratio)
    dr     = r_list[1:] - r_list[:-1]

    #p_list = np.array([psiL(r,a1,b1) for r in r_list])
    #dr     = r_list[1] - r_list[0]
    norm   = 2*np.pi*np.sum(0.5*(abs(p_list[:-1])**2 * r_list[:-1] + abs(p_list[1:])**2 * r_list[1:]) * dr)
    
    wf = lambda r: (1/np.sqrt(norm)) * psi(r,r0,a1,b1,a2,b2,l1,l2,m,mu,ratio)

    # Find the value rc at which the wavefunction attains its critical points (maximum or minimum)
    # Numerical derivative:
    dwfu = lambda r: (wf(r+ddr/2) - wf(r-ddr/2)) / ddr
    # Use weighted binary search to find zeros of the derivative:
    if abs(dwfu(0))<1e-8:
        # If there is a critical point at r=0, the binary search will miss it,
        # So, it is added manually in that case
        rc_search = [0.] + weighted_binary_search(dwfu,0.01,100,1,tol=1e-8,strict=True,quiet=True)
    else:
        rc_search = weighted_binary_search(dwfu,0,100,2,tol=1e-5,strict=True,quiet=True)
    # Sometimes the exponentially decaying tail results in a false positive (gradient is close to zero)
    # In that case, make sure the wavefunction isn't zero at that point also
    rc = [x for x in rc_search if wf(x)>1e-3]
    return wf, rc



## Search algorithm to solve for zeros of a function
def weighted_binary_search_interval(f,endpoints, tol=1e-14, max_iter=50000,quiet=False):
    st = time.time()
    fmin = f(endpoints[0])
    fmax = f(endpoints[1])
    if np.sign(fmin) * np.sign(fmax) > 0:
        if not quiet:
            print("WARNING: function evaluated at endpoints have the same sign! Terminating.")
            print(f"{time.time()-st} seconds")
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
                if not quiet:
                    print(f"Convergence after {i} iterations. Confidence range = {currentmax - currentmin}")
                    print(f"{time.time()-st} seconds")
                return 0.5*(currentmax+currentmin)
            elif np.sign(fmin) * np.sign(fret) <= 0:
                currentmax = ret
                fmax = fret
            else:
                currentmin = ret
                fmin = fret
            if i == max_iter-1:
                if not quiet:
                    print(f"WARNING: Max iteration number reached. Confidence range = {currentmax - currentmin}")
                    print(f"{time.time()-st} seconds")
                return 0.5*(currentmax+currentmin)

def weighted_binary_search(f,xmin,xmax,resolution,tol=1e-14,max_iter=50000,strict=False,quiet=False):
    test_array = np.arange(xmin,xmax,resolution)
    test_signs = np.sign(np.array(list(map(f, test_array))))
    if strict:
        intervals  = [[test_array[i], test_array[i+1]] for i in range(len(test_array)-1) if test_signs[i]*test_signs[i+1] < 0]
    else:
        intervals  = [[test_array[i], test_array[i+1]] for i in range(len(test_array)-1) if test_signs[i]*test_signs[i+1] <= 0]
    return [weighted_binary_search_interval(f,interval,tol,max_iter,quiet=quiet) for interval in intervals] 