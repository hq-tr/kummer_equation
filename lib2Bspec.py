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


# Read spectrum from data file
def read_spectrum(B0,flux,R0):
    try:
        with open(f"energies/eigen_B_0_{B0}_flux_{flux:.4f}_R0_{R0:.4f}.dat") as f:
            data = [list(map(float,x.split())) for x in f.readlines()]
            E = np.array([x[0] for x in data])
            m = np.array([x[1] for x in data])
    except FileNotFoundError:
        print(f"Spectrum data not found for B0={B0}, flux={flux}, R0={R0}")
        E = []
        m = []

    return E,m

# The continuity condition is described by the Wronskian.
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

