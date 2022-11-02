import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

df1 = pd.read_csv("Data/acceptance_mc.csv")
total_df = pd.read_csv("Data/signal.csv")

q2_min = 15.0
q2_max = 17.9

df = df1[df1['q2']<q2_max]
df = df[df['q2']>q2_min]

total_df = total_df[total_df['q2']<q2_max]
total_df = total_df[total_df['q2']<q2_max]

costhetal_accept = df['costhetal']
costhetak_accept = df['costhetak']
phi_accept = df['phi']
q2_accept = df['q2']

costhetal_total = total_df['costhetal']
costhetak_total = total_df['costhetak']
phi_total = total_df['phi']
q2_total = total_df['q2']

def thetal_dist(costhetal,Fl,Afb):
    thetal = np.arccos(costhetal)
    dist = ((3.0*Fl*(np.sin(thetal)**2))/4.0 +
              (3.0*(1-Fl)*(1.0+(np.cos(thetal)**2)))/8.0+
              (Afb*np.cos(thetal))
              )*np.sin(thetal)
    return dist

def thetak_dist(costhetak,Fl):
    thetak = np.arccos(costhetak)
    dist = (3.0*np.sin(thetak)*(2*Fl*(np.cos(thetak))**2 + 
                                (1-Fl)*(np.sin(thetak))**2)/4.0)
    return dist

def phi_dist(phi,Fl,At,Ai):
    phi2 = 2.0*phi
    dist = (1.0 + (1.0-Fl)*(At)**2*(np.cos(phi2))/2.0 + Ai*(np.sin(phi2)))
    return dist

y, bin_edges, patches = plt.hist(costhetak_accept,bins=6,density = True, histtype='step')
x = 0.5 * (bin_edges[1:] + bin_edges[:-1])

#plt.plot(x,y,'.')

from scipy.interpolate import lagrange
f = lagrange(x, y)
from numpy.polynomial.polynomial import Polynomial
L_polynomial_coefficients = Polynomial(f.coef[::-1]).coef
x_new = np.arange(x[0],x[-1],0.01)
'''plt.plot(x_new,f(x_new))
plt.xlabel('$cos\\theta_{k}$')
plt.ylabel('normed frequency')
plt.title('plot of $cos\\theta_{k}$ for $1.1<q^2<2.5$ bin')
plt.show()'''

y_new = np.array(y)/np.array(f(x))
'''plt.plot(x,y_new)
plt.xlabel('$cos\\theta_{k}$')
plt.ylabel('$F_{true}$')
plt.title('$\\frac{F_{observed}}{Acceptance function}$')
plt.show()'''
    
plt.show()
y_total, bin_edges_total, patches_total = plt.hist(costhetak_total,bins=100,density = True, histtype='step')
x_total = 0.5 * (bin_edges_total[1:] + bin_edges_total[:-1])
plt.plot(x_total,y_total,'.',label='signal')
scaling_factor = 1/sum((y_total/f(x_total))*(bin_edges_total[1]-bin_edges_total[0]))
y_total_acceptance = scaling_factor*y_total/f(x_total)
plt.plot(x_total,y_total_acceptance,'--',label='signal * inverse of acceptance function')
plt.legend()

popt, pcov = curve_fit(thetak_dist, x_total, y_total_acceptance)
plt.plot(x_total,thetak_dist(x_total,popt[0]))
plt.title(f'{q2_min} < q2 < {q2_max}, {popt[0]}')
plt.show()
print(popt)
'''frequency, bins, patches = plt.hist(costhetal_accept,density=True,bins=50)
x_values = []
for i in range(len(bins)):
    if i < len(bins)-1:
        x = (bins[i]+bins[i+1])/2
        x_values.append(x)

param, param_cov = curve_fit(thetal_dist,x_values,frequency)

print(param)

plt.plot(x_values,thetal_dist(x_values,param[0],param[1]))
plt.show()'''

'''frequency, bins, patches = plt.hist(costhetak_accept,density=True,bins=50)
x_values = []
for i in range(len(bins)):
    if i < len(bins)-1:
        x = (bins[i]+bins[i+1])/2
        x_values.append(x)

param, param_cov = curve_fit(thetak_dist,x_values,frequency)
plt.plot(x_values,thetak_dist(x_values,param[0]))
plt.show()'''

'''frequency, bins, patches = plt.hist(phi_accept,density=True,bins=50)
x_values = []
for i in range(len(bins)):
    if i < len(bins)-1:
        x = (bins[i]+bins[i+1])/2
        x_values.append(x)

param, param_cov = curve_fit(phi_dist,x_values,frequency)
plt.plot(x_values,phi_dist(x_values,param[0],param[1],param[2]))
plt.show()'''