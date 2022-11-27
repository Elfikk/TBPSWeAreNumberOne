import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

#read in your files
df_accept = pd.read_csv("Data/acceptance_mc.csv")
df_data = pd.read_csv("Data/signal.csv")

#set your bins range
q2_min = 0.1
q2_max = 0.98

#filter for specific bin range
df_accept = df_accept[df_accept['q2']<q2_max]
df_accept = df_accept[df_accept['q2']>q2_min]

df_data = df_data[df_data['q2']<q2_max]
df_data = df_data[df_data['q2']<q2_max]

#get out the angles
costhetal_accept = df_accept['costhetal']
costhetak_accept = df_accept['costhetak']
phi_accept = df_accept['phi']
q2_accept = df_accept['q2']

costhetal_data = df_data['costhetal']
costhetak_data = df_data['costhetak']
phi_data = df_data['phi']
q2_data = df_data['q2']

#angular dsitributions
def thetal_dist(costhetal,Fl,Afb):
    thetal = np.arccos(costhetal)
    dist = ((3.0*Fl*(np.sin(thetal)**2))/4.0 +
              (3.0*(1-Fl)*(1.0+(np.cos(thetal)**2)))/8.0+
              (Afb*np.cos(thetal))
              )*np.sin(thetal)
    return dist

def thetak_dist(costhetak,Fl):
    thetak = np.arccos(costhetak) #find theta_k
    dist = (3.0*np.sin(thetak)*(2*Fl*(np.cos(thetak))**2 + 
                                (1-Fl)*(np.sin(thetak))**2)/4.0)
    return dist

def phi_dist(phi,Fl,At,Ai):
    phi2 = 2.0*phi
    dist = (1.0 + (1.0-Fl)*(At)**2*(np.cos(phi2))/2.0 + Ai*(np.sin(phi2)))
    return dist

#plot the acceptance.csv for a specific angle of interest, within a q2 bin of interest
y, bin_edges, patches= plt.hist(costhetak_accept,bins=10,density = True, histtype='step')

#if we just want the values and we don't want to plot hist:
y, bin_edges = np.histogram(costhetak_accept,bins=10,density = True)
x = 0.5 * (bin_edges[1:] + bin_edges[:-1]) #get a list of the midpoints of the bins

#double check that the x are in the midpoints
#plt.plot(x,y,'.')

#fit the acceptance function to a lagrange polynomial of order 5
#might want to apply other fitting methods
#can change the order of the polynomial by changing the bin number
#for N bins, polynomial is of order N-1
from scipy.interpolate import lagrange
f = lagrange(x, y) #f is our lagrange polynomial fitted to points x and y
from numpy.polynomial.polynomial import Polynomial
L_polynomial_coefficients = Polynomial(f.coef[::-1]).coef
x_new = np.arange(x[0],x[-1],0.01) #generate a range of x values with smaller steps
plt.plot(x_new,f(x_new)) #plot lagrange polynomial
plt.xlabel('$cos\\theta_{k}$') #change this if you are using other angles
plt.ylabel('normed frequency')
plt.title(f'plot of $cos\\theta_k$ for {q2_min}$<q^2<${q2_max} bin') #change this if you are using other angles
plt.show()

#double check if the acceptance function works
#divides the bin heights from acceptance.csv with the polynomial
#we expect a flat line because ftrue is flat
y_flat = y/f(x)
plt.plot(x,y_flat,'.')
plt.xlabel('$cos\\theta_{k}$') #change this if you are using other angles
plt.ylabel('$F_{true}$')
plt.title('$\\frac{F_{observed}}{Acceptance function}$')
plt.show()
    

#plot histogram of data
y_data, bin_edges_data, patches_data = plt.hist(costhetak_data,bins=100,density = True, histtype='step')

#if you don't want to plot and just want the bin heights and bin edges
#y_total, bin_edges_total, patches_total = plt.hist(costhetak_data,bins=100,density = True)
x_data = 0.5 * (bin_edges_data[1:] + bin_edges_data[:-1])
plt.plot(x_data,y_data,'.',label='signal')
scaling_factor = 1/sum((y_data/f(x_data))*(bin_edges_data[1]-bin_edges_data[0]))#to make sure the final plot is normalised
y_data_acceptance = scaling_factor*y_data/f(x_data) #apply acceptance function
plt.plot(x_data,y_data_acceptance,'--',label='signal * inverse of acceptance function')
plt.legend()

popt, pcov = curve_fit(thetak_dist, x_data, y_data_acceptance) #scipy curvefit to get angular observables
plt.plot(x_data,thetak_dist(x_data,popt[0]))
plt.title(f'{q2_min} < q2 < {q2_max}, {popt[0]}')
plt.show()
print(popt) #print angular observables
