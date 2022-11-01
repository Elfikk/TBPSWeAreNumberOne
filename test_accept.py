import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Data/acceptance_mc.csv")
df1 = pd.read_csv("Data/total_dataset.csv")

costhetal_accept = df['costhetal']
costhetak_accept = df['costhetak']
phi_accept = df['phi']
q2_accept = df['q2']

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