#importing library's
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spc
# function for gaussian 
def gaussianfit(data,title):
    bin_height, bin_edges, patches = plt.hist(data, bins=1000,alpha=0)
    midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    plt.plot(midpoints,bin_height,'-',color='royalblue',label="q21")
    plt.xlim([8,16])
    
    length =len(midpoints)
    o = list(bin_height).index(max(list(bin_height)))
    
    furty = midpoints[o]        
    
    def f(x,a,b,c):
        return a*np.exp(-(x-b)**2/(2*c**2))
    
    x=np.arange(0,100,0.01)
    initialguess = [max(bin_height),furty,0.1]
    popt , pcov = spc.curve_fit(f,midpoints,bin_height,initialguess)
    
    plt.plot(x,f(x,popt[0],popt[1],popt[2]),initialguess)
    plt.xlim([8,16])
    plt.xlabel('qsquared')
    plt.ylabel('frequency')
    plt.title(title)
    plt.show()
    
    print('mean=',popt[1])
    print('sigma=',popt[2])
    
    print('Lower Bound qsquare value',popt[1]-3*popt[2])
    print('Upper Bound qsquare value',popt[1]+3*popt[2])

if __name__ == "__main__":

    #Need these files in current working directory!
    jpsi_path = "jpsi.csv"
    psi2s_path = "psi2s.csv"
    phimumu_path = "phimumu.csv"

    jpsi=pd.read_csv(jpsi_path,index_col=0)
    psi2s=pd.read_csv(psi2s_path,index_col=0)
    phimumu = pd.read_csv(phimumu_path,index_col=0)
    data = [jpsi, psi2s,phimumu]

    q21 = data[0]['q2']
    q22 = data[1]['q2']
    q23 = data [2]['q2']

    gaussianfit(q21,'Qsquare for Jpsi')
    gaussianfit(q22,'Qsquare for PSi2s')
