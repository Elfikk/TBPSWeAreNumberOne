# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:31:39 2022

@author: lizzi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from iminuit import Minuit
import math
from math import ceil


class Data:

    
    def __init__(self, df):
        
        """
        Initialises the dataset
        
        """
      
        self._df = pd.DataFrame(df)
        
    def angle(self, angle):
        
        """
        Returns a list of the angular data
        
        """
        
        return self._df[angle].tolist()
    
    def q2(self):
        
        """
        Returns a list of the angular data
        
        """
        
        return self._df["q2"].tolist()
        
        
    def sl(self, angle, qmin, qmax):
        
        """
        Slices the data according to the given angle and q2 values
        
        """
        
        df = self._df
        
        #filter for specific bin range
        
        df = df[df['q2']<qmax]
        df = df[df['q2']>qmin]
        
        #get out the angles
        
        if type(angle) == str:
    
             angle = df[angle]
            
        else:
        
            raise TypeError("'Angle' should be the header of the angle you are trying to investigate")
        
        return np.array(angle.tolist())
    
    def slq(self, qmin, qmax):
        
        """
        Slices the data according to the given angle and q2 values
        
        """
        
        df = self._df
        
        #filter for specific bin range
        
        df = df[df['q2']<qmax]
        df = df[df['q2']>qmin]
        
        #get out the angles
        
        angle = df["q2"]
            
        
        return np.array(angle.tolist())
    
    def get_x(self, angle, qmin, qmax):
        
        d = self.sl(angle, qmin, qmax)
        y, bin_edges = np.histogram(d,bins=6,density = True)
        x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        
        return x
    
    def get_y(self, angle, qmin, qmax):
        
        d = self.sl(angle, qmin, qmax)
        y, bin_edges = np.histogram(d,bins=6,density = True)
        
        return y
        
        
class Accept(Data):
    
    """
    Subclass of data for the acceptance dataset.
    
    """
    
    def __init__(self, df):

        Data.__init__(self, df)     
        

    def chebyshev(self, angle, qmin, qmax, order = 4, name = "Data", plot_p = False, plot_poly = False):
        
        """
        Function to calculate the Chebyshev polynomial for the accpetance data of a given bin and angle. This method contains 
        optional plotting to check the 
        """
        if angle == "q2":
            d = self.slq(qmin,qmax)
        else:
            
            d = self.sl(angle, qmin, qmax)
        
        #getting x and y values

        y, bin_edges = np.histogram(d,bins=int(1+ceil(3.322*np.log(len(d))/np.log(2))),density = True)
        x = 0.5 * (bin_edges[1:] + bin_edges[:-1]) #get a list of the midpoints of the bins
        # p = np.polynomial.legendre.Legendre.fit(x, y, order) #f is our Chebyshev polynomial fitted to points x and y of order 4
        p = np.polynomial.Chebyshev.fit(x, y, order)
        
        if plot_p == True:

            x_new = np.arange(x[0],x[-1],0.01) #generate a range of x values with smaller steps
            plt.plot(x_new,p(x_new)) #plot Chebyshev polynomial
            plt.xlabel(name)
            plt.ylabel('normed frequency')
            #plt.title(f'plot of {name} for {self._bin[0]}$<q^2<${self._bin[1]} bin')
            plt.show()
            
        #converting to polynomial form
        
        z = p.convert().coef
        # T0 = np.array([1,0,0,0,0,0])
        # T1 = np.array([0,1,0,0,0,0])
        # T2 = np.array([-1*0.5,0,3*0.5,0,0,0])
        # T3 = np.array([0,-3*0.5,0,5*0.5,0,0])
        # T4 = np.array([(1/8)*3,0,(1/8)*-30,0,(1/8)*35,0])
        # T5 = np.array([0,(1/8)*15,0,(1/8)*-70,0,(1/8)*63])
        T0 = np.array([1,0,0,0,0,0])
        T1 = np.array([0,1,0,0,0,0])
        T2 = np.array([-1,0,2,0,0,0])
        T3 = np.array([0,-3,0,4,0,0])
        T4 = np.array([1,0,-8,0,8,0])
        T5 = np.array([0,5,0,-20,0,16])
        poly = z[0]*T0 + z[1]*T1 + z[2]*T2 + z[3]*T3 + z[4]*T4
        
        if plot_poly == True:
        
            x = np.linspace(-1,1,1000)
            plt.hist(d,histtype="step",color="orange", bins=int(1+3.332*((np.log(len(d)))/(np.log(2)))))
            plt.plot(x,poly[0] + poly[1]*x + poly[2]*x**2 + poly[3]*x**3 + poly[4]*x**4)
            plt.show()
            
            
        return p, poly
    
    def chebyshevq(self, qmin, qmax, order = 5, name = "Data", plot_p = False, plot_poly = False):
        
        d = self.slq(qmin, qmax)
        
        #getting x and y values

        y, bin_edges = np.histogram(d,bins=int(1+ceil(3.322*np.log(len(d))/np.log(2))),density = True) #bins eran seis me cago en dios marik
        x = 0.5 * (bin_edges[1:] + bin_edges[:-1]) 
        # p = np.polynomial.legendre.Legendre.fit(x, y, order)#get a list of the midpoints of the bins
        p = np.polynomial.Chebyshev.fit(x, y, order) #f is our Chebyshev polynomial fitted to points x and y of order 4
        
        if plot_p == True:

            x_new = np.arange(x[0],x[-1],0.01) #generate a range of x values with smaller steps
            plt.plot(x_new,p(x_new)) #plot Chebyshev polynomial
            plt.xlabel(name)
            plt.ylabel('normed frequency')
            #plt.title(f'plot of {name} for {self._bin[0]}$<q^2<${self._bin[1]} bin')
            plt.show()
            
        #converting to polynomial form
        
        z = p.convert().coef
        # T0 = np.array([1,0,0,0,0,0])
        # T1 = np.array([0,1,0,0,0,0])
        # T2 = np.array([-1*0.5,0,3*0.5,0,0,0])
        # T3 = np.array([0,-3*0.5,0,5*0.5,0,0])
        # T4 = np.array([(1/8)*3,0,(1/8)*-30,0,(1/8)*35,0])
        # T5 = np.array([0,(1/8)*15,0,(1/8)*-70,0,(1/8)*63])
        T0 = np.array([1,0,0,0,0,0])
        T1 = np.array([0,1,0,0,0,0])
        T2 = np.array([-1,0,2,0,0,0])
        T3 = np.array([0,-3,0,4,0,0])
        T4 = np.array([1,0,-8,0,8,0])
        T5 = np.array([0,5,0,-20,0,16])
        poly = z[0]*T0 + z[1]*T1 + z[2]*T2 + z[3]*T3 + z[4]*T4 + z[5]*T5
        
        if plot_poly == True:
        
            x = np.linspace(-1,1,1000)
            plt.plot(x,poly[0] + poly[1]*x + poly[2]*x**2 + poly[3]*x**3 + poly[4]*x**4)
            plt.show()
            
            
        return p, poly