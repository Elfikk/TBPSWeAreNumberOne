from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from iminuit import Minuit

#%% LOADING DATA
#make sure to load in data that has passed through the ML model

data_path = 'Data/'
filename1 = f'{data_path}acceptance_mc.csv'
filename2 = f'{data_path}total_dataset.csv'
filename3 = f'{data_path}signal.csv'
df_ac = pd.read_csv(filename1)
title_ac = pd.read_csv(filename1, index_col=0, nrows=0).columns.tolist()
df_td = pd.read_csv(filename2)
title_td = pd.read_csv(filename2, index_col=0, nrows=0).columns.tolist()
df_sg = pd.read_csv(filename3)
title_sg = pd.read_csv(filename3, index_col=0, nrows=0).columns.tolist()

# df: dataframe
# ac: acceptance
# td: total data set
# sg: signal

#%% BINNING DATA

bins = []
bin_edges = ([[0.1,0.98], [1.1,2.5], [2.5,4.0], [4.0,6.0], [6.0,8.0], [15.0,17.0], [17.0,19.0], [11.0,12.5], [1.0,6.0], [15.0,17.9]])
# binning scheme given on the predictions page

def bin_q2(dataframe,edges):
    '''
    This function bins given data according to the input bin edges.
    '''
    for i in range(len(edges)):
        df1 = dataframe[dataframe['q2']<edges[i][1]]
        df1 = df1[df1['q2']>edges[i][0]]
        bins.append(df1)

# change df_sg to look at other files            
bin_q2(df_sg, bin_edges)

# we have 10 bins in total, labelled by 0,1,2,...,9
# e.g. bin 1 is bins[1]
#%% DEFINING FUNCTIONS
#angular distribution functions below

def costhetak(a):
    return a

def costhetal(b):
    return b

def phi(c):
    return c

#%%
#log likelihood function (needed for minuit)

def log_likelihood_costhetal(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf of costhetal
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    normalised_scalar_array = costhetal(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array)) #np log uses base e
# minimising negative log likelihood gives best the estimate of the parameters (fl and afb)
# we use negative log likelihood because it is computationally easier to find a minimum rather than a maximum

def log_likelihood_costhetak(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf of costhetak
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetak']
    normalised_scalar_array = costhetak(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array)) #np log uses base e

def log_likelihood_phi(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf of phi
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    """
    _bin = bins[int(_bin)]
    ctl = _bin['phi']
    normalised_scalar_array = phi(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array)) #np log uses base e
#%% FITTING AND CHECKING FOR CONVERGENCE

bin_number_to_check = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
bin_results_to_check = None

def negative_likelihood_fit(angular_dist, 
                            costhetal_flag = False, 
                            costhetak_flag = False, 
                            phi_flag = False,
                            sanity_check = False):
    """
    Returns the values of the angular observabels for each bin
    and minos plots with confidence intervals for each angular observable minimisation
    :angular_dist: function for angular distribution we want to fit for
    :costhetal: boolean, if True, then the fit will run for the observables in the costhetal distributino
    :costhetak: boolean, if True, then the fit will run for the observables in the costhetak distribution
    :phi: boolean, if True, then the fit will run for the observables in the phi distribution
    :santify_check: boolean, if True, then function wil plot the minos profile with confidence intervals
    """
    angular_dist.errordef = Minuit.LIKELIHOOD
    decimal_places = 3
    starting_point = [0.1,-0.1]
    if costhetal_flag==True:
        fls, fl_errs = [], []
        afbs, afb_errs = [], []
        for i in range(len(bins)):
            m = Minuit(log_likelihood_costhetal, fl=starting_point[0], afb=starting_point[1], _bin = i)
            m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
            m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad() # a minimisation algorithm
            m.hesse() # an algorithm to compute errors
            if sanity_check == True:
                if i == bin_number_to_check[i]:
                    bin_results_to_check = m
                    plt.figure(figsize=(8, 5))
                    plt.subplot(221)
                    bin_results_to_check.draw_mnprofile('afb', bound=3)
                    plt.subplot(222)
                    bin_results_to_check.draw_mnprofile('fl', bound=3)
                    plt.tight_layout()
                    plt.show()
                
                # This loop plots the Minos profiles. gray zone is confidence interval. 
                # not that important since we already have the uncertainties printed out
            fls.append(m.values[0])
            afbs.append(m.values[1])
            fl_errs.append(m.errors[0])
            afb_errs.append(m.errors[1])
            print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}, Fuction minimum considered valid: {m.fmin.is_valid}")
        return fls,fl_errs, afbs, afb_errs
    if costhetak_flag==True:
        fls, fl_errs = [], []
        afbs, afb_errs = [], []
        for i in range(len(bins)):
            m = Minuit(log_likelihood_costhetak, fl=starting_point[0], afb=starting_point[1], _bin = i)
            m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
            m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad() # a minimisation algorithm
            m.hesse() # an algorithm to compute errors
            if sanity_check == True:
                if i == bin_number_to_check[i]:
                    bin_results_to_check = m
                    plt.figure(figsize=(8, 5))
                    plt.subplot(221)
                    bin_results_to_check.draw_mnprofile('afb', bound=3)
                    plt.subplot(222)
                    bin_results_to_check.draw_mnprofile('fl', bound=3)
                    plt.tight_layout()
                    plt.show()
                
                # This loop plots the Minos profiles. gray zone is confidence interval. 
                # not that important since we already have the uncertainties printed out
            fls.append(m.values[0])
            afbs.append(m.values[1])
            fl_errs.append(m.errors[0])
            afb_errs.append(m.errors[1])
            print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}, Fuction minimum considered valid: {m.fmin.is_valid}")
        return fls,fl_errs, afbs, afb_errs
    if phi_flag==True:
        fls, fl_errs = [], []
        afbs, afb_errs = [], []
        for i in range(len(bins)):
            m = Minuit(log_likelihood_phi, fl=starting_point[0], afb=starting_point[1], _bin = i)
            m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
            m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad() # a minimisation algorithm
            m.hesse() # an algorithm to compute errors
            if sanity_check == True:
                if i == bin_number_to_check[i]:
                    bin_results_to_check = m
                    plt.figure(figsize=(8, 5))
                    plt.subplot(221)
                    bin_results_to_check.draw_mnprofile('afb', bound=3)
                    plt.subplot(222)
                    bin_results_to_check.draw_mnprofile('fl', bound=3)
                    plt.tight_layout()
                    plt.show()
                
                # This loop plots the Minos profiles. gray zone is confidence interval. 
                # not that important since we already have the uncertainties printed out
            fls.append(m.values[0])
            afbs.append(m.values[1])
            fl_errs.append(m.errors[0])
            afb_errs.append(m.errors[1])
            print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}, Fuction minimum considered valid: {m.fmin.is_valid}")
        return fls,fl_errs, afbs, afb_errs    
# This cell prints whether the minimum converged or not
# also prints fitted fl and afb values and associated errors.
# Might get error message saying 'invalid value encountered in log' 
# but doesn't really matter as long as end result converges.

#how to call for this function:
'''
fls, fl_errs, afbs, afb_errs = negative_likelihood_fit(costhetal, costhetal_flag = True, sanity_check = True)

'''

#%%
