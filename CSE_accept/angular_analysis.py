from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from iminuit import Minuit

#%% LOADING DATA

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

def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    #acceptance = 0.5  # acceptance "function"
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) #* acceptance
    normalised_scalar_array = scalar_array #* 2  # normalising scalar array to account for the non-unity acceptance function
    return normalised_scalar_array
# For now ignore acceptance, later need to implement it in this function.


def log_likelihood(fl, afb, _bin):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    """
    _bin = bins[int(_bin)]
    ctl = _bin['costhetal']
    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl)
    return - np.sum(np.log(normalised_scalar_array))
# minimising negative log likelihood gives best the estimate of the parameters (fl and afb)
# we use negative log likelihood because it is computationally easier to find a minimum rather than a maximum


#%% FITTING AND CHECKING

bin_number_to_check = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = [0.1,-0.1]
fls, fl_errs = [], []
afbs, afb_errs = [], []
for i in range(len(bins)):
    m = Minuit(log_likelihood, fl=starting_point[0], afb=starting_point[1], _bin = i)
    m.fixed['_bin'] = True  # fixing the bin number as we don't want to optimize it
    m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
    m.migrad() # a minimisation algorithm
    m.hesse() # an algorithm to compute errors
    #if i == bin_number_to_check[i]:
        #bin_results_to_check = m
        #plt.figure(figsize=(8, 5))
        #plt.subplot(221)
        #bin_results_to_check.draw_mnprofile('afb', bound=3)
        #plt.subplot(222)
        #bin_results_to_check.draw_mnprofile('fl', bound=3)
        #plt.tight_layout()
        #plt.show()
        
        # This loop plots the Minos profiles. gray zone is confidence interval. 
        # not that important since we already have the uncertainties printed out
    fls.append(m.values[0])
    afbs.append(m.values[1])
    fl_errs.append(m.errors[0])
    afb_errs.append(m.errors[1])
    print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}, Fuction minimum considered valid: {m.fmin.is_valid}")
    
# This cell prints whether the minimum converged or not
# also prints fitted fl and afb values and associated errors.
# Might get error message saying 'invalid value encountered in log' 
# but doesn't really matter as long as end result converges.

#%% PLOTTING FIT FOR EACH BIN

# plots the d2gamma_p_d2q2_dcostheta function (PDF for cos_theta_l) over each bin, 
# using fl and afb values obtained from minimising the negative log likelihood

bin_to_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
number_of_bins_in_hist = 25
for i in range(len(bin_to_plot)):
    cos_theta_l_bin = bins[bin_to_plot[i]]['costhetal']
    hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=number_of_bins_in_hist)
    x = np.linspace(-1, 1, number_of_bins_in_hist)
    pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
    y = d2gamma_p_d2q2_dcostheta(fl=fls[bin_to_plot[i]], afb=afbs[bin_to_plot[i]], cos_theta_l=x) * pdf_multiplier
    plt.plot(x, y, label=f'Fit for bin {bin_to_plot[i]}')
    plt.xlabel(r'$cos(\theta_l)$')
    plt.ylabel(r'Number of candidates')
    plt.legend()
    plt.grid()
    plt.show()
    
#%% PLOTTING RESULTS FOR ALL BINS

# plots the estimated fl and afb values for each bin, with associated uncertainties

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
ax1.errorbar(np.linspace(0, len(bins) - 1, len(bins)), fls, yerr=fl_errs, fmt='o', markersize=2, label=r'$F_L$', color='red')
ax2.errorbar(np.linspace(0, len(bins) - 1, len(bins)), afbs, yerr=afb_errs, fmt='o', markersize=2, label=r'$A_{FB}$', color='red')
ax1.grid()
ax2.grid()
ax1.set_ylabel(r'$F_L$')
ax2.set_ylabel(r'$A_{FB}$')
ax1.set_xlabel(r'Bin number')
ax2.set_xlabel(r'Bin number')
plt.tight_layout()
plt.show()