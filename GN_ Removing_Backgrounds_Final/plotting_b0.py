import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from selection_criteria import apply_all_selection
import xgboost as xgb
import os


'''
This script is used to plot the b0_candidate histograms.
'''



def exponential(x,A,d):
    exponential = A*np.exp(-x/d)
    return exponential

def gaussian(x,A,mu,std):
    gaussian = A*np.exp(-(x-mu)**2/(2*std**2))
    return gaussian

def gaussian_exponential(x,A1,mu,std,A2,d):
    exponential_1 = exponential(x,A2,d)
    gaussian_1 = gaussian(x,A1,mu,std)
    f = exponential_1+gaussian_1
    return f

# def two_gaussian_exponential(x,A1,mu1,std1,A2,d,A3,std2):
#     exponential_1 = exponential(x,A2,d)
#     gaussian_1 = gaussian(x,A1,mu1,std1)
#     gaussian_2 = gaussian(x,A3,mu1,std1)
#     f = exponential_1+gaussian_1+gaussian_2
#     return f

#For some reason writing each out is more accurate?
def two_gaussian_exponential(x,A1,mu1,std1,A2,d,A3,std2):
    exponential_1 = A2*np.exp(-x/d)
    gaussian_1 = A1*np.exp(-(x-mu1)**2/(2*std1**2))
    gaussian_2 = A3*np.exp(-(x-mu1)**2/(2*std2**2))
    f = exponential_1+gaussian_1+gaussian_2
    return f



def fit(x,y):
    '''
    Returns the params which decribe the expotenial and guassians which best fit the data.
    '''
    #Trial and error inital guess, I found that these numbers seemed to work best.
    p00=[0.01,5275,15]

    #guesses the parameters for a single gaussian
    par0, cov0 = curve_fit(gaussian,x,y,p0=p00)

    #uses the guesses for the gaussian, to find a more precise guess for a guassian and exponential. Again trail and error used for inital guess.
    p01=list(par0)+[1.7e5,1.9e2]
    par1, cov1 = curve_fit(gaussian_exponential,x,y,p0=p01)

    #Uses previous guesses to find the fit for two gaussians and an expoential.
    p02=list(par1)+[par1[0]]+[par1[2]]

    return curve_fit(two_gaussian_exponential,x,y,p0=p02)

def plot(x,y,dataset,param,bins_n,y_0,y_exp, name_of_graph,dataset_name):
    #Ploting graph
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x_smooth = np.linspace(min(x),max(x),10000)
    ax.plot(x,y, ".", color='green',label = dataset_name)

    n, bins = np.histogram(dataset['B0_M'],bins = bins_n)
    yerr = np.divide(np.sqrt(n), n,)* y
    ax.errorbar(x,y, yerr = yerr , ls = 'none', color = 'black', elinewidth=1)
    ax.plot(x_smooth,two_gaussian_exponential(x_smooth,*param), color='green', label = "Fitted (2 Gaussian's + Exponential)")
    ax.fill_between(x,y_0,y_exp, color = 'orange',label = 'Combinatorial Background')

    # Layout of graph - From Nick Khoze. ----------------------

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # Switch off ticks
    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    # Draw horizontal axis lines
    vals = ax.get_yticks()
    for tick in vals:
        ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    # Format y-axis label
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))

    #----------------------------------------------------------
    plt.xlabel(r'm($K^{+}\pi^{-}\mu^{+}\mu^{-}$) [MeV/$c^{2}$]')
    plt.ylabel(r'Frequency')
    plt.title(name_of_graph)
    plt.ylim(0)
    plt.legend()
    plt.savefig('B0_plot_graph/'+str(name_of_graph)+'.jpg', bbox_inches="tight", pad_inches=0, format="jpg", dpi=600)
    plt.show()

def remove_columns(dataset, reject_rows = []):
    #'B0_M', 'J_psi_M', 'q2','Kstar_M'
    dataset_modified = dataset.copy()
    dataset_modified.drop(columns=['Unnamed: 0.1','Unnamed: 0', 'year', 'B0_ID', 'B0_ENDVERTEX_NDOF','J_psi_ENDVERTEX_NDOF', 'Kstar_ENDVERTEX_NDOF'], inplace=True)
    if 'Unnamed: 0.2' in dataset_modified.columns:
        dataset_modified.drop(columns=['Unnamed: 0.2'], inplace=True)

    columns_list = dataset_modified.columns.tolist()
    for x in reject_rows:
        if x in columns_list:
            dataset_modified.drop(columns=[x],inplace = True)
    
    return dataset_modified


def high_corrolation_list_peaking_together(num):
    corrolation = pd.read_csv('Peaking_Together_Correlation/continuous_f1_score_peaking_together.csv')
    cols = corrolation.columns.tolist()
    values = {}
    for x in cols:
        values[x] = float(corrolation[x])

    values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    array_list = []
    for idx, key in enumerate(values):
        if(idx > num-1):
            array_list.append(key)
    return array_list

def high_corrolation_list_comb(num):
    corrolation = pd.read_csv('Comb_Correlation/continuous_f1_score_comb.csv')
    cols = corrolation.columns.tolist()
    values = {}
    for x in cols:
        values[x] = float(corrolation[x])

    values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    array_list = []
    for idx, key in enumerate(values):
        if(idx > num-1):
            array_list.append(key)
    return array_list

def high_corrolation_list_peaking_separate(num,name_of_background):
    corrolation = pd.read_csv('Peaking_Separate_Correlation/continuous_f1_score_{}.csv'.format(name_of_background))
    cols = corrolation.columns.tolist()
    values = {}
    for x in cols:
        values[x] = float(corrolation[x])

    values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))

    array_list = []
    for idx, key in enumerate(values):
        if(idx > num-1):
            array_list.append(key)
    return array_list

if __name__ == '__main__':
    total_data = pd.read_csv('../data/total_dataset.csv')
    #signal_ml = pd.read_csv('Signal_ML_Combo_removed_only.csv')
    s_c_dataset = apply_all_selection(total_data)

    #------------------------------
    '''
    Loads in the peaking model and comb models... and applies them.
    '''
    #boolean controls if the peaking model used is the based off multiple peaking models or just one combined one.
    separate = True
    directory = 'peaking_models'
    training = s_c_dataset.copy()
    if separate == True:
        string_background_models = ['jpsi_mu_k_swap','jpsi_mu_pi_swap','k_pi_swap','phimumu','pKmumu_piTok_kTop','pKmumu_piTop','Kmumu','Kstarp_pi0']
        count = [4,5,5,7,6,8,7,7]
        for idx,x in enumerate(string_background_models):
            peaking_model = xgb.XGBClassifier()
            peaking_model.load_model('Peaking_Models_Separate/peaking_model_'+x+'_removed_q2_range.model')
            temp_full = training
            remove_rows = high_corrolation_list_peaking_separate(count[idx],x)
            training = remove_columns(training, remove_rows)
            ypred = peaking_model.predict(training)
            training = temp_full.loc[ypred == 1]
    elif separate == False:
        peaking_model = xgb.XGBClassifier()
        peaking_model.load_model('Peaking_Models_Together/peaking_model_trained_together_removed_q2_range_13_features.model')
        temp_full = training
        remove_col_list = high_corrolation_list_peaking_together(13)
        training = remove_columns(training, remove_col_list)
        ypred = peaking_model.predict(training)
        training = temp_full.loc[ypred == 1]


    model = xgb.XGBClassifier()
    model.load_model('Comb_Model/comb_model_5350_removed_q2_range_29_features.model')
    
    temp_full = training

    remove_columns_array = high_corrolation_list_comb(29)
    remove_columns_array.append('B0_M')
    training = remove_columns(temp_full, remove_columns_array)
    ypred = model.predict(training)

    signal_ml = temp_full.loc[ypred == 1]


    #writes the ml signal to file (peaking and comb applied).
    signal_ml.to_csv('ML_SIGNAL_BDT.csv')
    #---------------------------------------------------

    bins_n = 100

    #Calculates histogram for the selection criteria applied dataset.
    n_b0_total,bin_edges_b0_total = np.histogram(s_c_dataset['B0_M'],bins = bins_n, density = True)
    mid_bin_total = (bin_edges_b0_total[1:]+bin_edges_b0_total[:-1])*0.5

    #Calculates histogram for the ml signal dataset.
    n_b0_signal,bin_edges_b0_signal = np.histogram(signal_ml['B0_M'],bins = bins_n, density = True)
    mid_bin_signal = (bin_edges_b0_signal[1:]+bin_edges_b0_signal[:-1])*0.5

    #Calculates histogram for the peaking model removed dataset.
    n_b0_peaking,bin_edges_b0_peaking = np.histogram(temp_full['B0_M'],bins = bins_n, density = True)
    mid_bin_peaking = (bin_edges_b0_peaking[1:]+bin_edges_b0_peaking[:-1])*0.5

    #calculates the parameters used for the fit.
    x_total, y_total = mid_bin_total,n_b0_total
    x_signal, y_signal = mid_bin_signal,n_b0_signal
    x_peaking, y_peaking = mid_bin_peaking,n_b0_peaking

    par_total,cov2_total = fit(x_total,y_total)
    par_signal,cov2_signal = fit(x_signal,y_signal)
    par_peaking,cov2_peaking = fit(x_peaking,y_peaking)

    #Gets the exponential terms
    y_exp_total=exponential(x_total,par_total[3],par_total[4])
    y_0_total=np.zeros(len(y_exp_total))

    y_exp_signal=exponential(x_signal,par_signal[3],par_signal[4])
    y_0_signal=np.zeros(len(y_exp_signal))

    y_exp_peaking=exponential(x_peaking,par_peaking[3],par_peaking[4])
    y_0_peaking=np.zeros(len(y_exp_peaking))

    plot(x_total,y_total,s_c_dataset,list(par_total),bins_n,y_0_total,y_exp_total, 'Total Dataset (selection criteria applied)','Total dataset (SC applied)')
    plot(x_signal,y_signal,signal_ml,list(par_signal),bins_n,y_0_signal,y_exp_signal, 'ML Signal (Combo + Peaking removed)','Ml Signal')
    plot(x_peaking,y_peaking,temp_full,list(par_peaking),bins_n,y_0_peaking,y_exp_peaking, 'ML Signal (Peaking removed)','Ml Signal')
