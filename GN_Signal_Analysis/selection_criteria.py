#imports for notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_scatter_density 
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter

#-------------------------Methods for all the selection criteria------------

def q2_ranges(dataset):
    q2 = dataset['q2']
    range1 = ((q2 >= 0.1) & (q2 < 0.98)) | ((q2 > 1.1) & (q2 < 8)) | ((q2 > 11) & (q2 < 12.5)) | ((q2 > 15) & (q2 < 19))
    dataset = dataset[range1]
    return dataset[range1]

def kstar_vertex_dof(dataset):
    cut_off = 12 * dataset["Kstar_ENDVERTEX_NDOF"].tolist()[0]
    return dataset[dataset['Kstar_ENDVERTEX_CHI2'] < cut_off]

def Kstar_mass(dataset):
     Kstar_M = dataset['Kstar_M']
     range1 = (Kstar_M > 792) & (Kstar_M < 992)
     return dataset[range1]

def B0_vertex_dof(dataset):
    cut_off = 6 * dataset["B0_ENDVERTEX_NDOF"].tolist()[0]
    return dataset[dataset['B0_ENDVERTEX_CHI2'] < cut_off]

def mu_plus_IP(dataset):
    return dataset[dataset['mu_plus_IPCHI2_OWNPV'] > 9]

def mu_minus_IP(dataset):
    return dataset[dataset['mu_minus_IPCHI2_OWNPV'] > 9]

def K_IP(dataset):
    return dataset[dataset['K_IPCHI2_OWNPV'] > 9]

def Pi_plus_IP(dataset):
    return dataset[dataset['Pi_IPCHI2_OWNPV'] > 9]

def B0_IP(dataset):
    return dataset[dataset['B0_IPCHI2_OWNPV'] < 16]

def B0_FD(dataset):
    return dataset[dataset['B0_FDCHI2_OWNPV'] > 121]

def KSTAR_FD(dataset):
    return dataset[dataset['Kstar_FDCHI2_OWNPV'] > 9]

def DIRA(dataset):
    return dataset[dataset['B0_DIRA_OWNPV'] > 0.9999]

def B0_M(dataset):
    B0_M = dataset['B0_M']
    range = (B0_M > 4850) & (B0_M < 5780)
    return dataset[range]

def J_Psi_vertex_dof(dataset):
    cut_off = 12 * dataset["J_psi_ENDVERTEX_NDOF"].tolist()[0]
    return dataset[dataset['J_psi_ENDVERTEX_CHI2'] < cut_off]

def J_Psi_FD(dataset):
    return dataset[dataset['J_psi_FDCHI2_OWNPV'] > 9]


#Import this function into your code, to get the filtered dataset. 'from selection_criteria import apply_all_selection'
def apply_all_selection(dataset):
    dataset = q2_ranges(dataset)
    dataset = kstar_vertex_dof(dataset)
    dataset = Kstar_mass(dataset)
    dataset = B0_vertex_dof(dataset)
    dataset = mu_plus_IP(dataset)
    dataset = mu_minus_IP(dataset)
    dataset = K_IP(dataset)
    dataset = Pi_plus_IP(dataset)
    dataset = B0_IP(dataset)
    dataset = B0_FD(dataset)
    dataset = KSTAR_FD(dataset)
    dataset = DIRA(dataset)
    dataset = B0_M(dataset)
    dataset = J_Psi_vertex_dof(dataset)
    dataset = J_Psi_FD(dataset)
    return dataset

#Displays the selection.
if __name__ == '__main__':
    data = pd.read_csv('../data/total_dataset.csv')
    filter = apply_all_selection(data)



    # Plotting the mass distribution over entire q^2 range
    n_b0,bin_edges_b0 = np.histogram(data['B0_M'],bins = 100, density = True)
    n_b0_t,bin_edges_b0_t = np.histogram(filter['B0_M'],bins = 100, density = True)

    #calculates the mid point of each bin.
    mid_bin = (bin_edges_b0[1:]+bin_edges_b0[:-1])*0.5
    mid_bin_t = (bin_edges_b0_t[1:]+bin_edges_b0_t[:-1])*0.5

    x = mid_bin
    y = n_b0

    x_t = mid_bin_t
    y_t = n_b0_t

    #Ploting graph
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax.errorbar(x,y, yerr = np.sqrt(y), ls = 'none', color = '#FF0000', elinewidth=1)
    ax.plot(x,y, ".", color='red',label = 'Total_dataset')
    ax.plot(x_t,y_t, ".", color='green', label = 'Filtered')
    ax.set_ylim(ymin=0)

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

    plt.xlabel(r'm($K^{+}\pi^{-}\mu^{+}\mu^{-}$) [GeV/$c^{2}$]')
    plt.ylabel(r'Frequency')
    plt.legend()
    plt.show()
