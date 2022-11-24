import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
from selection_criteria import apply_all_selection
import os

total_data = pd.read_csv('../data/total_dataset.csv')
signal = pd.read_csv('../Data/signal.csv')

'''
This graph generates the graphs which compare each feature of the total dataset and signal
'''

def clean_dataset(dataset, reject_rows = []):
    dataset_modified = dataset.copy()
    dataset_modified.drop(columns=['Unnamed: 0.1','Unnamed: 0', 'year', 'B0_ID'], inplace=True)
    if 'Unnamed: 0.2' in dataset_modified.columns:
        dataset_modified.drop(columns=['Unnamed: 0.2'], inplace=True)

    columns_list = dataset_modified.columns.tolist()
    for x in reject_rows:
        if x in columns_list:
            dataset_modified.drop(columns=[x],inplace = True)
    
    return dataset_modified

def plot(x_total,y_total,x_signal,y_signal,title):
    #Ploting graph
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x_total,y_total, ".", color='green',label = 'Total_Dataset.csv')
    ax.plot(x_signal,y_signal, ".", color='red',label = 'signal.csv')

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
    plt.title(title)
    plt.ylim(0)
    plt.legend()
    plt.savefig('Total_vs_Signal_Comparison/'+str(title)+'_comparison.jpg', bbox_inches="tight", pad_inches=0, format="jpg", dpi=600)
    print(str(title)+' graph saved.')

if __name__ == '__main__':
    total_data_clean = clean_dataset(total_data)
    signal_clean = clean_dataset(signal)

    cols = signal_clean.columns.tolist()

    BIN_N = 100

    for x in cols:
        total_n,bin_edges_total = np.histogram(total_data_clean[x],bins = BIN_N, density = True)
        mid_bin_total = (bin_edges_total[1:]+bin_edges_total[:-1])*0.5

        signal_n,bin_edges_signal = np.histogram(signal_clean[x],bins = BIN_N, density = True)
        mid_bin_signal = (bin_edges_signal[1:]+bin_edges_signal[:-1])*0.5

        plot(mid_bin_total,total_n,mid_bin_signal,signal_n,str(x)+' Comparison')
