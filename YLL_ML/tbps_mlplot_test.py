import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

filepath = '/Users/gordonlai/Documents/ICL/ICL_Y3/TBPSWeAreNumberOne/Data' #set your own file path

columns_to_remove_1 = [
    "Unnamed: 0",
    "Unnamed: 0.1", 
    "year",
    'B0_ID',
    'B0_ENDVERTEX_NDOF',
    'J_psi_ENDVERTEX_NDOF',
    'Kstar_ENDVERTEX_NDOF' #Yoinked from Ganels thank you Ganel
    ]

columns_to_remove = [
    "Unnamed: 0",
    "Unnamed: 0.1", 
    "Unnamed: 0.2",
    "year",
    'B0_ID',
    'B0_ENDVERTEX_NDOF',
    'J_psi_ENDVERTEX_NDOF',
    'Kstar_ENDVERTEX_NDOF' #Yoinked from Ganels thank you Ganel
    ]

# load real total dataset
total_df = pd.read_csv(filepath + '/total_dataset.csv')
total_df = total_df.drop(columns_to_remove_1, axis=1)

# load all simulated dataset
signal_df = pd.read_csv(filepath + '/signal.csv') # real signal we want, the good boi
signal_df['identity'] = 'signal'


# plot result
total_df2 = pd.read_csv(filepath + '/total_ml3_1.csv')

# filter using q^2 conditions
b0mass=total_df2['B0_M']
invarmass=total_df2['q2']
total_dataset_df_yes_peaking = total_df2[(invarmass > 0.98) & (invarmass < 1.10) | (invarmass > 8.0) & (invarmass < 11.0) | (invarmass > 12.5) & (invarmass < 15.0)]
total_df2 = total_df2.merge(total_dataset_df_yes_peaking, how='left', indicator=True)
total_df2 = total_df2[total_df2['_merge'] == 'left_only'] # line 16-17 from stackoverflow.com

# splitting the dataset into different categories
total_df_sig = total_df2[(total_df2['identity'] == 'signal')]
total_df_back = total_df2[(total_df2['identity'] != 'signal')]
total_df_comb = total_df2[(total_df2['identity'] == 'combinatorial')]
total_df_peaking = total_df2[(total_df2['identity'] != 'signal') & (total_df2['identity'] != 'combinatorial')]

# function to plot midpoints
def histmidpoints(data,colour,name,binn=100):
    bin_height, bin_edges, patches = plt.hist(data, bins=binn,alpha=0)
    midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return plt.plot(midpoints,bin_height,'-',color=colour,label=name)
    
# da plots
numbin=100
plt.figure()
plt.xlabel('B0_M')
plt.ylabel('Count')
# histograms
plt.hist(total_df2['B0_M'],color='royalblue',label='Total', histtype='step',bins=numbin,density=False) # total data
# plt.hist(signal_df['B0_M'],color='violet',label='Simulated Signal',histtype='step',bins=numbin) # simulated signal from signal.csv
plt.hist(total_df_sig['B0_M'],color='red',label='ML Predicted Signal', histtype='step',bins=numbin,density=False) # ML Predicted Signal on total
plt.hist(total_df_comb['B0_M'],color='cyan',label='ML Predicted Combinatorial', histtype='step',bins=numbin,density=False) # ML Predicted combinatorial background on total
plt.hist(total_df_peaking['B0_M'],color='green',label='ML Predicted Peaking',histtype='step',bins=numbin,density=False) # ML Predicted peaking background on total
# plt.hist(total_df_back['B0_M'],color='green',label='Background', histtype='step',bins=numbin)
'''# midpoints
histmidpoints(total_df2['B0_M'],'royalblue','Total')
histmidpoints(total_df_sig['B0_M'],'red','ML Predicted Signal')
histmidpoints(total_df_comb['B0_M'],'cyan','ML Predicted Combinatorial')
histmidpoints(total_df_peaking['B0_M'],'green','ML Predicted Peaking')
'''
plt.legend()
plt.show()