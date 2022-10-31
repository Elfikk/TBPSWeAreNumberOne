import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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
total_df2 = pd.read_csv(filepath + '/total_ml_2.csv')

total_df_sig = total_df2[(total_df2['identity'] == 'signal')]
total_df_back = total_df2[(total_df2['identity'] != 'signal')]
total_df_comb = total_df2[(total_df2['identity'] == 'comb.csv')]
total_df_peaking = total_df2[(total_df2['identity'] != 'signal') & (total_df2['identity'] != 'comb.csv')]

plt.figure()
plt.xlabel('B0_M')
plt.ylabel('Count')
plt.hist(total_df['B0_M'],color='royalblue',label='Total', histtype='step',bins=100) # total data
plt.hist(signal_df['B0_M'],color='violet',label='Simulated Signal',histtype='step',bins=100) # simulated signal from signal.csv
plt.hist(total_df_sig['B0_M'],color='red',label='ML Predicted Signal', histtype='step',bins=100) # ML Predicted Signal on total
plt.hist(total_df_comb['B0_M'],color='cyan',label='ML Predicted Combinatorial', histtype='step',bins=100) # ML Predicted combinatorial background on total
plt.hist(total_df_peaking['B0_M'],color='green',label='ML Predicted Peaking',histtype='step',bins=100) # ML Predicted peaking background on total
# plt.hist(total_df_back['B0_M'],color='green',label='Background', histtype='step',bins=100)
# sns.histplot(data=total_df2['B0_M'])
plt.legend()
plt.show()