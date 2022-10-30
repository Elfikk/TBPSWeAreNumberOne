import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import model_selection
from sklearn import metrics
import xgboost as xgb
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
'''
# construct training dataset
for f in os.listdir(filepath):
    if not f.startswith('total') and not f.startswith('signal') and not f.startswith('acceptance') and not f.startswith('.DS'):
        temp_df = pd.read_csv(filepath + '/' + f)
        temp_df['identity'] = 'background'
        signal_df = pd.concat([signal_df,temp_df])

signal_df.to_csv(filepath + '/total_sim.csv',index=False) # save training dataset
'''

# tidy up training dataset
sim_df = pd.read_csv(filepath + '/total_sim.csv')
sim_df = sim_df.drop(columns_to_remove, axis=1)
# print(sim_df)

# training ML dataset

# splitting the target out
sim_X = sim_df.drop('identity', axis=1)
sim_Y = sim_df['identity']

# split dataset for training and validation 
seed = 1
train_X,test_X,train_Y,test_Y = model_selection.train_test_split(sim_X, sim_Y, test_size = 0.33, random_state = seed)

# actually fitting the training data to the model
model = xgb.XGBClassifier()
model.fit(train_X,train_Y)

# predict the result
# y_pred = model.predict(test_X)
# accuracy = metrics.accuracy_score(test_Y,y_pred) # accuracy score, self explanatory
# print(accuracy)

# apply ML model to actual dataset
total_df['identity'] = model.predict(total_df)

# plot result
total_df_sig = total_df[(total_df['identity'] == 'signal')]
total_df_back = total_df[(total_df['identity'] == 'background')]

plt.figure()
plt.xlabel('B0_M')
plt.ylabel('Count')
plt.hist(total_df['B0_M'],color='royalblue',label='Total', histtype='step',bins=100)
plt.hist(total_df_sig['B0_M'],color='red',label='Predicted Signal', histtype='step',bins=100)
plt.hist(signal_df['B0_M'],color='violet',label='Simulated Signal',histtype='step',bins=100)
#plt.hist(total_df_back['B0_M'],color='green',label='Background', histtype='step',bins=100)
plt.legend()
plt.show()

