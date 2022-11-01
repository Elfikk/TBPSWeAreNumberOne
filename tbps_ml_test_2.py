import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import model_selection
from sklearn import metrics
import xgboost as xgb
import os
import seaborn as sns

filepath = '/Users/gordonlai/Documents/ICL/ICL_Y3/TBPSWeAreNumberOne/Data' #set your own file path

columns_to_remove_1 = [
    "Unnamed: 0",
    "Unnamed: 0.1", 
    "year",
    'B0_ID',
    'B0_ENDVERTEX_NDOF',
    'J_psi_ENDVERTEX_NDOF',
    'Kstar_ENDVERTEX_NDOF',
    '_merge' #Yoinked from Ganels thank you Ganel
    ]

columns_to_remove = [
    "Unnamed: 0",
    "Unnamed: 0.1", 
    "Unnamed: 0.2",
    "year",
    'B0_ID',
    'B0_ENDVERTEX_NDOF',
    'J_psi_ENDVERTEX_NDOF',
    'Kstar_ENDVERTEX_NDOF',
    '_merge' #Yoinked from Ganels thank you Ganel
    ]

# load real total dataset
total_df = pd.read_csv(filepath + '/total_dataset.csv')


# filter using q^2 conditions
b0mass=total_df['B0_M']
invarmass=total_df['q2']
total_dataset_df_yes_peaking = total_df[(invarmass > 0.98) & (invarmass < 1.10) | (invarmass > 8.0) & (invarmass < 11.0) | (invarmass > 12.5) & (invarmass < 15.0)]
total_df = total_df.merge(total_dataset_df_yes_peaking, how='left', indicator=True)
total_df = total_df[total_df['_merge'] == 'left_only'] # line 16-17 from stackoverflow.com

total_df = total_df.drop(columns_to_remove_1, axis=1)

# load all simulated dataset
signal_df = pd.read_csv(filepath + '/signal.csv') # real signal we want, the good boi
signal_df['identity'] = 'signal'

# construct training dataset
for f in os.listdir(filepath):
    if not f.startswith('total') and not f.startswith('signal') and not f.startswith('acceptance') and not f.startswith('.DS') and not f.startswith('comb'):
        temp_df = pd.read_csv(filepath + '/' + f)
        temp_df['identity'] = 'peaking'
        signal_df = pd.concat([signal_df,temp_df])
    elif f.startswith('comb'):
        temp_df = pd.read_csv(filepath + '/' + f)
        temp_df['identity'] = 'combinatorial'
        signal_df = pd.concat([signal_df,temp_df])

b0mass=signal_df['B0_M']
invarmass=signal_df['q2']
signal_dataset_df_yes_peaking = signal_df[(invarmass > 0.98) & (invarmass < 1.10) | (invarmass > 8.0) & (invarmass < 11.0) | (invarmass > 12.5) & (invarmass < 15.0)]
signal_df = signal_df.merge(signal_dataset_df_yes_peaking, how='left', indicator=True)
signal_df = signal_df[signal_df['_merge'] == 'left_only'] # line 16-17 from stackoverflow.com

# signal_df.to_csv(filepath + '/total_sim_4.csv',index=False) # save training dataset



# tidy up training dataset
sim_df = pd.read_csv(filepath + '/total_sim_4.csv')
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
y_pred = model.predict(test_X)
accuracy = metrics.accuracy_score(test_Y,y_pred) # accuracy score, self explanatory
# f1score = metrics.f1_score(test_Y,y_pred) # f1 score
print('Accuracy Score = ' +str(accuracy))
# print('F1 Score = ' +str(f1score))


# apply ML model to actual dataset
total_df['identity'] = model.predict(total_df)
# total_df.to_csv(filepath + '/total_ml_4.csv')


