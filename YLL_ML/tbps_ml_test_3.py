import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
import os
from sklearn import metrics
from sklearn import model_selection
import time

st = time.time() # to time how long it takes to run script, ignore

filepath = '/Users/gordonlai/Documents/ICL/ICL_Y3/TBPSWeAreNumberOne/Data' #set your own file path

columns_to_remove_1 = [
    "Unnamed: 0",
    "Unnamed: 0.1", 
    "year",
    'B0_ID',
    'B0_ENDVERTEX_NDOF',
    'J_psi_ENDVERTEX_NDOF',
    'Kstar_ENDVERTEX_NDOF',
    # '_merge' #Yoinked from Ganels thank you Ganel
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

'''
# filter using q^2 conditions
b0mass=total_df['B0_M']
invarmass=total_df['q2']
total_dataset_df_yes_peaking = total_df[(invarmass > 0.98) & (invarmass < 1.10) | (invarmass > 8.0) & (invarmass < 11.0) | (invarmass > 12.5) & (invarmass < 15.0)]
total_df = total_df.merge(total_dataset_df_yes_peaking, how='left', indicator=True)
total_df = total_df[total_df['_merge'] == 'left_only'] # line 16-17 from stackoverflow.com
'''
total_df = total_df.drop(columns_to_remove_1, axis=1)
total_df_clean = total_df.copy()
# identity = pd.array([])
total_df_new = pd.DataFrame(data=[],columns=total_df_clean.columns)
# load all simulated dataset
signal_df = pd.read_csv(filepath + '/signal.csv') # real signal we want, the good boi
signal_df['identity'] = 'signal'

# reorder listdir
fswap = [a for a in os.listdir(filepath) if a.endswith('swap.csv')]
fpeak = [b for b in os.listdir(filepath) if not b.endswith('swap.csv')]
ftot = fpeak + fswap
# loop through data file and perform ML each time
for f in ftot:
    training_df = pd.read_csv(filepath + '/signal.csv')
    training_df['identity'] = 'signal'
    if not f.startswith('total') and not f.startswith('signal') and not f.startswith('acceptance') and not f.startswith('.DS') and not f.startswith('comb'):
        print(f)
        temp_df = pd.read_csv(filepath + '/' + f)
        if f.endswith('swap.csv'):
            temp_df['identity'] = 'combinatorial'
        else:
            temp_df['identity'] = 'peaking'
        training_df = pd.concat([training_df,temp_df])
        # ML starts
        training_df = training_df.drop(columns_to_remove,axis=1)
        sim_X = training_df.drop('identity', axis=1) # splitting independent and dependent variables
        sim_Y = training_df['identity']
        seed=1
        train_X,test_X,train_Y,test_Y = model_selection.train_test_split(sim_X, sim_Y, test_size = 0.33, random_state = seed) # splitting data for training and validation
        model = lgbm.LGBMClassifier() # model used for training
        model.fit(train_X,train_Y) # actually training the model
        y_pred = model.predict(test_X) # predict results for test_Y, later on compare with actual test_Y
        accuracy = metrics.accuracy_score(test_Y,y_pred) # accuracy score, self explanatory
        print('Accuracy Score = ' +str(accuracy))
        # apply ML to acutal dataset
        total_df_clean['identity'] = model.predict(total_df_clean)
        # sorting the categories
        new_temp_df = total_df_clean[(total_df_clean['identity'] != 'signal')]
        total_df_new = pd.concat([total_df_new, new_temp_df])
        total_df_clean = total_df_clean[(total_df_clean['identity'] == 'signal')]
        total_df_clean = total_df_clean.drop('identity',axis=1)

total_df_clean['identity'] = 'signal'
total_df2 = pd.concat([total_df_clean,total_df_new])
total_df2.to_csv(filepath + '/total_ml2_3.csv')
# print(total_df2)


et = time.time()
elapsed_time = et - st
print('Exeuction time: ' + str(elapsed_time/60) +' mins')
