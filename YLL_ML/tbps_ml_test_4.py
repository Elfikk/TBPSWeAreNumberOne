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

total_df = total_df.drop(columns_to_remove_1, axis=1)
total_df_clean = total_df.copy()
training_df = pd.DataFrame(data=[])
signal_df = pd.read_csv(filepath + '/signal.csv')
signal_df['identity'] = 'signal'
comb_df = total_df[(total_df['B0_M'] > 5350.0)]
comb_df['identity'] = 'combinatorial'

# ML for signal vs peaking
for f in os.listdir(filepath):
    if not f.startswith('total') and not f.startswith('signal') and not f.startswith('acceptance') and not f.startswith('.DS') and not f.startswith('comb'):
        temp_df = pd.read_csv(filepath + '/' + f)
        temp_df['identity'] = 'peaking'
        training_df = pd.concat([training_df,temp_df])
training_df = pd.concat([training_df,comb_df,signal_df])
# ML starts
training_df = training_df.drop(columns_to_remove,axis=1)
sim_X = training_df.drop('identity', axis=1) # splitting independent and dependent variables
sim_Y = training_df['identity']
train_X,test_X,train_Y,test_Y = model_selection.train_test_split(sim_X, sim_Y, test_size = 0.33, random_state = 1) # splitting data for training and validation
model = lgbm.LGBMClassifier() # model used for training
model.fit(train_X,train_Y) # actually training the model
y_pred = model.predict(test_X) # predict results for test_Y, later on compare with actual test_Y
accuracy = metrics.accuracy_score(test_Y,y_pred) # accuracy score, self explanatory
print('Accuracy Score = ' +str(accuracy))
# apply ML to acutal dataset
total_df_clean['identity'] = model.predict(total_df_clean)

total_df_clean.to_csv(filepath + '/total_ml3_1.csv')

et = time.time()
print('Exeuction time: ' + str((et-st)/60) +' mins')

