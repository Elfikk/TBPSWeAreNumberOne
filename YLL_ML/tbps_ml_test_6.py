import pandas as pd
import xgboost as xgb
import lightgbm as lgbm
import os
from sklearn import metrics
from sklearn import model_selection
import time
import matplotlib.pyplot as plt

filepath = '/Users/gordonlai/Documents/ICL/ICL_Y3/TBPSWeAreNumberOne/Data' #set your own file path

comb_columns = ['mu_plus_ProbNNk', 'mu_plus_ProbNNmu', 'mu_plus_ProbNNe', 'mu_plus_ProbNNp', 'mu_minus_ProbNNk', 'mu_minus_ProbNNe', 'mu_minus_ProbNNp',
'K_ProbNNk', 'K_ProbNNpi', 'K_ProbNNmu', 'K_ProbNNe', 'Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNmu', 'Pi_ProbNNe', 'Pi_ProbNNp', 'Pi_PT', 'Pi_IPCHI2_OWNPV',
'B0_ENDVERTEX_CHI2', 'B0_PT', 'Kstar_M', 'J_psi_M', 'B0_IPCHI2_OWNPV', 'B0_DIRA_OWNPV', 'B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_ENDVERTEX_X',
'B0_ENDVERTEX_Y', 'q2']
comb_columns_2 = ['mu_plus_ProbNNk', 'mu_plus_ProbNNmu', 'mu_plus_ProbNNe', 'mu_plus_ProbNNp', 'mu_minus_ProbNNk', 'mu_minus_ProbNNe', 'mu_minus_ProbNNp',
'K_ProbNNk', 'K_ProbNNpi', 'K_ProbNNmu', 'K_ProbNNe', 'Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNmu', 'Pi_ProbNNe', 'Pi_ProbNNp', 'Pi_PT', 'Pi_IPCHI2_OWNPV',
'B0_ENDVERTEX_CHI2', 'B0_PT', 'Kstar_M', 'J_psi_M', 'B0_IPCHI2_OWNPV', 'B0_DIRA_OWNPV', 'B0_OWNPV_X', 'B0_OWNPV_Y', 'B0_ENDVERTEX_X',
'B0_ENDVERTEX_Y', 'q2','identity']

peaking_columns = ['K_ProbNNk', 'K_ProbNNpi', 'K_ProbNNp', 'Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNp', 'Pi_PT', 'Pi_PX', 'Pi_IPCHI2_OWNPV', 'B0_M', 'Kstar_M', 'costhetal',
'costhetak']
peaking_columns_2 = ['K_ProbNNk', 'K_ProbNNpi', 'K_ProbNNp', 'Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNp', 'Pi_PT', 'Pi_PX', 'Pi_IPCHI2_OWNPV', 'B0_M', 'Kstar_M', 'costhetal',
'costhetak','identity']


def loadpeakingall(file,initialdata = pd.DataFrame(data=[])):
    train = initialdata
    for f in os.listdir(file):
        if not f.startswith('total') and not f.startswith('signal') and not f.startswith('acceptance') and not f.startswith('.DS') and not f.startswith('comb'):
            temp_df = pd.read_csv(file + '/' + f)
            temp_df['identity'] = 'peaking'
            train = pd.concat([train,temp_df])
    return train

def selection_criteria_q2(data):
    invarmass=data['q2']
    total_dataset_df_yes_peaking = data[(invarmass > 0.98) & (invarmass < 1.10) | (invarmass > 8.0) & (invarmass < 11.0) | (invarmass > 12.5) & (invarmass < 15.0)]
    data = data.merge(total_dataset_df_yes_peaking, how='left', indicator=True)
    data = data[data['_merge'] == 'left_only'] # line 16-17 from stackoverflow.com
    data = data.drop('_merge',axis=1)
    return data

def MLtrain(realdata, trainerdf):
    # training the ML

    # splitting independent and dependent variables
    sim_X = trainerdf.drop('identity', axis=1) 
    sim_Y = trainerdf['identity']

    # splitting data for training and validation
    train_X,test_X,train_Y,test_Y = model_selection.train_test_split(sim_X, sim_Y, test_size = 0.33, random_state = 1) 

    # model used for training
    model = lgbm.LGBMClassifier() 
    model.fit(train_X,train_Y) # actually training the model
    y_pred = model.predict(test_X) # predict results for test_Y, later on compare with actual test_Y
    accuracy = metrics.accuracy_score(test_Y,y_pred) # accuracy score, self explanatory
    print('Accuracy Score = ' +str(accuracy))

    # apply to real dataset
    iden = model.predict(realdata)

    return iden

def b0plots(df,densities=False):
    # separate identity
    sig = df[(df['identity'] == 'signal')]
    peak = df[(df['identity'] == 'peaking')]
    com = df[(df['identity'] == 'combinatorial')]
    print('Number of signal candidate = ' +str(len(sig)))
    
    # plots
    numbin=100
    plt.figure()
    plt.xlabel('B0_M')
    plt.ylabel('Count')
    # histograms
    plt.hist(df['B0_M'],color='royalblue',label='Total', histtype='step',bins=numbin,density=densities) # total data
    plt.hist(sig['B0_M'],color='red',label='ML Predicted Signal', histtype='step',bins=numbin,density=densities) # ML Predicted Signal on total
    plt.hist(com['B0_M'],color='cyan',label='ML Predicted Combinatorial', histtype='step',bins=numbin,density=densities) # ML Predicted combinatorial background on total
    plt.hist(peak['B0_M'],color='green',label='ML Predicted Peaking',histtype='step',bins=numbin,density=densities) # ML Predicted peaking background on total
    plt.legend()
    plt.show()


if __name__ == '__main__' :
    # first ML for peaking
    total_df = pd.read_csv(filepath + '/total_dataset.csv')
    total_df = selection_criteria_q2(total_df)
    signal_df = pd.read_csv(filepath + '/signal.csv')
    signal_df['identity'] = 'signal'
    peaking_df = loadpeakingall(filepath,signal_df)
    peaking_df = peaking_df[peaking_columns_2]
    total_no_peak_df = total_df[peaking_columns]
    total_df['identity'] = MLtrain(total_no_peak_df, peaking_df)
    
    # split peaking and non peaking
    peaking = total_df[(total_df['identity'] == 'peaking')]
    signalcomb = total_df[(total_df['identity'] != 'peaking')]
    # signalcomb = signalcomb.drop('identity',axis=1)
    
    # second ML for comb
    comb_df = total_df[(total_df['B0_M'] > 5350.0)]
    comb_df['identity'] = 'combinatorial'
    signalcombtrain = pd.concat([comb_df, signal_df])
    signalcombtrain = signalcombtrain[comb_columns_2]
    signalcomb_filter = signalcomb[comb_columns]
    signalcomb['identity'] = MLtrain(signalcomb_filter,signalcombtrain)
    
    # split comb and non comb
    combinatorial = signalcomb[(signalcomb['identity'] == 'combinatorial')]
    signalpure = signalcomb[(signalcomb['identity'] != 'combinatorial')]
    total_df_classified = pd.concat([signalpure,combinatorial,peaking])
    # total_df_classified.to_csv(filepath + '/total_ml4_1.csv')
    # total_df_classified = selection_criteria_q2(total_df_classified)
    # print(total_df_classified)
    
    # plots
    b0plots(total_df_classified)













