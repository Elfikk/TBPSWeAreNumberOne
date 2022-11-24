#Imports

from copy import deepcopy
from pandas import DataFrame
import scipy
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, balanced_accuracy_score, roc_auc_score, make_scorer,log_loss
from sklearn.model_selection import GridSearchCV
from matplotlib.ticker import StrMethodFormatter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from scipy.optimize import curve_fit
from selection_criteria import apply_all_selection
from selection_criteria import q2_ranges

total_data = pd.read_csv('../data/total_dataset.csv')
signal = pd.read_csv('../Data/signal.csv')
jpsi_mu_k_swap = pd.read_csv('../Data/jpsi_mu_k_swap.csv')
jpsi_mu_pi_swap = pd.read_csv('../Data/jpsi_mu_pi_swap.csv')
k_pi_swap = pd.read_csv('../Data/k_pi_swap.csv')
phimumu = pd.read_csv('../Data/phimumu.csv')
pKmumu_piTok_kTop = pd.read_csv('../Data/pKmumu_piTok_kTop.csv')
pKmumu_piTop = pd.read_csv('../Data/pKmumu_piTop.csv')
Kmumu = pd.read_csv('../Data/Kmumu.csv')
Kstarp_pi0 = pd.read_csv('../Data/Kstarp_pi0.csv')
Jpsi_Kstarp_pi0 = pd.read_csv('../Data/Jpsi_Kstarp_pi0.csv')

'''
Script used to train a model to separate signal from background. The background used is a combined dataframe of all simulated backgrounds.
'''

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

def train_model(X_train,y_train,X_test,y_test):
    model = xgb.XGBClassifier('binary:logistic', missing = np.nan, seed = 42)

    model.fit(X_train,
            y_train,
            verbose = True,
            early_stopping_rounds = 10,
            eval_metric = 'logloss',
            eval_set = [(X_test,y_test)]
            )

    model.save_model('peaking_model_trained_together_removed_q2_range.model')

    return model

def high_corrolation_list(num):
    corrolation = pd.read_csv('continuous_f1_score_peaking_together.csv')
    cols = corrolation.columns.tolist()
    values = {}
    for x in cols:
        values[x] = float(corrolation[x])

    values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    array_list = []
    for idx, key in enumerate(values):
        if(idx <= num):
            array_list.append(key)
        else:
            break
    return array_list

if __name__ == '__main__':
    background_models = [jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap,phimumu,pKmumu_piTok_kTop,pKmumu_piTop,Kmumu,Kstarp_pi0,Jpsi_Kstarp_pi0]
    string_background_models = ['jpsi_mu_k_swap','jpsi_mu_pi_swap','k_pi_swap','phimumu','pKmumu_piTok_kTop','pKmumu_piTop','Kmumu','Kstarp_pi0','Jpsi_Kstarp_pi0']
    signal.loc[:, "target"] = 1
    remove_columns_array = high_corrolation_list(25)
    signal_clean = remove_columns(signal, ['B0_M', 'J_psi_M', 'q2','Kstar_M'])
    combine = signal_clean
    for idx,x in enumerate(background_models):
        x.loc[:, "target"] = 0

        x = q2_ranges(x)

        background_clean = remove_columns(x, ['B0_M', 'J_psi_M', 'q2','Kstar_M'])

        #We then combine the entire dataset.
        combine = pd.concat((combine,background_clean), ignore_index=True, axis=0)

    #We then shuffle the order and relabel the index.
    combine = combine.sample(frac=1)
    combine = combine.reset_index(drop=True)

    #CREATING THE TRAINING AND TEST DATA
    seed = 42
    test_size = 0.2
    y = combine['target']
    X = combine.drop(columns=['target'])
    #Creating the three different data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify = y)

    model = train_model(X_train,y_train,X_test,y_test)
