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
import training_model_peaking as TMP
from sklearn.metrics import roc_curve, roc_auc_score

'''This Script will plot the ROC curves for the peaking model using different amount of features. (Code for ROC curves from Trinity's code)
    The Background is trained separately.
'''

'''THIS FILE WILL REWRITE THE MODELS SO MAKE SURE TO RERUN THE TRAINING SCRIPTS AFTERWARDS'''

#total_data = pd.read_csv('../data/total_dataset.csv')
signal = pd.read_csv('../Data/signal.csv')
jpsi_mu_k_swap = pd.read_csv('../Data/jpsi_mu_k_swap.csv')
jpsi_mu_pi_swap = pd.read_csv('../Data/jpsi_mu_pi_swap.csv')
k_pi_swap = pd.read_csv('../Data/k_pi_swap.csv')
phimumu = pd.read_csv('../Data/phimumu.csv')
pKmumu_piTok_kTop = pd.read_csv('../Data/pKmumu_piTok_kTop.csv')
pKmumu_piTop = pd.read_csv('../Data/pKmumu_piTop.csv')
Kmumu = pd.read_csv('../Data/Kmumu.csv')
Kstarp_pi0 = pd.read_csv('../Data/Kstarp_pi0.csv')
#Jpsi_Kstarp_pi0 = pd.read_csv('../Data/Jpsi_Kstarp_pi0.csv')



def prepare_data(feature_amount,background,background_name):
    #background_models = [jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap,phimumu,pKmumu_piTok_kTop,pKmumu_piTop,Kmumu,Kstarp_pi0]
    #string_background_models = ['jpsi_mu_k_swap','jpsi_mu_pi_swap','k_pi_swap','phimumu','pKmumu_piTok_kTop','pKmumu_piTop','Kmumu','Kstarp_pi0','Jpsi_Kstarp_pi0']
    signal.loc[:, "target"] = 1
    background.loc[:, "target"] = 0
    signal_q_range = q2_ranges(signal)
    background_q_range = q2_ranges(background)
    remove_columns_array = high_corrolation_list(feature_amount,background_name)
    signal_clean = TMP.remove_columns(signal_q_range, remove_columns_array)
    background_clean = TMP.remove_columns(background_q_range, remove_columns_array)

    combine = pd.concat((signal_clean,background_clean), ignore_index=True, axis=0)
    #We then shuffle the order and relabel the index.
    combine = combine.sample(frac=1)
    combine = combine.reset_index(drop=True)
    print(combine.columns.tolist())
    #CREATING THE TRAINING AND TEST DATA
    seed = 42
    test_size = 0.2
    y = combine['target']
    X = combine.drop(columns=['target'])
    #Creating the three different data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify = y)

    return X_train, X_test, y_train, y_test

def high_corrolation_list(num,name_of_background):
    corrolation = pd.read_csv('Peaking_Separate_Correlation/continuous_f1_score_{}.csv'.format(name_of_background))
    cols = corrolation.columns.tolist()
    values = {}
    for x in cols:
        values[x] = float(corrolation[x])

    values = dict(sorted(values.items(), key=lambda item: item[1], reverse=True))
    array_list = []
    for idx, key in enumerate(values):
        if(idx > num-1):
            array_list.append(key)
    return array_list




if __name__ == '__main__':
    #background_models = [jpsi_mu_k_swap,jpsi_mu_pi_swap,k_pi_swap,phimumu,pKmumu_piTok_kTop,pKmumu_piTop,Kmumu,Kstarp_pi0]
    background = 'Kstarp_pi0'
    dataset = Kstarp_pi0
    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - Boosted Descision Tree ({})'.format(background))
    
    for feature_amount in range(1,10,1):
        #print('Training Combinatorial Model with {} features.'.format(feature_amount))
        X_train, X_test, y_train, y_test = prepare_data(feature_amount,dataset,background)
        model = TMP.train_model(X_train,y_train,X_test,y_test,background)
        y_score = model.predict_proba(X_test)[:,1]
        false_positive_rate, true_positive_rate, threshold = roc_curve(y_test, y_score)
        plt.plot(false_positive_rate, true_positive_rate, label = "{} features. roc_auc_score - {}".format(feature_amount,roc_auc_score(y_test, y_score)))
        print('(Peaking Model) roc_auc_score for Boosted Decision Tree: {} with {} features'.format(roc_auc_score(y_test, y_score),feature_amount))

    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig("ROC_CURVES_PEAKING_SEPARATE/ROC_CURVES_PEAKING_MODEL_SEPARATE_{}.jpg".format(background), bbox_inches="tight", pad_inches=0, format="jpg", dpi=600)
    plt.show()
