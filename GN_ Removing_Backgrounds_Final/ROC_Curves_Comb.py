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
import training_model_combo as TMC
from sklearn.metrics import roc_curve, roc_auc_score

THRESHOLD = 5350

'''This Script will plot the ROC curves for the comb model using different amount of features. (Code for ROC curves from Trinity's code)'''

'''THIS FILE WILL REWRITE THE MODELS SO MAKE SURE TO RERUN THE TRAINING SCRIPTS AFTERWARDS'''

def prepare_data():
    total_data = pd.read_csv('../data/total_dataset.csv')
    signal = pd.read_csv('../Data/signal.csv')

    comb_data = TMC.extract_combo_data(total_data,THRESHOLD)
    signal.loc[:, "target"] = 1
    comb_data.loc[:, "target"] = 0
    signal = q2_ranges(signal)
    comb_data = q2_ranges(comb_data)

    return signal, comb_data

def high_corrolation_list(num):
    corrolation = pd.read_csv('Comb_Correlation/continuous_f1_score_comb.csv')
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

def training_data(feature_amount):
    remove_columns_array = high_corrolation_list(feature_amount)
    remove_columns_array.append('B0_M')
    signal_clean = TMC.remove_columns(signal, remove_columns_array)
    combo_data_clean = TMC.remove_columns(comb, remove_columns_array)

    #We then combine the entire dataset.
    combine = pd.concat((signal_clean,combo_data_clean), ignore_index=True, axis=0)
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


if __name__ == '__main__':
    signal, comb = prepare_data()
    plt.subplots(1, figsize=(10,10))
    plt.title('Receiver Operating Characteristic - Boosted Descision Tree')
    
    for feature_amount in range(1,40,1):
        #print('Training Combinatorial Model with {} features.'.format(feature_amount))
        X_train, X_test, y_train, y_test = training_data(feature_amount)
        model = TMC.train_model(X_train,y_train,X_test,y_test)

        y_score = model.predict_proba(X_test)[:,1]
        false_positive_rate, true_positive_rate, threshold3 = roc_curve(y_test, y_score)
        plt.plot(false_positive_rate, true_positive_rate, label = "{} features. roc_auc_score - {}".format(feature_amount,roc_auc_score(y_test, y_score)))
        print('(Combinatorial Model) roc_auc_score for Boosted Decision Tree: {} with {} features'.format(roc_auc_score(y_test, y_score),feature_amount))

    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig("ROC_CURVES_COMB/ROC_CURVES_COMB_MODEL_FEATURE_AMOUNT.jpg", bbox_inches="tight", pad_inches=0, format="jpg", dpi=600)
    plt.show()
