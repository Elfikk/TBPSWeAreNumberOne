from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
from preprocessing import general_data_load, mass_exclusion_above

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import joblib
import os

def rf_train(model_name, datasets_signal, datasets_background, train_ratio = 0.8, 
             cols_to_drop = None, cols_to_keep = None, apply_q2 = True,
             extra_methods = {}, seed = None, plot_cm = False):
    #Function for general training of a Random Forest model. Params:
    #       model_name(str/None): File name for generated model. If None, the
    #                             model will not be saved.
    #      datasets_signal(list): List of string paths to the files used as the
    #                             the signal in training.
    #  datasets_background(list): List of string paths of the files used as the
    #                             background in training.
    #         train_ratio(float): Proportion of data from datasets to be used
    #                             for training. Expect a value from 0 to 1.
    #         cols_to_drop(list): Columns to drop in every single file.
    #         cols_to_keep(list): Alternative to cols_to_drop. Either cols_to_drop
    #                             or cols_to_keep must be passed.
    #             apply_q2(bool): If True, q^2 criteria are applied to the dataset
    #                             prior to training.
    #        extra_methods(dict): Any methods to be applied onto specific files. 
    #                             Dict keys should be path strings, with a list of
    #                             funcs to be applied to the Pandas dataframe.
    #                  seed(any): Specifies "random" state for the train-test split.
    #                             If None, no preset seed will be used.
    #              plot_cm(bool): If True, displays the Confusion matrix after 
    #                             training.

    data = general_data_load(datasets_background, datasets_signal, cols_to_drop,\
                             cols_to_keep, extra_methods, apply_q2)

    X, y = data.drop(columns=["Signal"]), data["Signal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 
        train_ratio, random_state = seed, stratify = y)

    forest = RandomForestClassifier(verbose=True)
    forest.fit(X_train, y_train)
    
    if model_name:
        joblib.dump(forest, model_name)

    print("Accuracy:", forest.score(X_test, y_test))

    predictions = forest.predict(X_test)

    print("F1-Score:", f1_score(y_test, predictions))

    if plot_cm:

        cm = confusion_matrix(y_test, predictions, labels = forest.classes_)

        disp = ConfusionMatrixDisplay(cm, display_labels = forest.classes_)

        disp.plot()
        plt.show()

def rf_optimise(model_name, datasets_signal, datasets_background, train_ratio = 0.8,
                cols_to_drop = None, cols_to_keep = None, apply_q2 = True, 
                extra_methods = {}, seed = None, plot_cm = False, 
                grid_search = RandomizedSearchCV, grid_search_params = {}):
    #Same parameters as rf_train, with the exception of,
    #    grid_search(CV class): Specifies type of grid search cross-validator
    #                           used for optimising the model.
    # grid_search_params(dict): param_grid for the grid search.

    data = general_data_load(datasets_background, datasets_signal, cols_to_drop,\
                             cols_to_keep, extra_methods, apply_q2)

    X, y = data.drop(columns=["Signal"]), data["Signal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 
        train_ratio, random_state = seed, stratify = y)
    
    forest = RandomForestClassifier(verbose=True)

    gs = grid_search(forest, grid_search_params)
    gs.fit(X_train, y_train)

    print(pd.DataFrame(gs.cv_results_))
    print(gs.best_params_)
    
    bestest_forest = gs.best_estimator_

    print("Accuracy:", gs.score(X_test, y_test))

    predictions = gs.predict(X_test)

    print("F1-Score:", f1_score(y_test, predictions))

    if model_name:
        joblib.dump(bestest_forest, model_name)

    if plot_cm:

        cm = confusion_matrix(y_test, predictions, labels = gs.classes_)

        disp = ConfusionMatrixDisplay(cm, display_labels = gs.classes_)

        disp.plot()
        plt.show()

def comb_train_example():
    #Example function of training combinatorial background, with
    #optimisation, with changing the number of trees for estimation.

    model_name = "test.joblib"

    datasets_signal = ["D:/Projekty/Coding/Python/TBPSWeAreNumberOne/data/signal.csv"]
    datasets_background = ["D:/Projekty/Coding/Python/TBPSWeAreNumberOne/data/total_dataset.csv"]

    extra_methods = {datasets_background[0]: [[mass_exclusion_above, 5350]]}

    columns_to_remove = [
        "Unnamed: 0",
        "Unnamed: 0.1", 
        "Unnamed: 0.1.1",
        "year",
        'B0_ID',
        'B0_ENDVERTEX_NDOF',
        'J_psi_ENDVERTEX_NDOF',
        'Kstar_ENDVERTEX_NDOF' #Yoinked from Ganels
    ]

    params = {"n_estimators": [10, 20, 30]}

    rf_optimise(model_name, datasets_signal, datasets_background, 
    cols_to_drop= columns_to_remove, extra_methods=extra_methods,
    plot_cm=True, grid_search=GridSearchCV, grid_search_params=params)

if __name__ == "__main__":

    comb_train_example()
