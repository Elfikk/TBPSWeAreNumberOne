import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import total_dataset_load, apply_q2_ranges, training_data
from selection_criteria import apply_all_selection

def rf_on_total_dataset(model_name, apply_q2 = True):
    #Plots the results of predictions of given Random Forest when applied to 
    #the real dataset. Intended for a background/signal classifier.

    forest = joblib.load(model_name)

    if apply_q2:
        total_df = apply_q2_ranges(total_dataset_load())
    else:
        total_df = total_dataset_load()

    predictions = forest.predict(total_df)
    
    signal = total_df.loc[predictions == 1]
    masses = signal["B0_M"]
    bins = list(np.linspace(5170, 5700, 100))

    plt.hist(masses, bins = bins, histtype = "step")
    plt.grid()
    plt.xlabel("B_0 Mass")
    plt.ylabel("Frequency")
    plt.show()

def rf_on_training_set(model_name, apply_q2 = True):
    #Plots the results of predictions of given Random Forest when 
    #applied to the training set.
    #model_name: str, Relative path to cwd
    #apply_q2: bool, Whether selection criteria with q^2 should be applied.
    #          True by default. 

    forest = joblib.load(model_name)

    if apply_q2:
        total_df = apply_q2_ranges(training_data())
    else:
        total_df = training_data()

    labels = total_df["Signal"]
    sim_signal = total_df.loc[labels == 1]

    total_df = total_df.drop(columns = ["Signal"])

    predictions = forest.predict(total_df)

    signal = total_df.loc[predictions == 1]

    masses = signal["B0_M"]
    no_filtering = total_df["B0_M"]
    mass_sim = sim_signal["B0_M"]

    bins = list(np.linspace(5170, 5700, 100))

    plt.hist(no_filtering, bins = bins, histtype = "step", label = "Raw Data")
    plt.hist(mass_sim, bins = bins, histtype = "step", label = "Simulated Signal Data")
    plt.hist(masses, bins = bins, histtype = "step", label = "ML Signal Prediction")
    plt.grid()
    plt.xlabel("B_0 Mass")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def rf_separate_models_plot(peaking_model, combo_model, dataset = "total",
                            apply_q2 = True):
    #Plots for separate peaking and combinatorial models. Massive pain with 
    #revisions due to 0/1 conventions being switched in some training.

    if dataset == "total":
        data = total_dataset_load()
    else:
        data = training_data().drop(columns=["Signal"])

    if apply_q2:
        data = apply_q2_ranges(data)

    total_mass = data["B0_M"]

    peaking = joblib.load(peaking_model)
    combo = joblib.load(combo_model)

    is_peaking = peaking.predict(data)    
    is_combo = combo.predict(data)

    signal = data.loc[(is_peaking == 0) & (is_combo == 1)]
    peaking_data = data.loc[is_peaking == 1]
    combo_data = data.loc[is_combo == 0]

    sig_mass = signal["B0_M"]
    peak_mass = peaking_data["B0_M"]
    comb_mass = combo_data["B0_M"]

    bins = list(np.linspace(5170, 5700, 100))

    plt.hist(total_mass, bins = bins, histtype = "step", label = "Raw Data")
    plt.hist(sig_mass, bins = bins, histtype = "step", label = "Predicted Signal")
    plt.hist(peak_mass, bins = bins, histtype = "step", label = "Predicted Peaking")
    plt.hist(comb_mass, bins = bins, histtype = "step", label = "Predicted Combinatorial")
    plt.grid()
    plt.xlabel("B_0 Mass")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def sep_models(model_folder = "SeparateModels"):

    #Plots for separate models, trained on each individual background dataset.
    #model_folder is string of path relative to current working directly.

    data = apply_q2_ranges(total_dataset_load())

    total_mass = data["B0_M"]    

    signal, background = apply_sep_models(data, model_folder)

    signal_mass = signal["B0_M"]
    back_mass = background["B0_M"]

    bins = list(np.linspace(5170, 5700, 100))   

    plt.hist(total_mass, bins = bins, histtype = "step", label = "Raw Data")#, density=True)
    plt.hist(signal_mass, bins = bins, histtype = "step", label = "Predicted Signal")#, density=True)
    plt.hist(back_mass, bins = bins, histtype = "step", label = "Predicted Background")#, density=True)
    plt.grid()
    plt.xlabel("B_0 Mass")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def apply_sep_models(data, model_folder = "SeparateModels"):

    #Applies separate models to passed data dataframe, given the path of the
    #folder holding the individual background models.

    model_dir = os.getcwd() + "/" + model_folder

    signal = data.copy()
    background = pd.DataFrame(columns = data.columns)

    for model in os.listdir(model_dir):

        print(model)

        forest = joblib.load(model_folder + "/" + model)

        predictions = forest.predict(signal)

        new_background = signal.loc[predictions == 0]
        signal = signal.loc[predictions == 1]

        background = pd.concat((new_background, background), ignore_index=True)

    return signal, background

def rf_exp_combo(peaking_model, combo_model, dataset = "total",
                            apply_q2 = True):

    # Plot for peaking model with exponential combo model (separate to peaking
    # vs combo due to needing to drop B0_M for exponential model)

    if dataset == "total":
        data = total_dataset_load()
    else:
        data = training_data().drop(columns=["Signal"])

    if apply_q2:
        data = apply_q2_ranges(data)

    total_mass = data["B0_M"]

    peaking = joblib.load(peaking_model)
    combo = joblib.load(combo_model)

    is_peaking = peaking.predict(data)    
    is_combo = combo.predict(data.drop(columns="B0_M"))

    signal = data.loc[(is_peaking == 0) & (is_combo == 1)]
    peaking_data = data.loc[is_peaking == 1]
    combo_data = data.loc[is_combo == 0]

    sig_mass = signal["B0_M"]
    peak_mass = peaking_data["B0_M"]
    comb_mass = combo_data["B0_M"]

    bins = list(np.linspace(5170, 5700, 100))

    plt.hist(total_mass, bins = bins, histtype = "step", label = "Raw Data", density= True)
    plt.hist(sig_mass, bins = bins, histtype = "step", label = "Predicted Signal", density= True)
    plt.hist(peak_mass, bins = bins, histtype = "step", label = "Predicted Peaking", density= True)
    plt.hist(comb_mass, bins = bins, histtype = "step", label = "Predicted Combinatorial", density= True)
    plt.grid()
    plt.xlabel("B_0 Mass")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def rf_exp_combo_sep(combo_model, peaking_folder = "SeparateModels", \
                     dataset = "total", apply_q2 = True):

    # Exponential model plots, but with separate peaking models applied on dataset.

    if dataset == "total":
        data = total_dataset_load()
    else:
        data = training_data().drop(columns=["Signal"])

    if apply_q2:
        data = apply_q2_ranges(data)

    total_mass = data["B0_M"]
    # data = apply_all_selection(data)

    signal, background = apply_sep_models(data, peaking_folder)
    combo = joblib.load(combo_model)

    # signal_masses = signal["B0_M"]

    combo_back_pred = combo.predict(signal.drop(columns = "B0_M"))

    combo_data = signal.loc[combo_back_pred == 0]
    signal = signal.loc[combo_back_pred == 1]
    peaking_data = background

    signal = apply_all_selection(signal)

    sig_mass = signal["B0_M"]
    peak_mass = peaking_data["B0_M"]
    comb_mass = combo_data["B0_M"]

    bins = list(np.linspace(5170, 5700, 100))

    plt.hist(total_mass, bins = bins, histtype = "step", label = "Raw Data", density= True)
    plt.hist(sig_mass, bins = bins, histtype = "step", label = "Predicted Signal", density= True)
    plt.hist(peak_mass, bins = bins, histtype = "step", label = "Predicted Peaking", density= True)
    plt.hist(comb_mass, bins = bins, histtype = "step", label = "Predicted Combinatorial", density= True)
    plt.grid()
    plt.xlabel("B_0 Mass")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    # model_name = "expcombo_default_q2.joblib" 
    # rf_on_total_dataset(model_name)

    # rf_on_training_set(model_name)

    # comb_model = "comb_default_with_q2.joblib"
    comb_model = "expcombo_default_q2.joblib"
    # peak_model = "peaking_default_with_q2.joblib"

    # rf_exp_combo(peak_model, comb_model)

    # sep_models("Test")
    rf_exp_combo_sep(comb_model)

