import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

from preprocessing import total_dataset_load, apply_q2_ranges, training_data

def rf_on_total_dataset(model_name, apply_q2 = True):
    #Plots the results of predictions of given Random Forest when applied to 
    #the real dataset.

    forest = joblib.load(model_name)

    if apply_q2:
        total_df = apply_q2_ranges(total_dataset_load())
    else:
        total_df = total_dataset_load()

    predictions = forest.predict(total_df)
    
    signal = total_df.loc[predictions == 1]
    masses = signal["B0_M"]

    plt.hist(masses, bins = 100)
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

if __name__ == "__main__":

    model_name = "default_with_q2.joblib" 

    rf_on_training_set(model_name)

