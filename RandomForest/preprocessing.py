import pandas as pd
import os
import matplotlib.pyplot as plt
import standard_removing as SR

#TL;DR Functions for loading datafiles, with general niceness i.e column removal.

def training_data():
    # Loading and preprocessing of data relevant to training Random Forest.
    
    #Expect a folder call "data" in the directory above - case sensitive!
    data_path = os.getcwd()[:-len("RandomForest")] + "data/"

    #Trash columns which should not be included in analysis.
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

    #Comment out the files you dont want to use.
    files_to_read = [
        # 'acceptance_mc.csv', 
        'comb.csv',
        'jpsi.csv',
        'Jpsi_Kstarp_pi0.csv',
        'jpsi_mu_k_swap.csv',
        'jpsi_mu_pi_swap.csv',
        'Kmumu.csv',
        'Kstarp_pi0.csv',
        'k_pi_swap.csv',
        'phimumu.csv',
        'pKmumu_piTok_kTop.csv',
        'pKmumu_piTop.csv',
        'psi2S.csv', 
        'signal.csv', 
        # 'total_dataset.csv'
    ]

    # Dataframes temporarily stored in a dictionary, with the keys being
    # the file name.
    dataframes = {}

    for file in files_to_read:
        
        path = data_path + file

        df = pd.read_csv(path)#, nrows = 100)

        for column in columns_to_remove:
            if column in df.columns:
                df = df.drop(columns = column)

        dataframes[file] = df

        # print(len(df.columns))

    #Signal column represents whether the data point is a signal (1) 
    #or a background (0) event.
    for file in dataframes:
        dataframes[file]["Signal"] = 0
        # print(dataframes[file].head(5))    
    dataframes["signal.csv"]["Signal"] = 1
    # print(dataframes["signal.csv"].head(5)) 
    
    combined_df = pd.concat(list(dataframes.values()), ignore_index=True)

    return combined_df

def total_dataset_load(nrows = None):
    #Loads the total dataset, dropping irrelevant columns.

    data_path = os.getcwd()[:-len("RandomForest")] + "data/"
    total_dataset_path = data_path + "total_dataset.csv"    

    if nrows:
        total_df = pd.read_csv(total_dataset_path, nrows = nrows)
    else:
        total_df = pd.read_csv(total_dataset_path)

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

    for column in columns_to_remove:
        if column in total_df.columns:
            total_df = total_df.drop(columns = column)

    return total_df

def apply_q2_ranges(combined_df):
    #Applies q^2 selection criteria.
    
    #Ranges are
    #0.98 < q^2 < 1.1, 8 < q^2 < 11, 12.5 < q^2 < 15

    q2 = combined_df["q2"]

    selected_df = combined_df[(0 < q2) & (q2 < 0.98) | (1.1 < q2) & (q2 < 8) | \
        (11 < q2) & (q2 < 12.5)| (q2 > 15)]

    return selected_df

def peaking_training():

    #Expect a folder call "data" in the directory above - case sensitive!
    data_path = os.getcwd()[:-len("RandomForest")] + "data/"

    #Trash columns which should not be included in analysis.
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

    #Comment out the files you dont want to use.
    files_to_read = [
        # 'acceptance_mc.csv', 
        'comb.csv',
        'jpsi.csv',
        'Jpsi_Kstarp_pi0.csv',
        'jpsi_mu_k_swap.csv',
        'jpsi_mu_pi_swap.csv',
        'Kmumu.csv',
        'Kstarp_pi0.csv',
        'k_pi_swap.csv',
        'phimumu.csv',
        'pKmumu_piTok_kTop.csv',
        'pKmumu_piTop.csv',
        'psi2S.csv', 
        'signal.csv', 
        # 'total_dataset.csv'
    ]

    peaking_files = [
        'jpsi.csv',
        'Jpsi_Kstarp_pi0.csv',
        'jpsi_mu_k_swap.csv',
        'jpsi_mu_pi_swap.csv',
        'Kmumu.csv',
        'Kstarp_pi0.csv',
        'k_pi_swap.csv',
        'phimumu.csv',
        'pKmumu_piTok_kTop.csv',
        'pKmumu_piTop.csv',
        'psi2S.csv', 
    ]

    # Dataframes temporarily stored in a dictionary, with the keys being
    # the file name.
    dataframes = {}

    for file in files_to_read:
        
        path = data_path + file

        df = pd.read_csv(path)#, nrows = 100)

        for column in columns_to_remove:
            if column in df.columns:
                df = df.drop(columns = column)

        dataframes[file] = df

    #Signal column represents whether the data point is a signal (1) 
    #or a background (0) event.
    for file in dataframes:
        if file in peaking_files:
            dataframes[file]["Peaking"] = 1
        else:
            dataframes[file]["Peaking"] = 0

    peaking_df = pd.concat(list(dataframes.values()), ignore_index=True)

    return peaking_df

def comb_training():
    # Classifies data as either combinatorial or not, giving a combined dataset 
    # from all the different backgrounds.

    #Expect a folder call "data" in the directory above - case sensitive!
    data_path = os.getcwd()[:-len("RandomForest")] + "data/"

    #Trash columns which should not be included in analysis.
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

    #Comment out the files you dont want to use.
    files_to_read = [
        # 'acceptance_mc.csv', 
        'comb.csv',
        'jpsi.csv',
        'Jpsi_Kstarp_pi0.csv',
        'jpsi_mu_k_swap.csv',
        'jpsi_mu_pi_swap.csv',
        'Kmumu.csv',
        'Kstarp_pi0.csv',
        'k_pi_swap.csv',
        'phimumu.csv',
        'pKmumu_piTok_kTop.csv',
        'pKmumu_piTop.csv',
        'psi2S.csv', 
        'signal.csv', 
        # 'total_dataset.csv'
    ]

    comb_files = [
        'comb.csv'
    ]

    # Dataframes temporarily stored in a dictionary, with the keys being
    # the file name.
    dataframes = {}

    for file in files_to_read:
        
        path = data_path + file

        df = pd.read_csv(path)#, nrows = 100)

        for column in columns_to_remove:
            if column in df.columns:
                df = df.drop(columns = column)

        dataframes[file] = df

    #Signal column represents whether the data point is a signal (1) 
    #or a background (0) event.
    for file in dataframes:
        if file in comb_files:
            dataframes[file]["Comb"] = 1
        else:
            dataframes[file]["Comb"] = 0

    comb_df = pd.concat(list(dataframes.values()), ignore_index=True)

    return comb_df

def column_remove(df, columns_to_remove = SR.columns_to_remove):
    # Removes passed columns from a dataframe. Use to remove redundant 
    # columns like year. columns_to_remove needs to be an iterable.

    for column in columns_to_remove:
        if column in df.columns:
            df = df.drop(columns = column)

    return df

def column_keep(df, columns_to_keep):
    # Removes all unpassed columns from a dataframe. Use to specify 
    # important columns for training.

    for column in df.columns:
        if column not in columns_to_keep:
            df = df.drop(columns = column)

    return df

def comb_total_training(apply_q2 = True, nrows = None):
    # Attempting to remove combinatorial background using B0_M > 5350 method, 
    # where we use the total dataset above a threshhold mass as our 

    data_path = os.getcwd()[:-len("RandomForest")] + "data/"

    total_set = total_dataset_load(nrows)
    if apply_q2:
        total_set = apply_q2_ranges(total_set)

    if nrows:
        signal_set = pd.read_csv(data_path + "signal.csv", nrows=nrows)
    else:
        signal_set = pd.read_csv(data_path + "signal.csv")
    if apply_q2:
        signal_set = apply_q2_ranges(signal_set)

    signal_set = column_remove(signal_set)

    B0_M = total_set["B0_M"]
    total_set = total_set.loc[B0_M > 5350]

    total_set["Signal"] = 0
    signal_set["Signal"] = 1

    training_set = pd.concat((total_set, signal_set))
    # training_set, masses = training_set.drop(columns = ["B0_M"]), training_set["B0_M"]

    return training_set#, masses

def general_data_load(datasets_background, datasets_signal, cols_to_drop, cols_to_keep,
                      extra_methods, apply_q2 = True):
    if cols_to_drop:
        column_method = column_remove
        columns = cols_to_drop
    elif cols_to_keep:
        column_method = column_keep
        columns = cols_to_keep

    dataframes = {}

    #To just have one file reading loop. This lines required Python 3.5
    #or greater!
    datasets = {**{dataset: 0 for dataset in datasets_background},\
                **{dataset: 1 for dataset in datasets_signal}}

    for file in datasets:
        df = pd.read_csv(file)
        if file in extra_methods:
            for method_list in extra_methods[file]:
                method = method_list[0]
                args = method_list[1:]
                df = method(df, *args)
        df = column_method(df, columns)
        df["Signal"] = datasets[file]
        dataframes[file] = df

    data = pd.concat(list(dataframes.values()), ignore_index=True)

    if apply_q2:
        data = apply_q2_ranges(data)

    return data

def mass_exclusion_above(df, *args):
    b0_m = df["B0_M"]
    return df.loc[b0_m > args[0]]

if __name__ == "__main__":

    # comb_df = training_data()

    # print(comb_df.shape)

    df = total_dataset_load(100000)

    print(df.shape)

    selected_df = apply_q2_ranges(df)

    print(selected_df.shape)

    #Plot as expected, we good.
    plt.plot(df["B0_M"], df["q2"], ".", label = "Base Data")
    plt.plot(selected_df["B0_M"], selected_df["q2"], ".", label = "q2 criteria applied")
    plt.legend()
    plt.grid()
    plt.show()