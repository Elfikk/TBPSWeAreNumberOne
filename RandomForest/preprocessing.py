import pandas as pd
import os
import matplotlib.pyplot as plt

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
        # 'comb.csv',
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
    #Applies selection criteria.
    
    #Ranges are
    #0.98 < q^2 < 1.1, 8 < q^2 < 11, 12.5 < q^2 < 15

    q2 = combined_df["q2"]

    selected_df = combined_df[(0 < q2) & (q2 < 0.98) | (1.1 < q2) & (q2 < 8) | (11 < q2) & (q2 < 12.5)| (q2 > 15)]

    # selected_df = combined_df.drop(combined_df.loc[to_exclude_df])

    return selected_df

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