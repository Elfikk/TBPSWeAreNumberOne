from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import f1_score
import os
import pandas as pd
import joblib
from preprocessing import apply_q2_ranges

#Training signal vs background on each background file provided.

files_to_read = [
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

data_path = os.getcwd()[:-len("RandomForest")] + "data/"

signal_path = data_path + "signal.csv"

signal = pd.read_csv(signal_path)

for column in columns_to_remove:
    if column in signal.columns:
        signal = signal.drop(columns = column)

signal["Signal"] = 1

for file in files_to_read:

    model_name = "default_q2_" + file[:-4] + ".joblib"

    background = pd.read_csv(data_path + file)

    for column in columns_to_remove:
        if column in background.columns:
            background = background.drop(columns = column)

    background["Signal"] = 0

    data = pd.concat((signal, background), ignore_index=True)

    data = apply_q2_ranges(data)

    X, y = data.drop(columns=["Signal"]), data["Signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, \
            random_state = 42, stratify = y)

    forest = RandomForestClassifier(verbose=True)

    forest.fit(X_train, y_train)

    joblib.dump(forest, model_name)

    accuracy = forest.score(X_test, y_test)
    predictions = forest.predict(X_test)
    f1 = f1_score(y_test, predictions)

    with open("scores.txt", "a") as f:
        f.write(model_name + "," + str(accuracy) + "," + str(f1))
        f.write("\n")