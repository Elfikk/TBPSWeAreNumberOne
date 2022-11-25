import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt

from rf_mass_plots import apply_sep_models
from selection_criteria import apply_all_selection
from preprocessing import column_remove

# file_name = "acceptance_mc.csv"
# peaking_folder = "SeparateModels"
# combo_model = "ExpCombo/expcombo_default_q2.joblib"

# data_path = os.getcwd()[:-len("RandomForest")] + "data/" 
# data = pd.read_csv(data_path + file_name)#, nrows = 100)
# data = column_remove(data)

# signal, background = apply_sep_models(data, peaking_folder)
# combo = joblib.load(combo_model)

# combo_back_pred = combo.predict(signal.drop(columns = "B0_M"))

# combo_data = signal.loc[combo_back_pred == 0]
# signal = signal.loc[combo_back_pred == 1]
# peaking_data = background

# signal = apply_all_selection(signal)

# signal.to_csv("FilteredAcceptanceDataBroken.csv")

signal = pd.read_csv("ML_SIGNAL_BDT(1).csv")

bins = ((.1, .98), (1.1, 2.5), (2.5, 4.), (4., 6.), (6., 8.), (15., 17.), \
        (17., 19.), (11., 12.5), (1., 6.), (15., 17.9))
freqs = [0 for i in range(0, len(bins))]
bin_nums = list(range(0, len(bins)))

q2 = signal["q2"]
# print(q2.head())
for i in range(len(q2)):
    for j in range(len(bins)):
        bin = bins[j]
        q2_i = q2.iloc[i]
        if bin[0] < q2_i and q2_i < bin[1]:
            freqs[j] = freqs[j] + 1
        # print(q2_i, freqs)

plt.grid()
plt.plot(bin_nums, freqs, "bo")
plt.show()