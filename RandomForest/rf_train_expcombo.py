from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import os
import pandas as pd
import joblib
from preprocessing import comb_total_training
import matplotlib.pyplot as plt
import numpy as np
from selection_criteria import apply_all_selection

#Training combinatorial background using by using dataset for >5350 mass units.

model_name = "expcombo_default_q2.joblib"
cwd_files = os.listdir()

training_set = comb_total_training()

training_set = apply_all_selection(training_set)

X, y = training_set.drop(columns=["Signal"]), training_set["Signal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, \
        random_state = 42, stratify = y)

#Gotta drop B0_M as the forest will easily find the artificial B0 mass 
#(trash in, trash out), giving artificially good results (and trash ones
# in lower mass range)
X_train_mass = X_train["B0_M"]
X_train = X_train.drop(columns = "B0_M")
X_test_mass = X_test["B0_M"]
X_test = X_test.drop(columns = "B0_M")

if model_name in cwd_files:
    forest = joblib.load(model_name)
else:
    forest = RandomForestClassifier(verbose=True)
    forest.fit(X_train, y_train)
    joblib.dump(forest, model_name)

print("Accuracy:", forest.score(X_test, y_test))

predictions = forest.predict(X_test)

print("F1-Score:", f1_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels = forest.classes_)

disp = ConfusionMatrixDisplay(cm, display_labels = forest.classes_)

# disp.plot()
# plt.show()

total_masses = X_test_mass
X_test["B0_M"] = X_test_mass
test_sig = X_test.loc[y_test == 1]
test_masses = test_sig["B0_M"]
predicted_sig = X_test.loc[predictions == 1]
predicted_masses = predicted_sig["B0_M"]
bins = list(np.linspace(5170, 5700, 100))

plt.hist(test_masses, bins = bins, histtype = "step", label = "Signal", density = True)
plt.hist(predicted_masses, bins= bins, histtype = "step", label = "Predicted", density = True)
# plt.hist(total_masses, bins= bins, histtype = "step", label = "Training Set", density = True)
plt.legend()
plt.grid()
plt.show()

