from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from preprocessing import comb_training, apply_q2_ranges

import matplotlib.pyplot as plt
import numpy as np

import joblib
import os

# Training for combo from given simulated background + signal files. 
# Classifies event as either 0 (not combo) or 1 (combo background event).
# This is a pain to work with so will be refactored in a later code revision.

#Files in the current directory. Want to check if model has been trained
#(retraining when I have the wanted the model is in fact, not that useful) 
cwd_files = os.listdir()

model_name = "comb_default_with_q2.joblib" 

#Get preprocessed data.
data = apply_q2_ranges(comb_training())

X, y = data.drop(columns=["Comb"]), data["Comb"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, \
        random_state = 42, stratify = y)

if model_name in cwd_files:
    forest = joblib.load(model_name)
    print("hello world")

else:
    #Initialise model. Verbose set to true to have an output on
    #training progress.
    forest = RandomForestClassifier(verbose=True)

    forest.fit(X_train, y_train)

    #Saves the model to a file called model_name - to not have 
    #to retrain every single time we run.
    joblib.dump(forest, model_name)

print("Accuracy:", forest.score(X_test, y_test))

predictions = forest.predict(X_test)

print("F1-Score:", f1_score(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels = forest.classes_)

disp = ConfusionMatrixDisplay(cm, display_labels = forest.classes_)

# disp.plot()

test_combo = X_test.loc[y_test == 1]
test_masses = test_combo["B0_M"]
predicted_combo = X_test.loc[predictions == 1]
predicted_masses = predicted_combo["B0_M"]
bins = list(np.linspace(5170, 5700, 100))

plt.hist(test_masses, bins = bins, histtype = "step", label = "Combo")
plt.hist(predicted_masses, bins= bins, histtype = "step", label = "Predicted")
plt.legend()
plt.grid()
plt.show()

