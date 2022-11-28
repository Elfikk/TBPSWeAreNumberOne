import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plots the top 10 most important features in the random forest model over
# the averages in each individual tree.

model_name = "peaking_default_with_q2.joblib"

forest = joblib.load(model_name)

#feature importances in forest are saved; these are already an average over
#individual trees, so no need to loop through.
importances = forest.feature_importances_ 
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

forest_importances = pd.Series(importances, index=forest.feature_names_in_).nlargest(10)
to_keep = forest_importances.index
std_series = pd.Series(std, index=forest.feature_names_in_)
new_std = []
for index in std_series.index:
    if index in to_keep:
        new_std.append(std_series[index])

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=new_std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.grid()
plt.show()