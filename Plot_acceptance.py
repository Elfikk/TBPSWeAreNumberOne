import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Data/acceptance_mc.csv")
df1 = pd.read_csv("Data/total_dataset.csv")

plt.hist(df1['phi'], bins=100)
plt.show()
plt.hist(df1['costhetal'],bins = 100)
plt.show()
plt.hist(df1['costhetak'],bins=100)
plt.show()
plt.hist(df['q2'], bins=100)
plt.show()
plt.hist(df1['q2'], bins=50)
plt.show()