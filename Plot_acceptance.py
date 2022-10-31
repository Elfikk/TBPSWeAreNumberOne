import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Data/acceptance_mc.csv")
df1 = pd.read_csv("Data/total_dataset.csv")


plt.hist(df['phi'], bins=100, histtype='step',label = 'acceptance.csv',density=True)  
plt.hist(df1['phi'], bins=100, histtype='step',label='total_dataset.csv',density=True)  
plt.xlabel('$\phi$')
plt.legend()
plt.show()

plt.hist(df['costhetak'], bins=100, histtype='step',label = 'acceptance.csv',density=True)  
plt.hist(df1['costhetak'], bins=100, histtype='step',label='total_dataset.csv',density=True)  
plt.xlabel('$cos\\theta_{k}$')
plt.legend()
plt.show()

plt.hist(df['costhetal'], bins=100, histtype='step',label = 'acceptance.csv',density=True)  
plt.hist(df1['costhetal'], bins=100, histtype='step',label='total_dataset.csv',density=True)  
plt.xlabel('$cos\\theta_{l}$')
plt.legend()
plt.show()

plt.hist(df['q2'], bins=100, histtype='step',label = 'acceptance.csv',density=True)  
#plt.hist(df1['q2'], bins=100, histtype='step',label='total_dataset.csv',density=True)  
plt.xlabel('$q^{2}$')
plt.legend()
plt.show()