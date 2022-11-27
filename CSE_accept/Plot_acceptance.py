import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_accept = pd.read_csv("Data/acceptance_mc.csv")
#df1 = pd.read_csv("Data/total_dataset.csv")

#set your bins range
#0  0.1 0.98
#1  1.1 2.5
#2  2.5 4.0
#3  4.0 6.0
#4  6.0 8.0
#5  15.0 17.0
#6  17.0 19.0
#7  11.0 12.5
#8  1.0 6.0
#9  15.0 17.9

q2_min = 0.1
q2_max = 0.98

#filter for specific bin range
df_accept = df_accept[df_accept['q2']<q2_max]
df_accept = df_accept[df_accept['q2']>q2_min]

plt.hist(df_accept['q2'],bins=50)
plt.xlabel('$q^2$')
plt.title('$q^2$')
plt.savefig('Data/plots/q2.png',dpi=600)
plt.show()

'''
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
plt.show()'''