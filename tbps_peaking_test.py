import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

datafilepath = '/Users/gordonlai/Documents/ICL/ICL_Y3/TBPSWeAreNumberOne/Data/' #set your own file path
total_dataset_df = pd.read_csv(datafilepath + 'total_dataset.csv') #load dataset
print(total_dataset_df.columns)

# assign columns to value
b0mass=total_dataset_df['B0_M']
invarmass=total_dataset_df['q2']

# filter the data using the q^2 criteria from angular analysis paper page 7
# 0.98 < q2 < 1.10, 8 < q2 < 11, 12.5 < q2 < 15
total_dataset_df_yes_peaking = total_dataset_df[(invarmass > 0.98) & (invarmass < 1.10) | (invarmass > 8.0) & (invarmass < 11.0) | (invarmass > 12.5) & (invarmass < 15.0)]
df_new = total_dataset_df.merge(total_dataset_df_yes_peaking, how='left', indicator=True)
df_new = df_new[df_new['_merge'] == 'left_only'] # line 16-17 from stackoverflow.com

# histogram stuff
bin_height, bin_edges, patches = plt.hist(b0mass, bins=100,alpha=0)
midpoints = 0.5 * (bin_edges[1:] + bin_edges[:-1])
bin_height_1, bin_edges_1, patches_1 = plt.hist(df_new['B0_M'],bins=100,alpha=0)
midpoints_1 = 0.5 * (bin_edges_1[1:] + bin_edges_1[:-1])

# plot histogram things
plt.grid()
plt.xlabel('B0_M')
plt.ylabel('Frequency')
plt.plot(midpoints,bin_height,'-',color='royalblue',label='Total Signal')
plt.plot(midpoints_1,bin_height_1,'-',color='salmon', label = 'Peaking Signal')
plt.legend(loc='best')
#plt.show()