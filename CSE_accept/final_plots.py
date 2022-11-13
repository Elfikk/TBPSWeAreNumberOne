import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd
import matplotlib.pyplot as pl
import matplotlib.patches

def error_rectangle(ax,x,y,width,height):
    #function to plot error bars as solid coloured blocks
    #ax is the figure to plot in
    #x is the list of x values
    #y is the list of y values
    #width is the list of q2 bin size
    #height is the list of error in angular observables
    for i in range(len(x)):
        ax.add_patch(matplotlib.patches.Rectangle(
            (x[i]-width[i], y[i]-height[i]), # position of the centre of rectangle
            width[i]*2,        # shows the q2 bin range
            height[i]*2,        # shows the error in angular observable
            color='m',         #set colour
            alpha=0.5,          #set transparency
            linewidth = 0))     #remove opaque border

q2_bin_edges = ([[0.1,0.98],
                 [1.1,2.5], 
                 [2.5,4.0], 
                 [4.0,6.0], 
                 [6.0,8.0], 
                 [15.0,17.0], 
                 [17.0,19.0], 
                 [11.0,12.5], 
                 [1.0,6.0], 
                 [15.0,17.9]])

q2_bin_middle = []
q2_bin_size = []
for i in range(len(q2_bin_edges)):
    q2_bin_middle.append(np.mean(q2_bin_edges[i])) #this is for plotting
    q2_bin_size.append(abs(q2_bin_edges[i][0]-q2_bin_middle[i]))

si_preds = {}
pi_preds = {}
with open("Data/std_predictions_si.json","r") as _f:
    si_preds = json.load(_f) #load the json file
with open("Data/std_predictions_pi.json","r") as _f:
    pi_preds = json.load(_f) #we aren't rly using this yet...
    
si_list = []
for _binNo in si_preds.keys():
    si_frame = pd.DataFrame(si_preds[_binNo]).transpose() #form a pandas dataframe
    si_list.append(si_frame) #if you want bin 0, use si_list[0]

#sanity checks
#print(si_list[0].loc['FL'][0]) #get value of FL from bin 0
#print(si_list[0].loc['FL'][1]) #get value of FL error from bin 0
#print(si_list[1].loc['FL'][0]) #get value of FL from bin 1

fl_predict = [] #list of FL values, in order of bin number
fl_predict_err =[]
afb_predict = [] #list of AFB values, in order of bin number
afb_predict_err = []

for i in range(len(si_list)):
    fl_predict.append(si_list[i].loc['FL'][0])
    fl_predict_err.append(si_list[i].loc['FL'][1])
    afb_predict.append(si_list[i].loc['AFB'][0])
    afb_predict_err.append(si_list[i].loc['AFB'][1])

def get_predicitons(filepath,
                    q2_bin_edges
                    fl=False,
                    afb=False,
                    s1=False,
                    s2=False,
                    s3=False,
                    s4=False,
                    s5=False,
                    s6=False,
                    s7=False,
                    s8=False):
    q2_bin_middle = []
    q2_bin_size = []
    for i in range(len(q2_bin_edges)):
        q2_bin_middle.append(np.mean(q2_bin_edges[i])) #this is for plotting
        q2_bin_size.append(abs(q2_bin_edges[i][0]-q2_bin_middle[i]))
    si_preds = {} #variable to store the json information in
    with open("Data/std_predictions_si.json","r") as _f:
        si_preds = json.load(_f) #load the json file
    si_list = []
    for _binNo in si_preds.keys():
        si_frame = pd.DataFrame(si_preds[_binNo]).transpose() #form a pandas dataframe
        si_list.append(si_frame) #if you want bin 0, use si_list[0]
    
angular_observable_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.set_xlim(0,20)
ax2.set_xlim(0,20)
ax1.set_ylim(0,1) #according to 2015 paper
ax2.set_ylim(-0.6,0.6) #according for 2015 paper
error_rectangle(ax1,q2_bin_middle,fl_predict,q2_bin_size,fl_predict_err)
error_rectangle(ax2,q2_bin_middle,afb_predict,q2_bin_size,afb_predict_err)
#ax1.errorbar(q2_bin_middle, fls, yerr=fls_err, fmt='None', markersize=2, label='$F_L$', color='black') #to plot our actual FL values
#ax2.errorbar(q2_bin_middle, afbs, yerr=afbs_err, fmt='None', markersize=2, label='$A_{FB}$', color='black') #to plot our actual AFB values
ax1.set_ylabel('$F_L$')
ax2.set_ylabel('$A_{FB}$')
ax1.set_xlabel('$q^2 [GeV^2/c^4]$')
ax2.set_xlabel('$q^2 [GeV^2/c^4]$')
plt.tight_layout()
plt.savefig('Data/prediction_plot.png',dpi=600) #save the figure
plt.show()

def plot_angular_observables(q2_bin_edges,observable,y,yerr,fl=False,afb=True):
    '''
    q2_bin_edges: (list) list of bin edges
    observable: (str) name of observable in capitals
    y: (list) list of observables from negative log likelihood fit
    y_err: (list) list of errors for the observables from negative log likelihood fit
    flags: if true, then plot for a specific observable
    '''
    q2_bin_middle = []
    q2_bin_size = []
    for i in range(len(q2_bin_edges)):
        q2_bin_middle.append(np.mean(q2_bin_edges[i])) #this is for plotting
        q2_bin_size.append(abs(q2_bin_edges[i][0]-q2_bin_middle[i]))
    si_preds = {}
    with open("Data/std_predictions_si.json","r") as _f:
        si_preds = json.load(_f) #load the json file
    si_list = []
    for _binNo in si_preds.keys():
        si_frame = pd.DataFrame(si_preds[_binNo]).transpose() #form a pandas dataframe
        si_list.append(si_frame) #if you want bin 0, use si_list[0]
    predict = [] #list of FL values, in order of bin number
    predict_err =[]
    for i in range(len(si_list)):
        predict.append(si_list[i].loc[observable][0])
        predict_err.append(si_list[i].loc[observable][1])
    