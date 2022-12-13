# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:30:14 2022

@author: lizzi
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches
import pandas as pd
from iminuit import Minuit
import scipy
from chebyshev_fitting import Data
from chebyshev_fitting import Accept
import json
from math import ceil

#%% Data:

#Change this to where acceptance is located
df_accept = pd.read_csv('../data/acceptance_mc.csv') 


#Change this to where the ml signal is located
df_data = pd.read_csv('ML_SIGNAL_BDT.csv')
# df_data = pd.read_csv('signal.csv')
#%% Acceptance:

acceptance = Accept(df_accept)
data = Data(df_data)
bins = []
bin_edges = ([[0.1,0.98], [1.1,2.5], [2.5,4.0], [4.0,6.0], [6.0,8.0], [15.0,17.0], [17.0,19.0], [11.0,12.5], [1.0,6.0], [15.0,17.9]])

def bin_q2(dataframe,edges, bins):
    '''
    This function bins given data according to the input bin edges.
    '''
    for i in range(len(edges)):
        df1 = dataframe[dataframe['q2']<edges[i][1]]
        df1 = df1[df1['q2']>edges[i][0]]
        bins.append(df1)
               
bin_q2(df_data, bin_edges, bins)

poly_l_list = []
poly_k_list = []
poly_p_list = []
poly_q_list = []

#Uncomment your choice of polynomial



for i in range(len(bin_edges)):
    b = bin_edges[i]
    poly_l = acceptance.chebyshev("costhetal", b[0], b[1], order = 4, name = "Data", plot_p = 0, plot_poly = 0)[1]
    poly_k = acceptance.chebyshev("costhetak", b[0], b[1], order = 4, name = "Data", plot_p = 0, plot_poly = 0)[1]
    poly_p = acceptance.chebyshev("phi", b[0], b[1], order = 4, name = "Data", plot_p = 0, plot_poly = 0)[1]
    poly_q = acceptance.chebyshevq(b[0], b[1], order = 5, name = "Data", plot_p = 0, plot_poly = 0)[1]
    poly_l_list.append(poly_l)
    poly_k_list.append(poly_k)
    poly_p_list.append(poly_p)
    poly_q_list.append(poly_q)

#%% Distributions: Theta - Phi

def ctl_dist(ctl,q2,fl,afb,_bin):
    c2tl = 2*ctl**2 - 1
    dist = 3/8 * (3/2 - 1/2*fl + 1/2*c2tl*(1-3*fl) + 8/3*afb*ctl)  
    
    poly_l = poly_l_list[int(_bin)]
    poly_q = poly_q_list[int(_bin)]

    acceptance_fn_l = poly_l[0] + poly_l[1]*ctl + poly_l[2]*ctl**2 + poly_l[3]*ctl**3 + poly_l[4]*ctl**4
    acceptance_fn_q = poly_q[0] + poly_q[1]*q2 + poly_q[2]*q2**2 + poly_q[3]*q2**3 + poly_q[4]*q2**4 + poly_q[5]*q2**5

    acceptance_fn = acceptance_fn_l*acceptance_fn_q
   
    normalised_dist = dist*acceptance_fn
    return normalised_dist

def ctk_dist(ctk,q2,fl,_bin):
    stk2 = 1-ctk**2
    ctk2 = ctk**2
    dist = 3/4 * ((1-fl)*stk2 + 2*fl*ctk2)
    
    poly_k = poly_k_list[int(_bin)]
    poly_q = poly_q_list[int(_bin)]
    
    acceptance_fn_k = poly_k[0] + poly_k[1]*ctk + poly_k[2]*ctk**2 + poly_k[3]*ctk**3 + poly_k[4]*ctk**4
    acceptance_fn_q = poly_q[0] + poly_q[1]*q2 + poly_q[2]*q2**2 + poly_q[3]*q2**3 + poly_q[4]*q2**4 + poly_q[5]*q2**5
    
    acceptance_fn = acceptance_fn_k*acceptance_fn_q
    normalised_dist = dist*acceptance_fn
    return normalised_dist
 
def phi_dist(phi,q2,s3,s9,_bin):
    dist = 1/(2*np.pi) * (1 + s3*np.cos(2*phi) + s9*np.sin(2*phi))
    
    poly_p = poly_p_list[int(_bin)]
    poly_q = poly_q_list[int(_bin)]

    acceptance_fn_p = poly_p[0] + poly_p[1]*phi + poly_p[2]*phi**2 + poly_p[3]*phi**3 + poly_p[4]*phi**4
    acceptance_fn_q = poly_q[0] + poly_q[1]*q2 + poly_q[2]*q2**2 + poly_q[3]*q2**3 + poly_q[4]*q2**4 + poly_q[5]*q2**5

    acceptance_fn = acceptance_fn_p*acceptance_fn_q
    normalised_dist = dist*acceptance_fn
    return normalised_dist

#%% Derived Quantities:
    
def Ps(Ss, FLs, P):
    for i in range(len(Ss)):
        P.append(Ss[i]/(np.sqrt(1-FLs[i])))

def Ps_E(Ss, FLs, Ss_E, FLs_E,P_E):
    for i in range(len(Ss)):
        A = ((Ss_E[i])/(np.sqrt(1-FLs[i])))
        B = (((Ss[i]*FLs_E[i])/2)*(1/(1-FLs[i])**(3/2)))
        P_E.append(np.sqrt(A**2 + B**2))

#%% Log-likelihood:

costhetal = data.angle("costhetal")
costhetak = data.angle("costhetak")
d_phi = data.angle("phi")
d_q2 = data.q2()


def log_likelihood(FL,AFB,S3,S4,S5,S7,S8,S9,_bin):
    bbin = bins[int(_bin)]
    ctl = bbin['costhetal']
    ctk = bbin['costhetak']
    phi = bbin['phi']
    q2 =  bbin["q2"]
    P = dist(ctk,ctl,phi,q2,FL,AFB,S3,S4,S5,S7,S8,S9,_bin)
    return - np.sum(np.log(P))

def decay_rate_eqn(ctk,ctl,phi,FL,AFB,S3,S4,S5,S7,S8,S9):
    ctk2 = ctk**2    
    ctl2 = ctl**2
    s2k = 2* np.sqrt(1-ctk2)*ctk
    s2l = 2* np.sqrt(1-ctl2)*ctl
    A = 0.75*(1-FL)*(1-ctk2) + FL*(ctk2) + 0.25*(1-FL)*(1-ctk2)*((2*ctl2)-1)
    B = -FL*(ctk2)*(2*ctl2-1) + S3*(1-ctk**2)*(1-ctl2)*np.cos(2*phi)
    C = S4*s2k*s2l*np.cos(phi) + S5*s2k*(np.sqrt(1-ctl**2))*np.cos(phi)
    D = (4/3)*AFB*(1-ctk2)*ctl + S7*s2k*(np.sqrt(1-ctl**2))*np.sin(phi)
    E = S8*s2k*s2l*np.sin(phi) + S9*(1-ctk2)*(1-ctl2)*np.sin(2*phi)
    F = (9/(32*np.pi))*(A+B+C+D+E)
    return F

def dist(ctk,ctl,phi,q2,FL,AFB,S3,S4,S5,S7,S8,S9,_bin):
    poly_l = poly_l_list[int(_bin)]
    poly_k = poly_k_list[int(_bin)]
    poly_p = poly_p_list[int(_bin)]
    poly_q = poly_q_list[int(_bin)]
    F = decay_rate_eqn(ctk,ctl,phi,FL,AFB,S3,S4,S5,S7,S8,S9)

    acceptance_fn_l = poly_l[0] + poly_l[1]*ctl + poly_l[2]*ctl**2 + poly_l[3]*ctl**3 + poly_l[4]*ctl**4
    acceptance_fn_k = poly_k[0] + poly_k[1]*ctk + poly_k[2]*ctk**2 + poly_k[3]*ctk**3 + poly_k[4]*ctk**4
    acceptance_fn_p = poly_p[0] + poly_p[1]*phi + poly_p[2]*phi**2 + poly_p[3]*phi**3 + poly_p[4]*phi**4
    acceptance_fn_q = poly_q[0] + poly_q[1]*q2 + poly_q[2]*q2**2 + poly_q[3]*q2**3 + poly_q[4]*q2**4 + poly_q[5]*q2**5
    
    acceptance_fn = acceptance_fn_l*acceptance_fn_k*acceptance_fn_p*acceptance_fn_q
    normalised_dist = (F*acceptance_fn)
    
    return normalised_dist


#%% Initial Values:

bin0 = [0.296448,-0.097052,0.010876,0.090919,0.252907,-0.020672,-0.002153,-0.000701]
bin1 = [0.760396,-0.137987,0.002373,-0.025359,0.054524,-0.027117,-0.006877,-0.000786]
bin2 = [0.796265,-0.017385,-0.010864,-0.151610,-0.193015,-0.019962,-0.006587,-0.000735]
bin3 = [0.711290,0.122155,-0.024751,-0.224204,-0.337140,-0.013383,-0.005062,-0.000706]
bin4 = [0.606965,0.239939,-0.039754,-0.259699,-0.403554,-0.008738,-0.003689,-0.000715]
bin5 = [0.348441,0.401914,-0.173464,-0.294319,-0.318728,-0.001377,0.000323,0.000292]
bin6 = [0.328081,0.318391,-0.251488,-0.310007,-0.226258,-0.000561,0.000119,0.000169]
bin7 = [0.435190,0.391390,-0.085975,-0.281589,-0.406803,-0.002194,0.001051,0.000449]
bin8 = [0.747644,0.004929,-0.012641,-0.142821,-0.176674,-0.019362,-0.006046,-0.000739]
bin9 = [0.340156,0.367672,-0.204963,-0.300427,-0.280936,-0.001039,0.000240, 0.000242]

st=[bin0,bin1,bin2,bin3,bin4,bin5,bin6,bin7,bin8,bin9]

#%% Minimisation:


bin_number_to_check = [0,1,2,3,4,5,6,7,8,9]
bin_results_to_check = None

log_likelihood.errordef = Minuit.LIKELIHOOD
decimal_places = 3
starting_point = st

FLs, FLs_E = [], []
S3s, S3s_E = [], []
S4s, S4s_E = [], []
S5s, S5s_E = [], []
AFBs, AFBs_E = [], []
S7s, S7s_E = [], []
S8s, S8s_E = [], []
S9s, S9s_E = [], []

for i in bin_number_to_check:
    print('Fitting for bin',i)
    m = Minuit(log_likelihood, FL=st[i][0], AFB=st[i][1], S3=st[i][2], S4=st[i][3], S5=st[i][4],S7=st[i][5], S8=st[i][6], S9=st[i][7],  _bin = i)
    m.fixed['_bin'] = True 
    m.limits=((-1.0, 1.0),(-1.0, 1.0),(-1.0, 1.0),(-1.0, 1.0),(-1.0, 1.0),(-1.0, 1.0),(-1.0, 1.0),(-1.0, 1.0), None)
    m.migrad() 
    m.hesse() 
    #if i == bin_number_to_check[i]:
    bin_results_to_check = m
    fig, ax = plt.subplots(3, 3, sharex='col', sharey='row')
    plt.subplot(331)
    bin_results_to_check.draw_mnprofile("FL", bound=3)
    plt.subplot(332)
    bin_results_to_check.draw_mnprofile('S3', bound=3)
    plt.subplot(333)
    bin_results_to_check.draw_mnprofile('S4', bound=3)
    plt.subplot(334)
    bin_results_to_check.draw_mnprofile('S5', bound=3)
    plt.subplot(335)
    bin_results_to_check.draw_mnprofile('AFB', bound=3)
    plt.subplot(336)
    bin_results_to_check.draw_mnprofile('S7', bound=3)
    plt.subplot(337)
    bin_results_to_check.draw_mnprofile('S8', bound=3)
    plt.subplot(338)
    bin_results_to_check.draw_mnprofile("S9", bound=3)
    # plt.tight_layout()
    plt.savefig(f'{i}_bin_observables.png',dpi=600)
    plt.show()
    
    FLs.append(m.values[0])
    AFBs.append(m.values[1])
    S3s.append(m.values[2])
    S4s.append(m.values[3])
    S5s.append(m.values[4])
    S7s.append(m.values[5])
    S8s.append(m.values[6])
    S9s.append(m.values[7])
    
    FLs_E.append(m.errors[0])
    AFBs_E.append(m.errors[1])
    S3s_E.append(m.errors[2])
    S4s_E.append(m.errors[3])
    S5s_E.append(m.errors[4])
    S7s_E.append(m.errors[5])
    S8s_E.append(m.errors[6])
    S9s_E.append(m.errors[7])
    
#%% Values:

print("FL:", FLs, "pm", FLs_E)
print("AFB:", AFBs, "pm", AFBs_E)
print("S3:",S3s, "pm", S3s_E)
print("S4:",S4s, "pm", S4s_E)
print("S5:",S5s, "pm", S5s_E)
print("S7:",S7s, "pm", S7s_E)
print("S8:",S8s, "pm", S8s_E)
print("S9:",S9s, "pm", S9s_E)
#%% P5

P5s = []
P5s_E = []

Ps(S5s,FLs, P5s)
Ps_E(S5s,FLs, S5s_E,FLs_E,P5s_E)

print("P5:", P5s, "pm", P5s_E)

#%% Comparison Code

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
            color='steelblue',         #set colour
            alpha=0.5, 
            #label="SM Prediction",#set transparency
            linewidth = 0)) 
        
        
q2_bin_middle = []
q2_bin_size = []
for i in range(len(bin_edges)):
    q2_bin_middle.append(np.mean(bin_edges[i])) #this is for plotting
    q2_bin_size.append(abs(bin_edges[i][0]-q2_bin_middle[i]))
    
si_preds = {}
# pi_preds = {} # if we need the pi.json
with open("std_predictions_si.json","r") as _f:
    si_preds = json.load(_f) #load the json file
#with open("Data/std_predictions_pi.json","r") as _f:
    #pi_preds = json.load(_f) #we aren't rly using this yet...
    
si_list = []
for _binNo in si_preds.keys():
    si_frame = pd.DataFrame(si_preds[_binNo]).transpose() #form a pandas dataframe
    si_list.append(si_frame) #if you want bin 0, call si_list[0]
    

fl_predict, fl_predict_err = [],[]
afb_predict, afb_predict_err = [],[]
s3_predict, s3_predict_err = [],[]
s4_predict, s4_predict_err = [],[]
s5_predict, s5_predict_err = [],[]
s7_predict, s7_predict_err = [],[]
s8_predict, s8_predict_err = [],[]
s9_predict, s9_predict_err = [],[]
p5_predict, p5_predict_err = [], []

for i in range(len(si_list)):
    fl_predict.append(si_list[i].loc['FL'][0])
    fl_predict_err.append(si_list[i].loc['FL'][1])
    afb_predict.append(si_list[i].loc['AFB'][0])
    afb_predict_err.append(si_list[i].loc['AFB'][1])
    s3_predict.append(si_list[i].loc['S3'][0])
    s3_predict_err.append(si_list[i].loc['S3'][1])
    s4_predict.append(si_list[i].loc['S4'][0])
    s4_predict_err.append(si_list[i].loc['S4'][1])
    s5_predict.append(si_list[i].loc['S5'][0])
    s5_predict_err.append(si_list[i].loc['S5'][1])
    s7_predict.append(si_list[i].loc['S7'][0])
    s7_predict_err.append(si_list[i].loc['S7'][1])
    s8_predict.append(si_list[i].loc['S8'][0])
    s8_predict_err.append(si_list[i].loc['S8'][1])
    s9_predict.append(si_list[i].loc['S9'][0])
    s9_predict_err.append(si_list[i].loc['S9'][1])
    
Ps(s5_predict, fl_predict, p5_predict)
Ps_E(s5_predict,fl_predict,s5_predict_err,fl_predict_err,p5_predict_err)
#%% Comparison 9x9

FLs_c = pd.read_csv('Bootstrap_confidence/FLs.csv') 
AFBs_c = pd.read_csv('Bootstrap_confidence/AFBs.csv') 
S3s_c = pd.read_csv('Bootstrap_confidence/S3s.csv') 
S4s_c = pd.read_csv('Bootstrap_confidence/S4s.csv') 
S5s_c = pd.read_csv('Bootstrap_confidence/S5s.csv') 
S7s_c = pd.read_csv('Bootstrap_confidence/S7s.csv') 
S8s_c = pd.read_csv('Bootstrap_confidence/S8s.csv') 
S9s_c = pd.read_csv('Bootstrap_confidence/S9s.csv') 

FLs_E = [[],[]]
AFBs_E = [[],[]]
S3s_E = [[],[]]
S4s_E = [[],[]]
S5s_E = [[],[]]
S7s_E = [[],[]]
S8s_E = [[],[]]
S9s_E = [[],[]]

for id, col in enumerate(FLs_c.columns.to_list()):
    arr = FLs_c[col][0].split('/')
    FLs_E[0].append(float(FLs[id]) - float(arr[0]))
    FLs_E[1].append(float(arr[1])- float(FLs[id]))

for id, col in enumerate(AFBs_c.columns.to_list()):
    arr = AFBs_c[col][0].split('/')
    AFBs_E[0].append(float(AFBs[id]) - float(arr[0]))
    AFBs_E[1].append(float(arr[1])- float(AFBs[id]))

for id, col in enumerate(S3s_c.columns.to_list()):
    arr = S3s_c[col][0].split('/')
    S3s_E[0].append(float(S3s[id]) - float(arr[0]))
    S3s_E[1].append(float(arr[1])- float(S3s[id]))

for id, col in enumerate(S4s_c.columns.to_list()):
    arr = S4s_c[col][0].split('/')
    S4s_E[0].append(float(S4s[id]) - float(arr[0]))
    S4s_E[1].append(float(arr[1])- float(S4s[id]))

for id, col in enumerate(S5s_c.columns.to_list()):
    arr = S5s_c[col][0].split('/')
    S5s_E[0].append(float(S5s[id]) - float(arr[0]))
    S5s_E[1].append(float(arr[1])- float(S5s[id]))

for id, col in enumerate(S7s_c.columns.to_list()):
    arr = S7s_c[col][0].split('/')
    S7s_E[0].append(float(S7s[id]) - float(arr[0]))
    S7s_E[1].append(float(arr[1])- float(S7s[id]))

for id, col in enumerate(S8s_c.columns.to_list()):
    arr = S8s_c[col][0].split('/')
    S8s_E[0].append(float(S8s[id]) - float(arr[0]))
    S8s_E[1].append(float(arr[1])- float(S8s[id]))

for id, col in enumerate(S9s_c.columns.to_list()):
    arr = S9s_c[col][0].split('/')
    S9s_E[0].append(float(S9s[id]) - float(arr[0]))
    S9s_E[1].append(float(arr[1])- float(S9s[id]))


fig, ((ax1, ax2,ax3),(ax4,ax5,ax6),(ax7,ax8, ax9)) = plt.subplots(3, 3)
ax1.errorbar(q2_bin_middle, FLs, xerr = q2_bin_size, yerr=FLs_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$F_L$', color='black')
error_rectangle(ax1,q2_bin_middle,fl_predict,q2_bin_size,fl_predict_err) #plots predicted values
ax2.errorbar(q2_bin_middle, AFBs, xerr = q2_bin_size, yerr=AFBs_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax2,q2_bin_middle,afb_predict,q2_bin_size,afb_predict_err) #plots predicted values
ax3.errorbar(q2_bin_middle, S3s, xerr = q2_bin_size, yerr=S3s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax3,q2_bin_middle,s3_predict,q2_bin_size,s3_predict_err) #plots predicted values
ax4.errorbar(q2_bin_middle, S4s, xerr = q2_bin_size, yerr=S4s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax4,q2_bin_middle,s4_predict,q2_bin_size,s4_predict_err) #plots predicted values
ax5.errorbar(q2_bin_middle, S5s, xerr = q2_bin_size, yerr=S5s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax5,q2_bin_middle,s5_predict,q2_bin_size,s5_predict_err) #plots predicted values
ax6.errorbar(q2_bin_middle, S7s, xerr = q2_bin_size, yerr=S7s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax6,q2_bin_middle,s7_predict,q2_bin_size,s7_predict_err) #plots predicted values
ax7.errorbar(q2_bin_middle, S8s, xerr = q2_bin_size, yerr=S8s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax7,q2_bin_middle,s8_predict,q2_bin_size,s8_predict_err) #plots predicted values
ax8.errorbar(q2_bin_middle, S9s, xerr = q2_bin_size, yerr=S9s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax8,q2_bin_middle,s9_predict,q2_bin_size,s9_predict_err) #plots predicted values
ax9.errorbar(q2_bin_middle, P5s, xerr = q2_bin_size, yerr=P5s_E, fmt='o', markersize=2, capsize = 2, elinewidth=0.5, label=r'$A_{FB}$', color='black')
error_rectangle(ax9,q2_bin_middle,p5_predict,q2_bin_size,p5_predict_err) #plots predicted values


ax1.set_xlim(0,20)
ax2.set_xlim(0,20)
ax3.set_xlim(0,20)
ax4.set_xlim(0,20)
ax5.set_xlim(0,20)
ax6.set_xlim(0,20)
ax7.set_xlim(0,20)
ax8.set_xlim(0,20)

ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
ax7.grid()
ax8.grid()

ax1.set_ylabel('$F_L$')
ax2.set_ylabel('$A_{FB}$')
ax3.set_ylabel('$S_3$')
ax4.set_ylabel('$S_4$')
ax5.set_ylabel('$S_5$')
ax6.set_ylabel('$S_7$')
ax7.set_ylabel('$S_8$')
ax8.set_ylabel('$S_9$')
ax9.set_ylabel('$P_5^{´}$')


plt.tight_layout()
plt.savefig(f'{i}_bin_observables.png',dpi=600)

plt.show()
#%% Comparison Individual

OBS = [FLs, AFBs, S3s, S4s, S5s, S7s, S8s, S9s, P5s]
OBS_E = [FLs_E, AFBs_E, S3s_E, S4s_E, S5s_E, S7s_E, S8s_E, S9s_E, P5s_E]

PR = [fl_predict, afb_predict, s3_predict, s4_predict,s5_predict,s7_predict,s8_predict,s9_predict,p5_predict]
PR_E = [fl_predict_err, afb_predict_err, s3_predict_err, s4_predict_err,s5_predict_err,s7_predict_err,s8_predict_err,s9_predict_err,p5_predict_err]

names = ['$F_L$', '$A_{FB}$', '$S_3$','$S_4$','$S_5$','$S_7$','$S_8$','$S_9$','$P_5^{´}$']

for i in range(len(OBS)):
    fig, ax1 = plt.subplots(1, 1)
    ax1.errorbar(q2_bin_middle, OBS[i], xerr = q2_bin_size, yerr=OBS_E[i], fmt='o', markersize=2, capsize = 2, elinewidth=0.5, color='black')
    error_rectangle(ax1,q2_bin_middle,PR[i],q2_bin_size,PR_E[i]) #plots predicted values
    leg = matplotlib.patches.Patch(color='steelblue', label='SM Predictions',alpha=0.5)
    plt.legend(handles=[leg])
    ax1.grid()
    ax1.set_xlim(0,20)
    ax1.set_ylabel(names[i])

    plt.tight_layout()
    plt.savefig(f'{names[i]}_observables.png',dpi=600)
    plt.show()
#%% STD

# def STD(val1,val2,err,unc,S):
#     for i in range(len(val1)):
#         A = abs(val1[i]-(val2[i]+unc[i]))
#         B = abs(val1[i]-(val2[i]-unc[i]))
#         S.append(min(A/err[i], B/err[i]))
    
# STD_FL = []
# STD_AFB = []
# STD_S3 = []
# STD_S4 = []
# STD_S5 = []
# STD_S7 = []
# STD_S8 = []
# STD_S9 = []
# STD_P5 = []

# STDs = [STD_FL, STD_AFB, STD_S3,STD_S4,STD_S5,STD_S7,STD_S8,STD_S9,STD_P5]
# for i in range(len(OBS)):
#     STD(OBS[i],PR[i],OBS_E[i],PR_E[i],STDs[i])
    

#%% Theta L
bin_to_plot = [0,1,2,3,4,5,6,7,8,9]    
print("θl:")
for i in range(len(bin_to_plot)):
    ctl_bin = bins[bin_to_plot[i]]['costhetal']
    q2ctl = bins[bin_to_plot[i]]["q2"]
    number_of_bins_in_hist = int(1 + ceil(3.322 * np.log(len(ctl_bin))/np.log(2)))
    hist, _bins, _ = plt.hist(ctl_bin, bins=number_of_bins_in_hist, density=1,histtype="step")
    x = np.linspace(-1, 1, number_of_bins_in_hist)
    x2 = np.linspace(bin_edges[i][0], bin_edges[i][1], number_of_bins_in_hist)
    pdf_multiplier = np.sum(hist) * (np.max(ctl_bin) - np.min(ctl_bin)) / number_of_bins_in_hist
    y = ctl_dist(fl=FLs[bin_to_plot[i]], afb = AFBs[bin_to_plot[i]], ctl=x, _bin=i, q2=x2) * pdf_multiplier
    plt.plot(x, y, label=f'Fit for bin {bin_to_plot[i]}')
    plt.xlabel(r'$cos\theta_l$')
    plt.ylabel(r'Number of candidates')
    plt.legend()
    plt.grid()
    plt.show()

    O = hist
    E = y
    chi_square =(sum((O-E)**2/E))/(len(O)-1)
    a,b = scipy.stats.ttest_ind(O,E)
    print("Red-χ2:", round(chi_square, 2),"Bin:", i)

 #%% Theta K
bin_to_plot = [0,1,2,3,4,5,6,7,8,9]


print("θk:")
for i in range(len(bin_to_plot)):
    ctk_bin = bins[bin_to_plot[i]]["costhetak"]
    number_of_bins_in_hist = int(1 + ceil(3.322 * np.log(len(ctk_bin))/np.log(2)))
    hist, _bins, _ = plt.hist(ctk_bin, bins=number_of_bins_in_hist, density=1, histtype="step")
    x = np.linspace(-1, 1, number_of_bins_in_hist)
    x2 = np.linspace(bin_edges[i][0], bin_edges[i][1], number_of_bins_in_hist)
    pdf_multiplier = np.sum(hist) * (np.max(ctk_bin) - np.min(ctk_bin)) / number_of_bins_in_hist
    y = ctk_dist(fl=FLs[bin_to_plot[i]], ctk=x,_bin=i,q2=x2) * pdf_multiplier
    plt.plot(x, y, label=f'Fit for bin {bin_to_plot[i]}')
    plt.xlabel(r'$cos\theta_l$')
    plt.ylabel(r'Number of candidates')
    plt.legend()
    plt.grid()
    plt.show()

    O = hist
    E = y
    chi_square = (sum((O-E)**2/E))/(len(O)-1)
    a,b = scipy.stats.ttest_ind(O,E)
    print("Red-χ2:", round(chi_square, 2),"Bin:", i)

#%% Phi
bin_to_plot = [0,1,2,3,4,5,6,7,8,9]


print("ϕ:")
for i in range(len(bin_to_plot)):
    phi_bin = bins[bin_to_plot[i]]['phi']
    number_of_bins_in_hist = int(1 + ceil(3.322 * np.log(len(phi_bin))/np.log(2)))
    hist, _bins, _ = plt.hist(phi_bin, bins=number_of_bins_in_hist , density=1, histtype="step")
    x = np.linspace(-np.pi, np.pi, number_of_bins_in_hist)
    x2 = np.linspace(bin_edges[i][0], bin_edges[i][1], number_of_bins_in_hist)
    pdf_multiplier = np.sum(hist) * (np.max(phi_bin) - np.min(phi_bin)) / number_of_bins_in_hist
    y = phi_dist(s3=S3s[bin_to_plot[i]], s9=S9s[bin_to_plot[i]], phi=x,_bin=i,q2=x2) * pdf_multiplier
    plt.plot(x, y, label=f'Fit for bin {bin_to_plot[i]}')
    plt.xlabel(r'phi')
    plt.ylabel(r'Number of candidates')
    plt.legend()
    plt.grid()
    plt.show()
    
    O = hist
    E = y
    chi_square = (sum((O-E)**2/E))/(len(O)-1)
    a,b = scipy.stats.ttest_ind(O,E)
    print("Red-χ2:", round(chi_square, 2),"Bin:", i)


