import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


fn = 'Data/total_dataset.csv'

def compare_P(fn):
    #function that compares the sum of momentum of final particles with the momentum of B0
    '''
    input parameters:

    fn: filepath to excel file

    output:
    new dataframe with columns
    ['B0_PX','B0_PY','B0_PZ','PXsum_final_particles','PYsum_final_particles','PZsum_final_particles']

    '''
    df = pd.read_csv(fn)
    df['PXsum_final_particles'] = df['mu_plus_PX'] + df['mu_minus_PX'] + df['K_PX'] + df['Pi_PX'] 
    df['PYsum_final_particles'] = df['mu_plus_PY'] + df['mu_minus_PY'] + df['K_PY'] + df['Pi_PY']
    df['PZsum_final_particles'] = df['mu_plus_PZ'] + df['mu_minus_PZ'] + df['K_PZ'] + df['Pi_PZ']  

    new_df = df[
        ['B0_PX',
        'B0_PY',
        'B0_PZ',
        'PXsum_final_particles',
        'PYsum_final_particles',
        'PZsum_final_particles']
        ].copy()
    return new_df

def IP(fn):
    #calculate impact parameter, IP, 
    #which is (perpendicular distance) between path of reconstructed B0
    #(line passing through end vertex with velocity of reconstructed B0 as direction vector)
    #and the production vertex (B0_OWNPV X,Y,Z)

    #IP of 0 means that the sum of momentum of final particles can be used to
    #join the end vertex to the production vertex

    #large value of IP means that the sum of momentum of final particles cannot
    #join the end vertex to the production vertex, so the reconstructed track quality is
    #not high
    '''
    input parameters:

    fn: filepath to excel file

    output:
    new dataframe with columns
    ['B0_PX',
    'B0_PY',
    'B0_PZ',
    'PXsum_final_particles',
    'PYsum_final_particles',
    'PZsum_final_particles',
    'IP']

    '''

    df1 = pd.read_csv(fn)
    df2 = compare_P(fn)

    #reconstruct the velocity of imaginary B0 from the momentum of its final particles
    df2['rec_B0_Vx'] = (df2['PXsum_final_particles'])/df1['B0_M']
    df2['rec_B0_Vy'] = (df2['PYsum_final_particles'])/df1['B0_M']
    df2['rec_B0_Vz'] = (df2['PZsum_final_particles'])/df1['B0_M']

    #Velocity magnitude
    df2['rec_B0_V'] = np.sqrt(df2['rec_B0_Vx']**2 + df2['rec_B0_Vy']**2 + df2['rec_B0_Vz']**2)

    #velocity direction unit vector
    df2['u_Vx'] = df2['rec_B0_Vx']/df2['rec_B0_V']
    df2['u_Vy'] = df2['rec_B0_Vy']/df2['rec_B0_V']
    df2['u_Vz'] = df2['rec_B0_Vz']/df2['rec_B0_V']

    #to find the impact parameter,IP:

    #Find vector,AP, where A is end vertex and P is production vertex
    df2['AP_x'] = df1['B0_ENDVERTEX_X']-df1['B0_OWNPV_X']
    df2['AP_y'] = df1['B0_ENDVERTEX_Y']-df1['B0_OWNPV_Y']
    df2['AP_z'] = df1['B0_ENDVERTEX_Z']-df1['B0_OWNPV_Z']

    #cross product to find IP
    IP = []
    for i in range(len(df2['rec_B0_Vx'])):
        AP_vector = np.array([df2['AP_x'][i],df2['AP_y'][i],df2['AP_z'][i]])
        V_unit_vector = np.array([df2['u_Vx'][i],df2['u_Vy'][i],df2['u_Vz'][i]])
        cross = np.cross(AP_vector,V_unit_vector)
        IP_value = np.sqrt(np.dot(cross,cross))
        IP.append(IP_value)
    
    #create new df that includes IP
    new_df = df2[
        ['B0_PX',
        'B0_PY',
        'B0_PZ',
        'PXsum_final_particles',
        'PYsum_final_particles',
        'PZsum_final_particles']
        ].copy()
    
    new_df['IP'] = IP
    return new_df

print(IP(fn))