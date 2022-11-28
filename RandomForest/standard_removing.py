columns_to_remove = [
        "Unnamed: 0",
        "Unnamed: 0.1", 
        "Unnamed: 0.1.1",
        "year",
        'B0_ID',
        'B0_ENDVERTEX_NDOF',
        'J_psi_ENDVERTEX_NDOF',
        'Kstar_ENDVERTEX_NDOF' #Yoinked from Ganels
    ]

model_features = {
    "jpsi_mu_k_swap": ['mu_plus_ProbNNk', 'mu_plus_ProbNNmu', 'K_ProbNNk', 'K_ProbNNmu'],
    "jpsi_mu_pi_swap": ['mu_minus_ProbNNpi', 'mu_minus_ProbNNmu', 'Pi_ProbNNpi', 'Pi_ProbNNmu', 'costhetal'],
    "k_pi_swap": ['K_ProbNNk', 'K_ProbNNpi', 'Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNp'],
    "Kmumu": ['Pi_PT', 'Pi_PX', 'Pi_IPCHI2_OWNPV', 'B0_M', 'B0_ENDVERTEX_CHI2', 'B0_OWNPV_Y', 'costhetak'],
    "Kstarp_pi0": ['Pi_PT', 'Pi_PX', 'Pi_IPCHI2_OWNPV', 'B0_M', 'B0_ENDVERTEX_CHI2', 'Kstar_M', 'Kstar_ENDVERTEX_CHI2'],
    "phimumu": ['Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNp', 'Pi_PT', 'B0_M', 'Kstar_M', 'costhetak'],
    "pKmumu_piTok_kTop": ['K_ProbNNk', 'K_ProbNNp', 'Pi_ProbNNk', 'Pi_ProbNNpi', 'B0_M', 'B0_OWNPV_Y'],
    "pKmumu_piTop": ['Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNp', 'Pi_P', 'Pi_PE', 'B0_M', 'Kstar_M', 'B0_OWNPV_Y'],
    "total_dataset": ['mu_plus_ProbNNk', 'mu_plus_ProbNNmu', 'mu_plus_ProbNNe', 'mu_plus_ProbNNp', 'mu_minus_ProbNNk',
                'mu_minus_ProbNNe', 'mu_minus_ProbNNp', 'K_ProbNNk', 'K_ProbNNpi', 'K_ProbNNmu', 'K_ProbNNe',
                'Pi_ProbNNk', 'Pi_ProbNNpi', 'Pi_ProbNNmu', 'Pi_ProbNNe', 'Pi_ProbNNp', 'Pi_PT', 'Pi_IPCHI2_OWNPV',
                'B0_ENDVERTEX_CHI2', 'B0_PT', 'Kstar_M', 'J_psi_M', 'B0_IPCHI2_OWNPV', 'B0_DIRA_OWNPV', 'B0_OWNPV_X',
                'B0_OWNPV_Y', 'B0_ENDVERTEX_X', 'B0_ENDVERTEX_Y', 'q2']
}