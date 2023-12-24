import numpy as np
from tqdm import tqdm

def positive_correction(pos_hsm_pairs, pos_thres_1, pos_thres_2):
    pos_pseudo = []
    for hsm in tqdm(pos_hsm_pairs):
        if hsm > pos_thres_2:
            pseudo = 0
        elif pos_thres_1 < hsm <= pos_thres_2:
            pseudo = (hsm - pos_thres_1) / (pos_thres_2 - pos_thres_1) 
        elif hsm <= pos_thres_1:
            pseudo = 1
        pos_pseudo.append(pseudo)
    pos_pseudo = np.array(pos_pseudo)
    return pos_pseudo
    
def negative_correction(neg_hsm_pairs, neg_thres_1, neg_thres_2):
    neg_pseudo = []
    for hsm in tqdm(neg_hsm_pairs):
        if hsm > neg_thres_2:
            pseudo = 1
        elif neg_thres_1 < hsm <= neg_thres_2:
            pseudo = (hsm - neg_thres_1) / (neg_thres_2 - neg_thres_1)
        elif hsm <= neg_thres_1:
            pseudo = 0
        neg_pseudo.append(pseudo)
    neg_pseudo = np.array(neg_pseudo)
    return neg_pseudo