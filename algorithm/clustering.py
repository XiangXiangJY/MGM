# -*- coding: utf-8 -*-
"""
Updated on 2/29/2024 
@author: yutah
"""

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import warnings
from scipy.optimize import linear_sum_assignment

'''
This file is used for clustering and obtaining the ARI, NMI, Purity and ACC
'''


def computeKMeans(X, y, max_state = 10):
    #kmeans clustering. returns LABELS, Each row contains 1 random_state
    n_clusters = np.unique(y).shape[0]
    LABELS = np.zeros([max_state, X.shape[0]])
    for random_state in range(max_state):
        myKM = KMeans(n_clusters = n_clusters, random_state = random_state,  n_init = 10).fit(X)
        LABELS[random_state, :] = myKM.labels_
    return LABELS


def computeARI(LABELS, y):
    #compute the ari
    ARI = np.zeros(LABELS.shape[0])
    for random_state in range(LABELS.shape[0]):
        ARI[random_state] = adjusted_rand_score(y, LABELS[random_state, :])
    return np.mean(ARI)


def computeNMI(LABELS, y):
    #normalized mutual information
    NMI = np.zeros(LABELS.shape[0])
    for random_state in range(LABELS.shape[0]):
        NMI[random_state] = normalized_mutual_info_score(y, LABELS[random_state, :])
    return np.mean(NMI)

def computePurity(LABELS, y):
    #purity
    PURITY = np.zeros(LABELS.shape[0])
    for random_state in range(LABELS.shape[0]):
        cm = contingency_matrix(y, LABELS[random_state, :])
        PURITY[random_state] =  np.sum(np.max(cm, axis = 0)) / y.shape[0]
    return np.mean(PURITY)
    

def computeACC(LABELS, y):
    #align labels using hunarian algorithm, and compute the accuracy
    ACC = np.zeros(LABELS.shape[0])
    for random_state in range(LABELS.shape[0]):
        cm = contingency_matrix(y, LABELS[random_state, :])
        row, col = linear_sum_assignment(cm, maximize = True)
        ACC[random_state] =  np.sum( cm[row, col]) / y.shape[0]
    return np.mean(ACC)


def computeClusteringScore(X, y, max_state = 10):
    '''
        X: n_samples x n_components matrix
        y: true labels
        max_state: numer of iteration
    '''
    with warnings.catch_warnings(): #there is some memory leakage in the recent kmeans implementation on sklearn, so the warning is removed.
        warnings.simplefilter("ignore")
        LABELS = computeKMeans(X, y, max_state)
    ari = computeARI(LABELS, y)
    nmi = computeNMI(LABELS, y)
    purity = computePurity(LABELS, y)
    acc = computeACC(LABELS, y)
    return ari, nmi, purity, acc,  LABELS