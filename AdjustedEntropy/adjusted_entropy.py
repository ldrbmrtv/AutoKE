import numpy as np
from scipy.stats import entropy
from math import log

def part(y):
    value, counts = np.unique(y, return_counts = True)
    return counts   

def inf(prob):
    return -log(prob, 2)

def entr(part):
    return entropy(part, base = 2)

def adj(part, mean_entr, max_entr):
    return (entr(part) - mean_entr)/(max_entr - mean_entr)

def adjusted_entropy(y):
    n = len(y)
    p = part(y)
    k = len(p)

    prob_max = (n-k+1)/n
    min_entr = prob_max*inf(prob_max) + (k-1)/n*inf(1/n)
    max_entr = inf(1/k)
    mean_entr = (max_entr + min_entr)/2
    return adj(p, mean_entr, max_entr)
