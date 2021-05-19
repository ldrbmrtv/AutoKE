import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.stats import entropy
import matplotlib.pyplot  as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import time

def get_entr(arr, n):
    part = np.fromiter(Counter(arr).values(), dtype=float)
    return 1 - entropy(part, base=n)

def get_labels(data, grid, scale):

    e_list = []
    min_pts_list = []
    labels_list = []
    n = data.shape[0]
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
    for e in grid[0]:
        for min_pts in grid[1]:
            dbscan = DBSCAN(eps=e, min_samples = min_pts)
            dbscan.fit(data)
            labels = dbscan.labels_
            if set(labels) == {-1}:
                continue
            if set(labels) == {0}:
                break
            e_list.append(e)
            min_pts_list.append(min_pts)
            labels_list.append(tuple(labels))
    
    df = pd.DataFrame()
    df['e'] = e_list
    df['min_pts'] = min_pts_list
    df['labels'] = labels_list
    
    parts = df['labels'].unique()
    probs = []
    entrs = []
    for part in parts:
        by_part = df[df['labels']==part]
        prob = len(by_part.index)/n
        entr = get_entr(part, n)
        probs.append(prob)
        entrs.append(entr)

    df_parts = pd.DataFrame()
    df_parts['part'] = parts
    df_parts['prob'] = probs
    df_parts['entr'] = entrs
    df_parts['exp_entr'] = df_parts['prob']*df_parts['entr']
    
    part_opt = df_parts.sort_values('exp_entr', ascending=False).values[0][0]
    by_part = df[df['labels']==part_opt]
    by_part.sort_values(['e', 'min_pts'], inplace=True)
    e_opt = by_part.values[0][0]
    min_pts_opt = by_part.values[0][1]

    return e_opt, min_pts_opt, part_opt

def get_scores(data, labels_true, labels_pred):

    return metrics.accuracy_score(labels_true, labels_pred)

def visualize(data, labels_pred, name):
    pca = PCA(2)
    data2d = pca.fit_transform(data)
    plt.figure()
    plt.scatter(data2d[:,0], data2d[:,1], c=labels_pred, s=2)
    plt.savefig(name)

def main(data, labels_true, grid, name, scale):
    e_opt, min_pts_opt, labels_pred = get_labels(data, grid, scale)
    score = get_scores(data, labels_true, labels_pred)
    visualize(data, labels_true, name + '_true')
    visualize(data, labels_pred, name + '_pred')
    return str(e_opt), str(min_pts_opt), str(score)

def read_csv(file_name, grid, scale):
    df = pd.read_csv('datasets\\' + file_name + '.csv')
    df[df.columns[-1]] = df[df.columns[-1]].map({'o': -1, 'n': 0})
    labels_true = df[df.columns[-1]].values
    start = time.time()
    res = main(df, labels_true, grid, file_name, scale)
    end = time.time()
    f = open('results.txt', 'a')
    f.write(file_name + ', ' + ', '.join(res) + ', ' + str(end - start) + '\n')
    f.close()

grid = [np.arange(1, 1000, 1), np.arange(2, 10, 1)]
read_csv('pen-global-unsupervised-ad', grid, False)
read_csv('breast-cancer-unsupervised-ad', grid, False)
read_csv('letter-unsupervised-ad', grid, False)
