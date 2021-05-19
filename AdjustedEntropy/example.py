from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score as sil, davies_bouldin_score as db
from sklearn.metrics import fowlkes_mallows_score
import numpy as np
from adj_entr import adj_entr
import matplotlib.pyplot as plt

def cluster(X, k):
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    return model.labels_
    
def get_best_k(k_list, score_list, minmax=max):
    if max:
        ind = np.argmax(score_list)
    else:
        ind = np.argmin(score_list)
    best_k = k_list[ind]
    return best_k

def cluster_and_evaluate(models, X, y, data_name, acc_metric):
    n = len(X)
    k_list = []
    sil_list = []
    db_list = []
    ae_list = []
    for k in range(2, n):
        labels = cluster(X, k)
        k_list.append(k)
        sil_list.append(sil(X, labels))
        db_list.append(db(X, labels))
        ae_list.append(adj_entr(labels))
        
    model = {}
    
    labels = cluster(X, get_best_k(k_list, sil_list))
    model['sil_acc'] = acc_metric(y, labels)
    
    labels = cluster(X, get_best_k(k_list, db_list))
    model['db_acc'] = acc_metric(y, labels)
    
    labels = cluster(X, get_best_k(k_list, ae_list))
    model['ae_acc'] = acc_metric(y, labels)

    models.append(model)

n = 150
models = []
k_list = []
for k in range(2, 16):
    print(k)
    k_list.append(k)
    X, y = make_blobs(n_samples=n, n_features=3, centers=k, random_state=0)
    cluster_and_evaluate(models, X, y, 'blobs', fowlkes_mallows_score)

sil_acc = [x['sil_acc'] for x in models]
db_acc = [x['db_acc'] for x in models]
ae_acc = [x['ae_acc'] for x in models]

print(np.mean(sil_acc))
print(np.mean(db_acc))
print(np.mean(ae_acc))

plt.plot(k_list, sil_acc, label='sil')
plt.plot(k_list, db_acc, label='db')
plt.plot(k_list, ae_acc, label='ae')
plt.legend()
plt.show()
