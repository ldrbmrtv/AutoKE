from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sil, davies_bouldin_score as db
from sklearn.metrics import fowlkes_mallows_score as fm
import numpy as np
from adjusted_entropy import adjusted_entropy as ae
import matplotlib.pyplot as plt

def cluster(X, k):
    model = KMeans(n_clusters=k, random_state=0).fit(X)
    return model.labels_
    
def get_best_k(k_list, scores, minmax=max):
    if max:
        ind = np.argmax(scores)
    else:
        ind = np.argmin(scores)
    best_k = k_list[ind]
    return best_k

def evaluate(model, k_list, scores, score, score_name, minmax=max):
    labels = cluster(X, get_best_k(k_list, scores, minmax))
    model[score_name] = score(y, labels)

def cluster_and_evaluate(models, X, y, score):
    n = len(X)
    k_list = range(2, n)
    sil_list = []
    db_list = []
    ae_list = []
    for k in k_list:
        labels = cluster(X, k)
        sil_list.append(sil(X, labels))
        db_list.append(db(X, labels))
        ae_list.append(ae(labels))
        
    model = {}
    evaluate(model, k_list, sil_list, score, 'sil')
    evaluate(model, k_list, db_list, score, 'db', min)
    evaluate(model, k_list, ae_list, score, 'ae')
    models.append(model)

def present(models, k_list, score_name):
    score = [x[score_name] for x in models]
    print(np.mean(score))
    plt.plot(k_list, score, label=score_name)
    
n = 150
max_k = 16
models = []
k_list = range(2, max_k)
for k in k_list:
    print(k)
    X, y = make_blobs(n_samples=n, n_features=3, centers=k, random_state=0)
    cluster_and_evaluate(models, X, y, fm)

present(models, k_list, 'sil')
present(models, k_list, 'db')
present(models, k_list, 'ae')

plt.legend()
plt.show()
