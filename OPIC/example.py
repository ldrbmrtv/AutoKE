from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
import time
import prophet
    
if __name__ == '__main__':

    n = 100
    d = 4
    k = 10
    
    data, target = make_blobs(n, d, k)
    
    start = time.time()
    k_inf = prophet.lpic(n)
    d_time = time.time() - start
    model = KMeans(n_clusters=k_inf).fit(data)
    labels = model.labels_
    sil = metrics.silhouette_score(data, labels)
    fm = metrics.fowlkes_mallows_score(target, labels)
    
    print('Time = {}'.format(d_time))
    print('Silhouette = {}'.format(sil))
    print('Fowlkes-Mallows = {}'.format(fm))
