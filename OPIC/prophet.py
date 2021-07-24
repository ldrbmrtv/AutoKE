from math import log
import numpy as np
from operator import itemgetter

def parts(n):
    a = [1]*n
    y = -1
    v = n
    while v > 0:
        v -= 1
        x = a[v] + 1
        while y >= 2 * x:
            a[v] = x
            y -= x
            v += 1
        w = v + 1
        while x <= y:
            a[v] = x
            a[w] = y
            yield a[:w + 1]
            x += 1
            y -= 1
        a[v] = x + y
        y = a[v] - 1
        yield a[:w]

def inf(part, n):

    arr = [log(x, n) for x in part]
    
    return np.mean(arr)

def lpic(n, cash = {}):

    file_name = 'mmry_lpic'
    
    if cash == {}:
        f = open(file_name + '.txt', 'r')
        for line in f.readlines():
            key, val = line.split(' ')
            cash[int(key)] = int(val)
        f.close()
        
    k = cash.get(n)

    if k == None:

        arr = []
        for i in range(1, n+1):
            arr.append([i, 0])

        _parts = parts(n)
        for part in _parts:
            arr[len(part)-1][1] += inf(part, n)
            
        arr.sort(key=itemgetter(1), reverse = True)
        k = arr[0][0]
        cash[n]=k
        f = open(file_name + '.txt', 'a+')
        f.write('{} {}\n'.format(n, k))

    return k
