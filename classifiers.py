import numpy as np
import signal_feature_extraction as sfe
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

n = 100
N = 100

#for weights see xgb_diagram
xgb_weights = [3,16,4,1,10]
xgb_weights = [x / sum(xgb_weights) for x in xgb_weights]
weights = [1,1,1,1,1]


def KNN(n,X,X_test,weights):
    for i in range(0, len(X)):
        X[i] = [a * b for a, b in zip(X[i], weights)]
        X_test[i] = [a * b for a, b in zip(X_test[i], weights)]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X_test)
    return indices


def Accuracy(N,weights):
    SNRs = []
    error_rates = []
    for i in range(0,N):
        error_rate = 0
        print(i)
        n = (i+1)*25
        e_ts, fs = sfe.random_fixture_sample(N)
        e_ts_noisy = sfe.noiser(e_ts, n)
        fs_noisy = sfe.fixture_sampler(e_ts_noisy)
        X = fs
        X_test = fs_noisy
        indices = KNN(N,X,X_test,weights)
        for j in range(0,len(indices)):
            bool = (indices[j][0] != j)
            error_rate = error_rate + int(bool)
        S = np.mean(np.abs(e_ts))
        SNRs.append(np.log10(S) / np.log10(n))
        error_rates.append(error_rate/100)
    if weights == weights:
        label = "Unweighted KNN"
    else:
        label = "Weighted KNN"
    plt.plot(SNRs,error_rates,"o",label = label)
    plt.ylabel("Error rate")
    plt.xlabel("SNR")
    plt.title("KNN Accuracy")

Accuracy(100,xgb_weights)
Accuracy(100,weights)
plt.show()