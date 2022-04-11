import numpy as np
import antropy as ant
import deconvolution_local as dcl

#used in random signal function
sr = 1e12
ts = 1000 #space units
lim = 1e-8 #limit of sample
t = np.linspace(0,lim,ts)
tau = 1e-11 #sample spacing (lim/ts)
n_samp = len(t)

def sevcik(data):
    max = np.max(data)
    min = np.min(data)
    x_i = []
    L = 0
    for i in range(0,len(data)):
        x_i.append((data[i]-min)/(max-min))
    for i in range(0,len(data)-1):
        L = L + x_i[i+1] - x_i[i]
    return L

def feature_extraction(data):
    return [ant.petrosian_fd(data), ant.katz_fd(data), ant.higuchi_fd(data), ant.detrended_fluctuation(data), sevcik(data)]

def random_fixture_sample(n):
    e_ts = []
    fs = []
    for i in range(0, n):
        e_ts.append(dcl.rts(t))
        fs.append(feature_extraction(e_ts[-1]))

    return e_ts, fs

def noiser(e_ts, k):
    for i in range(0,len(e_ts)):
        e_ts[i] = e_ts[i] + k*np.random.normal(0,1,len(e_ts[i]))
    return e_ts


def fixture_sampler(e_ts):
    fs = []
    for i in range(0,len(e_ts)):
        fs.append(feature_extraction(e_ts[i]))
    return fs

